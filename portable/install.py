"""chordgen ポータブル版のセットアップ（Python で書く部分）。

run.bat が環境変数をアプリのフォルダに向け、bootstrap.ps1 で runtime/python と runtime/node を置いたあと、
同梱の Python でこのファイルを実行する。手順は次のとおり。

1. GPU を見分けて、合う PyTorch を入れる（版と取得元は sources.json の torch.profiles）
   - AMD: ROCm 10.2.0 / torch 2.15.0（既定）。`--rocm 7.2.1` で ROCm 7.2.1 / torch 2.9.1 に戻せる
   - NVIDIA: CUDA 13.2 / torch 2.15.0（既定）。`--cuda 12.6` で古いドライバ向けの版
   - CPU
2. torch を固定する constraints を作り、残りの依存（pyproject.toml と requirements-*.txt）を入れる
3. media-dl を npm で入れ、`media-dl setup` で yt-dlp・ffmpeg・ffprobe・deno を MEDIA_DL_BIN_DIR に入れる
4. torch.cuda.is_available() を確かめて表示し、runtime/setup-complete.json を書く

冪等: 済んだ手順は runtime/setup-state.json を見て飛ばす。途中で失敗しても、もう一度実行すれば続きから進む。
GPU 名の判定などの純粋な関数は tests/test_portable_install.py で確かめる。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PORTABLE_DIR = Path(__file__).resolve().parent
APP_DIR = PORTABLE_DIR.parent

BACKENDS = ("amd", "nvidia", "cpu")
# 引数の代わりに使える環境変数
ENV_GPU = "CHORDGEN_GPU"  # auto / amd / nvidia / cpu
ENV_AMD_DEVICE = "CHORDGEN_AMD_DEVICE"  # gfx1201 など
ENV_ROCM = "CHORDGEN_ROCM"  # 10.2.0 / 7.2.1
ENV_CUDA = "CHORDGEN_CUDA"  # 13.2 / 12.6
ENV_MEDIA_DL_TGZ = "CHORDGEN_MEDIA_DL_TGZ"  # package.json の URL の代わりに使う tarball（URL かファイル）

ROCM_VERSIONS = {"10.2.0": "rocm-10.2.0", "7.2.1": "rocm-7.2.1"}
CUDA_VERSIONS = {"13.2": "cu132", "12.6": "cu126"}

# 上書きさせない（constraints で固定し、GPU を替えるときに外す）パッケージ
PINNED_PATTERN = re.compile(r"^(torch|torchaudio|torchvision|rocm|rocm-.+|amd-torch.*)$")
MEDIA_DL_PACKAGE = "@liasvistreamgamemusic-sketch/media-dl"
MEDIA_DL_BINARIES = ("yt-dlp", "ffmpeg", "ffprobe", "deno")
NODEPS_REQUIREMENTS = "requirements-nodeps.txt"


class SetupError(Exception):
    """利用者に見せて止める失敗。message はそのまま表示する。"""


# ---------------------------------------------------------------------------
# GPU の判定（純粋な関数）
# ---------------------------------------------------------------------------

# Win32_VideoController の Name → gfx ターゲット。AMD の PyTorch のインストール手順（ROCm 10.2.0）の
# 「GPU と device extra」の表と、ROCm 10.2.0 の互換表（Windows は gfx1030 も対象）から。
# 3 つ目は内蔵 GPU か（NVIDIA と同居するノート PC で NVIDIA を選ぶため）。
_AMD_GFX_TABLE: tuple[tuple[str, str, bool], ...] = (
    (r"\bRX\s*9070\b|\bAI\s*PRO\s*R9[67]00", "gfx1201", False),  # RX 9070 / 9070 XT / 9070 GRE、AI PRO R9700(S) / R9600(D)
    (r"\bRX\s*90[56]0\b", "gfx1200", False),  # RX 9060 XT / 9060 / 9050
    (r"\bRX\s*7900\b|\bPRO\s*W7[89]00\b", "gfx1100", False),  # RX 7900 XTX / XT / GRE、PRO W7900 / W7800
    (r"\bRX\s*7[78]00\b|\bPRO\s*W7700\b|\bPRO\s*V710\b", "gfx1101", False),  # RX 7800 XT / 7700 XT / 7700、W7700
    (r"\bRX\s*7600\b", "gfx1102", False),  # RX 7600 / 7600 XT
    (r"\bRX\s*6(800|900|950)\b|\bPRO\s*W6800\b", "gfx1030", False),  # RX 6800 / 6900 / 6950（互換表の gfx1030）
    (r"\b(8060S|8050S|8040S)\b", "gfx1151", True),  # Ryzen AI Max
    (r"\b(890M|880M)\b", "gfx1150", True),  # Ryzen AI 9 HX
    (r"\b(860M|820M)\b", "gfx1152", True),  # Ryzen AI 7 / AI 5
    (r"\b(780M|760M|740M)\b", "gfx1103", True),  # Ryzen 7040 / 8040 など
)
# 840M は gfx1152 と gfx1153 の両方にあり、名前では見分けられない
_AMD_AMBIGUOUS = (r"\b840M\b",)

_VENDOR_IDS = {"10DE": "nvidia", "1002": "amd", "1022": "amd"}


@dataclass(frozen=True)
class Adapter:
    name: str
    vendor: str  # "nvidia" / "amd" / "other"
    driver_version: str = ""

    @property
    def gfx(self) -> str | None:
        return amd_gfx_target(self.name)[0] if self.vendor == "amd" else None

    @property
    def integrated(self) -> bool:
        return self.vendor == "amd" and amd_gfx_target(self.name)[1]


def vendor_of(name: str, pnp_device_id: str = "", compat: str = "") -> str:
    """PNPDeviceID の VEN_xxxx を優先し、なければ名前と AdapterCompatibility で見分ける。"""
    m = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp_device_id or "")
    if m:
        return _VENDOR_IDS.get(m.group(1).upper(), "other")
    text = f"{name or ''} {compat or ''}".lower()
    if "nvidia" in text:
        return "nvidia"
    if re.search(r"\b(amd|radeon|advanced micro devices|ati technologies)\b", text):
        return "amd"
    return "other"


def amd_gfx_target(name: str) -> tuple[str | None, bool]:
    """AMD の GPU 名から（gfx ターゲット, 内蔵 GPU か）。分からなければ (None, False)。"""
    if any(re.search(p, name or "", re.IGNORECASE) for p in _AMD_AMBIGUOUS):
        return None, True
    for pattern, gfx, integrated in _AMD_GFX_TABLE:
        if re.search(pattern, name or "", re.IGNORECASE):
            return gfx, integrated
    return None, False


def parse_video_controllers(text: str) -> list[Adapter]:
    """PowerShell の `Get-CimInstance Win32_VideoController | ConvertTo-Json` の出力を読む（1 件なら配列でない）。"""
    text = (text or "").strip().lstrip("\ufeff")
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    out = []
    for d in data:
        name = str(d.get("Name") or "").strip()
        out.append(
            Adapter(
                name=name,
                vendor=vendor_of(name, str(d.get("PNPDeviceID") or ""), str(d.get("AdapterCompatibility") or "")),
                driver_version=str(d.get("DriverVersion") or ""),
            )
        )
    return out


def nvidia_driver_version(windows_version: str) -> tuple[int, int] | None:
    """Windows のドライバ版（例 32.0.15.7283）から NVIDIA の版（572.83）を取り出す。"""
    parts = (windows_version or "").split(".")
    if len(parts) < 4 or not all(p.isdigit() for p in parts[-2:]):
        return None
    digits = (parts[-2] + parts[-1].zfill(4))[-5:]
    if len(digits) < 5:
        return None
    return int(digits[:3]), int(digits[3:])


@dataclass
class BackendChoice:
    backend: str
    reason: str
    amd_device: str | None = None
    warnings: list[str] = field(default_factory=list)


def _names(xs: list[Adapter]) -> str:
    return "、".join(a.name for a in xs)


def choose_backend(adapters: list[Adapter], requested: str | None = None, amd_device: str | None = None) -> BackendChoice:
    """入れる PyTorch の種類を決める。requested（--gpu か CHORDGEN_GPU）が amd / nvidia / cpu ならそれに従う。"""
    req = (requested or "auto").strip().lower()
    if req not in ("auto", *BACKENDS):
        raise SetupError(f"GPU の指定が正しくありません: {requested!r}（auto / amd / nvidia / cpu のどれか）")
    amd_device = (amd_device or "").strip().lower() or None
    nvidia = [a for a in adapters if a.vendor == "nvidia"]
    amd = [a for a in adapters if a.vendor == "amd"]
    amd_known = [a for a in amd if a.gfx]
    amd_discrete = [a for a in amd_known if not a.integrated]

    def amd_choice(reason: str, warnings: list[str]) -> BackendChoice:
        candidates = amd_discrete or amd_known  # 単体の GPU を内蔵 GPU より優先する
        device = amd_device or (candidates[0].gfx if candidates else None)
        if not device:
            found = f"（見つかった GPU: {_names(amd)}）" if amd else ""
            raise SetupError(
                f"AMD の GPU の種類（gfx ターゲット）を名前から決められませんでした{found}。"
                "--amd-device gfx1201 のように指定してください（README の対応表を参照）"
            )
        return BackendChoice("amd", reason, device, warnings)

    if req == "amd":
        return amd_choice("指定により AMD を使います", [])
    if req in ("nvidia", "cpu"):
        return BackendChoice(req, f"指定により {req} を使います")

    if nvidia and amd_discrete:
        raise SetupError(
            f"NVIDIA の GPU（{_names(nvidia)}）と AMD の GPU（{_names(amd_discrete)}）の両方が見つかりました。"
            "どちらを使うかを指定してください。例: setup.bat --gpu amd / setup.bat --gpu nvidia"
        )
    if nvidia:
        warnings = [f"AMD の内蔵 GPU（{_names(amd)}）は使いません"] if amd else []
        return BackendChoice("nvidia", f"NVIDIA の GPU を見つけました: {_names(nvidia)}", warnings=warnings)
    if amd_known:
        unknown = [a for a in amd if not a.gfx]
        warnings = [f"種類の分からない AMD の GPU は使いません: {_names(unknown)}"] if unknown else []
        return amd_choice(f"AMD の GPU を見つけました: {_names(amd_known)}", warnings)
    if amd:
        if amd_device:
            return amd_choice(f"AMD の GPU を見つけました: {_names(amd)}", [])
        return BackendChoice(
            "cpu",
            "対応表にある GPU が見つかりません",
            warnings=[
                f"AMD の GPU（{_names(amd)}）の種類を名前から決められないので CPU 版を入れます。"
                "GPU で試すなら setup.bat --gpu amd --amd-device gfx1201 のように gfx ターゲットを指定してください"
            ],
        )
    return BackendChoice("cpu", "NVIDIA / AMD の GPU が見つかりません（CPU で動かします。時間がかかります）")


def nvidia_driver_warning(adapters: list[Adapter], min_driver: list[int] | None) -> str | None:
    if not min_driver:
        return None
    for a in adapters:
        if a.vendor != "nvidia":
            continue
        ver = nvidia_driver_version(a.driver_version)
        if ver and ver < tuple(min_driver):
            return (
                f"NVIDIA のドライバ {ver[0]}.{ver[1]:02d} は、この PyTorch に要る {min_driver[0]} より古いです。"
                "ドライバを更新するか、setup.bat --cuda 12.6 を使ってください"
            )
    return None


def hip_sdk_warnings(environ: dict[str, str], program_files_exists: bool) -> list[str]:
    """HIP SDK for Windows が入っていると、pip で入れた ROCm と DLL が食い違うことがある。"""
    out = []
    hip_vars = sorted(k for k in environ if k.upper() == "HIP_PATH" or k.upper().startswith("HIP_PATH_"))
    if hip_vars:
        out.append(f"HIP SDK の環境変数があります（{', '.join(f'{k}={environ[k]}' for k in hip_vars)}）")
    rocm_in_path = [p for p in environ.get("PATH", "").split(";") if re.search(r"\\AMD\\ROCm\\", p, re.IGNORECASE)]
    if rocm_in_path:
        out.append(f"PATH に HIP SDK のフォルダがあります（{'; '.join(rocm_in_path)}）")
    if program_files_exists:
        out.append(r"C:\Program Files\AMD\ROCm があります（HIP SDK for Windows）")
    return out


# ---------------------------------------------------------------------------
# PyTorch と依存の組み立て（純粋な関数）
# ---------------------------------------------------------------------------


def load_sources(path: Path | None = None) -> dict:
    return json.loads((path or PORTABLE_DIR / "sources.json").read_text(encoding="utf-8"))


def select_profile(sources: dict, backend: str, rocm: str | None = None, cuda: str | None = None) -> str:
    """GPU の種類と --rocm / --cuda から sources.json の torch.profiles の名前を選ぶ。"""
    torch = sources["torch"]
    if backend == "amd" and rocm:
        if rocm not in ROCM_VERSIONS:
            raise SetupError(f"--rocm は {' / '.join(ROCM_VERSIONS)} のどれかです: {rocm}")
        return ROCM_VERSIONS[rocm]
    if backend == "nvidia" and cuda:
        if cuda not in CUDA_VERSIONS:
            raise SetupError(f"--cuda は {' / '.join(CUDA_VERSIONS)} のどれかです: {cuda}")
        return CUDA_VERSIONS[cuda]
    return torch["defaults"][backend]


def torch_profile(sources: dict, name: str) -> dict:
    try:
        return sources["torch"]["profiles"][name]
    except KeyError:
        raise SetupError(f"sources.json に PyTorch の設定 {name!r} がありません") from None


def torch_pip_steps(profile: dict, amd_device: str | None) -> list[list[str]]:
    """pip install に渡す引数（{device} を gfx ターゲットに置き換える）。"""
    steps = []
    for step in profile["pip_steps"]:
        if any("{device}" in a for a in step) and not amd_device:
            raise SetupError("AMD の gfx ターゲットが決まっていません（--amd-device で指定してください）")
        steps.append([a.replace("{device}", amd_device or "") for a in step])
    return steps


def is_pinned(name: str) -> bool:
    return bool(PINNED_PATTERN.match(normalize_name(name)))


def constraints_text(installed: dict[str, str]) -> str:
    """入れた torch などの版を固定する constraints ファイルの中身。"""
    lines = ["# install.py が書く。GPU 用の torch を PyPI の torch で上書きさせないための固定"]
    lines += [f"{name}=={ver}" for name, ver in sorted(installed.items()) if ver and is_pinned(name)]
    return "\n".join(lines) + "\n"


def pyproject_dependencies(path: Path | None = None) -> list[str]:
    """pyproject.toml の project.dependencies。リポジトリ本体はインストールせず PYTHONPATH で読むので、依存だけ入れる。"""
    data = tomllib.loads((path or APP_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data.get("project", {}).get("dependencies", []))


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(lines: list[str]) -> list[str]:
    """要求の文字列からパッケージ名（正規化したもの）を取り出す。コメント・空行・オプションの行は飛ばす。"""
    names = []
    for line in lines:
        s = line.split("#", 1)[0].strip()
        if not s or s.startswith("-"):
            continue
        m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", s)
        if m:
            names.append(normalize_name(m.group(0)))
    return names


def requirement_files(portable_dir: Path | None = None) -> tuple[list[Path], Path | None]:
    """（依存を解決して入れるファイル, --no-deps で入れるファイル。中身がなければ None）。"""
    d = portable_dir or PORTABLE_DIR
    files = sorted(p for p in d.glob("requirements-*.txt") if p.name != NODEPS_REQUIREMENTS)
    nodeps = d / NODEPS_REQUIREMENTS
    has_nodeps = nodeps.exists() and requirement_names(nodeps.read_text(encoding="utf-8").splitlines())
    return files, (nodeps if has_nodeps else None)


def deps_pip_args(constraints: Path, deps: list[str], req_files: list[Path]) -> list[str]:
    args = ["-c", str(constraints), *deps]
    for f in req_files:
        args += ["-r", str(f)]
    return args


def media_dl_spec(package_json: Path | None = None) -> str:
    data = json.loads((package_json or APP_DIR / "package.json").read_text(encoding="utf-8"))
    spec = data.get("dependencies", {}).get(MEDIA_DL_PACKAGE)
    if not spec:
        raise SetupError(f"package.json の dependencies に {MEDIA_DL_PACKAGE} がありません")
    return spec


def fingerprint(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# torch や ROCm の wheel には site-packages から 140 文字ほどのパスがある。Windows の 260 文字の上限に収めるため、
# 長いパスが有効でなければアプリのフォルダのパスをこれより短くしてもらう
MAX_APP_PATH = 80


def long_path_warning(app_dir: str, enabled: bool) -> str | None:
    if enabled or len(app_dir) <= MAX_APP_PATH:
        return None
    return (
        f"アプリのフォルダのパスが長すぎます（{len(app_dir)} 文字）。PyTorch のファイルが Windows のパスの上限（260 文字）を"
        r"超えて入らないことがあります。C:\chordgen のような短い場所に移してから実行してください"
    )


def long_paths_enabled() -> bool:
    """Windows の「長いパス」（LongPathsEnabled）が有効か。読むだけで変えない。"""
    if os.name != "nt":
        return True
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def strip_setup_flag(argv: list[str]) -> list[str]:
    """run.bat --setup から来たときの --setup を落とす。"""
    return [a for a in argv if a != "--setup"]


# ---------------------------------------------------------------------------
# 実行（Windows の上で動く部分）
# ---------------------------------------------------------------------------


def say(msg: str = "") -> None:
    print(msg, flush=True)


def heading(msg: str) -> None:
    say()
    say(f"=== {msg} ===")


class Runner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, cmd: list, *, cwd: Path | None = None, env: dict | None = None, what: str = "") -> None:
        cmd = [str(c) for c in cmd]
        say("  > " + subprocess.list2cmdline(cmd))
        if self.dry_run:
            return
        try:
            r = subprocess.run(cmd, cwd=cwd, env=env)
        except OSError as e:
            raise SetupError(f"{what or cmd[0]} を起動できませんでした: {e}") from e
        if r.returncode != 0:
            raise SetupError(f"{what or '処理'}に失敗しました（終了コード {r.returncode}）。上の出力を確かめてください")

    def pip(self, *args: str, what: str) -> None:
        self.run([sys.executable, "-m", "pip", *args], what=what)

    def pip_install(self, *args: str, what: str) -> None:
        self.pip("install", "--disable-pip-version-check", "--no-input", *args, what=what)


class State:
    """runtime/setup-state.json。済んだ手順の記録。"""

    def __init__(self, path: Path, dry_run: bool = False):
        self.path = path
        self.dry_run = dry_run
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value) -> None:
        if value is None:
            self.data.pop(key, None)
        else:
            self.data[key] = value
        if self.dry_run:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


_LIST_CODE = (
    "import json,importlib.metadata as m\n"
    "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))\n"
)


def installed_distributions() -> dict[str, str]:
    """同梱の Python に入っているパッケージと版（入れた直後でも正しく読めるよう別のプロセスで調べる）。"""
    r = subprocess.run([sys.executable, "-c", _LIST_CODE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SetupError(f"入っているパッケージを読めませんでした: {r.stderr.strip()}")
    return {normalize_name(k): v for k, v in json.loads(r.stdout).items()}


def pinned_installed() -> dict[str, str]:
    return {k: v for k, v in installed_distributions().items() if is_pinned(k)}


def detect_adapters() -> list[Adapter]:
    """Win32_VideoController を PowerShell で読む。読めなければ空（CPU 扱い）。"""
    ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        "Get-CimInstance -ClassName Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"
    )
    try:
        r = subprocess.run([str(ps), "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, timeout=60)
        return parse_video_controllers(r.stdout.decode("utf-8", errors="replace"))
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        say(f"  GPU の一覧を読めませんでした（{e}）。--gpu で指定できます")
        return []


def step_torch(args, sources: dict, state: State, runner: Runner, runtime: Path) -> dict:
    heading("1/4 PyTorch")
    requested = args.gpu or os.environ.get(ENV_GPU) or "auto"
    amd_device = args.amd_device or os.environ.get(ENV_AMD_DEVICE)
    adapters = detect_adapters() if os.name == "nt" else []
    for a in adapters:
        extra = f"、{a.gfx}" if a.gfx else ""
        say(f"  GPU: {a.name}（{a.vendor}{extra}、ドライバ {a.driver_version or '不明'}）")
    choice = choose_backend(adapters, requested, amd_device)
    name = select_profile(sources, choice.backend, args.rocm or os.environ.get(ENV_ROCM), args.cuda or os.environ.get(ENV_CUDA))
    profile = torch_profile(sources, name)
    say(f"  {choice.reason} → {profile['label']}" + (f"（{choice.amd_device}）" if choice.amd_device else ""))
    if profile.get("driver"):
        say(f"  要るドライバ: {profile['driver']}")
    warnings = list(choice.warnings)
    if choice.backend == "amd":
        if choice.amd_device not in profile.get("devices", []):
            warnings.append(f"{choice.amd_device} は {name} の対応表にありません。動かなければ setup.bat --gpu cpu")
        if os.name == "nt":
            warnings += [
                f"{w}。pip で入れた ROCm と食い違って GPU が使えないことがあります。問題が出たら HIP SDK を外してください"
                for w in hip_sdk_warnings(dict(os.environ), Path(r"C:\Program Files\AMD\ROCm").exists())
            ]
    w = nvidia_driver_warning(adapters, profile.get("min_driver")) if choice.backend == "nvidia" else None
    if w:
        warnings.append(w)
    for w in warnings:
        say(f"  注意: {w}")

    expect = profile["expect"]
    constraints = runtime / "torch-constraints.txt"
    wanted = {"profile": name, "device": choice.amd_device}
    prev = state.get("torch") or {}
    have = {} if runner.dry_run else pinned_installed()
    same = all(have.get(k) == v for k, v in expect.items())
    if not args.force and same and constraints.exists() and {k: prev.get(k) for k in wanted} == wanted:
        say("  入っています（飛ばします）")
        return {**wanted, "backend": choice.backend, "versions": have}

    # 別の GPU 用・別の版の torch や ROCm の部品が入っていれば外してから入れる。
    # 同じ設定で途中まで入ったもの（前回の失敗）は外さず、pip に続きを入れさせる
    switched = bool(prev) and {k: prev.get(k) for k in wanted} != wanted
    wrong_torch = bool(have.get("torch")) and have.get("torch") != expect.get("torch")
    if have and (switched or wrong_torch or args.force):
        runner.pip("uninstall", "-y", *sorted(have), what="入っている PyTorch の削除")
    state.set("torch", None)
    state.set("deps", None)  # torch が変わったら依存も入れ直す

    steps = torch_pip_steps(profile, choice.amd_device)
    for i, step in enumerate(steps, 1):
        say(f"  ({i}/{len(steps)}) 大きなファイル（1〜3 GB）を取得します。数分〜数十分かかります")
        runner.pip_install(*step, what=f"PyTorch（{profile['label']}）のインストール")

    got = dict(expect) if runner.dry_run else pinned_installed()
    bad = {k: (got.get(k), v) for k, v in expect.items() if got.get(k) != v}
    if bad:
        detail = "、".join(f"{k}: {g or 'なし'}（期待 {v}）" for k, (g, v) in bad.items())
        raise SetupError(f"PyTorch の版が期待どおりではありません: {detail}")
    if not runner.dry_run:
        constraints.write_text(constraints_text(got), encoding="utf-8")
    result = {**wanted, "backend": choice.backend, "versions": got}
    state.set("torch", result)
    return result


def step_deps(args, state: State, runner: Runner, runtime: Path) -> None:
    heading("2/4 解析に使うライブラリ")
    constraints = runtime / "torch-constraints.txt"
    deps = pyproject_dependencies()
    req_files, nodeps = requirement_files()
    texts = [p.read_text(encoding="utf-8") for p in [*req_files, *([nodeps] if nodeps else [])]]
    fp = fingerprint(constraints.read_text(encoding="utf-8") if constraints.exists() else "", json.dumps(deps), *texts)
    if not args.force and state.get("deps") == fp:
        say("  入っています（飛ばします）")
        return

    before = {} if runner.dry_run else pinned_installed()
    # torch は constraints で固定する。満たせない依存があれば pip は torch を替えずにエラーにする
    runner.pip_install(*deps_pip_args(constraints, deps, req_files), what="依存のインストール")
    if nodeps:
        runner.pip_install("--no-deps", "-c", str(constraints), "-r", str(nodeps), what="依存（--no-deps）のインストール")
    after = {} if runner.dry_run else pinned_installed()
    if before != after:
        raise SetupError(
            f"依存のインストールで PyTorch が置き換わりました（{before} → {after}）。torch を要求するものを "
            "requirements-nodeps.txt に移し、setup.bat --force を実行してください"
        )
    state.set("deps", fp)


def url_reachable(url: str, timeout: float = 30.0) -> tuple[bool, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "chordgen-setup"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, str(getattr(e, "reason", e))


def step_media_dl(args, state: State, runner: Runner, runtime: Path) -> bool:
    heading("3/4 media-dl（URL からの音源の取得）")
    no_media_dl = "media-dl なしでも、音源ファイルを渡す使い方はできます（URL の入力だけが使えません）"
    if args.skip_media_dl:
        say(f"  --skip-media-dl なので飛ばします。{no_media_dl}")
        return False
    node = runtime / "node" / "node.exe"
    npm_cli = runtime / "node" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    bin_dir = Path(os.environ.get("MEDIA_DL_BIN_DIR") or runtime / "media-dl" / "bin")
    cli_js = APP_DIR / "node_modules" / Path(*MEDIA_DL_PACKAGE.split("/")) / "dist" / "cli.js"

    override = os.environ.get(ENV_MEDIA_DL_TGZ, "").strip()
    spec = override or media_dl_spec()
    fp = fingerprint((APP_DIR / "package.json").read_text(encoding="utf-8"), override)
    if args.force or state.get("media_dl_npm") != fp or not cli_js.exists():
        if re.match(r"https?://", spec):
            ok, why = url_reachable(spec)
            if not ok:
                say(f"  注意: media-dl の tarball を取得できません（{why}）: {spec}")
                say(f"  package.json の URL を直すか、環境変数 {ENV_MEDIA_DL_TGZ} に tarball の URL かファイルを指定して")
                say("  setup.bat をもう一度実行してください。")
                say(f"  {no_media_dl}")
                return False
        if override:
            cmd = [node, npm_cli, "install", "--no-save", "--no-audit", "--no-fund", override]
        elif (APP_DIR / "package-lock.json").exists():
            cmd = [node, npm_cli, "ci", "--no-audit", "--no-fund"]
        else:
            cmd = [node, npm_cli, "install", "--no-audit", "--no-fund"]
        try:
            runner.run(cmd, cwd=APP_DIR, what="media-dl のインストール（npm）")
        except SetupError as e:
            say(f"  注意: {e}")
            say(f"  {no_media_dl}")
            return False
        state.set("media_dl_npm", fp)
    else:
        say("  media-dl は入っています")

    missing = [b for b in MEDIA_DL_BINARIES if not (bin_dir / f"{b}.exe").exists()]
    if missing or args.force:
        # PATH を絞って呼ぶ。Windows 側の PATH にある ffmpeg を使わせず、必ずアプリのフォルダに入れるため
        env = dict(os.environ)
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        env["PATH"] = os.pathsep.join([str(node.parent), str(Path(sysroot) / "System32"), sysroot])
        env["MEDIA_DL_BIN_DIR"] = str(bin_dir)
        setup_args = ["setup", "--bin-dir", str(bin_dir)] + (["--force"] if args.force else [])
        try:
            runner.run([node, cli_js, *setup_args], env=env, what="media-dl setup（yt-dlp・ffmpeg・ffprobe・deno）")
        except SetupError as e:
            say(f"  注意: {e}")
            say(f"  {no_media_dl}")
            return False
    else:
        say(f"  yt-dlp・ffmpeg・ffprobe・deno は入っています（{bin_dir}）")
    return True


VERIFY_CODE = r"""
import json, torch
info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda, "devices": []}
if info["cuda_available"]:
    info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
try:
    import torchaudio
    info["torchaudio"] = torchaudio.__version__
except Exception as e:
    info["torchaudio_error"] = repr(e)
print(json.dumps(info))
"""


def step_verify(torch_info: dict, runner: Runner) -> dict:
    heading("4/4 確認（torch.cuda.is_available()）")
    if runner.dry_run:
        return {}
    r = subprocess.run([sys.executable, "-c", VERIFY_CODE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SetupError(f"torch を読み込めませんでした:\n{r.stderr.strip()}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    say(f"  torch {info['torch']}（HIP {info['hip']} / CUDA {info['cuda']}）、torchaudio {info.get('torchaudio')}")
    if info.get("torchaudio_error"):
        say(f"  注意: torchaudio を読み込めません: {info['torchaudio_error']}")
    say(f"  torch.cuda.is_available() = {info['cuda_available']}")
    for d in info["devices"]:
        say(f"  デバイス: {d}")
    backend = torch_info.get("backend")
    if backend != "cpu" and not info["cuda_available"]:
        if backend == "amd":
            say("  注意: GPU が使えません。README の「必要なもの」の Adrenalin の版を入れたか、--amd-device が GPU と")
            say("  合っているかを確かめてください。ROCm 7.2.1 に戻すなら setup.bat --rocm 7.2.1、CPU なら setup.bat --gpu cpu")
        else:
            say("  注意: GPU が使えません。NVIDIA のドライバの版を確かめてください（古いなら setup.bat --cuda 12.6）。")
            say("  CPU で動かすなら setup.bat --gpu cpu")
    return info


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="setup.bat", description="chordgen ポータブル版のセットアップ（冪等。再実行で続きから）")
    p.add_argument("--gpu", choices=["auto", *BACKENDS], help=f"入れる PyTorch の種類（既定は自動判定。環境変数 {ENV_GPU}）")
    p.add_argument("--amd-device", metavar="gfxXXXX", help=f"AMD の gfx ターゲット（例 gfx1201。環境変数 {ENV_AMD_DEVICE}）")
    p.add_argument("--rocm", choices=list(ROCM_VERSIONS), help=f"AMD で使う ROCm の版（既定 10.2.0。環境変数 {ENV_ROCM}）")
    p.add_argument("--cuda", choices=list(CUDA_VERSIONS), help=f"NVIDIA で使う CUDA の版（既定 13.2。環境変数 {ENV_CUDA}）")
    p.add_argument("--force", action="store_true", help="済んだ手順もやり直す")
    p.add_argument("--skip-media-dl", action="store_true", help="media-dl を入れない（URL の入力が使えなくなる）")
    p.add_argument("--dry-run", action="store_true", help="実行するコマンドを表示するだけ")
    return p.parse_args(strip_setup_flag(argv))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.amd_device and not re.fullmatch(r"gfx[0-9a-f]{3,4}", args.amd_device.lower()):
        say(f"--amd-device は gfx1201 のような形で指定してください: {args.amd_device}")
        return 2
    if os.name != "nt" and not args.dry_run:
        say("このセットアップは Windows 用です（開発は README の「開発」を参照）。--dry-run で手順だけ表示できます")
        return 2
    runtime = Path(os.environ.get("CHORDGEN_HOME") or APP_DIR) / "runtime"
    w = long_path_warning(str(APP_DIR), long_paths_enabled())
    if w:
        say(f"注意: {w}")
    runner = Runner(dry_run=args.dry_run)
    state = State(runtime / "setup-state.json", dry_run=args.dry_run)
    try:
        sources = load_sources()
        torch_info = step_torch(args, sources, state, runner, runtime)
        step_deps(args, state, runner, runtime)
        media_dl = step_media_dl(args, state, runner, runtime)
        verify = step_verify(torch_info, runner)
    except SetupError as e:
        say()
        say(f"セットアップに失敗しました: {e}")
        say("直してから setup.bat をもう一度実行すると、済んだところは飛ばして続きから進みます。")
        return 1
    except KeyboardInterrupt:
        say()
        say("中断しました。setup.bat をもう一度実行すると続きから進みます。")
        return 130

    if not args.dry_run:
        done = {
            "completedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "torch": torch_info,
            "verify": verify,
            "mediaDl": media_dl,
            "python": sys.version.split()[0],
        }
        (runtime / "setup-complete.json").write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    say()
    say("セットアップが終わりました。使い方: run.bat <音源のファイルか URL>（run.bat だけでヘルプ）")
    if not media_dl:
        say("（media-dl は入っていないので、URL の入力は使えません）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
