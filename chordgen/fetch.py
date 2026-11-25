"""media-dl（@liasvistreamgamemusic-sketch/media-dl）で URL から音源を取る。

media-dl は子プロセスとして呼び、`--json` の NDJSON（1 行 1 JSON）を読む。
- ダウンロード: `media-dl -f wav -o <dir> --json -- <URL>`
  標準出力に {"type":"progress",...} / {"type":"done","files":[...],"warnings":[...]} /
  {"type":"error","code":...,"userMessage":...,"detail":...} が出る。
- 情報: `media-dl info --json -- <URL>` は MediaInfo を 1 行の JSON で出す。失敗は
  標準エラーに「エラー [CODE] 文言」と出し、終了コード 1。
- 終了コード: 成功 0、失敗 1、引数の誤り 2、Ctrl+C による中断 130。
- プレイリストは `--playlist` を付けない限り 1 曲だけ（yt-dlp の --no-playlist）。

実行ファイルの探し方: 環境変数 CHORDGEN_MEDIA_DL → アプリのフォルダの node_modules →
PATH。Windows の npm が作る `media-dl.cmd` は shell=False で安全に渡せない（引数の `&` などを
cmd.exe が解釈する）ので、中に書かれた JS のエントリを取り出して node で直接呼ぶ。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

from chordgen.paths import app_home
from chordgen.types import FetchResult, TrackMeta

log = logging.getLogger(__name__)

PACKAGE_NAME = "@liasvistreamgamemusic-sketch/media-dl"
ENV_MEDIA_DL = "CHORDGEN_MEDIA_DL"  # 実行ファイルのパス、または JSON 配列（実行ファイル + 前置きの引数）
ENV_NODE = "CHORDGEN_NODE"  # node の実行ファイル（JS のエントリを直接呼ぶとき）

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130

# 中断のとき、子が自分で後片付け（一時フォルダの削除）を終えるのを待つ秒数
STOP_GRACE_SEC = 5.0

INSTALL_HINT = (
    "media-dl が見つかりません。アプリのフォルダで setup.bat を実行して入れてください"
    "（入れ直すときは run.bat --media-dl setup）。\n"
    f"別の場所にある場合は環境変数 {ENV_MEDIA_DL} で実行ファイルを指定してください。"
)
UPDATE_HINT = (
    "yt-dlp が古い可能性があります。run.bat から開いた環境（MEDIA_DL_BIN_DIR をアプリのフォルダに"
    "向けた状態）で `media-dl update` を実行して yt-dlp を更新し、もう一度試してください。"
)

_CANCELLED_MESSAGE = "ダウンロードを中止しました。"
_INFO_ERROR_RE = re.compile(r"エラー \[([A-Z_]+)\]\s*(.*)")
_SHIM_JS_RE = re.compile(r'"%~?dp0%?\\([^"]+?\.(?:js|mjs|cjs))"', re.IGNORECASE)


class FetchError(Exception):
    """取得の失敗。code は media-dl の code（または本モジュール独自の code）。"""

    def __init__(self, code: str, user_message: str, exit_code: int = EXIT_FAILED, detail: str = "") -> None:
        super().__init__(f"[{code}] {user_message}")
        self.code = code
        self.user_message = user_message
        self.exit_code = exit_code
        self.detail = detail


def is_url(source: str) -> bool:
    """http / https の URL なら True。Windows のパス（C:\\...）は False。"""
    if not isinstance(source, str):
        return False
    p = urlparse(source.strip())
    return p.scheme.lower() in ("http", "https") and bool(p.netloc)


# ---- 実行ファイルの解決 ----


def _node_exe_name() -> str:
    return "node.exe" if os.name == "nt" else "node"


def find_node(near: Path | None = None) -> str:
    """node を探す。CHORDGEN_NODE → near（.cmd の隣）→ アプリのフォルダの runtime → PATH。"""
    given = os.environ.get(ENV_NODE, "").strip()
    if given:
        if Path(given).is_file():
            return given
        raise FetchError("NODE_MISSING", f"{ENV_NODE} の指定先に node がありません: {given}")
    exe = _node_exe_name()
    home = app_home()
    candidates = [near / exe] if near else []
    candidates += [home / "runtime" / "node" / exe, home / "runtime" / "node" / "bin" / exe, home / "node" / exe]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which("node")
    if found:
        return found
    raise FetchError("NODE_MISSING", "Node.js（node）が見つかりません。アプリのフォルダに同梱するか PATH に追加してください。")


def _js_from_cmd_shim(shim: Path) -> Path | None:
    """npm の cmd-shim（media-dl.cmd）に書かれた JS のエントリの絶対パスを返す。"""
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _SHIM_JS_RE.search(text)
    if not m:
        return None
    rel = PureWindowsPath(m.group(1))
    return (shim.parent.joinpath(*rel.parts)).resolve()


def _command_for(path: Path) -> list[str]:
    """実行ファイルのパスから、shell=False で呼べるコマンドの前置きを作る。"""
    suffix = path.suffix.lower()
    if suffix in (".js", ".mjs", ".cjs"):
        return [find_node(), str(path)]
    if suffix == ".py":
        return [sys.executable, str(path)]
    if suffix in (".cmd", ".bat"):
        js = _js_from_cmd_shim(path)
        if js is None or not js.is_file():
            raise FetchError(
                "MEDIA_DL_MISSING",
                f"{path} から media-dl の JS のエントリを読み取れませんでした。"
                f"{ENV_MEDIA_DL} に node_modules/{PACKAGE_NAME}/dist/cli.js を指定してください。",
            )
        return [find_node(near=path.parent), str(js)]
    return [str(path)]


def _from_package_dir(pkg_dir: Path) -> list[str] | None:
    """node_modules/<パッケージ> の package.json の bin から JS のエントリを取る。"""
    try:
        pkg = json.loads((pkg_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    bin_field = pkg.get("bin")
    rel = bin_field.get("media-dl") if isinstance(bin_field, dict) else bin_field
    if not isinstance(rel, str):
        return None
    js = (pkg_dir / rel).resolve()
    return [find_node(), str(js)] if js.is_file() else None


def _from_explicit(value: str) -> list[str]:
    if value.startswith("["):
        try:
            parts = json.loads(value)
        except ValueError as e:
            raise FetchError("MEDIA_DL_MISSING", f"{ENV_MEDIA_DL} の JSON 配列を読めません: {e}") from e
        if not parts or not all(isinstance(p, str) and p for p in parts):
            raise FetchError("MEDIA_DL_MISSING", f"{ENV_MEDIA_DL} は文字列の配列で指定してください。")
        first = Path(parts[0])
        if first.is_file():
            return _command_for(first) + parts[1:]
        found = shutil.which(parts[0])
        if not found:
            raise FetchError("MEDIA_DL_MISSING", f"{ENV_MEDIA_DL} の実行ファイルが見つかりません: {parts[0]}")
        return _command_for(Path(found)) + parts[1:]
    path = Path(value)
    if not path.is_file():
        raise FetchError("MEDIA_DL_MISSING", f"{ENV_MEDIA_DL} の指定先が見つかりません: {value}")
    return _command_for(path)


def find_media_dl() -> list[str]:
    """media-dl を呼ぶコマンドの前置き（実行ファイル + 引数）を返す。見つからなければ FetchError。"""
    explicit = os.environ.get(ENV_MEDIA_DL, "").strip()
    if explicit:
        return _from_explicit(explicit)

    modules = app_home() / "node_modules"
    cmd = _from_package_dir(modules.joinpath(*PACKAGE_NAME.split("/")))
    if cmd:
        return cmd
    names = ["media-dl.cmd", "media-dl"] if os.name == "nt" else ["media-dl"]
    for name in names:
        p = modules / ".bin" / name
        if p.is_file():
            return _command_for(p)

    found = shutil.which("media-dl")
    if found:
        return _command_for(Path(found))
    raise FetchError("MEDIA_DL_MISSING", INSTALL_HINT)


# ---- 子プロセスの実行 ----


def _kill_tree(proc: subprocess.Popen) -> None:
    """子とその子（yt-dlp・ffmpeg）を止める。"""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _stop(proc: subprocess.Popen) -> None:
    """中断のときに子を止める。

    端末の Ctrl+C は子にも届き、media-dl は yt-dlp を止めて一時フォルダを消してから 130 で終わる。
    まずそれを待ち、終わらなければ SIGINT（POSIX のみ）、最後はプロセスツリーごと強制終了する。
    media-dl は 2 回目の SIGINT で後片付けせずに終わるので、先に SIGINT は送らない。
    """
    try:
        try:
            proc.wait(timeout=STOP_GRACE_SEC)
            return
        except subprocess.TimeoutExpired:
            pass
        if os.name != "nt":
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=STOP_GRACE_SEC)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        _kill_tree(proc)
    except BaseException:
        _kill_tree(proc)
        raise


def _run(args: list[str], on_event: Callable[[dict[str, Any]], None]) -> tuple[int, str]:
    """子を起動し、標準出力の JSON の行を on_event に渡す。JSON でない行は無視する。

    戻り値は（終了コード、標準エラーの末尾）。
    """
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except OSError as e:
        raise FetchError("MEDIA_DL_MISSING", f"media-dl を起動できませんでした: {e}\n{INSTALL_HINT}") from e

    err_tail: deque[str] = deque(maxlen=200)

    def drain_stderr() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            err_tail.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                on_event(obj)
        rc = proc.wait()
    except BaseException:
        # Ctrl+C（KeyboardInterrupt）や on_progress の例外。子を止めてから投げ直す
        _stop(proc)
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
    t.join(timeout=5)
    return rc, "\n".join(err_tail)


def _error_from_event(ev: dict[str, Any], rc: int) -> FetchError:
    code = str(ev.get("code") or "UNKNOWN")
    message = str(ev.get("userMessage") or "取得に失敗しました。")
    if code == "ENGINE_OUTDATED":
        message = f"{message}\n{UPDATE_HINT}"
    exit_code = EXIT_CANCELLED if (code == "CANCELLED" or rc == EXIT_CANCELLED) else EXIT_FAILED
    return FetchError(code, message, exit_code, str(ev.get("detail") or ""))


def _error_from_exit(rc: int, stderr: str) -> FetchError:
    """error の行がないまま失敗したとき。標準エラーの「エラー [CODE] 文言」を読む。"""
    if rc == EXIT_CANCELLED:
        return FetchError("CANCELLED", _CANCELLED_MESSAGE, EXIT_CANCELLED, stderr)
    if rc == EXIT_USAGE:
        first = next((s for s in stderr.splitlines() if s.strip()), "")
        return FetchError("USAGE", f"media-dl の引数が正しくありません: {first}", EXIT_USAGE, stderr)
    m = _INFO_ERROR_RE.search(stderr)
    if m:
        code, message = m.group(1), m.group(2).strip()
        # 「エラー [CODE] URL」の次の行に字下げした文言が来る形（ダウンロードの非 JSON 出力）
        tail = stderr[m.end() :].lstrip("\r\n").splitlines()
        if (not message or message.startswith("http")) and tail:
            message = tail[0].strip() or message
        if code == "ENGINE_OUTDATED":
            message = f"{message}\n{UPDATE_HINT}"
        return FetchError(code, message or "取得に失敗しました。", EXIT_FAILED, stderr)
    last = next((s for s in reversed(stderr.splitlines()) if s.strip()), "")
    return FetchError("UNKNOWN", f"media-dl が終了コード {rc} で終わりました。{last}".strip(), EXIT_FAILED, stderr)


# ---- 公開関数 ----


def _clean(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s or None


def _artist_from_info(info: dict[str, Any]) -> str | None:
    artist = _clean(info.get("artist"))
    if artist:
        return artist
    uploader = _clean(info.get("uploader"))
    if uploader and uploader.endswith(" - Topic"):  # YouTube の自動生成チャンネル
        uploader = _clean(uploader[: -len(" - Topic")])
    return uploader


def fetch_info(url: str) -> TrackMeta:
    """`media-dl info <URL> --json` で題名とアーティスト（なければ投稿者）を取る。"""
    if not is_url(url):
        raise FetchError("INVALID_INPUT", f"URL ではありません: {url}", EXIT_USAGE)
    cmd = find_media_dl()
    infos: list[dict[str, Any]] = []
    rc, stderr = _run([*cmd, "info", "--json", "--", url], infos.append)
    errors = [ev for ev in infos if ev.get("type") == "error"]
    if errors:
        raise _error_from_event(errors[0], rc)
    if rc != EXIT_OK:
        raise _error_from_exit(rc, stderr)
    info = next((ev for ev in reversed(infos) if "title" in ev), None)
    if info is None:
        raise FetchError("UNKNOWN", "media-dl info の出力を読み取れませんでした。", EXIT_FAILED, stderr)

    title = _clean(info.get("title"))
    if info.get("isPlaylist"):
        # プレイリストの題名は曲の題名ではないので使わない
        log.warning("プレイリストの URL です。最初の 1 曲だけを解析します: %s", url)
        title = None
    return TrackMeta(title=title, artist=_artist_from_info(info), source_url=url)


_ID_SUFFIX_RE = re.compile(r"\s*\[[A-Za-z0-9_-]+\]$")


def fetch_audio(
    url: str,
    dest_dir: Path,
    on_progress: Callable[[dict], None] | None = None,
) -> FetchResult:
    """`media-dl <URL> -f wav -o <dest_dir> --json` で wav を取る。

    on_progress には progress のイベント（type, url, status, percent, downloadedBytes, totalBytes,
    speedBps, etaSec）をそのまま渡す。プレイリストの URL は 1 曲だけ取る（media-dl の既定）。
    それでも複数のファイルが返ったときは最初のファイルを使い、警告を出す。
    """
    if not is_url(url):
        raise FetchError("INVALID_INPUT", f"URL ではありません: {url}", EXIT_USAGE)
    cmd = find_media_dl()
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    done: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def on_event(ev: dict[str, Any]) -> None:
        kind = ev.get("type")
        if kind == "progress":
            if on_progress is not None:
                on_progress(ev)
        elif kind == "done":
            done.append(ev)
        elif kind == "error":
            errors.append(ev)

    # プレイリストは --playlist を付けないので 1 曲だけ。-- の後に URL を置き、オプションと取り違えない
    rc, stderr = _run([*cmd, "-f", "wav", "-o", str(dest), "--json", "--", url], on_event)

    if errors:
        raise _error_from_event(errors[0], rc)
    if rc != EXIT_OK:
        raise _error_from_exit(rc, stderr)

    files = [str(f) for ev in done for f in (ev.get("files") or []) if isinstance(f, str) and f]
    for ev in done:
        for w in ev.get("warnings") or []:
            log.warning("media-dl: %s", w)
    if not files:
        raise FetchError("NO_FILE", "取得したファイルの場所が分かりませんでした。", EXIT_FAILED, stderr)
    if len(files) > 1:
        log.warning("複数のファイルを取得しました。最初の 1 つだけを解析します: %s", files[0])

    audio_path = Path(files[0])
    if not audio_path.is_absolute():
        audio_path = dest / audio_path
    if not audio_path.is_file():
        raise FetchError("NO_FILE", f"取得したファイルが見つかりません: {audio_path}", EXIT_FAILED, stderr)

    # 既定のファイル名は「題名 [ID].wav」。題名は fetch_info の値を優先する前提の仮の値
    title = _clean(_ID_SUFFIX_RE.sub("", audio_path.stem))
    return FetchResult(audio_path=audio_path, meta=TrackMeta(title=title, source_url=url))
