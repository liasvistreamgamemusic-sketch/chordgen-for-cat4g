"""portable/install.py の純粋な部分（GPU の判定、PyTorch の版の選び方、依存の組み立て）と設定ファイルの検査。"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PORTABLE = ROOT / "portable"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


inst = _load("chordgen_portable_install", PORTABLE / "install.py")
Adapter = inst.Adapter


def amd(name: str) -> Adapter:
    return Adapter(name, inst.vendor_of(name, "PCI\\VEN_1002&DEV_7550&SUBSYS_0"))


def nv(name: str, driver: str = "32.0.15.8180") -> Adapter:
    return Adapter(name, inst.vendor_of(name, "PCI\\VEN_10DE&DEV_2786"), driver)


# ---- メーカーの判定 ----


@pytest.mark.parametrize(
    ("name", "pnp", "compat", "expected"),
    [
        ("AMD Radeon RX 9070 XT", "PCI\\VEN_1002&DEV_7550", "Advanced Micro Devices, Inc.", "amd"),
        ("NVIDIA GeForce RTX 4070", "PCI\\VEN_10DE&DEV_2786", "NVIDIA", "nvidia"),
        ("Intel(R) UHD Graphics 770", "PCI\\VEN_8086&DEV_4680", "Intel Corporation", "other"),
        # PNPDeviceID が Intel なら名前に関わらず other
        ("Radeon-like name", "PCI\\VEN_8086&DEV_0001", "", "other"),
        # PNPDeviceID がないときは名前と AdapterCompatibility で見る
        ("NVIDIA GeForce RTX 3060", "", "", "nvidia"),
        ("AMD Radeon(TM) Graphics", "", "", "amd"),
        ("Some Adapter", "", "Advanced Micro Devices, Inc.", "amd"),
        ("Microsoft Basic Display Adapter", "ROOT\\BASICDISPLAY\\0000", "(Standard display types)", "other"),
        ("Parsec Virtual Display Adapter", "ROOT\\DISPLAY\\0000", "Parsec Cloud, Inc.", "other"),
    ],
)
def test_vendor_of(name, pnp, compat, expected):
    assert inst.vendor_of(name, pnp, compat) == expected


# ---- AMD の GPU 名 → gfx ターゲット ----


@pytest.mark.parametrize(
    ("name", "gfx", "integrated"),
    [
        ("AMD Radeon RX 9070 XT", "gfx1201", False),
        ("AMD Radeon RX 9070", "gfx1201", False),
        ("AMD Radeon RX 9070 GRE", "gfx1201", False),
        ("AMD Radeon AI PRO R9700", "gfx1201", False),
        ("AMD Radeon RX 9060 XT", "gfx1200", False),
        ("AMD Radeon RX 9060", "gfx1200", False),
        ("AMD Radeon RX 7900 XTX", "gfx1100", False),
        ("AMD Radeon RX 7900 GRE", "gfx1100", False),
        ("AMD Radeon PRO W7900", "gfx1100", False),
        ("AMD Radeon RX 7800 XT", "gfx1101", False),
        ("AMD Radeon RX 7700 XT", "gfx1101", False),
        ("AMD Radeon RX 7600", "gfx1102", False),
        ("AMD Radeon RX 6800 XT", "gfx1030", False),
        ("AMD Radeon(TM) 8060S Graphics", "gfx1151", True),
        ("AMD Radeon(TM) 890M Graphics", "gfx1150", True),
        ("AMD Radeon 780M Graphics", "gfx1103", True),
        ("AMD Radeon(TM) 860M", "gfx1152", True),
    ],
)
def test_amd_gfx_target(name, gfx, integrated):
    assert inst.amd_gfx_target(name) == (gfx, integrated)


@pytest.mark.parametrize(
    "name",
    [
        "AMD Radeon(TM) Graphics",  # 型番のない内蔵 GPU
        "AMD Radeon RX 580",
        "AMD Radeon RX 6600",
        "AMD Radeon RX 7900M",  # ノート PC 用は表にない
        "AMD Radeon(TM) 840M Graphics",  # gfx1152 と gfx1153 の両方にある
    ],
)
def test_amd_gfx_target_unknown(name):
    assert inst.amd_gfx_target(name)[0] is None


def test_every_mapped_gfx_is_a_rocm_10_2_device():
    sources = inst.load_sources()
    devices = set(sources["torch"]["profiles"]["rocm-10.2.0"]["devices"])
    assert {gfx for _, gfx, _ in inst._AMD_GFX_TABLE} <= devices


# ---- Win32_VideoController の JSON ----


def test_parse_video_controllers_single_object_with_bom():
    text = "\ufeff" + json.dumps(
        {
            "Name": "AMD Radeon RX 9070 XT",
            "AdapterCompatibility": "Advanced Micro Devices, Inc.",
            "PNPDeviceID": "PCI\\VEN_1002&DEV_7550&SUBSYS_24121EAE&REV_C0\\6&1",
            "DriverVersion": "32.0.21043.1001",
        }
    )
    [a] = inst.parse_video_controllers(text)
    assert a.vendor == "amd" and a.gfx == "gfx1201" and a.driver_version == "32.0.21043.1001"


def test_parse_video_controllers_list_and_empty():
    text = json.dumps(
        [
            {"Name": "NVIDIA GeForce RTX 4070", "PNPDeviceID": "PCI\\VEN_10DE&DEV_2786", "DriverVersion": None},
            {"Name": "AMD Radeon(TM) Graphics", "PNPDeviceID": "PCI\\VEN_1002&DEV_164E"},
        ]
    )
    adapters = inst.parse_video_controllers(text)
    assert [a.vendor for a in adapters] == ["nvidia", "amd"]
    assert adapters[0].driver_version == ""
    assert inst.parse_video_controllers("") == []
    assert inst.parse_video_controllers("  \r\n") == []


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        ("32.0.15.7283", (572, 83)),
        ("31.0.15.5222", (552, 22)),
        ("27.21.14.6079", (460, 79)),
        ("32.0.15.8180", (581, 80)),
        ("", None),
        ("abc", None),
    ],
)
def test_nvidia_driver_version(windows, expected):
    assert inst.nvidia_driver_version(windows) == expected


# ---- 入れる PyTorch の種類 ----


def test_choose_single_amd():
    c = inst.choose_backend([amd("AMD Radeon RX 9070 XT")])
    assert (c.backend, c.amd_device) == ("amd", "gfx1201")


def test_choose_amd_prefers_discrete_over_integrated():
    c = inst.choose_backend([amd("AMD Radeon 780M Graphics"), amd("AMD Radeon RX 7900 XTX")])
    assert (c.backend, c.amd_device) == ("amd", "gfx1100")


def test_choose_nvidia_with_amd_igpu_picks_nvidia():
    c = inst.choose_backend([amd("AMD Radeon 780M Graphics"), nv("NVIDIA GeForce RTX 4060 Laptop GPU")])
    assert c.backend == "nvidia"
    assert c.warnings  # 内蔵 GPU を使わない旨


def test_choose_both_discrete_requires_explicit():
    adapters = [amd("AMD Radeon RX 9070 XT"), nv("NVIDIA GeForce RTX 4070")]
    with pytest.raises(inst.SetupError, match="--gpu"):
        inst.choose_backend(adapters)
    assert inst.choose_backend(adapters, "nvidia").backend == "nvidia"
    c = inst.choose_backend(adapters, "amd")
    assert (c.backend, c.amd_device) == ("amd", "gfx1201")


def test_choose_none_is_cpu():
    assert inst.choose_backend([]).backend == "cpu"
    assert inst.choose_backend([Adapter("Microsoft Basic Display Adapter", "other")]).backend == "cpu"


def test_choose_unknown_amd_falls_back_to_cpu_with_hint():
    c = inst.choose_backend([amd("AMD Radeon RX 580")])
    assert c.backend == "cpu"
    assert any("--amd-device" in w for w in c.warnings)


def test_choose_unknown_amd_with_device_override():
    c = inst.choose_backend([amd("AMD Radeon RX 580")], amd_device="GFX1030")
    assert (c.backend, c.amd_device) == ("amd", "gfx1030")


def test_choose_requested_amd_needs_device():
    with pytest.raises(inst.SetupError, match="--amd-device"):
        inst.choose_backend([], "amd")
    assert inst.choose_backend([], "amd", "gfx1200").amd_device == "gfx1200"


def test_choose_amd_device_override_wins():
    c = inst.choose_backend([amd("AMD Radeon RX 9070 XT")], amd_device="gfx1200")
    assert c.amd_device == "gfx1200"


def test_choose_requested_cpu_and_invalid():
    assert inst.choose_backend([amd("AMD Radeon RX 9070 XT")], "CPU").backend == "cpu"
    assert inst.choose_backend([], " auto ").backend == "cpu"
    with pytest.raises(inst.SetupError):
        inst.choose_backend([], "intel")


def test_nvidia_driver_warning():
    old = [nv("NVIDIA GeForce RTX 3060", "31.0.15.5222")]  # 552.22
    assert "552.22" in inst.nvidia_driver_warning(old, [580, 0])
    assert inst.nvidia_driver_warning(old, [525, 0]) is None
    assert inst.nvidia_driver_warning(old, None) is None
    assert inst.nvidia_driver_warning([nv("NVIDIA GeForce RTX 5080", "32.0.15.9597")], [580, 0]) is None


def test_hip_sdk_warnings():
    env = {
        "HIP_PATH": r"C:\Program Files\AMD\ROCm\6.4\\",
        "HIP_PATH_64": r"C:\Program Files\AMD\ROCm\6.4\\",
        "PATH": r"C:\Windows\System32;C:\Program Files\AMD\ROCm\6.4\bin;C:\tools",
    }
    w = inst.hip_sdk_warnings(env, True)
    assert len(w) == 3
    assert "HIP_PATH" in w[0] and "HIP_PATH_64" in w[0]
    assert r"ROCm\6.4\bin" in w[1]
    assert inst.hip_sdk_warnings({"PATH": r"C:\Windows"}, False) == []


# ---- sources.json と PyTorch の版 ----


@pytest.fixture(scope="module")
def sources():
    return inst.load_sources()


def test_select_profile(sources):
    assert inst.select_profile(sources, "amd") == "rocm-10.2.0"
    assert inst.select_profile(sources, "amd", rocm="7.2.1") == "rocm-7.2.1"
    assert inst.select_profile(sources, "nvidia") == "cu132"
    assert inst.select_profile(sources, "nvidia", cuda="12.6") == "cu126"
    assert inst.select_profile(sources, "cpu", rocm="7.2.1") == "cpu"
    with pytest.raises(inst.SetupError):
        inst.select_profile(sources, "amd", rocm="6.4")


def test_every_profile_is_consistent(sources):
    profiles = sources["torch"]["profiles"]
    for backend, name in sources["torch"]["defaults"].items():
        assert profiles[name]["backend"] == backend
    for name, p in profiles.items():
        assert p["backend"] in inst.BACKENDS, name
        assert set(p["expect"]) == {"torch", "torchaudio"}, name  # torchvision は入れない
        flat = " ".join(a for step in p["pip_steps"] for a in step)
        assert "torchvision" not in flat, name
        for pkg, ver in p["expect"].items():
            local = ver.split("+")[1]
            assert re.search(rf"{pkg}(\[[^\]]*\])?(==|-){re.escape(ver.split('+')[0])}(\+|%2B){local}", flat), (name, pkg)
        for a in flat.split():
            if a.startswith("http"):
                assert a.startswith("https://"), a


def test_rocm_10_2_profile(sources):
    p = sources["torch"]["profiles"]["rocm-10.2.0"]
    [step] = inst.torch_pip_steps(p, "gfx1201")
    assert step[step.index("--index-url") + 1] == "https://stable.repo.amd.com/rocm/whl-next/"
    assert "torch[device-gfx1201]==2.15.0+rocm10.2.0" in step
    assert "torchaudio==2.11.1+rocm10.2.0" in step
    assert "gfx1201" in p["devices"] and "26.11.2" in p["driver"]
    with pytest.raises(inst.SetupError):
        inst.torch_pip_steps(p, None)


def test_rocm_7_2_1_profile_keeps_amd_wheels(sources):
    p = sources["torch"]["profiles"]["rocm-7.2.1"]
    urls = [a for step in inst.torch_pip_steps(p, None) for a in step if a.startswith("https://")]
    assert all(u.startswith("https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/") for u in urls)
    assert any("torch-2.9.1%2Brocm7.2.1-cp312" in u for u in urls)
    assert any("torchaudio-2.9.1%2Brocm7.2.1-cp312" in u for u in urls)
    assert "26.2.2" in p["driver"]


def test_python_and_node_sources(sources):
    py, node = sources["python"], sources["node"]
    assert py["version"].startswith("3.12.")
    assert "x86_64-pc-windows-msvc-install_only" in py["url"]
    node_ver = tuple(int(x) for x in node["version"].split("."))
    assert node_ver >= (20, 19, 0)
    assert node["url"].endswith(f"node-v{node['version']}-win-x64.zip")
    assert node["archive_root"] == f"node-v{node['version']}-win-x64"
    for s in (py, node):
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])


def test_pinned_and_constraints():
    for n in ["torch", "torchaudio", "torchvision", "rocm", "rocm-sdk-core", "rocm_sdk_device_gfx1201",
              "amd-torch-device-gfx12-0", "amd_torch_device_gfx1201"]:
        assert inst.is_pinned(n), n
    for n in ["numpy", "rocmlike", "torchcodec", "demucs"]:
        assert not inst.is_pinned(n), n
    text = inst.constraints_text(
        {"torch": "2.15.0+rocm10.2.0", "rocm-sdk-device-gfx1201": "10.2.0", "numpy": "2.3.0", "torchaudio": ""}
    )
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    assert lines == ["rocm-sdk-device-gfx1201==10.2.0", "torch==2.15.0+rocm10.2.0"]


# ---- 依存 ----


def test_pyproject_dependencies_has_no_torch():
    deps = inst.pyproject_dependencies()
    assert deps and "numpy" in deps
    assert not set(inst.requirement_names(deps)) & {"torch", "torchaudio", "torchvision"}


def test_requirement_files_do_not_install_torch():
    files, nodeps = inst.requirement_files()
    assert {f.name for f in files} >= {"requirements-analysis.txt", "requirements-lyrics.txt", "requirements-eval.txt"}
    names = set()
    for f in files:
        names |= set(inst.requirement_names(f.read_text(encoding="utf-8").splitlines()))
    assert not names & {"torch", "torchaudio", "torchvision"}
    assert {"demucs", "beat-this", "librosa", "openai-whisper", "stable-ts", "mir-eval"} <= names
    assert nodeps is None  # 今は --no-deps で入れるものはない


def test_requirement_names():
    lines = ["# c", "", "demucs>=4.1.0,<5", "beat_this  # x", "-r other.txt", "Stable.TS==2.19", "mir_eval[display]>=0.8"]
    assert inst.requirement_names(lines) == ["demucs", "beat-this", "stable-ts", "mir-eval"]


def test_deps_pip_args(tmp_path):
    c = tmp_path / "c.txt"
    args = inst.deps_pip_args(c, ["numpy"], [tmp_path / "r1.txt", tmp_path / "r2.txt"])
    assert args == ["-c", str(c), "numpy", "-r", str(tmp_path / "r1.txt"), "-r", str(tmp_path / "r2.txt")]


def test_media_dl_spec_points_to_release_tarball():
    spec = inst.media_dl_spec()
    assert re.fullmatch(
        r"https://github\.com/liasvistreamgamemusic-sketch/media-download/releases/download/"
        r"[^/]+/liasvistreamgamemusic-sketch-media-dl-\d+\.\d+\.\d+\.tgz",
        spec,
    )


# ---- 引数など ----


def test_parse_args_ignores_setup_flag():
    a = inst.parse_args(["--setup", "--gpu", "amd", "--amd-device", "gfx1201", "--rocm", "7.2.1"])
    assert (a.gpu, a.amd_device, a.rocm, a.force) == ("amd", "gfx1201", "7.2.1", False)
    with pytest.raises(SystemExit):
        inst.parse_args(["--gpu", "intel"])


def test_long_path_warning():
    assert inst.long_path_warning(r"C:\chordgen", False) is None
    long = "C:\\" + "x" * 100
    assert "C:\\chordgen" in inst.long_path_warning(long, False)
    assert inst.long_path_warning(long, True) is None


def test_fingerprint_changes():
    assert inst.fingerprint("a", "b") != inst.fingerprint("ab")
    assert inst.fingerprint("a", b"b") == inst.fingerprint("a", "b")
