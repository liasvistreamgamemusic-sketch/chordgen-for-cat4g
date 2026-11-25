"""run.bat / setup.bat / bootstrap.ps1 の中身の検査（Windows がなくても確かめられること）。

要件の表「起動とフォルダの外に書き出さない仕組み」の環境変数がすべて run.bat の中でアプリのフォルダに
向いているかを、run.bat の set の行を読んで展開して確かめる。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN_BAT = ROOT / "run.bat"
SETUP_BAT = ROOT / "setup.bat"
BOOTSTRAP = ROOT / "portable" / "bootstrap.ps1"

# 要件の表にある、アプリのフォルダに向ける環境変数
REQUIRED_DIR_VARS = [
    "MEDIA_DL_BIN_DIR",
    "XDG_CACHE_HOME",
    "HOME",
    "XDG_CONFIG_HOME",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "PIP_CACHE_DIR",
    "DENO_DIR",
    "TEMP",
    "TMP",
]
# .gitignore にあり、書き出しに使ってよいアプリのフォルダの中の場所
IGNORED_TOP = {"runtime", "models", "work", "output", "cache", "third_party", "node_modules"}

APP = r"D:\アプリ\chord gen (test)"  # 空白・日本語・括弧を含むパスで展開してみる
WINDOWS_ENV = {
    "PATH": r"C:\Windows\system32;C:\Program Files (x86)\Something",
    "SystemRoot": r"C:\Windows",
    "TEMP": r"C:\Users\u\AppData\Local\Temp",
}


def _text(p: Path) -> str:
    return p.read_bytes().decode("utf-8")


def _set_lines(text: str) -> list[tuple[str, str]]:
    """`set "NAME=VALUE"` の行（ブロックの外のもの）を順に返す。"""
    out = []
    for line in text.splitlines():
        m = re.match(r'^\s*set\s+"([^=]+)=(.*)"\s*$', line, re.IGNORECASE)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _expand(value: str, env: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        name, slice_ = m.group(1), m.group(2)
        v = env.get(name.upper(), "")
        if slice_ == ":~0,-1":
            v = v[:-1]
        elif slice_ == ": =":
            v = v.replace(" ", "")
        return v

    return re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)(:~0,-1|: =)?%", sub, value)


@pytest.fixture(scope="module")
def run_env() -> dict[str, str]:
    """run.bat の set を上から順に実行したあとの環境変数（%~dp0 はアプリのフォルダ + \\）。"""
    env = {k.upper(): v for k, v in WINDOWS_ENV.items()}
    for name, value in _set_lines(_text(RUN_BAT)):
        value = value.replace("%~dp0", APP + "\\")
        v = _expand(value, env)
        if v:
            env[name.upper()] = v
        else:
            env.pop(name.upper(), None)
    return env


def test_line_endings_are_crlf():
    for p in (RUN_BAT, SETUP_BAT, BOOTSTRAP):
        data = p.read_bytes()
        assert b"\n" in data
        assert data.count(b"\n") == data.count(b"\r\n"), f"{p.name} に LF だけの行があります"


def test_bootstrap_has_utf8_bom():
    # Windows PowerShell 5.1 は BOM のない .ps1 を ANSI（日本語の Windows では CP932）で読む
    assert BOOTSTRAP.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("bat", [RUN_BAT, SETUP_BAT])
def test_bat_is_ascii_and_starts_with_setlocal(bat):
    # cmd.exe は .bat をコンソールのコードページ（日本語の Windows では 932）で読む。UTF-8 の日本語があると
    # 行の区切りを読み違えて別の行がコマンドとして走ることを、WSL から cmd.exe で実際に確かめた。だから ASCII だけにする
    text = _text(bat)
    assert text.isascii()
    lines = text.splitlines()
    assert lines[0].lower() == "@echo off"
    if bat == RUN_BAT:
        assert lines[1].lower().startswith("setlocal"), "環境変数を run.bat の外に漏らさないため setlocal を先頭に置く"
    assert "chcp" not in text.lower(), "コンソールのコードページを変えない"


def test_endlocal_returns_exit_code():
    assert "\r\nendlocal & exit /b %CG_RC%\r\n" in _text(RUN_BAT)


def test_required_vars_point_inside_app(run_env):
    for name in REQUIRED_DIR_VARS:
        assert name in run_env, f"{name} を run.bat で設定していません"
        value = run_env[name]
        assert value.startswith(APP + "\\"), f"{name}={value} がアプリのフォルダの外です"
        top = value[len(APP) + 1 :].split("\\")[0]
        assert top in IGNORED_TOP, f"{name} の置き場所 {top} が .gitignore にありません"


def test_chordgen_home_and_python_settings(run_env):
    assert run_env["CHORDGEN_HOME"] == APP
    assert run_env["PYTHONNOUSERSITE"] == "1"
    assert run_env["PYTHONPATH"] == APP
    assert "PYTHONHOME" not in run_env
    assert run_env["CHORDGEN_PYTHON"] == APP + r"\runtime\python\python.exe"
    assert run_env["CHORDGEN_NODE"] == APP + r"\runtime\node\node.exe"


def test_extra_vars_point_inside_app(run_env):
    # 表にはないが、既定ではユーザーのフォルダに書き出すもの
    for name in ["NPM_CONFIG_CACHE", "NPM_CONFIG_PREFIX", "NPM_CONFIG_USERCONFIG", "CUDA_CACHE_PATH",
                 "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
        assert run_env[name].startswith(APP + "\\"), name


def test_path_order(run_env):
    parts = run_env["PATH"].split(";")
    assert parts[0] == APP + r"\runtime\python"
    assert APP + r"\runtime\node" in parts[:4]
    assert run_env["MEDIA_DL_BIN_DIR"] in parts[:4], "ffmpeg を共有するため MEDIA_DL_BIN_DIR を PATH の先頭側に置く"
    # Windows 側の PATH は後ろに残す
    assert parts[-2:] == WINDOWS_ENV["PATH"].split(";")


def test_no_sets_inside_parenthesized_blocks():
    # PATH に "Program Files (x86)" があるとブロックの中の %PATH% の展開で括弧が閉じてしまう
    depth = 0
    for line in _text(RUN_BAT).splitlines():
        s = line.strip()
        if s.lower().startswith("rem") or s.startswith("::"):
            continue
        if depth > 0:
            assert "%PATH%" not in s, line
        depth += s.count("(") - s.count(")") if not s.lower().startswith("for ") else 0


def test_flow_labels_exist():
    text = "\n".join(l for l in _text(RUN_BAT).splitlines() if not l.lstrip().lower().startswith("rem"))
    labels = set(re.findall(r"^:([A-Za-z_]+)\s*$", text.replace("\r\n", "\n"), re.MULTILINE))
    targets = set(re.findall(r"\bgoto\s+([A-Za-z_]+)", text, re.IGNORECASE)) | set(
        re.findall(r"\bcall\s+:([A-Za-z_]+)", text, re.IGNORECASE)
    )
    assert targets - {"eof"} <= labels


def test_run_invokes_cli_and_setup():
    text = _text(RUN_BAT)
    assert '"%CHORDGEN_PYTHON%" -P -s -m chordgen.cli %*' in text
    assert '"%CHORDGEN_PYTHON%" -P -s -m chordgen.cli --help' in text  # 引数なしならヘルプ
    assert r'"%CHORDGEN_HOME%\portable\install.py"' in text
    assert r'-ExecutionPolicy Bypass -File "%CHORDGEN_HOME%\portable\bootstrap.ps1"' in text
    assert r'if not exist "%CG_RUNTIME%\setup-complete.json" goto setup_auto' in text
    # パスは引用符で囲む
    for line in text.splitlines():
        if "%CHORDGEN_PYTHON%" in line and not line.lstrip().lower().startswith(("set ", "rem", "if ")):
            assert '"%CHORDGEN_PYTHON%"' in line, line


def test_setup_bat_delegates_to_run_bat():
    text = _text(SETUP_BAT)
    assert 'call "%~dp0run.bat" --setup %*' in text


def test_bootstrap_reads_sources_and_verifies_hash():
    text = _text(BOOTSTRAP)
    assert "sources.json" in text
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "ensurepip" in text
    assert "$ErrorActionPreference = 'Stop'" in text
