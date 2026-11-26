"""python -m chordgen.gui（起動・ポート・二重起動）と run.bat の画面の起動のテスト。"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from chordgen.gui import DEFAULT_PORT
from chordgen.gui.__main__ import InstanceLock, main, open_socket

ROOT = Path(__file__).resolve().parent.parent
RUN_BAT = ROOT / "run.bat"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_open_socket_skips_busy_default_port():
    busy = socket.socket()
    try:
        try:
            busy.bind(("127.0.0.1", DEFAULT_PORT))
            busy.listen(1)
        except OSError:
            pass  # ほかのものが使っている。それでも空いているポートを探せること
        s = open_socket("127.0.0.1", None)
        try:
            assert s.getsockname()[1] != DEFAULT_PORT
            assert s.getsockname()[0] == "127.0.0.1"
        finally:
            s.close()
    finally:
        busy.close()


def test_open_socket_explicit_busy_port_fails():
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        with pytest.raises(OSError):
            open_socket("127.0.0.1", busy.getsockname()[1])
    finally:
        busy.close()


def test_instance_lock(tmp_path):
    a, b = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert a.acquire()
    try:
        assert not b.acquire()
        assert b.running_url() is None  # server.json がない
        a.publish("http://127.0.0.1:1/?t=x", 1)
        assert b.running_url() is None  # 応えない
    finally:
        a.release()
    assert not (tmp_path / "server.json").exists()
    assert b.acquire()
    b.release()


def test_missing_dependencies_message(monkeypatch, capsys, app_home):
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # import すると ImportError
    assert main(["--no-browser"]) == 1
    err = capsys.readouterr().err
    assert "画面に要るライブラリが入っていません" in err and "setup.bat" in err


@pytest.fixture
def app_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CHORDGEN_HOME", str(home))
    return home


def _read_url(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        line = proc.stdout.readline().decode("utf-8", errors="replace")
        if not line:
            if proc.poll() is not None:
                break
            continue
        seen += line
        m = re.search(r"(http://127\.0\.0\.1:\d+/\?t=\S+)", line)
        if m:
            return m.group(1)
    raise AssertionError(f"URL が出ません: {seen}")


def test_end_to_end_with_real_server(app_home, tmp_path):
    """本物の uvicorn で起動し、合言葉の Cookie・ジョブ・二重起動・終了を確かめる。"""
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    env = {**os.environ, "CHORDGEN_HOME": str(app_home), "CHORDGEN_GUI_FAKE": "1", "CHORDGEN_GUI_FAKE_DELAY": "0.02",
           "PYTHONPATH": str(ROOT), "NO_PROXY": "*"}
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "chordgen.gui", "--no-browser", "--port", str(port)],
                            cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        url = _read_url(proc)
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, trust_env=False, follow_redirects=True) as c:
            assert c.get("/api/info").status_code == 403
            r = c.get(url.replace(base, ""))
            assert r.status_code == 200 and r.url.path == "/"  # 303 で / へ。Cookie が付いて画面が出る
            assert c.get("/api/info").json()["fake"] is True
            job_id = c.post("/api/jobs", files={"file": ("曲.wav", b"RIFF", "audio/wav")}).json()["id"]
            with c.stream("GET", f"/api/jobs/{job_id}/events") as s:
                body = s.read().decode("utf-8")
            assert '"status": "done"' in body
            assert c.get("/api/outputs").json()[0]["name"] == "曲.cat4g.json"
            assert c.get("/", headers={"Host": "evil.example"}).status_code == 400

            # 二つ目の起動は、起動中の画面の URL を出して終わる
            second = subprocess.run([sys.executable, "-m", "chordgen.gui", "--no-browser"], cwd=tmp_path, env=env,
                                    capture_output=True, timeout=60)
            assert second.returncode == 0
            assert url in second.stdout.decode("utf-8")

            assert c.post("/api/shutdown").json() == {"ok": True}
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert not (app_home / "work" / "gui" / "server.json").exists()


# ---- run.bat ----


def _code_lines() -> list[str]:
    text = RUN_BAT.read_bytes().decode("ascii")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().lower().startswith("rem")]


def test_run_bat_starts_gui_without_arguments():
    lines = _code_lines()
    run = lines.index(":run")
    assert lines[run + 1] == 'if "%~1"=="" goto gui'
    assert lines[run + 2] == 'if /i "%~1"=="--gui" goto gui_opts'
    gui = lines.index(":gui")
    assert lines[gui + 1] == '"%CHORDGEN_PYTHON%" -P -s -m chordgen.gui'
    assert 'if not "%CG_RC%"=="0" pause' in lines[gui + 1:gui + 4]  # 失敗したら窓を閉じずに止める
    call = lines.index(":gui_call")
    assert lines[call + 1] == '"%CHORDGEN_PYTHON%" -P -s -m chordgen.gui%CG_ARGS%'
    # 引数があれば今のまま CLI に渡す
    assert '"%CHORDGEN_PYTHON%" -P -s -m chordgen.cli %*' in lines


def test_run_bat_gui_options_go_through_collect_args():
    lines = _code_lines()
    i = lines.index(":gui_opts")
    assert lines[i + 1:i + 3] == ['set "CG_NEXT=gui_call"', "goto collect_args"]
