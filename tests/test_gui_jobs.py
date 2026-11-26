"""chordgen.gui.jobs のテスト（fastapi は要らない）。

子プロセスは、--progress-json と同じ形の行を出す偽の CLI（下の FAKE_CLI）と、
GUI の偽のジョブ（chordgen.gui.fake_cli）の両方で確かめる。
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from chordgen.gui.jobs import MAX_KEPT_JOBS, TERMINAL, Job, JobManager, JobOptions, NotCancellable, python_command
from test_pipeline_fakes import app_home  # noqa: F401

# 偽の CLI。環境変数 FAKE_PLAN（JSON）の steps を順に実行して、exit の終了コードで終わる
FAKE_CLI = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    plan = json.loads(os.environ["FAKE_PLAN"])
    if plan.get("argv_file"):
        with open(plan["argv_file"], "w", encoding="utf-8") as f:
            json.dump(sys.argv[1:], f)
    if plan.get("grandchild_pidfile"):
        gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open(plan["grandchild_pidfile"], "w") as f:
            f.write(str(gc.pid))
    for step in plan.get("steps", []):
        if "out" in step:
            sys.stdout.buffer.write((step["out"] + "\\n").encode("utf-8"))
            sys.stdout.flush()
        if "err" in step:
            sys.stderr.buffer.write(step["err"].encode("utf-8"))
            sys.stderr.flush()
        if "sleep" in step:
            time.sleep(step["sleep"])
    sys.exit(plan.get("exit", 0))
    """
)


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _progress(stage: str, status: str, index: int, total: int, **kw) -> str:
    return _j({"type": "progress", "stage": stage, "label": stage, "index": index, "total": total,
               "status": status, "elapsed": None, "message": None, "data": None, **kw})


@pytest.fixture
def make_manager(tmp_path, app_home):
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    managers: list[JobManager] = []

    def factory(plan: dict | None = None, *, command: list[str] | None = None, env_extra: dict | None = None,
                start: bool = True) -> JobManager:
        env = dict(os.environ)
        env["FAKE_PLAN"] = json.dumps(plan or {})
        env.update(env_extra or {})
        m = JobManager(gui_dir=app_home / "work" / "gui", command=command or [sys.executable, str(script)], env=env)
        managers.append(m)
        if start:
            m.start()
        return m

    yield factory
    for m in managers:
        m.close(timeout=10)


def _job(m: JobManager, source: str = "https://example.com/v/1", **kw) -> Job:
    job_id = m.new_id()
    kind = "url" if source.startswith("http") else "file"
    kw.setdefault("work_dir", m.job_work_dir(job_id))
    kw.setdefault("upload_dir", m.upload_dir(job_id))
    return Job(id=job_id, source_kind=kind, source=source, source_name=source, **kw)


def _wait(m: JobManager, job_id: str, statuses=TERMINAL, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = m.job_api(job_id, with_events=True)
        if d["status"] in statuses:
            return d
        time.sleep(0.02)
    raise AssertionError(f"ジョブが {statuses} になりません: {m.job_api(job_id)}")


def _logs(m: JobManager, job_id: str) -> list[str]:
    return [d["line"] for _, name, d in m.snapshot(job_id)[0] if name == "log"]


def test_success_and_args(make_manager, tmp_path, app_home):
    out = app_home / "output" / "曲.cat4g.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}", encoding="utf-8")
    argv_file = tmp_path / "argv.json"
    m = make_manager({
        "argv_file": str(argv_file),
        "steps": [
            {"out": "not json"},
            {"out": _progress("fetch", "start", 1, 3)},
            {"out": _progress("fetch", "progress", 1, 3, data={"percent": 10})},
            {"out": _progress("fetch", "progress", 1, 3, data={"percent": 90})},
            {"out": _progress("fetch", "done", 1, 3)},
            {"out": _j({"type": "warning", "message": "警告のテスト"})},
            {"err": "line1\nprogress 10%\rprogress 50%\rprogress 100%\nlast line\n"},
            {"out": _progress("write", "start", 3, 3)},
            {"out": _j({"type": "done", "output": str(out), "title": "取得した題名"})},
        ],
    })
    opts = JobOptions(chord_input="no_drums", bass="slash", snap=False, model="btc", language="ja", bars_per_line=2)
    job = m.submit(_job(m, options=opts, title="-始まりがハイフン", artist="私"))
    d = _wait(m, job.id)
    assert d["status"] == "done", d
    assert d["outputName"] == "曲.cat4g.json"
    assert d["title"] == "取得した題名"
    assert d["currentStage"] == {"stage": "write", "label": "write", "index": 3, "total": 3}
    kinds = [e["type"] for e in d["events"]]
    assert kinds.count("warning") == 1
    # 取得中の progress は最後の 1 件だけ残る
    fetch_progress = [e for e in d["events"] if e["type"] == "progress" and e["status"] == "progress"]
    assert len(fetch_progress) == 1 and fetch_progress[0]["data"] == {"percent": 90}
    assert all("at" in e for e in d["events"])
    logs = _logs(m, job.id)
    assert "line1" in logs and "last line" in logs and "not json" in logs

    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert argv[:2] == ["--progress-json", "--chord-input"]
    assert argv[-2:] == ["--", "https://example.com/v/1"]
    for a in ["--no-snap", "--language=ja", "--title=-始まりがハイフン", "--artist=私", f"--work-dir={job.work_dir}"]:
        assert a in argv
    assert argv[argv.index("--bass") + 1] == "slash" and argv[argv.index("--model") + 1] == "btc"
    assert argv[argv.index("--bars-per-line") + 1] == "2"
    # 終わったら作業フォルダを消す
    assert not job.work_dir.exists()
    saved = json.loads((app_home / "work" / "gui" / "jobs.json").read_text(encoding="utf-8"))
    assert saved["jobs"][0]["id"] == job.id and saved["jobs"][0]["status"] == "done"


def test_error_event(make_manager):
    m = make_manager({"steps": [
        {"out": _progress("probe", "start", 1, 2)},
        {"out": _progress("probe", "failed", 1, 2)},
        {"out": _j({"type": "error", "code": "STAGE_FAILED", "message": "音源の読み込みで失敗しました",
                    "stage": "probe", "exitCode": 1})},
    ], "exit": 1})
    d = _wait(m, m.submit(_job(m)).id)
    assert d["status"] == "failed"
    assert d["error"] == {"code": "STAGE_FAILED", "message": "音源の読み込みで失敗しました", "stage": "probe"}
    assert [e["type"] for e in d["events"]].count("error") == 1


def test_exit_without_error_event(make_manager):
    m = make_manager({"steps": [{"out": _progress("beats", "start", 1, 2)},
                                {"err": "Traceback (most recent call last):\nSegfaultish: boom\n"}], "exit": 3})
    d = _wait(m, m.submit(_job(m)).id)
    assert d["status"] == "failed"
    assert d["error"]["code"] == "PROCESS_FAILED" and d["error"]["stage"] == "beats"
    assert "終了コード 3" in d["error"]["message"] and "boom" in d["error"]["message"]


def test_done_with_nonzero_exit_is_failure(make_manager):
    m = make_manager({"steps": [{"out": _j({"type": "done", "output": "/x/y.cat4g.json", "title": None})}], "exit": 1})
    assert _wait(m, m.submit(_job(m)).id)["status"] == "failed"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # ゾンビ（親がまだ回収していない）は止まったものとみなす
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.skipif(os.name == "nt", reason="POSIX のプロセスグループで確かめる")
def test_cancel_running_kills_tree(make_manager, tmp_path):
    pidfile = tmp_path / "gc.pid"
    m = make_manager({"grandchild_pidfile": str(pidfile),
                      "steps": [{"out": _progress("chords", "start", 1, 1)}, {"sleep": 60}]})
    job = m.submit(_job(m))
    _wait(m, job.id, statuses={"running"})
    deadline = time.monotonic() + 10
    while not pidfile.exists() or not pidfile.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    gc_pid = int(pidfile.read_text())
    t0 = time.monotonic()
    m.cancel(job.id)
    d = _wait(m, job.id, timeout=10)
    assert d["status"] == "cancelled" and d["error"] is None
    assert time.monotonic() - t0 < 8
    deadline = time.monotonic() + 5
    while _alive(gc_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(gc_pid), "孫のプロセスが残っています"
    with pytest.raises(NotCancellable):
        m.cancel(job.id)


def test_queue_runs_one_at_a_time_and_cancel_queued(make_manager, tmp_path):
    argv_file = tmp_path / "argv.json"
    m = make_manager({"argv_file": str(argv_file), "steps": [{"sleep": 1.0}], "exit": 1})
    first = m.submit(_job(m, "https://example.com/1"))
    second = m.submit(_job(m, "https://example.com/2"))
    third = m.submit(_job(m, "https://example.com/3"))
    _wait(m, first.id, statuses={"running"})
    assert m.job_api(second.id)["queuePosition"] == 1
    assert m.job_api(third.id)["queuePosition"] == 2
    assert [j["id"] for j in m.list_api()] == [third.id, second.id, first.id]  # 新しい順
    m.cancel(second.id)
    assert m.job_api(second.id)["status"] == "cancelled"
    assert m.job_api(third.id)["queuePosition"] == 1
    _wait(m, first.id)
    _wait(m, third.id, statuses={"running"} | TERMINAL)
    assert m.job_api(second.id)["startedAt"] is None
    with pytest.raises(KeyError):
        m.cancel("nothing")


def test_log_throttle(make_manager):
    many = "".join(f"line {i}\n" for i in range(500))
    m = make_manager({"steps": [{"err": many}]})
    job = m.submit(_job(m))
    _wait(m, job.id)
    logs = _logs(m, job.id)
    assert 1 <= len(logs) <= 60
    assert any("省きました" in s for s in logs)


def test_uploads_are_removed_unless_keep_work(make_manager, app_home):
    m = make_manager({"steps": [{"out": _j({"type": "done", "output": "/x/a.cat4g.json", "title": "a"})}]})
    jobs = []
    for keep in (False, True):
        job = _job(m, options=JobOptions(keep_work=keep))
        job.upload_dir.mkdir(parents=True)
        (job.upload_dir / "song.wav").write_bytes(b"RIFF")
        job.source = str(job.upload_dir / "song.wav")
        job.source_kind = "file"
        jobs.append(m.submit(job))
    for job in jobs:
        _wait(m, job.id)
    assert not jobs[0].upload_dir.exists()
    assert (jobs[1].upload_dir / "song.wav").exists()


def test_restart_marks_interrupted_jobs_failed(make_manager, app_home):
    state = app_home / "work" / "gui" / "jobs.json"
    state.parent.mkdir(parents=True)
    leftover = app_home / "work" / "gui" / "uploads" / "aaaaaaaaaaaa"
    leftover.mkdir(parents=True)
    (leftover / "x.wav").write_bytes(b"RIFF")
    base = {"sourceKind": "url", "sourceName": "u", "options": {}, "title": None, "artist": None,
            "startedAt": None, "finishedAt": None, "outputName": None, "error": None, "events": []}
    state.write_text(json.dumps({"version": 1, "jobs": [
        {**base, "id": "aaaaaaaaaaaa", "status": "running", "createdAt": "2026-01-03T00:00:00.000Z",
         "currentStage": {"stage": "chords", "label": "コード認識", "index": 3, "total": 6},
         "events": [{"type": "progress", "stage": "chords", "status": "start"}]},
        {**base, "id": "bbbbbbbbbbbb", "status": "queued", "createdAt": "2026-01-02T00:00:00.000Z"},
        {**base, "id": "cccccccccccc", "status": "done", "createdAt": "2026-01-01T00:00:00.000Z",
         "outputName": "c.cat4g.json"},
        {"id": "../evil", "status": "done"},
        "broken",
    ]}), encoding="utf-8")
    m = make_manager()
    jobs = m.list_api()
    assert [j["id"] for j in jobs] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"]
    a = m.job_api("aaaaaaaaaaaa", with_events=True)
    assert a["status"] == "failed" and a["error"]["code"] == "SERVER_STOPPED" and a["error"]["stage"] == "chords"
    assert [e["type"] for e in a["events"]] == ["progress", "error"]
    assert m.job_api("bbbbbbbbbbbb")["status"] == "failed"
    assert m.job_api("cccccccccccc")["outputName"] == "c.cat4g.json"
    assert not leftover.exists()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert {j["id"]: j["status"] for j in saved["jobs"]}["aaaaaaaaaaaa"] == "failed"


def test_broken_state_file_is_ignored(make_manager, app_home):
    state = app_home / "work" / "gui" / "jobs.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json", encoding="utf-8")
    assert make_manager().list_api() == []


def test_keeps_latest_jobs_only(make_manager):
    m = make_manager({"steps": []}, start=False)
    m.start()
    for i in range(MAX_KEPT_JOBS + 5):
        job = _job(m)
        job.created_at = f"2026-01-01T00:00:{i // 10:02d}.{i % 10:03d}Z"
        job.status = "done"
        with m._lock:
            m._jobs[job.id] = job
            m._prune()
    assert len(m.list_api()) == MAX_KEPT_JOBS


# ---- GUI の偽のジョブ（chordgen.gui.fake_cli） ----


@pytest.fixture
def fake_manager(make_manager):
    return lambda: make_manager(command=python_command("chordgen.gui.fake_cli"),
                                env_extra={"CHORDGEN_GUI_FAKE_DELAY": "0.02"})


def test_fake_job_writes_output(fake_manager, app_home):
    m = fake_manager()
    job = m.submit(_job(m, "https://example.com/warn", title="偽の題名", artist="誰か"))
    d = _wait(m, job.id)
    assert d["status"] == "done", d
    assert d["outputName"] == "偽の題名.cat4g.json"
    doc = json.loads((app_home / "output" / "偽の題名.cat4g.json").read_text(encoding="utf-8"))
    assert doc["song"]["title"] == "偽の題名" and doc["song"]["artist"] == "誰か"
    starts = [e["stage"] for e in d["events"] if e["type"] == "progress" and e["status"] == "start"]
    assert starts == ["fetch", "probe", "beats", "chords", "key", "snap", "write"]
    assert any(e["type"] == "warning" for e in d["events"])


def test_fake_job_validates_with_schema(fake_manager, app_home):
    pytest.importorskip("jsonschema")
    from chordgen.songformat.validate import validate_document

    m = fake_manager()
    d = _wait(m, m.submit(_job(m, "https://example.com/ok")).id)
    validate_document(json.loads((app_home / "output" / d["outputName"]).read_text(encoding="utf-8")))


def test_fake_job_fails(fake_manager, tmp_path):
    m = fake_manager()
    d = _wait(m, m.submit(_job(m, "https://example.com/fail")).id)
    assert d["status"] == "failed"
    assert d["error"]["code"] == "UNAVAILABLE" and d["error"]["stage"] == "fetch"

    audio = tmp_path / "fail.wav"
    audio.write_bytes(b"RIFF")
    d = _wait(m, m.submit(_job(m, str(audio))).id)
    assert d["status"] == "failed" and d["error"]["stage"] == "chords"


def test_fake_job_with_lyrics_and_separation(fake_manager, tmp_path):
    m = fake_manager()
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("そらに\n", encoding="utf-8")
    job = _job(m, str(audio), lyrics_path=lyrics, options=JobOptions(bass="root_fix", has_lyrics=True))
    d = _wait(m, m.submit(job).id)
    starts = [e["stage"] for e in d["events"] if e["type"] == "progress" and e["status"] == "start"]
    assert starts == ["probe", "separate", "beats", "chords", "key", "snap", "bass", "lyrics", "write"]
    assert d["outputName"] == "song.cat4g.json"


def test_fake_job_cancel(fake_manager):
    m = fake_manager()
    job = m.submit(_job(m, "https://example.com/slow"))
    _wait(m, job.id, statuses={"running"})
    m.cancel(job.id)
    assert _wait(m, job.id, timeout=10)["status"] == "cancelled"
