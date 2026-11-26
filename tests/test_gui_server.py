"""chordgen.gui.server のテスト。ジョブは偽のジョブ（CHORDGEN_GUI_FAKE と同じ chordgen.gui.fake_cli）で走らせる。

fastapi・httpx・python-multipart がなければ skip する。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from chordgen.gui.info import DeviceProbe  # noqa: E402
from chordgen.gui.server import ServerConfig, create_app, upload_name  # noqa: E402
from test_pipeline_fakes import app_home  # noqa: E402,F401

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = "test-token-abc"
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "specs" / "format" / "samples" / "analyzed.cat4g.json"


class Gui:
    def __init__(self, client: TestClient, config: ServerConfig, opened: list, shutdowns: list):
        self.client = client
        self.config = config
        self.opened = opened
        self.shutdowns = shutdowns

    def wait(self, job_id: str, statuses=("done", "failed", "cancelled"), timeout: float = 20.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            d = self.client.get(f"/api/jobs/{job_id}").json()
            if d["status"] in statuses:
                return d
            time.sleep(0.03)
        raise AssertionError(f"ジョブが {statuses} になりません")


@pytest.fixture
def gui(app_home, monkeypatch):
    monkeypatch.setenv("CHORDGEN_GUI_FAKE_DELAY", "0.02")
    opened: list = []
    shutdowns: list = []
    config = ServerConfig(
        port=PORT, token=TOKEN, fake=True, ping_interval=0.3,
        device_probe=DeviceProbe(fixed={"kind": "cuda", "name": "Test GPU", "torchAvailable": True}),
        open_path=lambda p, select: opened.append((p, select)),
        request_shutdown=lambda: shutdowns.append(True),
    )
    app = create_app(config)
    with TestClient(app, base_url=BASE) as client:
        client.headers[config_header()] = TOKEN
        yield Gui(client, config, opened, shutdowns)


def config_header() -> str:
    from chordgen.gui import TOKEN_HEADER

    return TOKEN_HEADER


def _bare(gui: Gui, **kw) -> TestClient:
    """合言葉を持たないクライアント（同じアプリ）。"""
    return TestClient(gui.client.app, base_url=kw.get("base_url", BASE))


# ---- 守り ----


def test_api_requires_token(gui):
    bare = _bare(gui)
    r = bare.get("/api/info")
    assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN"
    assert bare.get("/api/info", headers={"X-Chordgen-Token": "wrong"}).status_code == 403
    bare.cookies.set("chordgen_token", "wrong")
    assert bare.get("/api/jobs").status_code == 403
    assert bare.post("/api/shutdown").status_code == 403
    assert gui.shutdowns == []


def test_token_url_sets_cookie_and_redirects(gui):
    bare = _bare(gui)
    r = bare.get("/")
    assert r.status_code == 403 and "コンソールに出ている URL から開いてください" in r.text
    assert bare.get("/?t=wrong", follow_redirects=False).status_code == 403
    r = bare.get(f"/?t={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"chordgen_token={TOKEN}")
    low = cookie.lower()
    assert "httponly" in low and "samesite=strict" in low and "path=/" in low
    assert r.headers["referrer-policy"] == "no-referrer"
    r = bare.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert bare.get("/api/info").status_code == 200  # Cookie で通る


def test_bad_host_is_rejected(gui):
    for host in ("evil.example:8765", "127.0.0.1:9999", "127.0.0.1", "attacker.localhost:8765"):
        r = gui.client.get("/api/info", headers={"Host": host})
        assert r.status_code == 400, host
        assert gui.client.get("/", headers={"Host": host}).status_code == 400
    assert gui.client.get("/api/info", headers={"Host": f"localhost:{PORT}"}).status_code == 200


def test_bad_origin_is_rejected_for_state_changes(gui):
    r = gui.client.post("/api/jobs", data={"url": "https://example.com/x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "BAD_ORIGIN"
    r = gui.client.post("/api/jobs", data={"url": "https://example.com/x"}, headers={"Origin": "null"})
    assert r.status_code == 403
    r = gui.client.delete("/api/outputs/a.cat4g.json", headers={"Origin": f"http://localhost:{PORT + 1}"})
    assert r.status_code == 403
    r = gui.client.post("/api/shutdown", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and gui.shutdowns == []
    r = gui.client.post("/api/jobs", data={"url": "https://example.com/x"}, headers={"Origin": BASE})
    assert r.status_code == 201
    gui.wait(r.json()["id"])


def test_static_files_are_not_cached_stale(gui):
    bare = _bare(gui)
    for path in ("/app.js", "/api.js", "/util.js", "/app.css"):
        r = bare.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == "no-cache", path
    bare.get(f"/?t={TOKEN}")
    r = bare.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    # 合言葉の Cookie を付けるリダイレクトと API は残さない
    assert bare.get(f"/?t={TOKEN}", follow_redirects=False).headers["cache-control"] == "no-store"
    assert gui.client.get("/api/info").headers["cache-control"] == "no-store"


def test_static_files_are_served(gui):
    bare = _bare(gui)
    r = bare.get("/util.js")
    assert r.status_code == 200
    assert bare.get("/index.html", follow_redirects=False).status_code == 303


# ---- 情報 ----


def test_info(gui):
    d = gui.client.get("/api/info").json()
    assert d["version"]
    assert d["device"] == {"kind": "cuda", "name": "Test GPU", "torchAvailable": True}
    assert d["mediaDlAvailable"] is True  # 偽のジョブのときは URL も試せる
    assert d["defaults"] == {"chordInput": "original", "bass": "off", "snap": True, "model": "chordmini",
                             "barsPerLine": 4, "language": None, "keepWork": False}
    assert [c["value"] for c in d["choices"]["chordInput"]] == ["original", "no_drums", "no_drums_vocals", "boost_other"]
    assert [c["value"] for c in d["choices"]["bass"]] == ["off", "root_fix", "slash"]
    assert [c["value"] for c in d["choices"]["model"]] == ["chordmini", "btc"]
    assert all(c["label"] and c["description"] for c in d["choices"]["bass"])
    assert [s["stage"] for s in d["stages"]] == ["fetch", "probe", "separate", "beats", "chords", "key", "snap",
                                                  "bass", "lyrics", "write"]
    assert d["limits"]["maxUploadBytes"] >= 1 << 30


def test_device_unknown_until_probed():
    p = DeviceProbe(fixed={"kind": "cpu", "name": None, "torchAvailable": False}, delay=0.3)
    assert p.get()["kind"] == "unknown"
    p.start()
    deadline = time.monotonic() + 5
    while p.get()["kind"] == "unknown":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert p.get()["kind"] == "cpu"


# ---- ジョブ ----


def test_url_job_completes(gui, app_home):
    r = gui.client.post("/api/jobs", data={"url": "https://example.com/warn", "title": "題名", "artist": "人",
                                           "snap": "false", "bass": "slash", "barsPerLine": "2", "language": "ja"})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    d = gui.wait(job_id)
    assert d["status"] == "done", d
    assert d["sourceKind"] == "url" and d["sourceName"] == "https://example.com/warn"
    assert d["options"] == {"chordInput": "original", "bass": "slash", "snap": False, "model": "chordmini",
                            "language": "ja", "barsPerLine": 2, "keepWork": False, "hasLyrics": False}
    assert d["outputName"] == "題名.cat4g.json" and d["title"] == "題名" and d["artist"] == "人"
    assert d["createdAt"].endswith("Z") and d["startedAt"] and d["finishedAt"]
    assert any(e["type"] == "warning" for e in d["events"])
    starts = [e["stage"] for e in d["events"] if e["type"] == "progress" and e["status"] == "start"]
    assert starts == ["fetch", "probe", "separate", "beats", "chords", "key", "bass", "write"]
    assert (app_home / "output" / "題名.cat4g.json").exists()
    assert gui.client.get("/api/jobs").json()[0]["id"] == job_id


def test_file_job_with_lyrics(gui, app_home):
    files = {"file": ("C:\\Users\\me\\私の曲.WAV", b"RIFF....", "audio/wav")}
    r = gui.client.post("/api/jobs", files=files, data={"lyrics": "そらに\r\nうかぶ\r\n", "keepWork": "true"})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    d = gui.wait(job_id)
    assert d["status"] == "done"
    assert d["sourceName"] == "私の曲.WAV" and d["sourceKind"] == "file"
    assert d["options"]["hasLyrics"] is True and d["options"]["keepWork"] is True
    stages = [e["stage"] for e in d["events"] if e["type"] == "progress" and e["status"] == "start"]
    assert "lyrics" in stages and "separate" in stages
    up = app_home / "work" / "gui" / "uploads" / job_id
    assert (up / "私の曲.wav").read_bytes() == b"RIFF...."
    assert (up / "lyrics.txt").read_text(encoding="utf-8") == "そらに\nうかぶ\n"  # keepWork なので残る
    assert d["outputName"] == "私の曲.cat4g.json"


def test_uploaded_file_is_removed_after_job(gui, app_home):
    r = gui.client.post("/api/jobs", files={"file": ("a.mp3", b"ID3", "audio/mpeg")})
    job_id = r.json()["id"]
    assert gui.wait(job_id)["status"] == "done"
    assert not (app_home / "work" / "gui" / "uploads" / job_id).exists()
    assert not (app_home / "work" / "gui" / "jobs" / job_id).exists()


@pytest.mark.parametrize(("data", "files", "code"), [
    ({}, None, "NO_SOURCE"),
    ({"url": "   "}, None, "NO_SOURCE"),
    ({"url": "https://example.com/x"}, {"file": ("a.wav", b"RIFF", "audio/wav")}, "BOTH_SOURCES"),
    ({"url": "ftp://example.com/x"}, None, "INVALID_URL"),
    ({"url": "C:\\music\\a.wav"}, None, "INVALID_URL"),
    ({"url": "https://example.com/x", "chordInput": "bogus"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "bass": "nope"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "model": "gpt"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "snap": "maybe"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "barsPerLine": "0"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "barsPerLine": "abc"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "language": "ja; rm"}, None, "INVALID_OPTION"),
    ({"url": "https://example.com/x", "title": "x" * 1000}, None, "INVALID_FIELD"),
    (None, {"file": ("empty.wav", b"", "audio/wav")}, "EMPTY_FILE"),
])
def test_bad_job_requests(gui, app_home, data, files, code):
    # multipart で送るため、ファイルがないときも空の欄を 1 つ付ける
    files = files if files is not None else {"dummy": ("", b"", "text/plain")}
    r = gui.client.post("/api/jobs", data=data or {}, files=files)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == code and err["message"]
    assert gui.client.get("/api/jobs").json() == []
    up = app_home / "work" / "gui" / "uploads"
    assert not up.exists() or list(up.iterdir()) == []


def test_json_body_is_rejected(gui):
    r = gui.client.post("/api/jobs", json={"url": "https://example.com/x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_FORM"


def test_too_large_upload(gui, monkeypatch, app_home):
    import chordgen.gui.server as server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    r = gui.client.post("/api/jobs", files={"file": ("a.wav", b"x" * 100, "audio/wav")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert list((app_home / "work" / "gui" / "uploads").iterdir()) == []


def test_failed_job(gui):
    job_id = gui.client.post("/api/jobs", data={"url": "https://example.com/fail"}).json()["id"]
    d = gui.wait(job_id)
    assert d["status"] == "failed"
    assert d["error"]["code"] == "UNAVAILABLE" and d["error"]["stage"] == "fetch" and d["error"]["message"]
    assert any(e["type"] == "error" for e in d["events"])


def test_queue_and_cancel(gui):
    first = gui.client.post("/api/jobs", data={"url": "https://example.com/slow/1"}).json()["id"]
    second = gui.client.post("/api/jobs", data={"url": "https://example.com/slow/2"}).json()["id"]
    gui.wait(first, statuses=("running",))
    d2 = gui.client.get(f"/api/jobs/{second}").json()
    assert d2["status"] == "queued" and d2["queuePosition"] == 1
    r = gui.client.post(f"/api/jobs/{second}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    r = gui.client.post(f"/api/jobs/{first}/cancel")
    assert r.status_code == 200
    assert gui.wait(first, timeout=10)["status"] == "cancelled"
    assert gui.client.post(f"/api/jobs/{first}/cancel").status_code == 409
    assert gui.client.post("/api/jobs/nothing/cancel").status_code == 404
    assert gui.client.get("/api/jobs/nothing").json()["error"]["code"] == "JOB_NOT_FOUND"


# ---- SSE ----


def _parse_sse(text: str) -> list[tuple[str | None, str, object]]:
    out = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        ev_id, name, data = None, "message", ""
        for line in block.splitlines():
            if line.startswith("id: "):
                ev_id = line[4:]
            elif line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        out.append((ev_id, name, json.loads(data)))
    return out


def test_sse_replays_and_closes_after_done(gui):
    job_id = gui.client.post("/api/jobs", data={"url": "https://example.com/warn"}).json()["id"]
    # 走っている途中から繋いでも、終わるまで流れて閉じる
    with gui.client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        live = _parse_sse(r.read().decode("utf-8"))
    names = [n for _, n, _ in live]
    assert names[-1] == "status" and live[-1][2]["status"] == "done"
    assert "progress" in names and "warning" in names and "log" in names
    assert "error" not in names
    log_ev = next(d for _, n, d in live if n == "log")
    assert isinstance(log_ev["line"], str)

    # 終わった後に繋ぐと、これまでの記録を全部流し直して、最後に status を送って閉じる
    r = gui.client.get(f"/api/jobs/{job_id}/events")
    replay = _parse_sse(r.text)
    assert replay[-1][1] == "status" and replay[-1][2]["status"] == "done"
    assert [n for _, n, _ in replay].count("status") == 1
    starts = [d["stage"] for _, n, d in replay if n == "progress" and d["status"] == "start"]
    assert starts[0] == "fetch" and starts[-1] == "write"
    assert any(n == "warning" for _, n, _ in replay)

    # Last-Event-ID からの続き（新しいものが無ければ status だけ）
    last_id = replay[-1][0]
    r = gui.client.get(f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": last_id})
    resumed = _parse_sse(r.text)
    assert [n for _, n, _ in resumed] == ["status"]


def test_sse_ping(gui):
    job_id = gui.client.post("/api/jobs", data={"url": "https://example.com/slow"}).json()["id"]
    with gui.client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        text = ""
        for chunk in r.iter_text():
            text += chunk
            if ": ping" in text:
                break
    assert ": ping" in text
    gui.client.post(f"/api/jobs/{job_id}/cancel")
    gui.wait(job_id)


def test_sse_unknown_job(gui):
    assert gui.client.get("/api/jobs/nothing/events").status_code == 404


# ---- 出力 ----


def _put_output(app_home: Path, name: str, mtime: float | None = None) -> Path:
    out = app_home / "output"
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_bytes(SAMPLE.read_bytes())
    if mtime is not None:
        import os

        os.utime(p, (mtime, mtime))
    return p


def test_outputs_list(gui, app_home):
    _put_output(app_home, "古い.cat4g.json", 1_700_000_000)
    _put_output(app_home, "新しい.cat4g.json", 1_800_000_000)
    (app_home / "output" / "壊れた.cat4g.json").write_text("{oops", encoding="utf-8")
    (app_home / "output" / "other.json").write_text("{}", encoding="utf-8")
    items = gui.client.get("/api/outputs").json()
    assert [i["name"] for i in items] == ["新しい.cat4g.json", "古い.cat4g.json"]
    it = items[0]
    assert (it["title"], it["artist"], it["originalKey"], it["bpm"], it["durationSec"]) == ("しろいくも", "サンプル", "A", 90, 64)
    assert it["chordCount"] > 0 and it["size"] > 0 and it["modifiedAt"].endswith("Z")


def test_output_get_and_download(gui, app_home):
    _put_output(app_home, "しろいくも.cat4g.json")
    name = quote("しろいくも.cat4g.json")
    r = gui.client.get(f"/api/outputs/{name}")
    assert r.status_code == 200 and r.json()["song"]["title"] == "しろいくも"
    assert "content-disposition" not in r.headers
    r = gui.client.get(f"/api/outputs/{name}?download=1")
    cd = r.headers["content-disposition"]
    assert cd.startswith("attachment;") and f"filename*=UTF-8''{name}" in cd and 'filename="' in cd
    cd.encode("latin-1")  # ヘッダは ASCII だけ


@pytest.mark.parametrize("name", ["..", "%2e%2e", ".hidden.cat4g.json", "a%5Cb.cat4g.json", "x.json",
                                  "a.cat4g.json%3Astream", ".cat4g.json"])
def test_output_bad_names(gui, app_home, name):
    _put_output(app_home, "a.cat4g.json")
    r = gui.client.get(f"/api/outputs/{name}")
    assert r.status_code in (400, 404), (name, r.status_code)
    assert r.json()["error"]["code"] in ("BAD_NAME", "NOT_FOUND", "OUTPUT_NOT_FOUND")
    r = gui.client.delete(f"/api/outputs/{name}")
    assert r.status_code in (400, 404, 405)
    assert (app_home / "output" / "a.cat4g.json").exists()


def test_output_traversal_outside(gui, app_home):
    secret = app_home / "secret.cat4g.json"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("{}", encoding="utf-8")
    for path in ("/api/outputs/../secret.cat4g.json", "/api/outputs/%2E%2E%2Fsecret.cat4g.json",
                 "/api/outputs/..%2Fsecret.cat4g.json", "/api/outputs/..%5Csecret.cat4g.json"):
        r = gui.client.get(path)
        assert r.status_code in (400, 404), path
        assert "format" not in r.text


def test_output_delete(gui, app_home):
    p = _put_output(app_home, "消す.cat4g.json")
    r = gui.client.delete(f"/api/outputs/{quote(p.name)}")
    assert r.status_code == 200 and not p.exists()
    assert gui.client.delete(f"/api/outputs/{quote(p.name)}").status_code == 404


def test_open_folder(gui, app_home):
    p = _put_output(app_home, "a.cat4g.json")
    assert gui.client.post("/api/open-folder", json={"target": "output"}).status_code == 200
    assert gui.client.post("/api/open-folder", json={"target": "work"}).status_code == 200
    assert gui.client.post("/api/open-folder", json={"target": "output", "name": "a.cat4g.json"}).status_code == 200
    assert [(Path(x).name, s) for x, s in gui.opened] == [("output", False), ("work", False), ("a.cat4g.json", True)]
    assert Path(gui.opened[2][0]) == p.resolve()
    for body in ({"target": "home"}, {"target": "work", "name": "a.cat4g.json"},
                 {"target": "output", "name": "../x.cat4g.json"}, []):
        assert gui.client.post("/api/open-folder", json=body).status_code == 400, body
    assert gui.client.post("/api/open-folder", json={"target": "output", "name": "no.cat4g.json"}).status_code == 404


def test_shutdown(gui):
    job_id = gui.client.post("/api/jobs", data={"url": "https://example.com/slow"}).json()["id"]
    gui.wait(job_id, statuses=("running",))
    assert gui.client.post("/api/shutdown").json() == {"ok": True}
    deadline = time.monotonic() + 10
    while not gui.shutdowns:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert gui.wait(job_id)["status"] == "cancelled"


def test_unknown_api_path(gui):
    r = gui.client.get("/api/nothing")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize(("given", "expected"), [
    ("song.wav", "song.wav"),
    ("C:\\Users\\me\\曲.MP3", "曲.mp3"),
    ("../../etc/passwd", "passwd"),
    ("a|b?.flac", "a_b_.flac"),
    ("x:y.flac", "y.flac"),
    ("CON.wav", "_CON.wav"),
    (".wav", "wav"),
    ("..", "audio"),
    ("noext", "noext"),
])
def test_upload_name(given, expected):
    assert upload_name(given) == expected
