"""chordgen.fetch のテスト。偽の media-dl（小さな Python スクリプト）を CHORDGEN_MEDIA_DL で差し替える。"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from chordgen import fetch
from chordgen.fetch import FetchError, fetch_audio, fetch_info, find_media_dl, is_url

URL = "https://example.com/watch?v=abc&list=xyz"

# 偽の media-dl。FAKE_PLAN（JSON ファイル）の指示どおりに出力して終わる。
# 標準出力の各行の {out} は -o のフォルダ（JSON 文字列の中身として）に置き換える。
FAKE = textwrap.dedent(
    """
    import json, os, sys, time
    plan = json.load(open(os.environ["FAKE_PLAN"], encoding="utf-8"))
    argv = sys.argv[1:]
    with open(os.environ["FAKE_ARGV"], "w", encoding="utf-8") as f:
        json.dump(argv, f)
    out = argv[argv.index("-o") + 1] if "-o" in argv else ""
    for name in plan.get("create", []):
        with open(os.path.join(out, name), "wb") as f:
            f.write(b"RIFF")
    if plan.get("pidfile"):
        with open(plan["pidfile"], "w") as f:
            f.write(str(os.getpid()))
    esc = json.dumps(out)[1:-1]
    for line in plan.get("stdout", []):
        sys.stdout.buffer.write((line.replace("{out}", esc) + "\\n").encode("utf-8"))
        sys.stdout.flush()
    if plan.get("stderr"):
        sys.stderr.buffer.write(plan["stderr"].encode("utf-8"))
    time.sleep(plan.get("sleep", 0))
    sys.exit(plan.get("exit", 0))
    """
)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """plan を受け取って偽の media-dl を設定する関数を返す。呼び出し後の argv は .argv() で読む。"""
    script = tmp_path / "fake_media_dl.py"
    script.write_text(FAKE, encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    argv_path = tmp_path / "argv.json"
    monkeypatch.setenv(fetch.ENV_MEDIA_DL, json.dumps([sys.executable, str(script)]))
    monkeypatch.setenv("FAKE_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_ARGV", str(argv_path))

    class Fake:
        def set(self, **plan):
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

        def argv(self):
            return json.loads(argv_path.read_text(encoding="utf-8"))

    return Fake()


def ev(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


def test_is_url():
    assert is_url("https://example.com/a")
    assert is_url("HTTP://example.com")
    assert is_url("  https://youtu.be/x  ")
    assert not is_url("C:\\music\\a.wav")
    assert not is_url("/home/a.wav")
    assert not is_url("a.wav")
    assert not is_url("ftp://example.com/a")
    assert not is_url("https://")
    assert not is_url(Path("a.wav"))  # type: ignore[arg-type]


def test_fetch_audio_success_with_progress(fake, tmp_path):
    dest = tmp_path / "work"
    name = "曲名 [abc123XYZ_-].wav"
    fake.set(
        create=[name],
        stdout=[
            ev(type="progress", url=URL, status="downloading", percent=33.2, downloadedBytes=1, totalBytes=3,
               speedBps=10, etaSec=1),
            "[download] これは JSON ではない行",
            "{壊れた JSON",
            "[1, 2]",
            ev(type="progress", url=URL, status="postprocessing", percent=None, downloadedBytes=None,
               totalBytes=None, speedBps=None, etaSec=None),
            '{"type":"done","url":"%s","files":["{out}/%s"],"warnings":[]}' % (URL, name),
        ],
    )
    got = []
    res = fetch_audio(URL, dest, on_progress=got.append)
    assert res.audio_path == (dest / name).resolve()
    assert res.audio_path.is_file()
    assert res.meta.title == "曲名"
    assert res.meta.source_url == URL
    assert [g["status"] for g in got] == ["downloading", "postprocessing"]
    assert got[0]["percent"] == 33.2
    argv = fake.argv()
    assert argv[:2] == ["-f", "wav"]
    assert argv[argv.index("-o") + 1] == str(dest.resolve())
    assert "--json" in argv
    assert "--playlist" not in argv
    assert argv[-2:] == ["--", URL]


def test_fetch_audio_multiple_files_uses_first(fake, tmp_path, caplog):
    fake.set(
        create=["a.wav", "b.wav"],
        stdout=['{"type":"done","url":"u","files":["{out}/a.wav","{out}/b.wav"],"warnings":["w1"]}'],
    )
    with caplog.at_level("WARNING", logger="chordgen.fetch"):
        res = fetch_audio(URL, tmp_path / "w")
    assert res.audio_path.name == "a.wav"
    assert "最初の 1 つ" in caplog.text
    assert "w1" in caplog.text


def test_fetch_audio_error_event(fake, tmp_path):
    fake.set(
        stdout=[ev(type="error", url=URL, code="UNAVAILABLE", userMessage="コンテンツが見つからないか、非公開です。",
                   detail="ERROR: x")],
        exit=1,
    )
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    e = ei.value
    assert e.code == "UNAVAILABLE"
    assert e.user_message == "コンテンツが見つからないか、非公開です。"
    assert e.exit_code == 1
    assert e.detail == "ERROR: x"


def test_fetch_audio_engine_outdated(fake, tmp_path):
    fake.set(
        stdout=[ev(type="error", url=URL, code="ENGINE_OUTDATED", userMessage="yt-dlp の更新が必要かもしれません。",
                   detail="")],
        exit=1,
    )
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.code == "ENGINE_OUTDATED"
    assert ei.value.user_message.startswith("yt-dlp の更新が必要かもしれません。")
    assert "media-dl update" in ei.value.user_message
    assert ei.value.exit_code == 1


def test_fetch_audio_cancelled(fake, tmp_path):
    fake.set(
        stdout=[ev(type="error", url=URL, code="CANCELLED", userMessage="ダウンロードを中止しました。", detail="")],
        exit=130,
    )
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.code == "CANCELLED"
    assert ei.value.exit_code == 130


def test_fetch_audio_usage_error_exit_2(fake, tmp_path):
    fake.set(stderr="未対応の出力形式です: wav\n\n使い方:\n  media-dl <URL...>\n", exit=2)
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.exit_code == 2
    assert ei.value.code == "USAGE"
    assert "未対応の出力形式です" in ei.value.user_message


def test_fetch_audio_failure_without_json(fake, tmp_path):
    fake.set(stdout=["garbage"], stderr="something broke\n", exit=1)
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.code == "UNKNOWN"
    assert ei.value.exit_code == 1
    assert "something broke" in ei.value.user_message


def test_fetch_audio_done_without_files(fake, tmp_path):
    fake.set(stdout=[ev(type="done", url=URL, files=[], warnings=["保存先ファイルのパスを特定できませんでした。"])])
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.code == "NO_FILE"


def test_fetch_audio_done_file_missing(fake, tmp_path):
    fake.set(stdout=['{"type":"done","url":"u","files":["{out}/none.wav"],"warnings":[]}'])
    with pytest.raises(FetchError) as ei:
        fetch_audio(URL, tmp_path / "w")
    assert ei.value.code == "NO_FILE"


def test_fetch_audio_rejects_non_url(fake, tmp_path):
    with pytest.raises(FetchError) as ei:
        fetch_audio("C:\\a.wav", tmp_path)
    assert ei.value.code == "INVALID_INPUT"


def test_fetch_audio_stops_child_on_exception(fake, tmp_path, monkeypatch):
    """on_progress の例外（Ctrl+C と同じ経路）で子プロセスを止める。"""
    monkeypatch.setattr(fetch, "STOP_GRACE_SEC", 0.5)
    pidfile = tmp_path / "pid"
    fake.set(
        pidfile=str(pidfile),
        stdout=[ev(type="progress", url=URL, status="downloading", percent=1.0)],
        sleep=60,
    )

    def boom(_):
        raise KeyboardInterrupt

    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        fetch_audio(URL, tmp_path / "w", on_progress=boom)
    assert time.monotonic() - t0 < 20
    pid = int(pidfile.read_text())
    if os.name != "nt":
        with pytest.raises(OSError):
            os.kill(pid, 0)


def test_fetch_info(fake):
    info = {
        "id": "abc", "title": "  曲の題名 ", "uploader": "投稿者", "durationSec": 200, "isPlaylist": False,
        "artist": None, "formats": [],
    }  # fmt: skip
    fake.set(stdout=["not json", json.dumps(info, ensure_ascii=False)])
    meta = fetch_info(URL)
    assert meta.title == "曲の題名"
    assert meta.artist == "投稿者"
    assert meta.source_url == URL
    assert fake.argv() == ["info", "--json", "--", URL]


def test_fetch_info_prefers_artist_and_strips_topic(fake):
    fake.set(stdout=[json.dumps({"title": "T", "uploader": "U", "artist": "A", "isPlaylist": False})])
    assert fetch_info(URL).artist == "A"
    fake.set(stdout=[json.dumps({"title": "T", "uploader": "歌手 - Topic", "artist": None})])
    assert fetch_info(URL).artist == "歌手"


def test_fetch_info_playlist_has_no_title(fake):
    fake.set(stdout=[json.dumps({"title": "再生リスト", "uploader": "U", "isPlaylist": True, "playlistCount": 3})])
    meta = fetch_info(URL)
    assert meta.title is None
    assert meta.artist == "U"


def test_fetch_info_error_on_stderr(fake):
    fake.set(stderr="エラー [ENGINE_OUTDATED] yt-dlp の更新が必要かもしれません。\n  ERROR: nsig extraction failed\n",
             exit=1)
    with pytest.raises(FetchError) as ei:
        fetch_info(URL)
    assert ei.value.code == "ENGINE_OUTDATED"
    assert "media-dl update" in ei.value.user_message
    assert ei.value.exit_code == 1


def test_fetch_info_exit_2(fake):
    fake.set(stderr="info には URL を 1 つ指定してください。\n", exit=2)
    with pytest.raises(FetchError) as ei:
        fetch_info(URL)
    assert ei.value.exit_code == 2


# ---- 実行ファイルの探し方 ----


def test_explicit_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(fetch.ENV_MEDIA_DL, str(tmp_path / "nothing"))
    with pytest.raises(FetchError) as ei:
        find_media_dl()
    assert ei.value.code == "MEDIA_DL_MISSING"


def test_explicit_js_uses_node(monkeypatch, tmp_path):
    js = tmp_path / "cli.js"
    js.write_text("")
    node = tmp_path / "node.exe"
    node.write_text("")
    monkeypatch.setenv(fetch.ENV_MEDIA_DL, str(js))
    monkeypatch.setenv(fetch.ENV_NODE, str(node))
    assert find_media_dl() == [str(node), str(js)]


def _clear_env(monkeypatch, home: Path):
    monkeypatch.delenv(fetch.ENV_MEDIA_DL, raising=False)
    monkeypatch.setenv("CHORDGEN_HOME", str(home))


def test_app_folder_package_json(monkeypatch, tmp_path):
    _clear_env(monkeypatch, tmp_path)
    pkg = tmp_path / "node_modules" / "@liasvistreamgamemusic-sketch" / "media-dl"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "dist" / "cli.js").write_text("")
    (pkg / "package.json").write_text(json.dumps({"bin": {"media-dl": "./dist/cli.js"}}))
    node = tmp_path / "runtime" / "node" / ("node.exe" if os.name == "nt" else "node")
    node.parent.mkdir(parents=True)
    node.write_text("")
    monkeypatch.delenv(fetch.ENV_NODE, raising=False)
    assert find_media_dl() == [str(node), str((pkg / "dist" / "cli.js").resolve())]


def test_cmd_shim_is_called_through_node(monkeypatch, tmp_path):
    """npm の media-dl.cmd は cmd.exe を通さず、中の JS を node で呼ぶ。"""
    _clear_env(monkeypatch, tmp_path / "home")
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    js = tmp_path / "node_modules" / "@liasvistreamgamemusic-sketch" / "media-dl" / "dist" / "cli.js"
    js.parent.mkdir(parents=True)
    js.write_text("")
    shim = bin_dir / "media-dl.cmd"
    shim.write_text(
        '@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n'
        'IF EXIST "%dp0%\\node.exe" (\r\n  SET "_prog=%dp0%\\node.exe"\r\n) ELSE (\r\n  SET "_prog=node"\r\n)\r\n'
        'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  '
        '"%dp0%\\..\\@liasvistreamgamemusic-sketch\\media-dl\\dist\\cli.js" %*\r\n'
    )
    node = bin_dir / "node.exe"
    node.write_text("")
    monkeypatch.delenv(fetch.ENV_NODE, raising=False)
    if os.name != "nt":
        monkeypatch.setattr(fetch, "_node_exe_name", lambda: "node.exe")
    monkeypatch.setenv(fetch.ENV_MEDIA_DL, str(shim))
    assert find_media_dl() == [str(node), str(js.resolve())]


@pytest.mark.skipif(os.name == "nt", reason="POSIX の実行ファイルで確かめる")
def test_path_lookup_and_missing(monkeypatch, tmp_path):
    _clear_env(monkeypatch, tmp_path / "home")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    with pytest.raises(FetchError) as ei:
        find_media_dl()
    assert ei.value.code == "MEDIA_DL_MISSING"
    assert "setup.bat" in ei.value.user_message
    exe = bin_dir / "media-dl"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert find_media_dl() == [str(exe)]
