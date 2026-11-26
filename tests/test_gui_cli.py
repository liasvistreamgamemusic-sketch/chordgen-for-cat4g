"""chordgen.cli の --progress-json（GUI が子プロセスとして読む出力）のテスト。各段は偽物。"""

from __future__ import annotations

import json
import sys

import pytest

from chordgen.cli import main
from test_pipeline_fakes import FakeFetchError, app_home, audio_file, fakes, make_fakes  # noqa: F401


def _events(out: str) -> list[dict]:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]  # 標準出力は JSON の行だけ
    assert all(isinstance(e, dict) and "type" in e for e in events)
    return events


def test_success_events(fakes, audio_file, app_home, capsys):
    assert main([str(audio_file), "--device", "cpu", "--title", "テスト曲", "--progress-json"]) == 0
    out, err = capsys.readouterr()
    events = _events(out)
    progress = [e for e in events if e["type"] == "progress"]
    starts = [e["stage"] for e in progress if e["status"] == "start"]
    assert starts == ["probe", "beats", "chords", "key", "snap", "write"]
    first = progress[0]
    assert set(first) >= {"stage", "label", "index", "total", "status", "elapsed", "message", "data"}
    assert (first["label"], first["index"], first["total"]) == ("音源の読み込み", 1, 6)
    done = events[-1]
    path = app_home / "output" / "テスト曲.cat4g.json"
    assert done == {"type": "done", "output": str(path.resolve()), "title": "テスト曲"}
    # 人向けの表示は標準エラーに出る（書き出した場所の行も）
    assert "[1/6] 音源の読み込み" in err and "書き出しました" in err


def test_library_prints_do_not_break_stdout(fakes, audio_file, app_home, capsys):
    orig = sys.modules["chordgen.chords"].recognize_chords

    def noisy(*a, **kw):
        print("library noise on stdout")
        return orig(*a, **kw)

    sys.modules["chordgen.chords"].recognize_chords = noisy
    assert main([str(audio_file), "--device", "cpu", "--progress-json"]) == 0
    out, err = capsys.readouterr()
    _events(out)
    assert "library noise on stdout" in err


def test_warning_event(fakes, audio_file, app_home, capsys):
    fakes.fail["estimate_key"] = RuntimeError("key boom")
    assert main([str(audio_file), "--device", "cpu", "--progress-json"]) == 0
    events = _events(capsys.readouterr().out)
    warnings = [e for e in events if e["type"] == "warning"]
    assert len(warnings) == 1 and "キー推定" in warnings[0]["message"] and "key boom" in warnings[0]["message"]
    assert any(e["type"] == "progress" and e["stage"] == "key" and e["status"] == "skipped" for e in events)
    assert events[-1]["type"] == "done"


def test_required_failure_event(fakes, audio_file, capsys):
    fakes.fail["detect_beats"] = RuntimeError("beat boom")
    assert main([str(audio_file), "--device", "cpu", "--progress-json"]) == 1
    out, err = capsys.readouterr()
    events = _events(out)
    assert events[-1] == {"type": "error", "code": "STAGE_FAILED",
                          "message": "ビート検出で失敗しました: RuntimeError: beat boom", "stage": "beats", "exitCode": 1}
    assert any(e["type"] == "progress" and e["status"] == "failed" for e in events)
    assert "ビート検出で失敗しました" in err


@pytest.mark.parametrize(("code", "exit_code"), [("ENGINE_OUTDATED", 1), ("ABORTED", 130)])
def test_fetch_error_event(fakes, capsys, code, exit_code):
    fakes.fail["fetch_audio"] = FakeFetchError(code, "取得できませんでした（テスト）", exit_code=exit_code)
    assert main(["https://example.com/v/1", "--device", "cpu", "--progress-json"]) == exit_code
    ev = _events(capsys.readouterr().out)[-1]
    assert (ev["type"], ev["code"], ev["stage"], ev["exitCode"]) == ("error", code, "fetch", exit_code)
    assert "取得できませんでした（テスト）" in ev["message"]
    assert ("media-dl update" in ev["message"]) is (code == "ENGINE_OUTDATED")


def test_validation_error_event(fakes, audio_file, capsys):
    from jsonschema.exceptions import ValidationError

    def bad_write(doc, path):
        raise ValidationError("-1 is less than the minimum of 1", path=["song", "bpm"])

    sys.modules["chordgen.songformat"].write_document = bad_write
    assert main([str(audio_file), "--device", "cpu", "--progress-json"]) == 1
    ev = _events(capsys.readouterr().out)[-1]
    assert (ev["code"], ev["stage"], ev["exitCode"]) == ("VALIDATION_FAILED", "write", 1)
    assert "検証" in ev["message"] and "$.song.bpm" in ev["message"]


def test_usage_errors(fakes, tmp_path, audio_file, capsys):
    assert main(["x.wav", "--bass", "nope", "--progress-json"]) == 2
    ev = _events(capsys.readouterr().out)[-1]
    assert (ev["type"], ev["code"], ev["exitCode"]) == ("error", "USAGE", 2) and "引数の誤り" in ev["message"]

    assert main([str(tmp_path / "none.wav"), "--progress-json"]) == 2
    assert _events(capsys.readouterr().out)[-1]["code"] == "SOURCE_NOT_FOUND"

    assert main([str(audio_file), "--lyrics", str(tmp_path / "none.txt"), "--progress-json"]) == 2
    assert _events(capsys.readouterr().out)[-1]["code"] == "LYRICS_NOT_FOUND"


def test_interrupt_event(fakes, audio_file, capsys):
    fakes.fail["recognize_chords"] = KeyboardInterrupt()
    assert main([str(audio_file), "--device", "cpu", "--progress-json"]) == 130
    ev = _events(capsys.readouterr().out)[-1]
    assert (ev["code"], ev["exitCode"]) == ("INTERRUPTED", 130)


def test_without_flag_stdout_is_unchanged(fakes, audio_file, app_home, capsys):
    assert main([str(audio_file), "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 1 and "書き出しました" in out


def test_work_dir_option(fakes, audio_file, tmp_path, app_home, capsys):
    work = tmp_path / "w"
    assert main([str(audio_file), "--device", "cpu", "--bass", "root_fix", f"--work-dir={work}", "--keep-work"]) == 0
    assert fakes.args_of("separate")[1].is_relative_to(work)
