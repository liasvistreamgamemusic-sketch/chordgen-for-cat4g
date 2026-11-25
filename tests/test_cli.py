"""chordgen.cli のテスト。各段は偽物（test_pipeline_fakes）に差し替える。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chordgen.cli import main
from test_pipeline_fakes import FakeFetchError, app_home, audio_file, fakes, make_fakes  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


def test_help(capsys):
    assert main(["--help"]) == 0
    assert "使い方" in capsys.readouterr().out


def test_module_help_runs_without_torch():
    proc = subprocess.run([sys.executable, "-m", "chordgen", "--help"], cwd=ROOT, capture_output=True)
    assert proc.returncode == 0
    assert "音源" in proc.stdout.decode("utf-8")


@pytest.mark.parametrize("argv", [[], ["x.wav", "--bass", "nope"], ["x.wav", "--bars-per-line", "0"],
                                  ["x.wav", "--model", "other"]])
def test_bad_arguments(argv, capsys):
    assert main(argv) == 2
    assert "引数の誤り" in capsys.readouterr().err


def test_missing_source_file(fakes, tmp_path, capsys):
    assert main([str(tmp_path / "nothing.wav")]) == 2
    assert "見つかりません" in capsys.readouterr().err
    assert fakes.calls == []


def test_missing_lyrics_file(fakes, audio_file, tmp_path, capsys):
    assert main([str(audio_file), "--lyrics", str(tmp_path / "none.txt")]) == 2
    assert "歌詞" in capsys.readouterr().err


def test_success(fakes, audio_file, app_home, capsys):
    assert main([str(audio_file), "--device", "cpu", "--title", "テスト曲", "--artist", "私"]) == 0
    out, err = capsys.readouterr()
    path = app_home / "output" / "テスト曲.cat4g.json"
    assert path.exists()
    assert str(path) in out and "cat4g" in out
    assert len(out.strip().splitlines()) == 1
    assert "[1/6] 音源の読み込み" in err and "[6/6] 書き出し" in err
    result = fakes.args_of("build_document")[0]
    assert (result.meta.title, result.meta.artist) == ("テスト曲", "私")


def test_options_are_passed(fakes, audio_file, tmp_path, capsys):
    lyr = tmp_path / "lyrics.txt"
    lyr.write_bytes("\ufeffそらに\r\nうかぶ\r\n".encode("utf-8"))  # BOM つき・CRLF
    out = tmp_path / "o.cat4g.json"
    code = main([str(audio_file), "--lyrics", str(lyr), "-o", str(out), "--chord-input", "no_drums",
                 "--bass", "root_fix", "--no-snap", "--model", "btc", "--device", "cpu", "--language", "ja",
                 "--bars-per-line", "2", "--keep-work"])
    assert code == 0
    assert out.exists()
    assert fakes.args_of("align_lyrics")[1] == "そらに\nうかぶ\n"
    assert fakes.args_of("align_lyrics")[3] == "ja"
    assert fakes.args_of("make_chord_input")[2] == "no_drums"
    assert fakes.args_of("apply_bass")[2] == "root_fix"
    assert fakes.args_of("recognize_chords")[2] == "btc"
    assert fakes.args_of("build_document")[1] == 2
    assert "snap_to_beats" not in fakes.names()
    assert fakes.args_of("separate")[1].exists()  # --keep-work で残る
    ex = json.loads(out.read_text(encoding="utf-8"))["song"]["extensions"]
    assert ex["analyzer.options"]["chordInput"] == "no_drums"


def test_optional_failure_still_writes(fakes, audio_file, app_home, capsys):
    fakes.fail["estimate_key"] = RuntimeError("key boom")
    assert main([str(audio_file), "--device", "cpu"]) == 0
    out, err = capsys.readouterr()
    assert "警告" in err and "キー推定" in err
    assert (app_home / "output" / "my song.cat4g.json").exists()


def test_required_failure(fakes, audio_file, capsys):
    fakes.fail["detect_beats"] = RuntimeError("beat boom")
    assert main([str(audio_file), "--device", "cpu"]) == 1
    err = capsys.readouterr().err
    assert "ビート検出で失敗しました" in err and "beat boom" in err
    assert "Traceback" not in err


def test_verbose_prints_traceback(fakes, audio_file, capsys):
    fakes.fail["recognize_chords"] = RuntimeError("chord boom")
    assert main([str(audio_file), "--device", "cpu", "-v"]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_keyboard_interrupt(fakes, audio_file, capsys):
    fakes.fail["recognize_chords"] = KeyboardInterrupt()
    assert main([str(audio_file), "--device", "cpu"]) == 130
    assert "中断" in capsys.readouterr().err


@pytest.mark.parametrize(("code", "exit_code"), [("ENGINE_OUTDATED", 1), ("BAD_URL", 2), ("ABORTED", 130)])
def test_fetch_error(fakes, capsys, code, exit_code):
    fakes.fail["fetch_audio"] = FakeFetchError(code, "取得できませんでした（テスト）", exit_code=exit_code)
    assert main(["https://example.com/v/1", "--device", "cpu"]) == exit_code
    err = capsys.readouterr().err
    assert "取得できませんでした（テスト）" in err and code in err
    assert ("media-dl update" in err) is (code == "ENGINE_OUTDATED")


def test_url_success_and_progress_is_throttled(fakes, app_home, capsys):
    fakes.progress_events = [{"type": "progress", "percent": float(i)} for i in range(100)]
    assert main(["https://example.com/v/1", "--device", "cpu"]) == 0
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if "取得中" in ln]
    assert 1 <= len(lines) <= 11  # 10% ごと程度に間引く
    assert "[1/7] 音源の取得" in err
    assert (app_home / "output" / "Fake_Song.cat4g.json").exists()


def test_validation_error_lists_fields(fakes, audio_file, app_home, capsys):
    from jsonschema.exceptions import ValidationError

    def bad_write(doc, path):
        raise ValidationError("-1 is less than the minimum of 1", path=["song", "bpm"])

    sys.modules["chordgen.songformat"].write_document = bad_write
    assert main([str(audio_file), "--device", "cpu"]) == 1
    err = capsys.readouterr().err
    assert "検証" in err and "$.song.bpm" in err and "minimum" in err
    assert not (app_home / "output" / "my song.cat4g.json").exists()


def test_validation_error_with_errors_list(fakes, audio_file, capsys):
    from jsonschema.exceptions import ValidationError

    class DocumentValidationError(ValueError):
        def __init__(self, errors):
            super().__init__("invalid")
            self.errors = errors

    def bad_write(doc, path):
        raise DocumentValidationError([ValidationError("bad a", path=["song", "a"]),
                                       ValidationError("bad b", path=["song", "sections", 0, "b"])])

    sys.modules["chordgen.songformat"].write_document = bad_write
    assert main([str(audio_file), "--device", "cpu"]) == 1
    err = capsys.readouterr().err
    assert "$.song.a: bad a" in err and "$.song.sections[0].b: bad b" in err
