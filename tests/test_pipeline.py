"""chordgen.pipeline のテスト。各段は偽物（test_pipeline_fakes）に差し替える。"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from chordgen import pipeline
from chordgen.pipeline import PipelineOptions, StageError, analyze, run, safe_filename
from test_pipeline_fakes import FakeFetchError, app_home, audio_file, fakes, make_fakes  # noqa: F401

CPU = dict(device="cpu")


def test_import_does_not_load_stage_modules_or_torch():
    code = (
        "import sys, chordgen.pipeline, chordgen.cli\n"
        "heavy = ['torch', 'chordgen.chords', 'chordgen.separation', 'chordgen.beats', 'chordgen.lyrics',"
        " 'chordgen.fetch', 'chordgen.songformat', 'chordgen.audio_io', 'chordgen.key', 'chordgen.bass',"
        " 'chordgen.postprocess']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parent.parent)


def test_default_order_file_input(fakes, audio_file):
    events = []
    result = analyze(audio_file, options=PipelineOptions(**CPU), on_progress=events.append)
    assert fakes.names() == [
        "probe_audio", "resolve_meta", "detect_beats", "recognize_chords", "estimate_key", "snap_to_beats",
    ]
    # ビートは原音、コード認識も（既定は original）原音
    assert fakes.args_of("detect_beats")[0] == audio_file
    assert fakes.args_of("recognize_chords") == (audio_file, "cpu", "chordmini")
    assert result.meta.title == "my song"
    assert result.key.tonic == "C"
    assert all(c.extras.get("snapped") for c in result.chords)
    assert result.lyrics is None

    # 進捗は段ごとに start → done、何段目かが 1 から順に
    starts = [(e.stage, e.index, e.total) for e in events if e.status == "start"]
    assert starts == [("probe", 1, 5), ("beats", 2, 5), ("chords", 3, 5), ("key", 4, 5), ("snap", 5, 5)]
    dones = [e.stage for e in events if e.status == "done"]
    assert dones == [s for s, _, _ in starts]
    assert all(e.elapsed is not None and e.elapsed >= 0 for e in events if e.status == "done")


def test_extras_are_namespaced_and_json(fakes, audio_file):
    result = analyze(audio_file, options=PipelineOptions(**CPU))
    ex = result.extras
    assert all(k.startswith("analyzer.") for k in ex)
    assert ex["analyzer.chordModel"] == "chordmini"
    assert ex["analyzer.options"]["chordInput"] == "original"
    assert ex["analyzer.options"]["separate"] is False
    assert set(ex["analyzer.stageSeconds"]) == {"probe", "beats", "chords", "key", "snap"}
    assert "analyzer.sourceUrl" not in ex
    json.dumps(ex, allow_nan=False)


def test_no_snap(fakes, audio_file):
    analyze(audio_file, options=PipelineOptions(snap=False, **CPU))
    assert "snap_to_beats" not in fakes.names()


def test_model_option_is_passed(fakes, audio_file):
    analyze(audio_file, options=PipelineOptions(chord_model="btc", **CPU))
    assert fakes.args_of("recognize_chords")[2] == "btc"


@pytest.mark.parametrize(
    ("opts", "lyrics", "expect_separate"),
    [
        (dict(), None, False),
        (dict(), "そらに\nうかぶ", True),  # 歌詞があれば vocals のために分ける
        (dict(chord_input="no_drums"), None, True),
        (dict(bass_mode="root_fix"), None, True),
        (dict(separate=True), None, True),
        (dict(separate=False), "そらに", False),
    ],
)
def test_separate_auto(fakes, audio_file, opts, lyrics, expect_separate):
    analyze(audio_file, lyrics_text=lyrics, options=PipelineOptions(**opts, **CPU))
    assert ("separate" in fakes.names()) is expect_separate


def test_chord_input_uses_made_input(fakes, audio_file, app_home):
    analyze(audio_file, options=PipelineOptions(chord_input="no_drums_vocals", **CPU))
    names = fakes.names()
    assert names.index("separate") < names.index("make_chord_input") < names.index("recognize_chords")
    made = fakes.args_of("recognize_chords")[0]
    assert made.name == "no_drums_vocals.wav"
    assert fakes.args_of("make_chord_input")[2] == "no_drums_vocals"
    assert fakes.args_of("estimate_key")[0] == made
    assert fakes.args_of("detect_beats")[0] == audio_file  # ビートは原音のまま
    # 作業フォルダは work_dir()/<安全な名前> の下
    stems_dir = Path(fakes.args_of("separate")[1])
    assert stems_dir.parent.parent == app_home / "work"
    assert stems_dir.parent.name.startswith("my song-")


def test_original_does_not_call_make_chord_input(fakes, audio_file):
    analyze(audio_file, lyrics_text="うた", options=PipelineOptions(**CPU))
    assert "make_chord_input" not in fakes.names()
    assert fakes.args_of("recognize_chords")[0] == audio_file


def test_bass_runs_after_snap(fakes, audio_file):
    result = analyze(audio_file, options=PipelineOptions(bass_mode="slash", **CPU))
    names = fakes.names()
    assert names.index("snap_to_beats") < names.index("detect_bass") < names.index("apply_bass")
    assert fakes.args_of("detect_bass")[0].name == "bass.wav"
    assert fakes.args_of("apply_bass")[2] == "slash"
    assert all(c.extras.get("snapped") for c in fakes.args_of("apply_bass")[0])  # 拍にそろえた後のコード
    assert all(c.extras.get("bass") == "slash" for c in result.chords)


def test_lyrics_use_vocals_and_language(fakes, audio_file):
    result = analyze(audio_file, lyrics_text="そらに\n\nうかぶ\n", options=PipelineOptions(lyrics_language="ja", **CPU))
    path, text, device, lang = fakes.args_of("align_lyrics")
    assert path.name == "vocals.wav"
    assert text == "そらに\n\nうかぶ\n"
    assert (device, lang) == ("cpu", "ja")
    assert [ln.text for ln in result.lyrics] == ["そらに", "うかぶ"]


def test_lyrics_without_separation_use_original(fakes, audio_file):
    analyze(audio_file, lyrics_text="うた", options=PipelineOptions(separate=False, **CPU))
    assert fakes.args_of("align_lyrics")[0] == audio_file


def test_blank_lyrics_are_ignored(fakes, audio_file):
    result = analyze(audio_file, lyrics_text=" \n ", options=PipelineOptions(**CPU))
    assert "align_lyrics" not in fakes.names() and "separate" not in fakes.names()
    assert result.lyrics is None


@pytest.mark.parametrize(
    "opts", [dict(separate=False, chord_input="no_drums"), dict(separate=False, bass_mode="slash"),
             dict(chord_input="nope"), dict(bars_per_line=0)]
)
def test_invalid_options(fakes, audio_file, opts):
    with pytest.raises(ValueError):
        analyze(audio_file, options=PipelineOptions(**opts, **CPU))
    assert fakes.calls == []


# ---- 任意の段・必須の段の失敗 -------------------------------------------------------

def test_key_failure_is_skipped(fakes, audio_file):
    fakes.fail["estimate_key"] = RuntimeError("key boom")
    events = []
    result = analyze(audio_file, options=PipelineOptions(**CPU), on_progress=events.append)
    assert result.key is None
    assert "snap_to_beats" in fakes.names()  # 後の段は続く
    assert result.extras["analyzer.skippedStages"] == ["key"]
    assert any("key boom" in w for w in result.extras["analyzer.warnings"])
    assert [e.stage for e in events if e.status == "skipped"] == ["key"]


def test_lyrics_failure_keeps_plain_lyrics(fakes, audio_file):
    fakes.fail["align_lyrics"] = RuntimeError("whisper boom")
    result = analyze(audio_file, lyrics_text="そらに\nうかぶ", options=PipelineOptions(**CPU))
    assert [ln.text for ln in result.lyrics] == ["そらに", "うかぶ"]
    assert all(t is None for ln in result.lyrics for t in ln.char_times)
    assert all(len(ln.char_times) == len(ln.text) for ln in result.lyrics)
    assert result.extras["analyzer.skippedStages"] == ["lyrics"]


def test_bass_failure_keeps_chords(fakes, audio_file):
    fakes.fail["detect_bass"] = RuntimeError("bass boom")
    result = analyze(audio_file, options=PipelineOptions(bass_mode="root_fix", **CPU))
    assert [c.label for c in result.chords] == ["N", "C:maj", "A:min7", "G:maj"]
    assert "apply_bass" not in fakes.names()
    assert result.extras["analyzer.skippedStages"] == ["bass"]


def test_separation_failure_only_for_lyrics_is_skipped(fakes, audio_file):
    fakes.fail["separate"] = RuntimeError("demucs boom")
    result = analyze(audio_file, lyrics_text="うた", options=PipelineOptions(**CPU))
    assert fakes.args_of("align_lyrics")[0] == audio_file
    assert result.extras["analyzer.skippedStages"] == ["separate"]


def test_separation_failure_with_bass_skips_bass(fakes, audio_file):
    fakes.fail["separate"] = RuntimeError("demucs boom")
    result = analyze(audio_file, options=PipelineOptions(bass_mode="slash", **CPU))
    assert "detect_bass" not in fakes.names()
    assert result.extras["analyzer.skippedStages"] == ["separate", "bass"]


def test_separation_failure_for_chord_input_is_fatal(fakes, audio_file):
    fakes.fail["separate"] = RuntimeError("demucs boom")
    with pytest.raises(StageError) as ei:
        analyze(audio_file, options=PipelineOptions(chord_input="boost_other", **CPU))
    assert ei.value.stage == "separate"
    assert "recognize_chords" not in fakes.names()


@pytest.mark.parametrize(("func", "stage"), [("probe_audio", "probe"), ("detect_beats", "beats"),
                                             ("recognize_chords", "chords")])
def test_required_failure_raises(fakes, audio_file, func, stage):
    err = RuntimeError("boom")
    fakes.fail[func] = err
    events = []
    with pytest.raises(StageError) as ei:
        analyze(audio_file, options=PipelineOptions(**CPU), on_progress=events.append)
    assert ei.value.stage == stage
    assert ei.value.__cause__ is err
    assert events[-1].status == "failed" and events[-1].stage == stage


def test_keyboard_interrupt_passes_through(fakes, audio_file):
    fakes.fail["recognize_chords"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        analyze(audio_file, options=PipelineOptions(**CPU))


def test_missing_file(fakes, tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "nothing.wav", options=PipelineOptions(**CPU))
    with pytest.raises(FileNotFoundError):
        analyze(str(tmp_path / "nothing.wav"), options=PipelineOptions(**CPU))


# ---- URL --------------------------------------------------------------------------

URL = "https://example.com/watch?v=abc"


def test_url_calls_fetch(fakes, app_home):
    events = []
    result = analyze(URL, options=PipelineOptions(**CPU), on_progress=events.append)
    names = fakes.names()
    assert names[:3] == ["fetch_info", "fetch_audio", "probe_audio"]
    dest = Path(fakes.args_of("fetch_audio")[1])
    assert dest.parent.parent == app_home / "work"
    assert dest.parent.name.startswith("Fake_Song-")  # 題名の "/" は置き換える
    assert fakes.args_of("resolve_meta")[1].source_url == URL
    assert result.meta.title == "Fake/Song"
    assert result.meta.artist == "Fake Artist"
    assert result.extras["analyzer.sourceUrl"] == URL
    assert [e.data for e in events if e.status == "progress"] == [{"type": "progress", "percent": 50.0}]
    assert events[0].stage == "fetch" and events[0].index == 1 and events[0].total == 6
    assert not dest.parent.exists()  # keep_work=False なので作業フォルダは消える


def test_url_keep_work(fakes):
    analyze(URL, options=PipelineOptions(keep_work=True, **CPU))
    dest = Path(fakes.args_of("fetch_audio")[1])
    assert (dest / "fake.wav").exists()


def test_custom_work_dir(fakes, tmp_path):
    wd = tmp_path / "mywork"
    analyze(URL, options=PipelineOptions(work_dir=wd, keep_work=True, **CPU))
    assert Path(fakes.args_of("fetch_audio")[1]).parent.parent == wd


def test_fetch_error_is_wrapped_with_cause(fakes):
    err = FakeFetchError("ENGINE_OUTDATED", "yt-dlp が古いです", exit_code=1)
    fakes.fail["fetch_info"] = err
    with pytest.raises(StageError) as ei:
        analyze(URL, options=PipelineOptions(**CPU))
    assert ei.value.stage == "fetch" and ei.value.__cause__ is err


def test_file_work_dir_removed_after_failure(fakes, audio_file, app_home):
    fakes.fail["recognize_chords"] = RuntimeError("boom")
    with pytest.raises(StageError):
        analyze(audio_file, lyrics_text="うた", options=PipelineOptions(**CPU))
    assert list((app_home / "work").iterdir()) == []


# ---- GPU のメモリ -------------------------------------------------------------------

def test_empty_cache_between_stages(fakes, audio_file, monkeypatch):
    calls = []
    cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append(1))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    analyze(audio_file, options=PipelineOptions(**CPU))
    assert len(calls) == 5  # 段ごとに 1 回


# ---- run と出力先 --------------------------------------------------------------------

def test_run_default_output_and_numbering(fakes, audio_file, app_home):
    p1 = run(audio_file, title="あの:曲?", options=PipelineOptions(**CPU))
    assert p1 == app_home / "output" / "あの_曲_.cat4g.json"
    p2 = run(audio_file, title="あの:曲?", options=PipelineOptions(**CPU))
    assert p2 == app_home / "output" / "あの_曲_ (2).cat4g.json"
    assert p1.exists() and p2.exists()
    doc = json.loads(p1.read_text(encoding="utf-8"))
    assert doc["song"]["title"] == "あの:曲?"
    assert fakes.args_of("build_document")[1] == 4


def test_run_explicit_path_overwrites(fakes, audio_file, tmp_path):
    out = tmp_path / "out" / "x.cat4g.json"
    out.parent.mkdir()
    out.write_text("old", encoding="utf-8")
    assert run(audio_file, out, options=PipelineOptions(bars_per_line=2, **CPU)) == out
    assert out.read_text(encoding="utf-8") != "old"
    assert fakes.args_of("build_document")[1] == 2


def test_run_to_directory(fakes, audio_file, tmp_path):
    out = run(audio_file, tmp_path, options=PipelineOptions(**CPU))
    assert out == tmp_path / "my song.cat4g.json"


def test_run_progress_includes_write(fakes, audio_file):
    events = []
    run(audio_file, options=PipelineOptions(**CPU), on_progress=events.append)
    starts = [(e.stage, e.index, e.total) for e in events if e.status == "start"]
    assert starts[-1] == ("write", 6, 6)


def test_run_rejects_unknown_kwargs(fakes, audio_file):
    with pytest.raises(TypeError):
        run(audio_file, options=PipelineOptions(**CPU), lyrics="x")


@pytest.mark.parametrize(
    ("name", "expect"),
    [("a/b:c?", "a_b_c_"), ("CON", "_CON"), ("aux.txt", "_aux.txt"), ("", "untitled"), (None, "untitled"),
     ("abc. ", "abc"), ("  曲  の  名前 ", "曲 の 名前"), ("..", "untitled"), ("tab\there", "tab_here")],
)
def test_safe_filename(name, expect):
    assert safe_filename(name) == expect


def test_safe_filename_length():
    assert len(safe_filename("あ" * 300)) == 100


def test_defaults_are_in_one_place():
    o = PipelineOptions()
    assert (o.chord_input, o.bass_mode, o.snap, o.chord_model, o.separate, o.bars_per_line) == (
        pipeline.DEFAULT_CHORD_INPUT, pipeline.DEFAULT_BASS_MODE, pipeline.DEFAULT_SNAP,
        pipeline.DEFAULT_CHORD_MODEL, None, pipeline.DEFAULT_BARS_PER_LINE,
    )
    assert (o.chord_input, o.bass_mode, o.snap) == ("original", "off", True)


# ---- 段の中の warnings.warn ----------------------------------------------------------

def test_stage_warnings_are_recorded_once(fakes, audio_file, monkeypatch, caplog):
    import logging
    import warnings

    beats_mod = sys.modules["chordgen.beats"]
    orig = beats_mod.detect_beats

    def detect_beats(audio_path, device):
        logging.getLogger("chordgen.beats").warning("ログにも出す警告")
        warnings.warn("ログにも出す警告", stacklevel=2)
        warnings.warn("warn だけの警告", stacklevel=2)
        return orig(audio_path, device)

    monkeypatch.setattr(beats_mod, "detect_beats", detect_beats)
    with caplog.at_level(logging.WARNING, logger="chordgen"):
        result = analyze(audio_file, options=PipelineOptions(**CPU))
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs.count("ログにも出す警告") == 1  # 段のモジュールが出したものだけ
    assert "ビート検出: ログにも出す警告" not in msgs
    assert "ビート検出: warn だけの警告" in msgs
    assert result.extras["analyzer.warnings"] == ["ビート検出: ログにも出す警告", "ビート検出: warn だけの警告"]
