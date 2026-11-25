"""本物の songformat（と audio_io・postprocess）を組み合わせて、.cat4g.json が書き出され検証を通るかを見る。

音源は自作の正弦波。ビート・コード・キー・歌詞の位置合わせなど重い段は偽物のまま。
本物のモジュールがまだ無い・import できないときは飛ばす。
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from chordgen.pipeline import PipelineOptions, run
from test_pipeline_fakes import app_home, make_fakes  # noqa: F401

REAL = ("songformat", "postprocess", "audio_io")


def _real_available() -> bool:
    try:
        sf = importlib.import_module("chordgen.songformat")
        importlib.import_module("chordgen.postprocess")
        importlib.import_module("chordgen.audio_io")
    except Exception:
        return False
    return all(hasattr(sf, n) for n in ("build_document", "validate_document", "write_document"))


pytestmark = pytest.mark.skipif(not _real_available(), reason="本物の songformat などがまだありません")


@pytest.fixture
def sine_wav(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.arange(int(sr * 8.0)) / sr
    y = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    p = tmp_path / "in" / "自作のテスト音.wav"
    p.parent.mkdir(parents=True)
    sf.write(p, y, sr)
    return p


@pytest.fixture
def real_fakes(make_fakes):
    return make_fakes(exclude=REAL)


def _validate(path):
    from chordgen.songformat import validate_document

    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_document(doc)
    return doc


def _chords(doc):
    return [c for s in doc["song"]["sections"] for ln in s["lines"] for c in ln["chords"]]


def test_without_lyrics(real_fakes, sine_wav, app_home):
    out = run(sine_wav, options=PipelineOptions(device="cpu"))
    assert out == app_home / "output" / "自作のテスト音.cat4g.json"
    doc = _validate(out)
    song = doc["song"]
    assert doc["format"] == "cat4g-song" and doc["version"] == 1
    assert song["title"] == "自作のテスト音"
    assert song["bpm"] == 120
    assert song["originalKey"] == "C"
    assert [c["chord"] for c in _chords(doc)] == ["C", "Am7", "G"]  # N は置かない
    ext = song["extensions"]
    assert ext["analyzer.chordModel"] == "chordmini"
    assert ext["analyzer.options"]["chordInput"] == "original"
    assert "probe" in ext["analyzer.stageSeconds"]


def test_with_lyrics(real_fakes, sine_wav):
    out = run(sine_wav, lyrics_text="そらにうかぶ\nしろいくも", title="題", artist="私",
              options=PipelineOptions(device="cpu"))
    doc = _validate(out)
    lyrics = [ln["lyrics"] for s in doc["song"]["sections"] for ln in s["lines"]]
    assert "そらにうかぶ" in lyrics and "しろいくも" in lyrics
    assert doc["song"]["artist"] == "私"


def test_lyrics_alignment_failure_still_valid(real_fakes, sine_wav):
    real_fakes.fail["align_lyrics"] = RuntimeError("whisper boom")
    out = run(sine_wav, lyrics_text="そらにうかぶ\nしろいくも", options=PipelineOptions(device="cpu"))
    doc = _validate(out)
    lyrics = [ln["lyrics"] for s in doc["song"]["sections"] for ln in s["lines"]]
    assert "そらにうかぶ" in lyrics
    assert doc["song"]["extensions"]["analyzer.skippedStages"] == ["lyrics"]


def test_key_failure_and_url_still_valid(real_fakes):
    real_fakes.fail["estimate_key"] = RuntimeError("key boom")
    # 偽の fetch_audio は中身のない WAV を置くので、本物の audio_io には自作の正弦波を渡す
    import sys

    import soundfile as sf

    orig = sys.modules["chordgen.fetch"].fetch_audio

    def fetch_audio(url, dest_dir, on_progress=None):
        res = orig(url, dest_dir, on_progress)
        sr = 22050
        sf.write(res.audio_path, np.zeros(sr * 4, dtype=np.float32), sr)
        return res

    sys.modules["chordgen.fetch"].fetch_audio = fetch_audio
    url = "https://example.com/watch?v=abc"
    out = run(url, options=PipelineOptions(device="cpu"))
    doc = _validate(out)
    song = doc["song"]
    assert song["title"] == "Fake/Song"
    assert song["sourceUrl"] == url
    assert song["extensions"]["analyzer.sourceUrl"] == url
    assert song["originalKey"] is None
