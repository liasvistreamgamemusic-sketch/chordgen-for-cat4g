"""align_lyrics を Whisper のモデルで実際に動かすテスト。

重みのダウンロードが要るので、環境変数 CHORDGEN_TEST_WHISPER_MODEL にモデル名（tiny など）を
設定したときだけ走らせる。重みは models_dir()/whisper（CHORDGEN_HOME があればその下）に置かれる。
音は合成した無音とトーンだけで、歌詞は自作の短い文。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

MODEL = os.environ.get("CHORDGEN_TEST_WHISPER_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="CHORDGEN_TEST_WHISPER_MODEL が未設定")

LYRICS = "朝の光が窓を照らす\n\n小さな声で歌をうたう 🎵\n"
LYRICS_EN = "Morning light is shining\nI sing a quiet song\n"


@pytest.fixture(scope="module")
def deps():
    pytest.importorskip("torch")
    pytest.importorskip("stable_whisper")


def _write(path, audio, sr):
    import soundfile as sf

    sf.write(str(path), audio.astype(np.float32), sr)
    return path


def _check(result, lyrics):
    from chordgen.lyrics import split_lyrics_lines

    assert [r.text for r in result] == split_lyrics_lines(lyrics)
    for r in result:
        assert len(r.char_times) == len(r.text)
    flat = [t for r in result for t in r.char_times if t is not None]
    assert flat == sorted(flat)


def test_silence_returns_untimed_lines(deps, tmp_path):
    from chordgen.lyrics import align_lyrics

    wav = _write(tmp_path / "silence.wav", np.zeros(16000 * 8), 16000)
    result = align_lyrics(wav, LYRICS, "cpu", model_name=MODEL)
    _check(result, LYRICS)
    assert all(t is None for r in result for t in r.char_times)


def test_tone_44k_stereo_with_language_detection(deps, tmp_path):
    from chordgen.lyrics import align_lyrics

    sr = 44100
    t = np.arange(sr * 6) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    wav = _write(tmp_path / "tone.wav", np.stack([tone, tone * 0.5], axis=1), sr)
    # language=None で歌詞にかなも漢字もないので、音から判定する経路を通る
    result = align_lyrics(wav, LYRICS_EN, "cpu", model_name=MODEL)
    _check(result, LYRICS_EN)


def test_empty_lyrics(deps, tmp_path):
    from chordgen.lyrics import align_lyrics

    wav = _write(tmp_path / "silence.wav", np.zeros(16000), 16000)
    assert align_lyrics(wav, "\n\n", "cpu", model_name=MODEL) == []
