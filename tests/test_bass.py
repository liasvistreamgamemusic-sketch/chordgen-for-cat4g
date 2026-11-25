"""chordgen.bass.detect_bass を合成したベースの音で確かめる。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chordgen.bass import detect_bass
from chordgen.types import BeatGrid

BEAT = 0.5  # 120 BPM


def _midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _note(f0: float, dur: float, sr: int, amps=(1.0, 0.5, 0.3, 0.2), cents: float = 0.0) -> np.ndarray:
    """倍音つきの正弦波。弦をはじいたように立ち上がって減衰させる。"""
    t = np.arange(int(round(dur * sr))) / sr
    f = f0 * 2 ** (cents / 1200)
    x = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(amps))
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 1.5)
    return (x * env).astype(np.float64)


def _render(midis: list[int | None], sr: int = 44100, seed: int = 0, **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for m in midis:
        n = int(round(BEAT * sr))
        if m is None:
            parts.append(np.zeros(n))
        else:
            parts.append(_note(_midi_hz(m), BEAT, sr, **kw))
    y = np.concatenate(parts)
    y = 0.3 * y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    return y + rng.standard_normal(len(y)) * 1e-4


def _beats(n: int) -> BeatGrid:
    return BeatGrid(beat_times=[i * BEAT for i in range(n + 1)], downbeat_indices=list(range(0, n + 1, 4)),
                    bpm=120.0, beats_per_bar=4)


def _write(tmp_path: Path, y: np.ndarray, sr: int = 44100, name: str = "bass.wav", stereo: bool = False) -> Path:
    p = tmp_path / name
    sf.write(p, np.stack([y, 0.8 * y], axis=1) if stereo else y, sr)
    return p


# E1(28)〜G3(55) から 12 のピッチクラスをすべて含むように選ぶ
SCALE = [28, 41, 30, 43, 32, 45, 34, 47, 36, 49, 38, 51, 40, 53, 55, 29, 31, 33, 35, 37, 39, 42, 44, 46]


def test_detect_pitch_class_per_beat(tmp_path: Path) -> None:
    path = _write(tmp_path, _render(SCALE))
    notes = detect_bass(path, _beats(len(SCALE)))
    assert len(notes) == len(SCALE)
    assert [n.pitch_class for n in notes] == [m % 12 for m in SCALE]
    for i, n in enumerate(notes):
        assert n.start == pytest.approx(i * BEAT)
        assert n.end == pytest.approx((i + 1) * BEAT)
        assert n.confidence is not None and 0.5 < n.confidence <= 1.0


def test_silent_beats_are_none(tmp_path: Path) -> None:
    midis = [33, None, 40, None, 28]
    notes = detect_bass(_write(tmp_path, _render(midis)), _beats(len(midis)))
    assert [n.pitch_class for n in notes] == [9, None, 4, None, 4]
    assert notes[1].confidence is not None and notes[1].confidence < 0.3


def test_noise_beat_is_none(tmp_path: Path) -> None:
    sr = 44100
    y = _render([36, 36, 43, 43], sr=sr)
    rng = np.random.default_rng(1)
    n = int(BEAT * sr)
    y[n:2 * n] = rng.standard_normal(n) * 0.1  # 2 拍目は雑音だけ
    notes = detect_bass(_write(tmp_path, y), _beats(4))
    assert [x.pitch_class for x in notes] == [0, None, 7, 7]


def test_detuned_and_missing_fundamental(tmp_path: Path) -> None:
    # 全体が 35 セント高く、基音が弱い（分離したベースでよくある）
    midis = [28, 31, 33, 35, 38, 40, 45, 50]
    y = _render(midis, sr=48000, amps=(0.15, 1.0, 0.6, 0.4, 0.2), cents=35.0)
    notes = detect_bass(_write(tmp_path, y, sr=48000, stereo=True), _beats(len(midis)))
    assert [n.pitch_class for n in notes] == [m % 12 for m in midis]


def test_beats_not_covering_audio(tmp_path: Path) -> None:
    path = _write(tmp_path, _render([33, 40]))
    assert detect_bass(path, BeatGrid(beat_times=[], downbeat_indices=[], bpm=0.0, beats_per_bar=4)) == []
    assert detect_bass(path, BeatGrid(beat_times=[0.0], downbeat_indices=[0], bpm=0.0, beats_per_bar=4)) == []
    # 音源より後ろの拍は音なし
    notes = detect_bass(path, BeatGrid(beat_times=[0.0, 0.5, 1.0, 5.0, 5.5], downbeat_indices=[0],
                                       bpm=120.0, beats_per_bar=4))
    assert notes[0].pitch_class == 9
    assert notes[1].pitch_class == 4
    assert notes[3].pitch_class is None
