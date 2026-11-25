"""chordgen.beats のテスト。整形の純粋関数は常に走り、モデルが要るものは無ければ skip。"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chordgen import beats as B

SR = 22050


# ---------------------------------------------------------------------------
# 合成音
# ---------------------------------------------------------------------------


def _add(y: np.ndarray, s: np.ndarray, t: float) -> None:
    i = int(t * SR)
    n = min(len(s), len(y) - i)
    if n > 0:
        y[i:i + n] += s[:n]


def _tone(midi: int, dur: float, amp: float, decay: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12)
    return amp * np.exp(-t * decay) * sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4))


def synth_song(bpm: float, beats_per_bar: int, dur: float = 24.0, offset: float = 0.3) -> np.ndarray:
    """クリック（1 拍目を強く、キックつき）と、小節ごとに変わる和音・ベースの合成音。"""
    rng = np.random.default_rng(0)
    y = np.zeros(int(dur * SR))
    ioi = 60.0 / bpm
    prog = [(57, (57, 60, 64)), (53, (53, 57, 60)), (48, (48, 52, 55)), (55, (55, 59, 62))]
    t, k = offset, 0
    while t < dur - 0.1:
        pos, bar = k % beats_per_bar, k // beats_per_bar
        root, chord = prog[bar % len(prog)]
        n = int(0.03 * SR)
        env = np.exp(-np.arange(n) / SR * 150)
        click_f, click_a = (1500, 0.6) if pos == 0 else (1000, 0.25)
        _add(y, click_a * env * np.sin(2 * np.pi * click_f * np.arange(n) / SR), t)
        if pos == 0:
            m = np.arange(int(0.15 * SR)) / SR
            _add(y, 0.9 * np.exp(-m * 25) * np.sin(2 * np.pi * 55 * m * (1 + np.exp(-m * 40))), t)
            _add(y, _tone(root - 12, ioi * beats_per_bar, 0.5, 1.0), t)
        else:
            m = int(0.1 * SR)
            _add(y, 0.3 * rng.standard_normal(m) * np.exp(-np.arange(m) / SR * 30), t)
        for note in chord:
            _add(y, _tone(note, ioi * 0.9, 0.15, 2.0), t)
        t += ioi
        k += 1
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


@pytest.fixture(scope="module")
def song_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("beats")
    sf.write(d / "s120_4.wav", synth_song(120, 4), SR)
    sf.write(d / "s100_3.wav", synth_song(100, 3), SR)
    sf.write(d / "s90_4.wav", synth_song(90, 4), SR)
    return d


# ---------------------------------------------------------------------------
# 整形の純粋関数
# ---------------------------------------------------------------------------


def test_clean_beats_sorts_and_dedups():
    out = B.clean_beats([1.0, 0.5, 0.5, 1.5, 1.52, -0.1, 2.0, 9.0, float("nan")], duration=3.0)
    assert out == [0.5, 1.0, 1.5, 2.0]


def test_fill_beat_gaps_inserts_missing_beats():
    beats = [0.5 * i for i in range(10)] + [0.5 * i for i in range(13, 20)]
    out = B.fill_beat_gaps(beats)
    assert np.allclose(out, [0.5 * i for i in range(20)])


def test_extrapolate_beats_within_duration():
    beats = [2.0 + 0.5 * i for i in range(10)]  # 2.0〜6.5
    out = B.extrapolate_beats(beats, duration=8.2)
    assert np.allclose(out, [0.5 * i for i in range(17)])  # 0.0〜8.0
    assert out[0] >= 0 and out[-1] <= 8.2


def test_downbeats_to_indices_nearest_beat():
    beats = [0.5 * i for i in range(12)]
    assert B.downbeats_to_indices(beats, [0.02, 1.98, 4.01, 5.0, 30.0]) == [0, 4, 8, 10]


@pytest.mark.parametrize(
    "indices, expected",
    [
        ([0, 4, 8, 12, 16], 4),
        ([1, 4, 7, 10, 13, 16], 3),
        ([0, 6, 12, 18], 6),
        ([0, 4, 12, 20, 24, 28], 4),  # 8 拍の抜けは数えない
        ([0, 4, 8, 11, 15, 18, 22], 4),  # 3 拍が混ざっても最頻は 4
        ([0, 3, 7], 4),  # 同数なら 4
        ([5], 4),
        ([], 4),
    ],
)
def test_infer_beats_per_bar(indices, expected):
    assert B.infer_beats_per_bar(indices) == expected


def test_regularize_fills_missing_and_pickup():
    # 2 拍の弱起、ダウンビートが所々抜けている
    out = B.regularize_downbeats([2, 6, 14, 22, 26], n_beats=32, beats_per_bar=4)
    assert out == [2, 6, 10, 14, 18, 22, 26, 30]


def test_regularize_ignores_single_wrong_downbeat():
    out = B.regularize_downbeats([0, 4, 9, 12, 16, 20], n_beats=24, beats_per_bar=4)
    assert out == [0, 4, 8, 12, 16, 20]


def test_regularize_keeps_persistent_phase_change():
    # 12 の後に 2/4 の小節が入り、以後は 14, 18, 22 …
    out = B.regularize_downbeats([0, 4, 8, 12, 14, 18, 22, 26], n_beats=30, beats_per_bar=4)
    assert out == [0, 4, 8, 12, 14, 18, 22, 26]


def test_regularize_avoids_one_beat_bar():
    out = B.regularize_downbeats([0, 4, 8, 13, 17, 21], n_beats=25, beats_per_bar=4)
    assert out == [0, 4, 8, 13, 17, 21]  # 8→13 は 5 拍の小節にする（1 拍の小節を作らない）
    out = B.regularize_downbeats([0, 4, 8, 12, 17, 21, 25], n_beats=28, beats_per_bar=4)
    assert out == [0, 4, 8, 12, 17, 21, 25]


def test_regularize_without_downbeats():
    assert B.regularize_downbeats([], n_beats=10, beats_per_bar=4) == [0, 4, 8]
    assert B.regularize_downbeats([0, 4], n_beats=0, beats_per_bar=4) == []


def test_estimate_bpm_removes_frame_quantization():
    # 90 BPM（0.6667 秒）を 20 ms 刻みに丸めた拍
    beats = np.round(np.arange(40) * (60 / 90) / 0.02) * 0.02
    assert B.estimate_bpm(beats) == pytest.approx(90.0, abs=0.3)
    assert 60 / np.median(np.diff(beats)) > 90.5  # 中央値だけだと偏る


def test_beat_unit_for():
    assert B.beat_unit_for(4) == 4
    assert B.beat_unit_for(3) == 4
    assert B.beat_unit_for(6) == 8


def test_build_beat_grid_full():
    rng = np.random.default_rng(1)
    ioi = 0.5
    true = np.arange(3, 50) * ioi  # 1.5 秒から 24.5 秒
    beats = list(true + rng.normal(0, 0.005, len(true)))
    del beats[20:22]  # 途中の 2 拍が抜けている
    beats.append(beats[5] + 0.01)  # 重複に近い拍
    downs = [t for i, t in enumerate(true) if (i + 3) % 4 == 0 and i not in (8, 9, 10, 11, 12)]
    grid = B.build_beat_grid(beats, downs, duration=26.2)

    assert grid.bpm == pytest.approx(120, abs=1)
    assert grid.beats_per_bar == 4 and grid.beat_unit == 4
    bt = np.array(grid.beat_times)
    assert np.all(np.diff(bt) > 0)
    assert bt[0] < ioi and bt[-1] <= 26.2 and 26.2 - bt[-1] < ioi
    assert len(bt) == 53  # 0.0〜26.0 を 0.5 秒刻み
    assert np.allclose(bt, np.arange(53) * ioi, atol=0.03)
    assert grid.downbeat_indices == list(range(0, 53, 4))


def test_build_beat_grid_without_downbeats():
    grid = B.build_beat_grid([1.0 + 0.6 * i for i in range(20)], None, duration=13.0)
    assert grid.bpm == pytest.approx(100, abs=0.5)
    assert grid.beats_per_bar == 4
    assert grid.downbeat_indices == list(range(0, len(grid.beat_times), 4))


def test_build_beat_grid_three_four():
    beats = [0.6 * i for i in range(30)]
    grid = B.build_beat_grid(beats, beats[1::3], duration=18.0)
    assert grid.beats_per_bar == 3
    assert grid.downbeat_indices[:3] == [1, 4, 7]


def test_build_beat_grid_almost_empty():
    with pytest.warns(UserWarning):
        grid = B.build_beat_grid([], None, duration=4.0)
    assert grid.bpm == pytest.approx(B.DEFAULT_BPM)
    assert grid.beat_times[0] == 0.0 and grid.beat_times[-1] <= 4.0
    assert grid.downbeat_indices[0] == 0


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------


def test_load_audio_mono_wav(tmp_path):
    stereo = np.stack([np.ones(1000), np.zeros(1000)], axis=1).astype(np.float32) * 0.5
    sf.write(tmp_path / "a.wav", stereo, 16000)
    y, sr = B.load_audio_mono(tmp_path / "a.wav")
    assert sr == 16000 and y.dtype == np.float32 and y.shape == (1000,)
    assert np.allclose(y, 0.25, atol=1e-4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_load_audio_mono_via_ffmpeg(tmp_path):
    sf.write(tmp_path / "a.wav", synth_song(120, 4, dur=3.0), SR)
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path / "a.wav"), str(m4a)],
                   check=True, shell=False)
    y, sr = B.load_audio_mono(m4a)
    assert sr == SR and abs(len(y) / sr - 3.0) < 0.1


# ---------------------------------------------------------------------------
# 検出（モデル・librosa が要る）
# ---------------------------------------------------------------------------


def _beat_this_available() -> bool:
    try:
        import beat_this.inference  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _detect_strict(path: Path):
    """Beat This! で検出する。フォールバックしたら（重みを取れないなど）skip。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid = B.detect_beats(path, "cpu")
    if any("librosa" in str(x.message) for x in w):
        pytest.skip("Beat This! を使えず librosa にフォールバックした")
    return grid


@pytest.mark.skipif(not _beat_this_available(), reason="beat_this / torch がない")
@pytest.mark.parametrize("name, bpm, bpb, first_down", [
    ("s120_4.wav", 120, 4, 0.3),
    ("s90_4.wav", 90, 4, 0.3),
    ("s100_3.wav", 100, 3, 0.3),
])
def test_detect_beats_beat_this(song_dir, name, bpm, bpb, first_down):
    grid = _detect_strict(song_dir / name)
    assert grid.bpm == pytest.approx(bpm, abs=2)
    assert grid.beats_per_bar == bpb
    downs = [grid.beat_times[i] for i in grid.downbeat_indices]
    assert any(abs(t - first_down) < 0.07 for t in downs[:2])
    bar = bpb * 60 / bpm
    assert np.allclose(np.diff(downs), bar, atol=0.07)


def test_detect_beats_librosa_fallback(song_dir, monkeypatch):
    pytest.importorskip("librosa")

    def broken(*args, **kwargs):
        raise ImportError("beat_this がない（テスト）")

    monkeypatch.setattr(B, "_detect_with_beat_this", broken)
    with pytest.warns(UserWarning, match="librosa"):
        grid = B.detect_beats(song_dir / "s120_4.wav", "cpu")
    assert grid.bpm == pytest.approx(120, abs=2)
    assert grid.beats_per_bar == 4
    assert grid.downbeat_indices == list(range(0, len(grid.beat_times), 4))


def test_detect_beats_without_any_backend(song_dir, monkeypatch):
    def broken(*args, **kwargs):
        raise ImportError("ない（テスト）")

    monkeypatch.setattr(B, "_detect_with_beat_this", broken)
    monkeypatch.setattr(B, "_detect_with_librosa", broken)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="librosa"):
        B.detect_beats(song_dir / "s120_4.wav", "cpu")
