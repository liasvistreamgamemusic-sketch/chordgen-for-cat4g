"""chordgen.separation を合成した音で確かめる。

Demucs を実際に動かすテストは demucs（と torch）が import できなければ skip する。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chordgen import separation
from chordgen.separation import MODE_GAINS, OTHER_BOOST_GAIN, PEAK_TARGET, make_chord_input, separate
from chordgen.types import STEM_NAMES, Stems

SR = 44100
# ステムごとに別の周波数の正弦波にして、組み直した音に何が入っているかを見分ける
FREQS = {"vocals": 440.0, "drums": 1000.0, "bass": 55.0, "other": 261.63}


def _fake_stems(tmp_path: Path, seconds: float = 2.0, amp: float = 0.9) -> Stems:
    t = np.arange(int(seconds * SR)) / SR
    paths = {}
    for name in STEM_NAMES:
        y = amp * np.sin(2 * np.pi * FREQS[name] * t)
        p = tmp_path / f"{name}.wav"
        sf.write(p, np.stack([y, y], axis=1).astype(np.float32), SR, subtype="FLOAT")
        paths[name] = p
    return Stems(paths=paths, sample_rate=SR)


def _amplitude(x: np.ndarray, freq: float) -> float:
    """x に含まれる freq の正弦波の振幅（射影で求める）。"""
    t = np.arange(len(x)) / SR
    c = np.cos(2 * np.pi * freq * t)
    s = np.sin(2 * np.pi * freq * t)
    return float(2 * np.hypot(x @ c, x @ s) / len(x))


def test_original_returns_input(tmp_path: Path) -> None:
    stems = _fake_stems(tmp_path)
    src = tmp_path / "song.mp3"
    assert make_chord_input(src, stems, "original", tmp_path / "out") == src


@pytest.mark.parametrize("mode", ["no_drums", "no_drums_vocals", "boost_other"])
def test_mix_modes(tmp_path: Path, mode: str) -> None:
    stems = _fake_stems(tmp_path)  # 足すと 1.0 を大きく超える
    out = make_chord_input(tmp_path / "song.wav", stems, mode, tmp_path / "out")
    assert out.is_file() and out.parent == tmp_path / "out"
    data, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == SR
    assert data.shape == (int(2.0 * SR), 2)
    peak = np.max(np.abs(data))
    assert peak <= 1.0
    assert peak == pytest.approx(PEAK_TARGET, abs=1e-3)

    mono = data.mean(axis=1)
    amps = {name: _amplitude(mono, f) for name, f in FREQS.items()}
    gains = MODE_GAINS[mode]
    used = [n for n in STEM_NAMES if n in gains]
    for name in STEM_NAMES:
        if name not in gains:
            assert amps[name] < 1e-3, name
    # 使ったステムどうしの大きさの比が倍率の比になっている
    ref = used[0]
    for name in used[1:]:
        assert amps[name] / amps[ref] == pytest.approx(gains[name] / gains[ref], rel=0.01)


def test_boost_other_gain() -> None:
    assert MODE_GAINS["boost_other"]["other"] == OTHER_BOOST_GAIN
    assert set(MODE_GAINS["boost_other"]) == set(STEM_NAMES)
    assert set(MODE_GAINS["no_drums"]) == {"vocals", "bass", "other"}
    assert set(MODE_GAINS["no_drums_vocals"]) == {"bass", "other"}


def test_quiet_stems_are_raised_without_clipping(tmp_path: Path) -> None:
    stems = _fake_stems(tmp_path, amp=0.01)
    out = make_chord_input(tmp_path / "song.wav", stems, "no_drums", tmp_path / "out")
    data, _ = sf.read(out, dtype="float64")
    assert np.max(np.abs(data)) == pytest.approx(PEAK_TARGET, abs=1e-3)


def test_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_chord_input(tmp_path / "song.wav", _fake_stems(tmp_path), "vocals_only", tmp_path)  # type: ignore[arg-type]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_read_audio_falls_back_to_ffmpeg(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    t = np.arange(SR) / SR
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 220 * t), SR)
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(wav), str(m4a)],
                   check=True, shell=False)
    data, sr = separation.read_audio(m4a)
    assert sr == SR
    assert data.ndim == 2 and data.shape[0] == 1
    assert abs(data.shape[1] - SR) < SR * 0.1


def test_separate_reuses_existing_result(tmp_path: Path) -> None:
    # 分離の結果と記録が out_dir にあれば、demucs も torch も使わずに返す
    src = tmp_path / "song.wav"
    sf.write(src, np.zeros(SR, dtype=np.float32), SR)
    out_dir = tmp_path / "stems"
    out_dir.mkdir()
    for name in STEM_NAMES:
        sf.write(out_dir / f"{name}.wav", np.zeros((SR, 2), dtype=np.float32), SR)
    (out_dir / separation.MANIFEST_NAME).write_text(json.dumps({
        "version": separation._MANIFEST_VERSION,
        "source_sha256": separation._file_sha256(src),
        "settings": separation._settings(),
        "sample_rate": SR,
    }), encoding="utf-8")

    stems = separate(src, out_dir, "cpu")
    assert stems.sample_rate == SR
    assert stems.paths == {n: out_dir / f"{n}.wav" for n in STEM_NAMES}

    # 音源が違えば再利用しない
    assert separation._load_cached(out_dir, "0" * 64) is None


class _FakeSeparator:
    """demucs.api.Separator の代わり。入力を 4 等分してステムにする。"""

    samplerate = SR
    audio_channels = 2

    def __init__(self) -> None:
        self.calls = 0

    def separate_tensor(self, wav, sr):
        from demucs.audio import convert_audio

        self.calls += 1
        if sr != self.samplerate:
            wav = convert_audio(wav, sr, self.samplerate, self.audio_channels)
        return wav, {n: wav / 4 for n in STEM_NAMES}


def test_separate_writes_stems_with_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("demucs.audio")
    fake = _FakeSeparator()
    monkeypatch.setattr(separation, "_get_separator", lambda device: fake)

    src = tmp_path / "mono22k.wav"  # モノラル・22.05 kHz でもモデルに合わせて直す
    sf.write(src, 0.5 * np.sin(2 * np.pi * 110 * np.arange(22050 * 2) / 22050), 22050)
    stems = separate(src, tmp_path / "stems", "cpu")
    assert fake.calls == 1
    for name in STEM_NAMES:
        info = sf.info(stems.paths[name])
        assert info.samplerate == SR and info.channels == 2
        assert info.frames == pytest.approx(2 * SR, abs=2)
    separate(src, tmp_path / "stems", "cpu")
    assert fake.calls == 1  # 2 回目は再利用


def test_separate_with_demucs(tmp_path: Path) -> None:
    """本物の htdemucs で分ける（初回は重みを取得する）。CPU で十数秒かかる。"""
    pytest.importorskip("torch")
    pytest.importorskip("demucs.api")
    sec = 10.0  # 区切り（約 7.8 秒）を 2 つにまたがる長さ
    t = np.arange(int(sec * SR)) / SR
    bass = 0.3 * np.sin(2 * np.pi * 55 * t) + 0.15 * np.sin(2 * np.pi * 110 * t)
    keys = 0.1 * sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63))
    kick = np.zeros_like(t)
    for s in range(0, len(t), SR // 2):
        n = min(4000, len(t) - s)
        kick[s:s + n] += 0.5 * np.sin(2 * np.pi * 60 * np.arange(n) / SR) * np.exp(-np.arange(n) / 600)
    mix = bass + keys + kick
    src = tmp_path / "mix.wav"
    sf.write(src, np.stack([mix, mix], axis=1) * 0.8, SR)

    stems = separate(src, tmp_path / "stems", "cpu")
    assert set(stems.paths) == set(STEM_NAMES)
    assert stems.sample_rate == SR
    mtimes = {}
    for name, p in stems.paths.items():
        info = sf.info(p)
        assert info.frames == len(t) and info.samplerate == SR and info.channels == 2
        mtimes[name] = p.stat().st_mtime_ns
    # 4 つを足すと元に近い
    total = sum(sf.read(p, dtype="float64")[0] for p in stems.paths.values())
    orig = sf.read(src, dtype="float64")[0]
    assert np.sqrt(np.mean((total - orig) ** 2)) < 0.1 * np.sqrt(np.mean(orig ** 2))

    again = separate(src, tmp_path / "stems", "cpu")
    assert again.paths == stems.paths
    assert all(p.stat().st_mtime_ns == mtimes[n] for n, p in again.paths.items())
