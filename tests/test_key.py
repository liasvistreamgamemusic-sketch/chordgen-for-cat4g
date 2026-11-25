"""chordgen.key のテスト。合成した和音とコード列だけを使う。"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from chordgen import key as K
from chordgen.types import ChordSegment

SR = 22050
_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6,
       "G": 7, "Ab": 8, "G#": 8, "A": 9, "Bb": 10, "B": 11}
_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def seq(labels: list[str], dur: float = 2.0) -> list[ChordSegment]:
    return [ChordSegment(i * dur, (i + 1) * dur, label) for i, label in enumerate(labels)]


def synth_chords(chords: list[list[int]], dur: float = 1.5, repeat: int = 2) -> np.ndarray:
    """MIDI ノート番号の和音を、倍音つきの減衰音で順に鳴らす。"""
    t = np.arange(int(dur * SR)) / SR
    parts = []
    for notes in chords:
        s = sum(
            sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * h * t) / h ** 1.5 for h in range(1, 6))
            for m in notes
        )
        parts.append(s * np.exp(-t * 1.5))
    y = np.concatenate(parts * repeat)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def triad(root: int, minor: bool = False, seventh: bool = False) -> list[int]:
    notes = [root - 12, root, root + (3 if minor else 4), root + 7]
    if seventh:
        notes.append(root + 10)
    return notes


def cadence_audio(tonic: int, minor: bool) -> np.ndarray:
    """i(I) - iv(IV) - V7 - i(I) の和音（tonic は 48〜59 の MIDI 番号）。"""
    return synth_chords([
        triad(tonic, minor), triad(tonic + 5, minor), triad(tonic + 7, False, True), triad(tonic, minor),
    ])


# ---------------------------------------------------------------------------
# Harte 記法
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label, expected", [
    ("C:maj", (0, {0, 4, 7}, 0)),
    ("C", (0, {0, 4, 7}, 0)),
    ("A:min7", (9, {9, 0, 4, 7}, 9)),
    ("C:maj/3", (0, {0, 4, 7}, 4)),
    ("Bb:hdim7", (10, {10, 1, 4, 8}, 10)),
    ("F#:sus4", (6, {6, 11, 1}, 6)),
    ("Db:7", (1, {1, 5, 8, 11}, 1)),
    ("C:(1,b3,5)", (0, {0, 3, 7}, 0)),
    ("C:maj(9,*5)", (0, {0, 2, 4}, 0)),
    ("G:min/b7", (7, {7, 10, 2, 5}, 5)),
])
def test_parse_harte(label, expected):
    root, pcs, bass = K.parse_harte(label)
    assert (root, set(pcs), bass) == expected


@pytest.mark.parametrize("label", ["N", "X", "", "H:maj", "C:foo", "C:maj(15)"])
def test_parse_harte_unusable(label):
    assert K.parse_harte(label) is None


# ---------------------------------------------------------------------------
# コード列からのキー
# ---------------------------------------------------------------------------

EXPECTED_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
EXPECTED_MINOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


@pytest.mark.parametrize("tonic", range(12))
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_key_from_chords_all_keys_and_spelling(tonic, mode):
    q = "min" if mode == "minor" else "maj"
    name = lambda pc: _SHARP[pc % 12]  # 入力のラベルはわざとシャープ表記
    labels = [f"{name(tonic)}:{q}", f"{name(tonic + 5)}:{q}", f"{name(tonic + 7)}:7", f"{name(tonic)}:{q}",
              f"{name(tonic + (9 if mode == 'major' else 8))}:{'min' if mode == 'major' else 'maj'}",
              f"{name(tonic + 7)}:7", f"{name(tonic)}:{q}"]
    est = K.estimate_key_from_chords(seq(labels))
    expected = (EXPECTED_MAJOR if mode == "major" else EXPECTED_MINOR)[tonic]
    assert (est.tonic, est.mode) == (expected, mode)
    assert 0.0 <= est.confidence <= 1.0


@pytest.mark.parametrize("labels, tonic, mode", [
    (["C:maj", "F:maj", "G:7", "C:maj", "A:min", "D:min7", "G:7", "C:maj"], "C", "major"),
    (["A:min", "D:min", "E:7", "A:min", "F:maj", "D:min", "E:7", "A:min"], "A", "minor"),
    (["Eb:maj", "Ab:maj", "Bb:7", "Eb:maj", "C:min", "F:min7", "Bb:7", "Eb:maj"], "Eb", "major"),
    (["E:min", "C:maj", "D:maj", "B:7", "E:min", "A:min", "B:7", "E:min"], "E", "minor"),
    (["C:maj", "G:maj/3", "A:min", "F:maj"] * 4, "C", "major"),
])
def test_key_from_chords_progressions(labels, tonic, mode):
    est = K.estimate_key_from_chords(seq(labels))
    assert (est.tonic, est.mode) == (tonic, mode)


def test_key_from_chords_duration_weighting():
    # C と Am を行き来する。長く鳴る方が主和音とみなされる
    a_long = [ChordSegment(0, 8, "A:min"), ChordSegment(8, 9, "C:maj"), ChordSegment(9, 10, "E:7"),
              ChordSegment(10, 18, "A:min"), ChordSegment(18, 19, "C:maj"), ChordSegment(19, 27, "A:min")]
    est = K.estimate_key_from_chords(a_long)
    assert (est.tonic, est.mode) == ("A", "minor")


def test_key_from_chords_none_when_no_pitched_chords():
    assert K.estimate_key_from_chords(seq(["N", "X", "N"])) is None
    assert K.estimate_key_from_chords([]) is None
    assert K.chord_pitch_distribution(seq(["N", "C:maj"])) is not None


def test_key_from_scores_confidence_range():
    flat = K.key_from_scores(np.zeros(24))
    assert 0.0 <= flat.confidence <= 1.0 and flat.confidence == 0.0
    s = np.zeros(24)
    s[9 + 12] = 0.9
    s[0] = 0.7
    est = K.key_from_scores(s)
    assert (est.tonic, est.mode) == ("A", "minor") and est.confidence > 0.85  # 差 0.2、1 位 0.9


# ---------------------------------------------------------------------------
# 音源からのキー
# ---------------------------------------------------------------------------

AUDIO_CASES = [("C", "major"), ("A", "minor"), ("Eb", "major"), ("F#", "minor"), ("Ab", "major")]


def _midi_tonic(name: str) -> int:
    return 48 + _PC[name]


@pytest.mark.parametrize("tonic, mode", AUDIO_CASES)
def test_chroma_numpy_key(tonic, mode):
    y = cadence_audio(_midi_tonic(tonic), mode == "minor")
    est = K.estimate_key_from_chroma(K.chroma_numpy(y, SR))
    assert (est.tonic, est.mode) == (tonic, mode)


@pytest.mark.parametrize("tonic, mode", AUDIO_CASES)
def test_chroma_librosa_key(tonic, mode):
    pytest.importorskip("librosa")
    y = cadence_audio(_midi_tonic(tonic), mode == "minor")
    est = K.estimate_key_from_chroma(K.chroma_librosa(y, SR))
    assert (est.tonic, est.mode) == (tonic, mode)


@pytest.mark.parametrize("tonic, mode", AUDIO_CASES[:3])
def test_estimate_key_from_file(tmp_path, tonic, mode):
    path = tmp_path / "k.wav"
    sf.write(path, cadence_audio(_midi_tonic(tonic), mode == "minor"), SR)
    est = K.estimate_key(path)
    assert (est.tonic, est.mode) == (tonic, mode)
    assert 0.0 <= est.confidence <= 1.0


def test_estimate_key_with_chords_resolves_ambiguity(tmp_path):
    # Am F C G は音だけでは C メジャーと区別しにくい。コード列で Am が長く、E7 があれば A マイナー
    path = tmp_path / "amb.wav"
    sf.write(path, synth_chords([triad(57, True), triad(53), triad(48), triad(55)]), SR)
    chords = [ChordSegment(0, 6, "A:min"), ChordSegment(6, 7, "F:maj"), ChordSegment(7, 8, "C:maj"),
              ChordSegment(8, 9, "E:7"), ChordSegment(9, 15, "A:min")]
    est = K.estimate_key(path, chords)
    assert (est.tonic, est.mode) == ("A", "minor")
