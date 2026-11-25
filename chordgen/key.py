"""キーの推定。

音源のクロマ（12 音の強さ）の平均と、調ごとの音の分布（キープロファイル）との相関を
24 キーで比べ、いちばん高いものを選ぶ（Krumhansl-Schmuckler 法）。
- 音源のクロマには Krumhansl-Kessler のプロファイルを使う（倍音で各音がにじむ音響の
  クロマには、なだらかな KK が合う）
- コード列（Harte 記法）からの分布には Temperley（Kostka-Payne）のプロファイルを使う
  （記号の分布で調べられたもので、調の外の音をはっきり下げる）
- コード列が渡されたら、両方の相関を重みつきで足す
クロマは librosa の chroma_cqt で取る。librosa がないときは numpy の STFT で近似する。

tonic の綴りは、そのキーの調号で普通に使う名前にする。調号の数が少ない方を選び、
同じ 6 つになるところはメジャーは F#（Gb ではなく）、マイナーは Eb（D# ではなく）にする。
  メジャー: C Db D Eb E F F# G Ab A Bb B
  マイナー: C C# D Eb E F F# G G# A Bb B
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from chordgen.types import ChordSegment, KeyEstimate

log = logging.getLogger(__name__)

MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")

# Krumhansl & Kessler (1982)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Temperley (2007)、Kostka-Payne の教科書の曲から
TKP_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
TKP_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

AUDIO_WEIGHT = 0.4  # chords があるときの音源の相関の重み（残りがコード列）
ROOT_BONUS = 1.0  # コード列の分布で、ルート音に足す重み（コードの長さ × この値）
EDGE_BONUS = 1.0  # 最初と最後のコードのルートに足す重み（平均のコードの長さ × この値）

# ---------------------------------------------------------------------------
# Harte 記法
# ---------------------------------------------------------------------------

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 0, 9: 2, 10: 4, 11: 5, 12: 7, 13: 9}
_SHORTHAND = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "minmaj7": (0, 3, 7, 11),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 2), "maj9": (0, 4, 7, 11, 2), "min9": (0, 3, 7, 10, 2),
    "11": (0, 4, 7, 10, 2, 5), "min11": (0, 3, 7, 10, 2, 5),
    "13": (0, 4, 7, 10, 2, 9), "maj13": (0, 4, 7, 11, 2, 9), "min13": (0, 3, 7, 10, 2, 9),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "1": (0,), "5": (0, 7),
}
_LABEL_RE = re.compile(r"^([A-G])([#b]*)(?::([^(/]*))?(?:\(([^)]*)\))?(?:/(.+))?$")
_DEGREE_RE = re.compile(r"^([#b]*)(\d+)$")


def _degree(text: str) -> int | None:
    m = _DEGREE_RE.match(text.strip())
    if not m or int(m.group(2)) not in _DEGREE_SEMITONES:
        return None
    acc = m.group(1).count("#") - m.group(1).count("b")
    return (_DEGREE_SEMITONES[int(m.group(2))] + acc) % 12


def parse_harte(label: str) -> tuple[int, frozenset[int], int] | None:
    """Harte 記法のラベルを (ルート, 構成音の集合, ベース音) の音高クラス（0=C）にする。

    "N"（無音）・"X"（不明）・読めないラベルは None。
    """
    m = _LABEL_RE.match(label.strip())
    if not m:
        return None
    letter, acc, shorthand, degrees, bass = m.groups()
    root = (_LETTER_PC[letter] + acc.count("#") - acc.count("b")) % 12
    if shorthand:
        if shorthand not in _SHORTHAND:
            return None
        intervals = set(_SHORTHAND[shorthand])
    elif degrees is not None:
        intervals = set()  # "C:(1,3,5)" は括弧の中だけ
    else:
        intervals = set(_SHORTHAND["maj"])  # "C" は C:maj
    if degrees:
        for item in degrees.split(","):
            item = item.strip()
            if not item:
                continue
            omit = item.startswith("*")
            iv = _degree(item.lstrip("*"))
            if iv is None:
                return None
            (intervals.discard if omit else intervals.add)(iv)
    bass_iv = 0
    if bass:
        b = _degree(bass)
        if b is None:
            return None
        bass_iv = b
        intervals.add(b)
    if not intervals:
        return None
    return root, frozenset((root + i) % 12 for i in intervals), (root + bass_iv) % 12


# ---------------------------------------------------------------------------
# 分布とキーの照合（純粋関数）
# ---------------------------------------------------------------------------


def chord_pitch_distribution(chords: Sequence[ChordSegment]) -> np.ndarray | None:
    """コード列から 12 音の分布を作る。各構成音にコードの長さを足し、ルートには重みを足す。

    N / X・読めないラベルは使わない。使えるコードがなければ None。
    """
    dist = np.zeros(12)
    parsed = []
    for c in chords:
        p = parse_harte(c.label)
        dur = float(c.end) - float(c.start)
        if p is None or not dur > 0:
            continue
        root, pcs, _ = p
        parsed.append((root, dur))
        for pc in pcs:
            dist[pc] += dur
        dist[root] += ROOT_BONUS * dur
    if not parsed:
        return None
    mean_dur = sum(d for _, d in parsed) / len(parsed)
    dist[parsed[0][0]] += EDGE_BONUS * mean_dur
    dist[parsed[-1][0]] += EDGE_BONUS * mean_dur
    return dist / dist.sum()


def key_scores(dist: np.ndarray, major: np.ndarray = KK_MAJOR,
               minor: np.ndarray = KK_MINOR) -> np.ndarray:
    """24 キーとの相関。添字 0〜11 は C〜B のメジャー、12〜23 は C〜B のマイナー。"""
    x = np.asarray(dist, dtype=float)
    if x.shape != (12,) or np.allclose(x, x[0]):
        return np.zeros(24)
    out = np.empty(24)
    for tonic in range(12):
        out[tonic] = np.corrcoef(x, np.roll(major, tonic))[0, 1]
        out[12 + tonic] = np.corrcoef(x, np.roll(minor, tonic))[0, 1]
    return out


def key_from_scores(scores: np.ndarray) -> KeyEstimate:
    """相関の最大のキーを選ぶ。

    confidence は 1 位と 2 位の差 m と 1 位の相関 r から (1 - exp(-m / 0.05)) × max(r, 0)。
    差 0.05 で約 0.63、0.1 で約 0.86 になり、全体に合っていなければ（r が低ければ）下がる。
    """
    order = np.argsort(scores)[::-1]
    best, second = int(order[0]), int(order[1])
    margin = float(scores[best] - scores[second])
    conf = (1.0 - np.exp(-margin / 0.05)) * max(float(scores[best]), 0.0)
    tonic, minor = best % 12, best >= 12
    return KeyEstimate(
        tonic=(MINOR_NAMES if minor else MAJOR_NAMES)[tonic],
        mode="minor" if minor else "major",
        confidence=round(float(np.clip(conf, 0.0, 1.0)), 3),
    )


def estimate_key_from_chroma(chroma: np.ndarray) -> KeyEstimate:
    """12 音の平均クロマ（0=C）からキーを出す。"""
    return key_from_scores(key_scores(chroma, KK_MAJOR, KK_MINOR))


def estimate_key_from_chords(chords: Sequence[ChordSegment]) -> KeyEstimate | None:
    """コード列（Harte 記法）だけからキーを出す。使えるコードがなければ None。"""
    dist = chord_pitch_distribution(chords)
    if dist is None:
        return None
    return key_from_scores(key_scores(dist, TKP_MAJOR, TKP_MINOR))


# ---------------------------------------------------------------------------
# 音源のクロマ
# ---------------------------------------------------------------------------


def _frames_mean(chroma: np.ndarray) -> np.ndarray:
    """(12, T) のクロマを、フレームごとに最大値で正規化してから平均する。無音のフレームは除く。"""
    energy = chroma.sum(axis=0)
    if energy.size == 0 or energy.max() <= 0:
        return np.zeros(12)
    keep = energy > 1e-3 * energy.max()
    c = chroma[:, keep]
    c = c / np.maximum(c.max(axis=0, keepdims=True), 1e-12)
    return c.mean(axis=1)


def chroma_numpy(y: np.ndarray, sr: int, fmin: float = 55.0, fmax: float = 5000.0) -> np.ndarray:
    """STFT の振幅を一番近い半音へまとめた平均クロマ（librosa がないときの代わり）。"""
    y = np.asarray(y, dtype=np.float64)
    n_fft = int(2 ** round(np.log2(sr * 0.37)))  # 22.05 kHz で 8192
    hop = n_fft // 2
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    pcs = np.round(12 * np.log2(freqs[band] / 440.0) + 69).astype(int) % 12
    onehot = np.zeros((12, int(band.sum())))
    onehot[pcs, np.arange(len(pcs))] = 1.0
    window = np.hanning(n_fft)
    starts = np.arange(0, len(y) - n_fft + 1, hop)
    cols = []
    for i in range(0, len(starts), 256):  # メモリを抑えるため少しずつ
        idx = starts[i:i + 256, None] + np.arange(n_fft)[None, :]
        mag = np.abs(np.fft.rfft(y[idx] * window, axis=1))[:, band]
        cols.append(onehot @ mag.T)
    return _frames_mean(np.concatenate(cols, axis=1))


def chroma_librosa(y: np.ndarray, sr: int) -> np.ndarray:
    """librosa の chroma_cqt の平均。"""
    import librosa

    target = 22050
    if sr != target:
        y = librosa.resample(y, orig_sr=sr, target_sr=target)
    chroma = librosa.feature.chroma_cqt(y=y, sr=target, hop_length=2048, norm=None)
    return _frames_mean(chroma)


def audio_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        return chroma_librosa(y, sr)
    except ImportError:
        log.info("librosa がないので numpy のクロマでキーを推定します")
        return chroma_numpy(y, sr)


def estimate_key(audio_path: Path, chords: list[ChordSegment] | None = None) -> KeyEstimate:
    """音源（とあればコード列）からキーを推定する。"""
    from chordgen.beats import load_audio_mono

    y, sr = load_audio_mono(Path(audio_path))
    scores = key_scores(audio_chroma(y, sr), KK_MAJOR, KK_MINOR)
    if chords:
        dist = chord_pitch_distribution(chords)
        if dist is not None:
            scores = AUDIO_WEIGHT * scores + (1 - AUDIO_WEIGHT) * key_scores(dist, TKP_MAJOR, TKP_MINOR)
    return key_from_scores(scores)
