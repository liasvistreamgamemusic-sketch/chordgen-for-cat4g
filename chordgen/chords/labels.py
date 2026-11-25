"""BTC の 170 クラスの番号と Harte 記法のラベルの対応。

並びは BTC / ChordMini の idx2voca_chord と同じ（番号 = ルート × 14 + 種類、168 = X、169 = N）。
ルートはシャープ表記、メジャーは "C" ではなく "C:maj" と書く。
"""

from __future__ import annotations

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = (
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
    "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
)
NUM_CLASSES = 170
X_INDEX = 168
N_INDEX = 169


def index_to_label(index: int) -> str:
    """クラス番号を Harte 記法にする（例 1 → "C:maj"、9*14+6 → "A:min7"）。"""
    i = int(index)
    if i == N_INDEX:
        return "N"
    if i == X_INDEX:
        return "X"
    if not 0 <= i < X_INDEX:
        raise ValueError(f"クラス番号が範囲外: {index}")
    root, quality = divmod(i, len(QUALITIES))
    return f"{PITCH_CLASSES[root]}:{QUALITIES[quality]}"


def label_to_index(label: str) -> int:
    """index_to_label の逆。表にないラベルは ValueError。"""
    if label == "N":
        return N_INDEX
    if label == "X":
        return X_INDEX
    root, sep, quality = label.partition(":")
    if not sep or root not in PITCH_CLASSES or quality not in QUALITIES:
        raise ValueError(f"170 クラスにないラベル: {label!r}")
    return PITCH_CLASSES.index(root) * len(QUALITIES) + QUALITIES.index(quality)


VOCAB: tuple[str, ...] = tuple(index_to_label(i) for i in range(NUM_CLASSES))
