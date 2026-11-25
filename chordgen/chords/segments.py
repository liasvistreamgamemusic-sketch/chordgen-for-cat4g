"""フレームごとの予測を区間（ChordSegment）にまとめる。numpy だけで動く。"""

from __future__ import annotations

import numpy as np

from chordgen.types import ChordSegment

from .labels import index_to_label


def majority_filter(indices: np.ndarray, kernel_size: int) -> np.ndarray:
    """クラス番号の多数決フィルタ（ChordMini の majority_filter_indices と同じ結果）。

    窓の中で最も多い番号にする。同数なら中央の番号が候補にあればそれを、なければ小さい番号を選ぶ。
    端は端の値で埋める。
    """
    values = np.asarray(indices, dtype=np.int64)
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if values.ndim != 1 or kernel_size == 1 or values.size < kernel_size:
        return values.copy()
    pad = kernel_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, (pad, pad), mode="edge"), kernel_size)
    out = np.empty_like(values)
    for i, window in enumerate(windows):
        labels, counts = np.unique(window, return_counts=True)
        candidates = labels[counts == counts.max()]
        out[i] = values[i] if values[i] in candidates else int(candidates[0])
    return out


def _runs(indices: np.ndarray) -> list[list[int]]:
    """[番号, 始まり, 終わり（含まない）] の並び。"""
    if indices.size == 0:
        return []
    change = np.flatnonzero(np.diff(indices)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [indices.size]))
    return [[int(indices[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def merge_short_runs(indices: np.ndarray, min_frames: int, probs: np.ndarray | None = None) -> np.ndarray:
    """min_frames より短い区間を隣の区間に吸収させる。

    ChordMini の test.py は短い区間を捨てて隙間を残すが、ここでは隙間を作らないよう、
    前後の区間のうち、その区間での確率（probs[フレーム, 番号] の平均）が高いほうに付ける。
    probs がなければ長いほうに付ける。短い区間から順に処理する。
    """
    out = np.asarray(indices, dtype=np.int64).copy()
    if min_frames <= 1:
        return out
    while True:
        runs = _runs(out)
        if len(runs) <= 1:
            return out
        lengths = [e - s for _, s, e in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_frames:
            return out
        _, s, e = runs[k]
        neighbors = [runs[j] for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        if probs is not None:
            best = max(neighbors, key=lambda r: float(probs[s:e, r[0]].mean()))
        else:
            best = max(neighbors, key=lambda r: r[2] - r[1])
        out[s:e] = best[0]


def frames_to_segments(
    indices: np.ndarray,
    frame_duration: float,
    probs: np.ndarray | None = None,
    total_duration: float | None = None,
) -> list[ChordSegment]:
    """同じ番号が続くフレームを 1 区間にまとめる。

    フレーム i の時刻は i * frame_duration（ChordMini の .lab と同じ）。
    confidence は区間の各フレームでの、その番号の確率の平均。
    total_duration を渡すと、最後の区間の終わりをそこまで伸ばす（1 フレーム未満の端数）。
    """
    indices = np.asarray(indices, dtype=np.int64)
    segments: list[ChordSegment] = []
    for idx, s, e in _runs(indices):
        conf = float(probs[s:e, idx].mean()) if probs is not None else None
        segments.append(
            ChordSegment(start=s * frame_duration, end=e * frame_duration, label=index_to_label(idx), confidence=conf)
        )
    if segments and total_duration is not None and total_duration > segments[-1].end:
        segments[-1].end = float(total_duration)
    return segments
