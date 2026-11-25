"""BeatGrid から、時刻 ⇄ 拍の位置 ⇄ 小節を引ける格子を作る。

拍の位置は「格子の先頭の拍を 0 とした拍の番号」（小数可）。拍の検出範囲の外は、
拍の間隔の中央値で拍を前後に延ばした仮の拍で埋める。最初に検出した小節の 1 拍目より
前は、拍子どおりの長さの仮の小節を前に足す（弱起のコードはその小節に入る）。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from statistics import median

from chordgen.types import BeatGrid

DEFAULT_PERIOD = 0.5  # 拍も BPM も使えないときの拍の間隔（120 BPM）


@dataclass
class Grid:
    beats: list[float]  # 拍の時刻（前後に延ばしたもの込み）
    bar_starts: list[int]  # 各小節の先頭の拍の番号。最後の要素は最後の小節の終わり
    first_detected_bar: int  # 検出した最初の小節の 1 拍目が何番目の小節か
    period: float

    @property
    def n_bars(self) -> int:
        return len(self.bar_starts) - 1

    def beat_pos(self, t: float) -> float:
        b = self.beats
        i = min(max(bisect_right(b, t) - 1, 0), len(b) - 2)
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def time_at(self, pos: float) -> float:
        b = self.beats
        i = min(max(math.floor(pos), 0), len(b) - 2)
        return b[i] + (pos - i) * (b[i + 1] - b[i])

    def bar_of_pos(self, pos: float) -> int:
        k = bisect_right(self.bar_starts, pos) - 1
        return min(max(k, 0), self.n_bars - 1)

    def bar_of_time(self, t: float) -> int:
        return self.bar_of_pos(round(self.beat_pos(t), 2))

    def bar_start_pos(self, bar: int) -> int:
        return self.bar_starts[bar]

    def bar_start_time(self, bar: int) -> float:
        return self.beats[self.bar_starts[bar]]


def _period(times: list[float], bpm: float) -> float:
    diffs = [b - a for a, b in zip(times, times[1:]) if b - a > 1e-6]
    if diffs:
        return median(diffs)
    if math.isfinite(bpm) and bpm > 0:
        return 60.0 / bpm
    return DEFAULT_PERIOD


def build_grid(beat_grid: BeatGrid, t_min: float, t_max: float) -> Grid:
    """t_min〜t_max を覆う格子を作る。"""
    bpb = beat_grid.beats_per_bar if beat_grid.beats_per_bar and beat_grid.beats_per_bar >= 1 else 4
    bpm = float(beat_grid.bpm) if beat_grid.bpm is not None else float("nan")

    # 拍を昇順に並べ、同じ時刻の拍はまとめる（1 拍目の印は残す）
    down_set = {i for i in beat_grid.downbeat_indices if 0 <= i < len(beat_grid.beat_times)}
    pairs = sorted(
        (float(t), i in down_set) for i, t in enumerate(beat_grid.beat_times) if math.isfinite(float(t))
    )
    beats: list[float] = []
    downs: list[int] = []
    for t, is_down in pairs:
        if beats and t - beats[-1] <= 1e-6:
            if is_down and (not downs or downs[-1] != len(beats) - 1):
                downs.append(len(beats) - 1)
            continue
        beats.append(t)
        if is_down:
            downs.append(len(beats) - 1)

    period = _period(beats, bpm)
    if not beats:
        beats = [max(0.0, t_min)]
    if not downs:
        downs = [0]

    # 前へ延ばす: t_min を覆い、仮の小節が拍子どおりの長さになるまで
    n_pre = 0
    while beats[0] > t_min:
        beats.insert(0, beats[0] - period)
        n_pre += 1
    extra = (-(downs[0] + n_pre)) % bpb
    for _ in range(extra):
        beats.insert(0, beats[0] - period)
    n_pre += extra
    downs = [d + n_pre for d in downs]

    # 後ろへ延ばす: t_max を越え、最後の拍が小節の区切りになるまで
    while len(beats) < 2 or beats[-1] <= t_max:
        beats.append(beats[-1] + period)
    while (len(beats) - 1 - downs[-1]) % bpb:
        beats.append(beats[-1] + period)

    starts: list[int] = list(range(downs[0] % bpb, downs[0], bpb))
    first_detected_bar = len(starts)
    for d, d_next in zip(downs, downs[1:] + [len(beats) - 1]):
        gap = d_next - d
        if d_next == len(beats) - 1 or gap > bpb * 1.5:
            # 1 拍目が見つからない長い区間と最後の小節より後ろは、拍子どおりに区切る
            starts.extend(range(d, d_next, bpb))
        else:
            starts.append(d)
    starts.append(len(beats) - 1)
    return Grid(beats=beats, bar_starts=starts, first_detected_bar=first_detected_bar, period=period)
