"""予測と正解の区間から mir_eval.chord の指標を出す（純粋関数。モデルは使わない）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chordgen.types import ChordSegment

# mir_eval.chord の比較関数の名前。どれも長さで重みづけした正解率にする。
# 上の 7 つは分数ベースを見ないので、ベースの使い方（slash）を比べるため転回形まで見る 2 つを足す
METRICS = ("root", "majmin", "thirds", "triads", "sevenths", "tetrads", "mirex", "majmin_inv", "sevenths_inv")


@dataclass
class SongScore:
    """1 曲の結果。scores[指標] は正解率（比べられる区間がなければ None）、
    durations[指標] はその指標で比べた長さ（秒。majmin で対象外のコードなどは除く）。"""

    scores: dict[str, float | None] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)


def fill_gaps(segments: list[ChordSegment], start: float, end: float) -> list[ChordSegment]:
    """[start, end] の範囲に切り詰め、すき間を N で埋める。"""
    out: list[ChordSegment] = []
    t = start
    for s in sorted(segments, key=lambda x: x.start):
        a, b = max(s.start, t), min(s.end, end)
        if b <= a:
            continue
        if a > t + 1e-9:
            out.append(ChordSegment(t, a, "N"))
        out.append(ChordSegment(a, b, s.label, s.confidence))
        t = b
    if t < end - 1e-9:
        out.append(ChordSegment(t, end, "N"))
    return out


def _to_arrays(segs: list[ChordSegment]) -> tuple[np.ndarray, list[str]]:
    return np.array([[s.start, s.end] for s in segs], dtype=float).reshape(-1, 2), [s.label for s in segs]


def score_song(reference: list[ChordSegment], estimated: list[ChordSegment]) -> SongScore:
    """正解の範囲（最初の開始〜最後の終了）で予測を比べる。予測のない部分は N とみなす。"""
    import mir_eval

    if not reference:
        raise ValueError("正解の区間がありません")
    start = min(s.start for s in reference)
    end = max(s.end for s in reference)
    ref = fill_gaps(reference, start, end)
    est = fill_gaps(estimated, start, end)
    ref_iv, ref_lab = _to_arrays(ref)
    est_iv, est_lab = _to_arrays(est)
    for lab in set(ref_lab) | set(est_lab):
        mir_eval.chord.encode(lab)  # 読めないラベルはここで例外にする
    iv, ref_lab, est_lab = mir_eval.util.merge_labeled_intervals(ref_iv, ref_lab, est_iv, est_lab)
    durations = mir_eval.util.intervals_to_durations(iv)
    result = SongScore()
    for metric in METRICS:
        comp = np.asarray(getattr(mir_eval.chord, metric)(ref_lab, est_lab), dtype=float)
        valid = comp >= 0
        total = float(durations[valid].sum())
        result.durations[metric] = total
        result.scores[metric] = float((comp[valid] * durations[valid]).sum() / total) if total > 0 else None
    return result


@dataclass
class Aggregate:
    mean: dict[str, float | None]  # 曲ごとの値の平均（曲を同じ重みで）
    weighted: dict[str, float | None]  # 全曲の長さで重みづけ
    minimum: dict[str, float | None]  # 曲ごとの値の最小


def aggregate(songs: list[SongScore]) -> Aggregate:
    mean: dict[str, float | None] = {}
    weighted: dict[str, float | None] = {}
    minimum: dict[str, float | None] = {}
    for m in METRICS:
        vals = [s.scores[m] for s in songs if s.scores.get(m) is not None]
        mean[m] = float(np.mean(vals)) if vals else None
        minimum[m] = float(np.min(vals)) if vals else None
        num = sum(s.scores[m] * s.durations[m] for s in songs if s.scores.get(m) is not None)
        den = sum(s.durations[m] for s in songs if s.scores.get(m) is not None)
        weighted[m] = float(num / den) if den > 0 else None
    return Aggregate(mean=mean, weighted=weighted, minimum=minimum)
