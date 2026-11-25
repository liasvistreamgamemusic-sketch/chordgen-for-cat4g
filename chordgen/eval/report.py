"""結果の表・CSV・JSON（純粋関数とファイルの書き出し）。"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from chordgen.eval.conditions import Condition, baseline
from chordgen.eval.metrics import METRICS, Aggregate, SongScore, aggregate

# 条件を基準と比べて「上がった / 下がった」と数える差（正解率の差。0.001 = 0.1 ポイント）
_TIE = 0.001

Results = dict[Condition, dict[str, SongScore]]


def _pct(v: float | None) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def _delta(v: float | None, b: float | None) -> str:
    return "     " if v is None or b is None else f"{(v - b) * 100:+5.1f}"


def format_tables(
    results: Results,
    songs: list[str],
    metrics: tuple[str, ...] = METRICS,
    top: int | None = None,
) -> str:
    """指標ごとに、条件（平均の高い順）× 曲の表を作る。値は %、Δ は基準の条件との差（ポイント）。"""
    conds = sorted(results)
    base = baseline(conds)
    aggs = {c: aggregate([results[c][s] for s in songs]) for c in conds}
    buf = io.StringIO()
    buf.write("曲: " + "  ".join(f"S{i + 1}={name}" for i, name in enumerate(songs)) + "\n")
    if base is not None:
        buf.write(f"基準: {base.id}（input/snap/bass/model/key）。Δ は基準との差（ポイント）\n")
        buf.write("上/下: 基準より上がった曲 / 下がった曲の数\n")
    width = max([len(c.id) for c in conds] + [9])
    for metric in metrics:
        buf.write(f"\n== {metric} ==\n")
        header = f"{'condition':<{width}}  mean   Δmean  weight   min  上/下"
        for i in range(len(songs)):
            header += f"  {'S' + str(i + 1):>5}  {'Δ':>5}"
        buf.write(header + "\n")
        order = sorted(conds, key=lambda c: -(aggs[c].mean[metric] if aggs[c].mean[metric] is not None else -1))
        if top:
            order = order[:top]
        for c in order:
            a = aggs[c]
            b = aggs[base] if base is not None else None
            up = down = 0
            cells = ""
            for s in songs:
                v = results[c][s].scores.get(metric)
                bv = results[base][s].scores.get(metric) if base is not None else None
                if v is not None and bv is not None:
                    up += v - bv > _TIE
                    down += bv - v > _TIE
                cells += f"  {_pct(v)}  {_delta(v, bv)}"
            line = (
                f"{c.id:<{width}}  {_pct(a.mean[metric])}  {_delta(a.mean[metric], b.mean[metric] if b else None)}"
                f"  {_pct(a.weighted[metric])}  {_pct(a.minimum[metric])}  {up:>2}/{down:<2}"
            )
            buf.write(line + cells + "\n")
    return buf.getvalue()


def results_to_rows(results: Results, songs: list[str]) -> list[dict]:
    """CSV 向けの縦長の行。song は曲名か __mean__（曲の平均）/ __weighted__（長さで重みづけ）。"""
    rows: list[dict] = []
    for c in sorted(results):
        agg = aggregate([results[c][s] for s in songs])
        for s in songs:
            sc = results[c][s]
            for m in METRICS:
                rows.append({"condition": c.id, **c.as_dict(), "song": s, "metric": m,
                             "score": sc.scores.get(m), "duration": sc.durations.get(m)})
        for name, vals in (("__mean__", agg.mean), ("__weighted__", agg.weighted), ("__min__", agg.minimum)):
            for m in METRICS:
                rows.append({"condition": c.id, **c.as_dict(), "song": name, "metric": m,
                             "score": vals[m], "duration": None})
    return rows


def results_to_json(results: Results, songs: list[str], meta: dict) -> dict:
    conds = []
    for c in sorted(results):
        agg: Aggregate = aggregate([results[c][s] for s in songs])
        conds.append({
            "id": c.id,
            **c.as_dict(),
            "perSong": {s: {"scores": results[c][s].scores, "durations": results[c][s].durations} for s in songs},
            "mean": agg.mean,
            "weighted": agg.weighted,
            "min": agg.minimum,
        })
    return {**meta, "songs": songs, "metrics": list(METRICS), "conditions": conds}


def write_reports(results: Results, songs: list[str], out_dir: Path, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = results_to_rows(results, songs)
    csv_path = out_dir / "results.csv"
    # Excel で開けるよう BOM つきの UTF-8
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["condition"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(results_to_json(results, songs, meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, json_path
