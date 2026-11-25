"""python -m chordgen.eval の入口。使い方は chordgen.eval の docstring。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from chordgen.eval.conditions import AXES, enumerate_conditions, parse_filters
from chordgen.eval.metrics import METRICS


def build_parser() -> argparse.ArgumentParser:
    axes = "\n".join(f"  {k:<6} {', '.join(v)}" for k, v in AXES.items())
    p = argparse.ArgumentParser(
        prog="python -m chordgen.eval",
        description="コード認識の正解率を条件ごとに比べる（mir_eval）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"条件の軸と値:\n{axes}",
    )
    p.add_argument("--dataset", type=Path, help="曲名.wav と曲名.lab を置いたフォルダ")
    p.add_argument("--conditions", nargs="+", metavar="軸=値,値",
                   help="比べる条件を絞る（例 input=original,no_drums snap=beat model=chordmini）")
    p.add_argument("--metrics", default=",".join(METRICS),
                   help=f"表に出す指標（カンマ区切り。CSV / JSON にはすべて出す）。既定 {','.join(METRICS)}")
    p.add_argument("--top", type=int, default=0, help="表に出す条件の数（平均の高い順。0 はすべて）")
    p.add_argument("--device", default=None, help="cuda / cpu（既定は自動）")
    p.add_argument("--out", type=Path, default=None, help="結果の置き場所（既定 作業フォルダ/eval/results/<日時>）")
    p.add_argument("--refresh", action="store_true", help="キャッシュを消して重い処理をやり直す")
    p.add_argument("--save-labs", action="store_true", help="条件ごとの予測を .lab で結果の置き場所に書く")
    p.add_argument("--list-conditions", action="store_true", help="条件の一覧を出して終わる")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        conditions = enumerate_conditions(parse_filters(args.conditions))
        metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
        bad = [m for m in metrics if m not in METRICS]
        if bad:
            raise ValueError(f"知らない指標です: {', '.join(bad)}")
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    if args.list_conditions:
        for c in conditions:
            print(c.id)
        print(f"{len(conditions)} 条件（input/snap/bass/model/key）", file=sys.stderr)
        return 0
    if args.dataset is None:
        print("エラー: --dataset を指定してください", file=sys.stderr)
        return 2
    try:
        import mir_eval  # noqa: F401
    except ImportError:
        print("エラー: mir_eval が入っていません（pip install mir_eval）", file=sys.stderr)
        return 1

    from chordgen.eval.labfile import write_lab
    from chordgen.eval.report import format_tables, write_reports
    from chordgen.eval.runner import SongCache, find_songs, run, stderr_log
    from chordgen.paths import work_dir

    try:
        songs, warnings = find_songs(args.dataset)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        stderr_log(f"注意: {w}")
    if not songs:
        print("エラー: 音源と .lab の組が見つかりません", file=sys.stderr)
        return 2
    names = [s.name for s in songs]
    if len(set(names)) != len(names):
        print("エラー: 曲名が重複しています", file=sys.stderr)
        return 2

    if args.device:
        device = args.device
    else:
        from chordgen.device import get_device

        device = get_device()
    cache_root = work_dir() / "eval" / "cache"
    if args.refresh:
        for s in songs:
            SongCache(s, cache_root, device, stderr_log).clear()
    stderr_log(f"{len(songs)} 曲 × {len(conditions)} 条件（device={device}）")

    output = run(songs, conditions, cache_root, device, log=stderr_log)
    if not output.results:
        print("エラー: 採点できた条件がありません", file=sys.stderr)
        for c, reason in output.failures.items():
            print(f"  {c.id}: {reason}", file=sys.stderr)
        return 1

    out_dir = args.out or (work_dir() / "eval" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    meta = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "device": device,
        "failures": {c.id: r for c, r in output.failures.items()},
    }
    csv_path, json_path = write_reports(output.results, names, out_dir, meta)
    if args.save_labs:
        for cond, per_song in output.predictions.items():
            for song, segs in per_song.items():
                write_lab(out_dir / "labs" / cond.id.replace("/", "_") / f"{song}.lab", segs)

    print(format_tables(output.results, names, metrics, top=args.top or None))
    if output.failures:
        # 同じ原因の条件はまとめて出す（すべては JSON の failures にある）
        by_reason: dict[str, list[str]] = {}
        for c, reason in output.failures.items():
            by_reason.setdefault(reason, []).append(c.id)
        print(f"\n採点できなかった条件（{len(output.failures)}）:")
        for reason, ids in by_reason.items():
            more = f" ほか {len(ids) - 1} 条件" if len(ids) > 1 else ""
            print(f"  {reason}\n    {ids[0]}{more}")
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
