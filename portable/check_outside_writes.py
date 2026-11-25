"""アプリのフォルダの外に書き出していないかを確かめる道具（要件「起動とフォルダの外に書き出さない仕組み」）。

%USERPROFILE%\\.cache・%LOCALAPPDATA%・%APPDATA%（と %USERPROFILE% の直下）の中身を実行の前後で比べ、
新しくできたフォルダやファイルを一覧にする。読むだけで、何も消さない。

使い方（アプリのフォルダで。Python は同梱のものでも Windows に入っているものでもよい）:
    runtime\\python\\python.exe portable\\check_outside_writes.py run -- run.bat 自作の曲.wav
    runtime\\python\\python.exe portable\\check_outside_writes.py snapshot   （実行の前）
    runtime\\python\\python.exe portable\\check_outside_writes.py diff       （実行の後）

ほかのアプリが同じ時間に書いたものも出るので、名前を見て判断する（%LOCALAPPDATA%\\Temp の下は特に多い）。
終了コード: Temp・Microsoft の下を除いて新しいものがなければ 0、あれば 1。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = APP_DIR / "cache" / "outside-writes" / "snapshot.json"
# 見る深さ。新しいフォルダはその中までは並べない
DEFAULT_DEPTH = 2
# 他のアプリも頻繁に書くので、別に分けて表示する場所
NOISY = {"temp", "microsoft", "packages"}


def watch_roots(environ: dict[str, str] | None = None) -> list[tuple[str, int]]:
    """（見る場所, 深さ）の一覧。run.bat は HOME などを変えるが、USERPROFILE などは変えないのでそれを使う。"""
    env = environ if environ is not None else dict(os.environ)
    out: list[tuple[str, int]] = []
    profile = env.get("USERPROFILE") or str(Path.home())
    out.append((str(Path(profile) / ".cache"), DEFAULT_DEPTH))
    for key in ("LOCALAPPDATA", "APPDATA"):
        if env.get(key):
            out.append((env[key], DEFAULT_DEPTH))
    # ~/.matplotlib、~/.deno、~/.npmrc のような直下のもの
    out.append((profile, 1))
    return out


def snapshot_tree(roots: list[tuple[str, int]]) -> dict[str, dict]:
    """各 root の下を depth まで並べる。読めない場所は飛ばす。キーは絶対パス。"""
    entries: dict[str, dict] = {}
    for root, depth in roots:
        entries[root] = {"dir": os.path.isdir(root), "root": True}
        _walk(root, depth, entries)
    return entries


def _walk(path: str, depth: int, entries: dict[str, dict]) -> None:
    if depth <= 0:
        return
    try:
        it = list(os.scandir(path))
    except OSError:
        return
    for e in it:
        try:
            is_dir = e.is_dir(follow_symlinks=False)
        except OSError:
            continue
        entries[e.path] = {"dir": is_dir}
        if is_dir:
            _walk(e.path, depth - 1, entries)


def diff_snapshots(before: dict[str, dict], after: dict[str, dict]) -> list[str]:
    """after にだけあるパス。新しいフォルダの中身は親だけにまとめる。"""
    new = sorted(p for p in after if p not in before)
    out: list[str] = []
    for p in new:
        if any(p.startswith(parent + os.sep) or p.startswith(parent + "/") for parent in out):
            continue
        out.append(p)
    return out


def is_noisy(path: str) -> bool:
    parts = path.replace("\\", "/").lower().split("/")
    return any(p in NOISY for p in parts)


def classify(new: list[str], app_dir: Path = APP_DIR) -> tuple[list[str], list[str]]:
    """（このアプリが書いた疑いのあるもの, Temp などの参考）。アプリのフォルダの中は除く。"""
    app = str(app_dir).lower()
    outside = [p for p in new if not p.lower().startswith(app)]
    return [p for p in outside if not is_noisy(p)], [p for p in outside if is_noisy(p)]


def report(new: list[str], app_dir: Path = APP_DIR) -> str:
    main, noisy = classify(new, app_dir)
    mine = main + noisy
    lines = []
    if not mine:
        lines.append("フォルダの外に新しいものはできていません。")
        return "\n".join(lines)
    if main:
        lines.append("新しくできたもの（このアプリが書いた可能性があります）:")
        lines += [f"  {p}" for p in main]
    if noisy:
        lines.append("参考（Temp・Microsoft の下。ほかのアプリが書いたことが多い）:")
        lines += [f"  {p}" for p in noisy]
    return "\n".join(lines)


def save(entries: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="アプリのフォルダの外に書き出していないかを確かめる")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="今の状態を記録する（実行の前）")
    s.add_argument("--out", type=Path, default=DEFAULT_SNAPSHOT)
    d = sub.add_parser("diff", help="記録と今を比べる（実行の後）")
    d.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    r = sub.add_parser("run", help="記録してからコマンドを実行し、比べる")
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- のあとに実行するコマンド（例: -- run.bat 曲.wav）")
    args = p.parse_args(argv)

    roots = watch_roots()
    if args.cmd == "snapshot":
        save(snapshot_tree(roots), args.out)
        print(f"記録しました: {args.out}")
        return 0
    if args.cmd == "diff":
        if not args.snapshot.exists():
            print(f"記録がありません。先に snapshot を実行してください: {args.snapshot}")
            return 2
        new = diff_snapshots(load(args.snapshot), snapshot_tree(roots))
        print(report(new))
        return 1 if classify(new)[0] else 0

    cmd = [c for c in args.command if c != "--"] if args.command else []
    if not cmd:
        print("実行するコマンドを -- のあとに書いてください。例: run -- run.bat 曲.wav")
        return 2
    before = snapshot_tree(roots)
    print("実行: " + subprocess.list2cmdline(cmd))
    # .bat を動かすため Windows では cmd.exe を通す
    rc = subprocess.run(subprocess.list2cmdline(cmd) if os.name == "nt" else cmd, shell=os.name == "nt").returncode
    print(f"（終了コード {rc}）")
    new = diff_snapshots(before, snapshot_tree(roots))
    print(report(new))
    return 1 if classify(new)[0] else 0


if __name__ == "__main__":
    sys.exit(main())
