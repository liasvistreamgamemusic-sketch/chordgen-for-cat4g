"""偽の CLI（CHORDGEN_GUI_FAKE=1 のときに GUI が本物の代わりに起動する。画面の開発・テスト用）。

本物の CLI と同じ引数を受け取り、--progress-json と同じ形で数秒かけて全段の progress を出して、
specs/format/samples/analyzed.cat4g.json を元にした文書を output_dir() に書く。
音源・URL・題名に次の語が入っていると振る舞いを変える（大文字・小文字は区別しない）。
- fail: 失敗する（URL なら音源の取得、ファイルならコード認識で）
- warn: 警告を出す
- slow: 1 段にかける時間を 4 倍にする（中止を試すとき）
1 段あたりの秒数は環境変数 CHORDGEN_GUI_FAKE_DELAY（既定 0.5）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from chordgen import __version__
from chordgen.cli import EXIT_FAIL, EXIT_OK, _JsonEmitter, _setup_stdio, build_parser
from chordgen.gui import ENV_FAKE_DELAY
from chordgen.pipeline import STAGE_LABELS, PipelineOptions, ProgressEvent, default_output_path, plan_stages
from chordgen.paths import schema_path


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    _setup_stdio()
    args = build_parser().parse_args(argv)
    out = _JsonEmitter(sys.stdout, progress_interval=0.0)
    try:
        delay = max(0.0, float(os.environ.get(ENV_FAKE_DELAY, "0.5")))
    except ValueError:
        delay = 0.5
    words = f"{args.source} {args.title or ''}".lower()
    if "slow" in words:
        delay *= 4
    is_url = args.source.startswith(("http://", "https://"))
    lyrics = None
    if args.lyrics is not None:
        lyrics = Path(args.lyrics).read_text(encoding="utf-8-sig")
    opts = PipelineOptions(chord_input=args.chord_input, bass_mode=args.bass, snap=not args.no_snap,
                           chord_model=args.model, bars_per_line=args.bars_per_line)
    stages = plan_stages(opts, is_url=is_url, has_lyrics=bool(lyrics and lyrics.strip())) + ["write"]
    fail_at = ("fetch" if is_url else "chords") if "fail" in words else None
    if args.work_dir is not None:  # 片付けを試せるよう、作業フォルダに何か置く
        wd = Path(args.work_dir) / "fake-work"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "note.txt").write_text("偽のジョブの作業ファイル\n", encoding="utf-8")

    def emit(stage: str, status: str, **kw) -> None:
        ev = ProgressEvent(stage, STAGE_LABELS[stage], stages.index(stage) + 1, len(stages), status, **kw)
        out.progress(ev)

    _err(f"偽のジョブ（{__version__}）: {args.source}")
    for stage in stages:
        label = STAGE_LABELS[stage]
        _err(f"[{stages.index(stage) + 1}/{len(stages)}] {label}…")
        emit(stage, "start")
        t0 = time.perf_counter()
        if stage == "fetch":
            for pct in range(0, 101, 20):
                time.sleep(delay / 6)
                emit(stage, "progress", data={"type": "progress", "status": "downloading", "percent": float(pct),
                                              "downloadedBytes": pct * 10000, "totalBytes": 1000000})
                print(f"\r  取得中 {pct}%", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr, flush=True)
        else:
            time.sleep(delay)
        if stage == fail_at:
            elapsed = time.perf_counter() - t0
            emit(stage, "failed", elapsed=elapsed, message="偽の失敗")
            if stage == "fetch":
                code, msg = "UNAVAILABLE", "音源を取得できませんでした: 動画を再生できません（偽のジョブ）"
            else:
                code, msg = "STAGE_FAILED", f"{label}で失敗しました: RuntimeError: 偽の失敗"
            _err(f"エラー: {msg}")
            out.error(code, msg, stage, EXIT_FAIL)
            return EXIT_FAIL
        if stage == "key" and "warn" in words:
            msg = "キー推定: 確からしさが低いので、キーは参考にしてください（偽の警告）"
            _err(f"警告: {msg}")
            out.warning(msg)
        if stage == "write":
            path = _write(args, is_url)
            emit(stage, "done", elapsed=time.perf_counter() - t0)
            _err(f"{path} に書き出しました。")
            out.done(path, args.title or _default_title(args, is_url))
            return EXIT_OK
        emit(stage, "done", elapsed=time.perf_counter() - t0)
    return EXIT_OK


def _default_title(args, is_url: bool) -> str:
    return "偽の曲（URL）" if is_url else Path(args.source).stem


def _write(args, is_url: bool) -> Path:
    sample = schema_path().parent / "samples" / "analyzed.cat4g.json"
    doc = json.loads(sample.read_text(encoding="utf-8"))
    title = args.title or _default_title(args, is_url)
    song = doc["song"]
    song["title"] = title
    song["artist"] = args.artist or song.get("artist")
    if is_url:
        song["sourceUrl"] = args.source
    song["notes"] = "偽のジョブ（CHORDGEN_GUI_FAKE=1）で作った文書。中身はサンプルのまま"
    doc["exportedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    doc["appVersion"] = f"chordgen-for-cat4g {__version__} (fake)"
    path = default_output_path(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
