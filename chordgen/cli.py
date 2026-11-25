"""コマンドライン: chordgen <音源ファイルか URL> [オプション]

終了コード: 成功 0、失敗 1、引数の誤り 2、Ctrl+C 130（URL の取得の失敗は media-dl の終了コード）。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from chordgen import __version__
from chordgen.pipeline import (
    BASS_MODES,
    CHORD_INPUT_MODES,
    CHORD_MODELS,
    DEFAULT_BARS_PER_LINE,
    DEFAULT_BASS_MODE,
    DEFAULT_CHORD_INPUT,
    DEFAULT_CHORD_MODEL,
    MAX_BARS_PER_LINE,
    PipelineOptions,
    ProgressEvent,
    StageError,
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

_CHORD_INPUT_HELP = {
    "original": "原音のまま",
    "no_drums": "ドラムだけ除く",
    "no_drums_vocals": "ドラムとボーカルを除く",
    "boost_other": "other（ギター・ピアノなど）を大きくして組み直す",
}
_BASS_HELP = {"off": "使わない", "root_fix": "ルート音を補正する", "slash": "ルートと違えば分数コードにする"}


class _JapaneseHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        return super().add_usage(usage, actions, groups, prefix if prefix is not None else "使い方: ")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # 引数の誤りは終了コード 2
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: 引数の誤り: {message}\n")


def _bars_per_line(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数を指定してください: {text!r}") from None
    if not 1 <= n <= MAX_BARS_PER_LINE:
        raise argparse.ArgumentTypeError(f"1〜{MAX_BARS_PER_LINE} を指定してください: {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    def choices_help(table: dict[str, str], default: str) -> str:
        return "、".join(f"{k}={v}" for k, v in table.items()) + f"（既定 {default}）"

    p = _Parser(
        prog="chordgen",
        description="音源を解析して、cat4g に読み込めるコード譜の下書き（.cat4g.json）を作ります。",
        epilog="例:\n  chordgen 曲.wav --lyrics 歌詞.txt\n  chordgen https://example.com/watch?v=xxxx --title 題名",
        formatter_class=_JapaneseHelpFormatter,
        add_help=False,
    )
    p._positionals.title = "引数"
    p._optionals.title = "オプション"
    p.add_argument("-h", "--help", action="help", help="このヘルプを表示して終わります")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}", help="版を表示して終わります")
    p.add_argument("source", metavar="音源", help="音源ファイルのパスか URL（URL は media-dl で取得します）")
    p.add_argument("--lyrics", metavar="歌詞.txt", type=Path,
                   help="歌詞のテキストファイル（UTF-8。BOM つきも可）。あればコードを歌詞の位置に置きます")
    p.add_argument("--title", metavar="題名", help="題名（なければ URL の情報、音源のタグ、ファイル名の順に使います）")
    p.add_argument("--artist", metavar="名前", help="アーティスト")
    p.add_argument("-o", "--output", metavar="出力先", type=Path,
                   help="出力するファイルかフォルダ（既定は output フォルダの <題名>.cat4g.json。"
                        "同名があれば連番をつけます。ファイルを指定したときは上書きします）")
    p.add_argument("--chord-input", choices=CHORD_INPUT_MODES, default=DEFAULT_CHORD_INPUT,
                   help="コード認識に入れる音。" + choices_help(_CHORD_INPUT_HELP, DEFAULT_CHORD_INPUT))
    p.add_argument("--bass", choices=BASS_MODES, default=DEFAULT_BASS_MODE,
                   help="ベースの認識結果の使い方。" + choices_help(_BASS_HELP, DEFAULT_BASS_MODE))
    p.add_argument("--no-snap", action="store_true", help="コードの切り替えを拍にそろえない")
    p.add_argument("--model", choices=CHORD_MODELS, default=DEFAULT_CHORD_MODEL,
                   help=f"コード認識のモデル。chordmini=ChordMini の BTC（CL full）、btc=元の BTC（比較用）（既定 {DEFAULT_CHORD_MODEL}）")
    p.add_argument("--device", choices=("cuda", "cpu"), help="推論に使うデバイス（既定は GPU があれば cuda）")
    p.add_argument("--language", metavar="言語", help="歌詞の言語（例 ja。既定は自動判定）")
    p.add_argument("--bars-per-line", metavar="小節数", type=_bars_per_line, default=DEFAULT_BARS_PER_LINE,
                   help=f"歌詞がないときに 1 行に入れる小節の数（既定 {DEFAULT_BARS_PER_LINE}）")
    p.add_argument("--keep-work", action="store_true", help="作業フォルダ（取得した音源・分離した音）を消さずに残す")
    p.add_argument("-v", "--verbose", action="store_true", help="詳しいログと、失敗したときのトレースバックを出す")
    return p


def _setup_stdio() -> None:
    """Windows のコンソールやリダイレクト先でも日本語で落ちないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


class _ProgressPrinter:
    """段ごとに 1 行を標準エラーに出す。取得中の progress は間引く。"""

    def __init__(self, stream: TextIO, verbose: bool, interval: float = 2.0):
        self.stream = stream
        self.verbose = verbose
        self.interval = interval
        self._last_fetch = 0.0
        self._last_percent: float | None = None

    def __call__(self, ev: ProgressEvent) -> None:
        head = f"[{ev.index}/{ev.total}] {ev.label}"
        if ev.status == "start":
            self._print(f"{head}…")
        elif ev.status == "done" and self.verbose and ev.elapsed is not None:
            self._print(f"{head} 終わり（{ev.elapsed:.1f} 秒）")
        elif ev.status == "progress" and ev.data:
            self._fetch_progress(ev.data)
        # skipped / failed は pipeline のログ（警告）と main のエラー表示で示す

    def _fetch_progress(self, data: dict[str, Any]) -> None:
        percent = _percent(data)
        now = time.monotonic()
        if now - self._last_fetch < self.interval:
            # 間隔が短くても、10% 以上進んだときは出す
            if percent is None or self._last_percent is None or percent - self._last_percent < 10:
                return
        self._last_fetch = now
        self._last_percent = percent
        parts = ["  取得中"]
        if percent is not None:
            parts.append(f"{percent:.0f}%")
        for k in ("speed", "eta"):
            if data.get(k) not in (None, ""):
                parts.append(f"{k}={data[k]}")
        if len(parts) > 1:
            self._print(" ".join(parts))

    def _print(self, text: str) -> None:
        print(text, file=self.stream, flush=True)


def _percent(data: dict[str, Any]) -> float | None:
    """media-dl の progress から割合（0〜100）を取り出す。キーの名前の違いを吸収する。"""
    for k in ("percent", "percentage", "progress"):
        v = data.get(k)
        if isinstance(v, (int, float)):
            return float(v) * (100.0 if k == "progress" and v <= 1 else 1.0)
        if isinstance(v, str):
            try:
                return float(v.strip().rstrip("%"))
            except ValueError:
                pass
    done = data.get("downloadedBytes", data.get("downloaded"))
    total = data.get("totalBytes", data.get("total"))
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return 100.0 * done / total
    return None


class _WarningFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            prefix = "警告: " if record.levelno == logging.WARNING else "エラー: "
            msg = prefix + record.getMessage()
            if record.exc_info:
                msg += "\n" + self.formatException(record.exc_info)
            return msg
        return super().format(record)


def _setup_logging(verbose: bool) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_WarningFormatter("%(levelname)s %(name)s: %(message)s"))
    log = logging.getLogger("chordgen")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG if verbose else logging.WARNING)
    return handler


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _cause_chain(exc: BaseException):
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def _find_fetch_error(exc: BaseException) -> BaseException | None:
    fetch = sys.modules.get("chordgen.fetch")
    cls = getattr(fetch, "FetchError", None) if fetch is not None else None
    for e in _cause_chain(exc):
        if cls is not None and isinstance(e, cls):
            return e
        if type(e).__name__ == "FetchError" and hasattr(e, "user_message"):
            return e
    return None


def _validation_errors(exc: BaseException) -> list[str] | None:
    """JSON Schema の検証の失敗なら、通らなかった項目の一覧を返す。"""
    try:
        from jsonschema.exceptions import ValidationError
    except ImportError:
        ValidationError = None  # type: ignore[assignment]
    for e in _cause_chain(exc):
        errors = getattr(e, "errors", None)
        if isinstance(errors, (list, tuple)) and errors:
            return [_describe_one(x) for x in errors]
        if ValidationError is not None and isinstance(e, ValidationError):
            return [_describe_one(e)]
        if type(e).__name__.endswith("ValidationError"):
            return [str(e)]
    return None


def _describe_one(err: Any) -> str:
    path = getattr(err, "json_path", None)
    msg = getattr(err, "message", None)
    if path is not None and msg is not None:
        return f"{path}: {msg}"
    return str(err)


def _read_lyrics(path: Path) -> str:
    # utf-8-sig は BOM があれば外し、なければ普通の UTF-8 として読む
    return path.read_text(encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    _setup_stdio()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # --help / --version は 0、引数の誤りは 2
        return e.code if isinstance(e.code, int) else EXIT_USAGE

    handler = _setup_logging(args.verbose)
    try:
        return _main(args, parser)
    except KeyboardInterrupt:
        _err("中断しました。")
        return EXIT_INTERRUPTED
    finally:
        logging.getLogger("chordgen").removeHandler(handler)


def _main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from chordgen import pipeline

    lyrics_text = None
    if args.lyrics is not None:
        try:
            lyrics_text = _read_lyrics(args.lyrics)
        except FileNotFoundError:
            _err(f"エラー: 歌詞のファイルが見つかりません: {args.lyrics}")
            return EXIT_USAGE
        except UnicodeDecodeError:
            _err(f"エラー: 歌詞のファイルを UTF-8 として読めません: {args.lyrics}")
            return EXIT_USAGE
        except OSError as e:
            _err(f"エラー: 歌詞のファイルを読めません: {args.lyrics}（{e}）")
            return EXIT_USAGE

    try:
        pipeline.classify_source(args.source)
    except FileNotFoundError as e:
        _err(f"エラー: {e}（URL なら http:// か https:// から書いてください）")
        return EXIT_USAGE

    options = PipelineOptions(
        device=args.device,
        chord_input=args.chord_input,
        bass_mode=args.bass,
        snap=not args.no_snap,
        chord_model=args.model,
        lyrics_language=args.language,
        keep_work=args.keep_work,
        bars_per_line=args.bars_per_line,
    )
    try:
        options.validate()
    except ValueError as e:
        _err(f"エラー: {e}")
        return EXIT_USAGE

    t0 = time.perf_counter()
    try:
        out = pipeline.run(
            args.source,
            args.output,
            lyrics_text=lyrics_text,
            title=args.title,
            artist=args.artist,
            options=options,
            on_progress=_ProgressPrinter(sys.stderr, args.verbose),
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return _report_failure(e, args.verbose)

    if args.verbose:
        _err(f"合計 {time.perf_counter() - t0:.1f} 秒")
    print(f"{out.resolve()} に書き出しました。cat4g の読み込みでこのファイルを開いてください。", flush=True)
    return EXIT_OK


def _report_failure(exc: BaseException, verbose: bool) -> int:
    fe = _find_fetch_error(exc)
    if fe is not None:
        code = getattr(fe, "code", None)
        msg = getattr(fe, "user_message", None) or str(fe)
        _err(f"エラー: 音源を取得できませんでした: {msg}" + (f"（{code}）" if code else ""))
        if code == "ENGINE_OUTDATED" and "update" not in msg:
            _err("  media-dl update で yt-dlp を更新してから、もう一度試してください。")
        if verbose:
            traceback.print_exception(exc, file=sys.stderr)
        exit_code = getattr(fe, "exit_code", None)
        return exit_code if isinstance(exit_code, int) and exit_code != 0 else EXIT_FAIL

    problems = _validation_errors(exc)
    if problems is not None:
        _err("エラー: 書き出す文書が JSON Schema の検証を通らなかったので、書き出しませんでした。通らなかった項目:")
        for line in problems:
            _err(f"  {line}")
    elif isinstance(exc, StageError):
        _err(f"エラー: {exc.label}で失敗しました: {type(exc.cause).__name__}: {exc.cause}")
    else:
        _err(f"エラー: {type(exc).__name__}: {exc}")
    if verbose:
        traceback.print_exception(exc, file=sys.stderr)
    else:
        _err("  詳しくは -v をつけて実行してください。")
    return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
