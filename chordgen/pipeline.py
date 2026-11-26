"""音源から AnalysisResult を作り、.cat4g.json に書き出すまでをつなぐ。

各段のモジュールは関数の中で遅延 import する（torch などが入っていなくても
`import chordgen.pipeline` が動き、使わない段のライブラリが無くても解析できるように）。
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import logging
import re
import shutil
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_args

from chordgen.types import (
    AlignedLine,
    AnalysisResult,
    BassMode,
    ChordInputMode,
    TrackMeta,
)

logger = logging.getLogger(__name__)

# ---- 既定値（ここだけで決める）----------------------------------------------
# 効果がまだ確かめられていない処理（分離した音をコード認識に使う、ベースで補正する）は切り、
# 拍にそろえる処理だけ入れる。要件「進め方 4」の比較の結果で、ここを書き換えて既定を変える。
DEFAULT_CHORD_INPUT: ChordInputMode = "original"
DEFAULT_BASS_MODE: BassMode = "off"
DEFAULT_SNAP = True
DEFAULT_CHORD_MODEL = "chordmini"
DEFAULT_BARS_PER_LINE = 4

CHORD_INPUT_MODES: tuple[str, ...] = get_args(ChordInputMode)
BASS_MODES: tuple[str, ...] = get_args(BassMode)
CHORD_MODELS = ("chordmini", "btc")
MAX_BARS_PER_LINE = 64  # 仕様書 14 章の measures の上限

# 段の名前と表示名（進捗の表示に使う）
STAGE_LABELS = {
    "fetch": "音源の取得",
    "probe": "音源の読み込み",
    "separate": "音源の分離",
    "beats": "ビート検出",
    "chords": "コード認識",
    "key": "キー推定",
    "snap": "拍にそろえる",
    "bass": "ベースで補正",
    "lyrics": "歌詞の位置合わせ",
    "write": "書き出し",
}


@dataclass
class PipelineOptions:
    """解析の条件。既定値は上の DEFAULT_* にまとめてある。"""

    device: str | None = None  # None は自動（GPU があれば cuda）
    chord_input: ChordInputMode = DEFAULT_CHORD_INPUT
    bass_mode: BassMode = DEFAULT_BASS_MODE
    snap: bool = DEFAULT_SNAP
    chord_model: str = DEFAULT_CHORD_MODEL
    # None は「必要なときだけ」（chord_input が original 以外、bass_mode が off 以外、歌詞があるとき）
    separate: bool | None = None
    lyrics_language: str | None = None
    work_dir: Path | None = None  # 作業フォルダの親。None は paths.work_dir()
    keep_work: bool = False  # False なら解析の後に作業フォルダを消す
    bars_per_line: int = DEFAULT_BARS_PER_LINE

    def validate(self) -> None:
        """組み合わせの誤りを解析を始める前に弾く。"""
        if self.chord_input not in CHORD_INPUT_MODES:
            raise ValueError(f"chord_input が不正です: {self.chord_input!r}")
        if self.bass_mode not in BASS_MODES:
            raise ValueError(f"bass_mode が不正です: {self.bass_mode!r}")
        if not 1 <= int(self.bars_per_line) <= MAX_BARS_PER_LINE:
            raise ValueError(f"bars_per_line は 1〜{MAX_BARS_PER_LINE} にしてください: {self.bars_per_line}")
        if self.separate is False and self.chord_input != "original":
            raise ValueError(f"chord_input={self.chord_input!r} には音源の分離が要ります（separate=False と両立しません）")
        if self.separate is False and self.bass_mode != "off":
            raise ValueError(f"bass_mode={self.bass_mode!r} には音源の分離が要ります（separate=False と両立しません）")

    def needs_separation(self, *, has_lyrics: bool) -> bool:
        if self.separate is not None:
            return self.separate
        return self.chord_input != "original" or self.bass_mode != "off" or has_lyrics


@dataclass
class ProgressEvent:
    """on_progress に渡す進捗。

    status: "start"（段の前）/ "done"（段の後）/ "skipped"（任意の段が失敗して飛ばした）/
            "failed"（必須の段が失敗した）/ "progress"（段の途中。URL の取得の progress など）
    """

    stage: str
    label: str
    index: int  # 何段目か（1 始まり）
    total: int
    status: str
    elapsed: float | None = None
    message: str | None = None
    data: dict[str, Any] | None = None


ProgressCallback = Callable[[ProgressEvent], None]


class StageError(RuntimeError):
    """必須の段が失敗した。元の例外は __cause__（と cause）に入る。"""

    def __init__(self, stage: str, cause: BaseException):
        self.stage = stage
        self.label = STAGE_LABELS.get(stage, stage)
        self.cause = cause
        super().__init__(f"{self.label}で失敗しました: {cause}")


class _CollectHandler(logging.Handler):
    """段の実行中に chordgen.* のログに出た警告の文を集める。"""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.add(record.getMessage())
        except Exception:
            pass


class _StageRunner:
    """段の実行・時間の計測・進捗の通知・GPU のメモリの解放をまとめる。"""

    def __init__(self, stages: list[str], on_progress: ProgressCallback | None):
        self.stages = stages
        self.on_progress = on_progress
        self.seconds: dict[str, float] = {}
        self.warnings: list[str] = []
        self.skipped: list[str] = []

    def emit(self, stage: str, status: str, **kw: Any) -> None:
        if self.on_progress is None:
            return
        index = self.stages.index(stage) + 1 if stage in self.stages else 0
        event = ProgressEvent(stage, STAGE_LABELS.get(stage, stage), index, len(self.stages), status, **kw)
        try:
            self.on_progress(event)
        except Exception:  # 表示の不具合で解析を止めない
            logger.debug("on_progress が失敗しました", exc_info=True)

    def skip(self, stage: str, reason: str) -> None:
        """段を実行せずに飛ばす（前の段が飛んで入力が無いときなど）。"""
        msg = f"{STAGE_LABELS.get(stage, stage)}を飛ばしました: {reason}"
        logger.warning(msg)
        self.warnings.append(msg)
        self.skipped.append(stage)
        self.emit(stage, "skipped", message=reason)

    def _call(self, stage: str, fn: Callable[[], Any]) -> Any:
        """fn を呼び、段の中で出た warnings.warn を警告の一覧に足す。

        段のモジュールがログにも同じ文を出していれば二重に表示しないよう、ログに出なかった
        ものだけをこのモジュールのログで警告する。
        """
        collect = _CollectHandler()
        root = logging.getLogger("chordgen")
        root.addHandler(collect)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return fn()
        finally:
            root.removeHandler(collect)
            label = STAGE_LABELS.get(stage, stage)
            for w in caught:
                # ライブラリの後始末の漏れ（unclosed file など）は利用者に関係がないので出さない
                if issubclass(w.category, ResourceWarning):
                    continue
                text = str(w.message)
                if text not in collect.messages:
                    logger.warning("%s: %s", label, text)
                self.warnings.append(f"{label}: {text}")

    def run(self, stage: str, fn: Callable[[], Any], *, optional: bool = False, default: Any = None) -> Any:
        self.emit(stage, "start")
        t0 = time.perf_counter()
        try:
            value = self._call(stage, fn)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.seconds[stage] = round(elapsed, 3)
            if not optional:
                self.emit(stage, "failed", elapsed=elapsed, message=str(e))
                raise StageError(stage, e) from e
            msg = f"{STAGE_LABELS.get(stage, stage)}に失敗したので飛ばしました: {type(e).__name__}: {e}"
            logger.warning(msg)
            logger.debug("%s の詳細", stage, exc_info=True)
            self.warnings.append(msg)
            self.skipped.append(stage)
            self.emit(stage, "skipped", elapsed=elapsed, message=f"{type(e).__name__}: {e}")
            return default
        finally:
            _release_gpu()
        elapsed = time.perf_counter() - t0
        self.seconds[stage] = round(elapsed, 3)
        self.emit(stage, "done", elapsed=elapsed)
        return value


def _mod(name: str):
    """chordgen.<name> を遅延 import する（sys.modules にあればそれを使う）。"""
    return importlib.import_module(f"chordgen.{name}")


def _release_gpu() -> None:
    """段の間で GPU のメモリを空ける。torch が import 済みのときだけ触る。"""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("torch.cuda.empty_cache に失敗しました", exc_info=True)


def _resolve_device(prefer: str | None) -> str:
    from chordgen.device import get_device

    try:
        return get_device(prefer)
    except ImportError:  # torch が無い環境。torch を使う段はそれぞれの段で失敗する
        return "cpu"


_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def classify_source(source: str | Path) -> tuple[str, str | Path]:
    """("file", Path) か ("url", str) を返す。どちらでもなければ FileNotFoundError。

    ファイルとして在るものは fetch を import せずにファイルとして扱う。
    """
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"音源ファイルが見つかりません: {source}")
        return "file", source
    s = str(source).strip()
    p = Path(s)
    if p.is_file():
        return "file", p
    try:
        from chordgen.fetch import is_url
    except ImportError:  # fetch がまだ無い・壊れているときの控え
        url = bool(_URL_RE.match(s))
    else:
        url = is_url(s)
    if url:
        return "url", s
    raise FileNotFoundError(f"音源ファイルが見つかりません: {s}")


_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str | None, *, max_len: int = 100, fallback: str = "untitled") -> str:
    """Windows でもファイル名に使える文字列にする（使えない文字は "_"）。"""
    s = unicodedata.normalize("NFC", name or "")
    s = _FORBIDDEN_CHARS.sub("_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s[:max_len].rstrip(" .")  # Windows は末尾の空白・点を許さない
    if not s or set(s) <= {"_", "."}:
        s = fallback
    if s.split(".")[0].upper() in _WINDOWS_RESERVED:
        s = "_" + s
    return s


def _job_dir_name(display_name: str | None, key: str) -> str:
    """作業フォルダの名前。同じ名前の別の音源とぶつからないよう、入力の短いハッシュを足す。"""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{safe_filename(display_name, max_len=60, fallback='audio')}-{digest}"


def _plain_lines(lyrics_text: str) -> list[AlignedLine]:
    """位置合わせに失敗したときの歌詞（時刻なし）。歌詞が消えないよう、文字だけは残す。"""
    lines = [ln.rstrip() for ln in lyrics_text.splitlines()]
    return [AlignedLine(text=ln, char_times=[None] * len(ln)) for ln in lines if ln.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def plan_stages(options: PipelineOptions, *, is_url: bool, has_lyrics: bool) -> list[str]:
    """実行する段の名前を順に返す。"""
    stages = ["fetch"] if is_url else []
    stages.append("probe")
    if options.needs_separation(has_lyrics=has_lyrics):
        stages.append("separate")
    stages += ["beats", "chords", "key"]
    if options.snap:
        stages.append("snap")
    if options.bass_mode != "off":
        stages.append("bass")
    if has_lyrics:
        stages.append("lyrics")
    return stages


def analyze(
    source: str | Path,
    *,
    lyrics_text: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    options: PipelineOptions | None = None,
    on_progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """音源（ファイルか URL）を解析する。要件「処理の流れ」の 1〜7。

    必須の段（音源の取得・読み込み、ビート、コード認識）が失敗したら StageError。
    任意の段（キー推定、ベース、歌詞の位置合わせ）が失敗したら警告してその段だけ飛ばす。
    """
    return _analyze(source, lyrics_text=lyrics_text, title=title, artist=artist,
                    options=options, on_progress=on_progress, extra_stages=())[0]


def _analyze(
    source: str | Path,
    *,
    lyrics_text: str | None,
    title: str | None,
    artist: str | None,
    options: PipelineOptions | None,
    on_progress: ProgressCallback | None,
    extra_stages: tuple[str, ...],
) -> tuple[AnalysisResult, _StageRunner]:
    from chordgen import APP_NAME, __version__

    opts = options or PipelineOptions()
    lyrics = lyrics_text if lyrics_text and lyrics_text.strip() else None
    has_lyrics = lyrics is not None
    opts.validate()

    kind, src = classify_source(source)
    is_url = kind == "url"
    do_separate = opts.needs_separation(has_lyrics=has_lyrics)
    runner = _StageRunner(plan_stages(opts, is_url=is_url, has_lyrics=has_lyrics) + list(extra_stages), on_progress)
    device = _resolve_device(opts.device)

    base_work = Path(opts.work_dir) if opts.work_dir else _mod("paths").work_dir()
    job_dir: Path | None = None
    stage_err: BaseException | None = None
    try:
        # 1. 音源の取得と読み込み、題名の確定
        fetched: TrackMeta | None = None
        if is_url:
            url = str(src)

            def do_fetch():
                fetch = _mod("fetch")
                info = fetch.fetch_info(url)
                nonlocal job_dir
                job_dir = base_work / _job_dir_name(info.title, url)
                res = fetch.fetch_audio(
                    url, job_dir / "download",
                    on_progress=lambda d: runner.emit("fetch", "progress", data=d),
                )
                # 取得前の情報を優先し、足りないところを取得後の情報で埋める
                m = res.meta or TrackMeta()
                meta = TrackMeta(
                    title=info.title or m.title,
                    artist=info.artist or m.artist,
                    source_url=info.source_url or m.source_url or url,
                )
                return res.audio_path, meta

            audio_path, fetched = runner.run("fetch", do_fetch)
        else:
            audio_path = Path(src)
            job_dir = base_work / _job_dir_name(audio_path.stem, str(audio_path.resolve()))

        assert job_dir is not None

        def do_probe():
            audio_io = _mod("audio_io")
            info = audio_io.probe_audio(audio_path)
            user = TrackMeta(title=title or None, artist=artist or None, source_url=None)
            return info, audio_io.resolve_meta(user=user, fetched=fetched, audio=info)

        audio, meta = runner.run("probe", do_probe)

        # 2. 音源の分離と、コード認識に入れる音
        stems = None
        chord_path: Path = audio.path
        if do_separate:
            # コード認識に分離した音を使うときは必須（黙って原音に戻すと比較の条件が変わる）。
            # ベース・歌詞のためだけなら任意（失敗したらベースを飛ばし、歌詞は原音で合わせる）。
            needed = opts.chord_input != "original"

            def do_separate_():
                separation = _mod("separation")
                st = separation.separate(audio.path, job_dir / "stems", device)
                cp = audio.path
                if opts.chord_input != "original":
                    cp = separation.make_chord_input(audio.path, st, opts.chord_input, job_dir / "chord_input")
                return st, cp

            out = runner.run("separate", do_separate_, optional=not needed, default=None)
            if out is not None:
                stems, chord_path = out

        # 3. ビート（原音で。drums を使うかは決めていない）
        beats = runner.run("beats", lambda: _mod("beats").detect_beats(audio.path, device))

        # 4. コード認識とキー推定
        chords = runner.run(
            "chords", lambda: _mod("chords").recognize_chords(chord_path, device, model=opts.chord_model)
        )
        # キーはコード認識と同じ音で推定する（比べる条件をコード認識とそろえる。ドラムを除いた音のほうが和音を拾いやすい）
        key = runner.run("key", lambda: _mod("key").estimate_key(chord_path, chords), optional=True)

        # 5・6. 拍にそろえてからベースで補正する（要件の 5→6 とは逆）。
        # ベースは拍ごとに出るので、先にコードの境目を拍にそろえておくと、各コードの区間が
        # 拍の区間の和になり、ベースの音とコードの区間の対応がずれない。逆の順だと、
        # ベースで分けた分数コード（C → C/E など）や直したルートを、後の拍そろえが
        # 動かしたりまとめたりしてしまう。
        if opts.snap:
            chords = runner.run("snap", lambda: _mod("postprocess").snap_to_beats(chords, beats))
        if opts.bass_mode != "off":
            if stems is None or "bass" not in stems.paths:
                runner.skip("bass", "分離したベースの音がありません")
            else:
                def do_bass():
                    notes = _mod("bass").detect_bass(stems.paths["bass"], beats)
                    return _mod("postprocess").apply_bass(chords, notes, opts.bass_mode)

                chords = runner.run("bass", do_bass, optional=True, default=chords)

        # 7. 歌詞の位置合わせ（分離してあれば vocals で）
        aligned: list[AlignedLine] | None = None
        if has_lyrics:
            vocal_path = stems.paths.get("vocals", audio.path) if stems is not None else audio.path
            aligned = runner.run(
                "lyrics",
                lambda: _mod("lyrics").align_lyrics(vocal_path, lyrics, device, language=opts.lyrics_language),
                optional=True,
                default=None,
            )
            if aligned is None:  # 失敗しても歌詞の文字は残す（時刻なし）
                aligned = _plain_lines(lyrics)

        extras: dict[str, Any] = {
            "analyzer.name": APP_NAME,
            "analyzer.version": __version__,
            "analyzer.chordModel": opts.chord_model,
            "analyzer.device": device,
            "analyzer.options": {
                "chordInput": opts.chord_input,
                "bassMode": opts.bass_mode,
                "snap": opts.snap,
                "separate": do_separate,
                "lyricsLanguage": opts.lyrics_language,
                "barsPerLine": opts.bars_per_line,
            },
            "analyzer.stageSeconds": dict(runner.seconds),
        }
        if is_url:
            extras["analyzer.sourceUrl"] = meta.source_url or str(src)
        if runner.skipped:
            extras["analyzer.skippedStages"] = list(runner.skipped)
        if runner.warnings:
            extras["analyzer.warnings"] = list(runner.warnings)

        result = AnalysisResult(
            audio=audio, meta=meta, beats=beats, key=key, chords=list(chords), lyrics=aligned,
            extras=_jsonable(extras),
        )
        return result, runner
    except BaseException as e:
        stage_err = e
        raise
    finally:
        if job_dir is not None and not opts.keep_work:
            _remove_job_dir(job_dir, base_work)
        elif job_dir is not None and job_dir.exists() and stage_err is None:
            logger.info("作業フォルダを残しました: %s", job_dir)


def _remove_job_dir(job_dir: Path, base: Path) -> None:
    """作業フォルダを消す。作業フォルダの親の下にあるものだけ。"""
    try:
        if job_dir.exists() and job_dir.resolve().is_relative_to(base.resolve()) and job_dir.resolve() != base.resolve():
            shutil.rmtree(job_dir, ignore_errors=True)
    except OSError:
        logger.debug("作業フォルダを消せませんでした: %s", job_dir, exc_info=True)


def default_output_path(title: str | None, out_dir: Path | None = None) -> Path:
    """output_dir()/<題名>.cat4g.json。同名があれば上書きせず「 (2)」などの連番をつける。

    解析の条件を変えて何度も出すとき（進め方 4）に前の結果を消さないため。
    -o で明示したときは上書きする。
    """
    base_dir = out_dir if out_dir is not None else _mod("paths").output_dir()
    stem = safe_filename(title)
    path = base_dir / f"{stem}.cat4g.json"
    n = 2
    while path.exists():
        path = base_dir / f"{stem} ({n}).cat4g.json"
        n += 1
    return path


def run(source: str | Path, output_path: str | Path | None = None, **kwargs: Any) -> Path:
    """analyze → songformat.build_document → write_document。書き出したパスを返す。

    output_path が None なら default_output_path、既存のフォルダならその中に既定の名前で置く。
    kwargs は analyze と同じ（lyrics_text, title, artist, options, on_progress）。
    """
    options: PipelineOptions = kwargs.pop("options", None) or PipelineOptions()
    result, runner = _analyze(source, options=options, extra_stages=("write",),
                              lyrics_text=kwargs.pop("lyrics_text", None),
                              title=kwargs.pop("title", None), artist=kwargs.pop("artist", None),
                              on_progress=kwargs.pop("on_progress", None))
    if kwargs:
        raise TypeError(f"知らない引数です: {', '.join(kwargs)}")

    if output_path is None:
        path = default_output_path(result.meta.title or result.audio.path.stem)
    else:
        path = Path(output_path)
        if path.is_dir():
            path = default_output_path(result.meta.title or result.audio.path.stem, path)

    def do_write():
        songformat = _mod("songformat")
        doc = songformat.build_document(result, bars_per_line=options.bars_per_line)
        path.parent.mkdir(parents=True, exist_ok=True)
        songformat.write_document(doc, path)
        return path

    return runner.run("write", do_write)


__all__ = [
    "PipelineOptions",
    "ProgressEvent",
    "StageError",
    "analyze",
    "run",
    "classify_source",
    "default_output_path",
    "plan_stages",
    "safe_filename",
    "STAGE_LABELS",
]
