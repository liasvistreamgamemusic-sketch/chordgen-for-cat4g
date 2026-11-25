"""歌詞の位置合わせ（処理の流れの 7）。

利用者が渡した歌詞を音源に強制的に合わせ（forced alignment）、行ごとに各文字の時刻を返す。
Whisper は純粋な PyTorch 実装の openai-whisper を使い（ROCm 版の torch でも動く）、
位置合わせは stable-ts の ``model.align`` で行う。align で時刻が取れた文字が少ないときは、
Whisper で書き起こして歌詞と文字単位で DP マッチングする方法も試し、時刻の取れた文字が
多いほうを採る。faster-whisper（CTranslate2）は CUDA 専用なので使わない。

torch・whisper・stable_whisper は align_lyrics() の中で読み込む。モデルに依存しない部分
（split_lyrics_lines、guess_language、build_aligned_lines）は import だけで使える。
"""

from __future__ import annotations

import bisect
import difflib
import logging
import os
import sys
import types
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chordgen.types import AlignedLine

log = logging.getLogger(__name__)

# モデルの大きさ。引数 → 環境変数 → 既定の順に決める。
# 既定は large-v3。turbo（large-v3-turbo）はデコーダが 4 層しかなく、単語の時刻の精度が
# large-v3 より落ちるという報告がある。align はデコードを繰り返さないので large-v3 でも重くなく、
# align は fp32 で動くが、重み（約 6GB）と作業領域を足しても VRAM 16GB に収まる
WHISPER_MODEL_ENV = "CHORDGEN_WHISPER_MODEL"
DEFAULT_WHISPER_MODEL = "large-v3"

SAMPLE_RATE = 16000  # Whisper の入力

# align で時刻の取れた文字の割合がこれより低いと、書き起こし + DP マッチングも試す
FALLBACK_BELOW = 0.5
# 時刻の取れた文字の割合がこれより低い行は、合わせ損ねとみなして全部 None にする。
# 書き起こしとの照合で偶然一致した数文字だけに時刻が付くのと、歌われない行（注記など）に
# align が詰め込んだ時刻を防ぐ。歌われた行は align ならほぼ全文字、書き起こしでも漢字と
# かなの違いを除いて半分以上の文字が一致する
MIN_LINE_RATIO = 0.5
MIN_LINE_RATIO_CHARS = 4  # この文字数より短い行には上の判定を使わない
# 長さがこれ未満の単語は時刻が取れなかったものとみなす（stable-ts は失敗した単語を長さ 0 で返す）
MIN_WORD_DUR = 1e-3
# LCS の表の大きさの上限（int32 で約 100MB）。超えたら difflib で近似する
_LCS_MAX_CELLS = 25_000_000


@dataclass(frozen=True)
class TimedWord:
    """Whisper が返した単語（日本語ではトークン）1 つと、その開始・終了の時刻（秒）。"""

    text: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# モデルに依存しない部分
# ---------------------------------------------------------------------------


def split_lyrics_lines(lyrics_text: str) -> list[str]:
    """歌詞を行に分ける。行頭末の空白だけ除き、空行（連の区切り）は捨てる。文字は変えない。"""
    text = lyrics_text.removeprefix("\ufeff")  # BOM
    return [s for line in text.splitlines() if (s := line.strip())]


def guess_language(lines: Sequence[str]) -> str | None:
    """歌詞の文字から言語を推定する。かな・漢字があれば "ja"、ハングルがあれば "ko"、なければ None。

    None のときは Whisper の自動判定に任せる。漢字だけの中国語の歌詞は "ja" になるので、
    その場合は呼び出し側が language="zh" を渡す。
    """
    has_hangul = has_han = False
    for line in lines:
        for ch in line:
            cp = ord(ch)
            if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F:
                return "ja"  # ひらがな・カタカナ（半角を含む）
            if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
                has_hangul = True
            elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
                has_han = True
    if has_hangul:
        return "ko"
    if has_han:
        return "ja"
    return None


def _is_timed(ch: str) -> bool:
    """時刻を付ける文字か（文字・数字・結合記号）。空白・記号・絵文字は付けない。"""
    return unicodedata.category(ch)[0] in "LNM"


def _key(ch: str) -> str:
    """照合用の正規化（NFKC、大文字小文字、カタカナ → ひらがな）。"""
    s = unicodedata.normalize("NFKC", ch).casefold()
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _word_chars(words: Sequence[TimedWord]) -> tuple[list[str], list[float]]:
    """単語の列を文字の列にし、各文字の時刻を単語の中で線形補間する。"""
    keys: list[str] = []
    times: list[float] = []
    for w in words:
        if not (w.end - w.start >= MIN_WORD_DUR):  # 長さ 0・NaN は時刻なし
            continue
        chars = [c for c in w.text if _is_timed(c)]
        n = len(chars)
        for k, c in enumerate(chars):
            keys.append(_key(c))
            times.append(w.start + (w.end - w.start) * k / n)
    return keys, times


def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> list[tuple[int, int]]:
    """2 つの列の最長共通部分列で対応する添字の組（両方とも昇順）。"""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    if list(a) == list(b):
        return [(i, i) for i in range(n)]
    if n * m > _LCS_MAX_CELLS:
        sm = difflib.SequenceMatcher(None, list(a), list(b), autojunk=False)
        return [(blk.a + k, blk.b + k) for blk in sm.get_matching_blocks() for k in range(blk.size)]

    vocab: dict[str, int] = {}
    ai = np.array([vocab.setdefault(k, len(vocab)) for k in a], dtype=np.int32)
    bi = np.array([vocab.setdefault(k, len(vocab)) for k in b], dtype=np.int32)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        # dp[i][j] = max(dp[i-1][j], dp[i-1][j-1] + 一致, dp[i][j-1]) は j についての累積最大で書ける
        t = np.maximum(dp[i - 1, 1:], dp[i - 1, :-1] + (bi == ai[i - 1]))
        dp[i, 1:] = np.maximum.accumulate(t)

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if ai[i - 1] == bi[j - 1] and dp[i, j] == dp[i - 1, j - 1] + 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _keep_monotonic(times: list[float | None]) -> list[float | None]:
    """時刻の最長の非減少部分列だけを残し、ほかを None にする（前後の逆転を直す）。"""
    idx = [i for i, t in enumerate(times) if t is not None]
    if not idx:
        return list(times)
    tails: list[float] = []  # 長さ k+1 の部分列の末尾の最小値
    tail_pos: list[int] = []
    prev: dict[int, int | None] = {}
    for i in idx:
        t = times[i]
        assert t is not None
        k = bisect.bisect_right(tails, t)
        prev[i] = tail_pos[k - 1] if k > 0 else None
        if k == len(tails):
            tails.append(t)
            tail_pos.append(i)
        else:
            tails[k] = t
            tail_pos[k] = i
    keep: set[int] = set()
    cur: int | None = tail_pos[-1]
    while cur is not None:
        keep.add(cur)
        cur = prev[cur]
    return [t if i in keep else None for i, t in enumerate(times)]


def _fill_inner_gaps(times: list[float | None]) -> list[float | None]:
    """時刻のある文字にはさまれた None を線形補間する。両端の None はそのまま。"""
    out = list(times)
    anchors = [i for i, t in enumerate(out) if t is not None]
    for a, b in zip(anchors, anchors[1:]):
        ta, tb = out[a], out[b]
        assert ta is not None and tb is not None
        for k in range(a + 1, b):
            out[k] = ta + (tb - ta) * (k - a) / (b - a)
    return out


def _drop_failed_lines(times: list[float | None], by_line: dict[int, list[int]]) -> list[float | None]:
    """時刻の取れた文字の割合が MIN_LINE_RATIO より低い行を全部 None にする。"""
    out = list(times)
    for ks in by_line.values():
        n_timed = sum(out[k] is not None for k in ks)
        if len(ks) >= MIN_LINE_RATIO_CHARS and n_timed < MIN_LINE_RATIO * len(ks):
            for k in ks:
                out[k] = None
    return out


def build_aligned_lines(lines: Sequence[str], words: Sequence[TimedWord]) -> list[AlignedLine]:
    """単語の時刻の列と歌詞の行から AlignedLine を作る（モデルに依存しない）。

    lines は split_lyrics_lines() の結果（各行がそのまま AlignedLine.text になる）。
    words は音源の時刻順に並んだ単語で、書き起こしでも align の結果でもよい。歌詞と単語を
    文字単位で照合し（NFKC・大文字小文字・カタカナとひらがなの違いは同一視）、一致した文字に
    単語の中で線形補間した時刻を写す。そのあと行をまたいで時刻が単調に増えるよう外れた時刻を
    捨て、行の中ではさまれた文字を補間する。空白・記号・絵文字と、時刻の取れなかった行は None。
    """
    # 歌詞の側の、時刻を付ける文字（行の番号、行の中の位置）
    pos: list[tuple[int, int]] = []
    lyr_keys: list[str] = []
    for li, text in enumerate(lines):
        for ci, ch in enumerate(text):
            if _is_timed(ch):
                pos.append((li, ci))
                lyr_keys.append(_key(ch))

    hyp_keys, hyp_times = _word_chars(words)
    times: list[float | None] = [None] * len(pos)
    for i, j in _lcs_pairs(lyr_keys, hyp_keys):
        times[i] = hyp_times[j]
    by_line: dict[int, list[int]] = {}
    for k, (li, _) in enumerate(pos):
        by_line.setdefault(li, []).append(k)

    # 合わせ損ねた行を先に除いてから単調性を直し（外れた行の時刻に正しい行が引きずられない）、
    # もう一度合わせ損ねを判定して、行の中ではさまれた文字を補間する
    times = _drop_failed_lines(times, by_line)
    times = _keep_monotonic(times)
    times = _drop_failed_lines(times, by_line)
    for ks in by_line.values():
        for k, t in zip(ks, _fill_inner_gaps([times[k] for k in ks])):
            times[k] = t

    char_times: list[list[float | None]] = [[None] * len(text) for text in lines]
    for (li, ci), t in zip(pos, times):
        if t is not None:
            char_times[li][ci] = round(float(t), 3)
    return [AlignedLine(text=text, char_times=ct) for text, ct in zip(lines, char_times)]


def _timed_ratio(lines: Sequence[AlignedLine]) -> float:
    """時刻を付ける文字のうち、時刻が取れた割合。"""
    total = got = 0
    for line in lines:
        for ch, t in zip(line.text, line.char_times):
            if _is_timed(ch):
                total += 1
                got += t is not None
    return got / total if total else 0.0


# ---------------------------------------------------------------------------
# モデルを使う部分
# ---------------------------------------------------------------------------


def resolve_model_name(model_name: str | None = None) -> str:
    """使う Whisper のモデル名（"tiny"・"base"・"large-v3"・"turbo" など、または .pt のパス）。"""
    return model_name or os.environ.get(WHISPER_MODEL_ENV) or DEFAULT_WHISPER_MODEL


def _load_audio_16k(audio_path: Path) -> np.ndarray:
    """16kHz の float32 モノラルで読む（Whisper の入力。ffmpeg の CLI は使わない）。"""
    try:
        from chordgen.audio_io import load_mono
    except ImportError:
        load_mono = None
    if load_mono is not None:
        return np.ascontiguousarray(load_mono(Path(audio_path), SAMPLE_RATE), dtype=np.float32)

    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF

        mono = AF.resample(torch.from_numpy(mono), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(mono, dtype=np.float32)


def _use_cpu_timing_kernels() -> None:
    """Whisper の DTW と median filter を CPU 版に固定する。

    GPU のときの Triton のカーネルは Windows・ROCm では入らないか動かないことがある。
    whisper は RuntimeError なら CPU 版に切り替えるので、Triton の代わりに RuntimeError を
    投げる部品を差し込む（行列は小さく、CPU 版でも遅くならない）。
    """
    import whisper.timing

    def _unavailable(*args, **kwargs):
        raise RuntimeError("chordgen: Triton のカーネルは使わない")

    whisper.timing.dtw_cuda = _unavailable
    stub = types.ModuleType("whisper.triton_ops")
    stub.median_filter_cuda = _unavailable  # type: ignore[attr-defined]
    stub.dtw_kernel = _unavailable  # type: ignore[attr-defined]
    sys.modules["whisper.triton_ops"] = stub


# 歌詞にかな・漢字・ハングルがないときに、音からの判定で選ばない言語
_CJK_LANGS = frozenset({"ja", "zh", "yue", "ko"})


def _detect_language(model, audio: np.ndarray) -> str | None:
    """音の大きい 30 秒の区間（最大 3 つ）で Whisper に言語を判定させる。

    guess_language() が None のとき（歌詞にかな・漢字・ハングルがない）に使うので、
    文字の合わない ja・zh・ko は候補から外す（日本人の歌う英語の歌詞が ja と判定されるのを防ぐ）。
    """
    import torch
    import whisper

    win = 30 * SAMPLE_RATE
    starts = list(range(0, max(len(audio) - win, 0) + 1, win // 2))
    rms = [float(np.sqrt(np.mean(audio[s : s + win] ** 2) + 1e-12)) for s in starts]
    chosen = [starts[i] for i in np.argsort(rms)[::-1][:3]]
    total: dict[str, float] = {}
    for s in chosen:
        seg = whisper.pad_or_trim(torch.from_numpy(audio[s : s + win]))
        mel = whisper.log_mel_spectrogram(seg, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        for lang, p in probs.items():
            total[lang] = total.get(lang, 0.0) + float(p)
    cands = {k: v for k, v in total.items() if k not in _CJK_LANGS}
    return max(cands, key=cands.__getitem__) if cands else None


def _words_of(result) -> list[TimedWord]:
    """stable-ts の WhisperResult から単語の列を取り出す。"""
    if result is None:
        return []
    words: list[TimedWord] = []
    for seg in result.segments:
        for w in seg.words or []:
            if w.start is None or w.end is None:
                continue
            words.append(TimedWord(text=w.word, start=float(w.start), end=float(w.end)))
    return words


def align_lyrics(
    audio_path: Path,
    lyrics_text: str,
    device: str,
    language: str | None = None,
    *,
    model_name: str | None = None,
) -> list[AlignedLine]:
    """歌詞のテキストを音源に位置合わせし、行ごとに各コードポイントの時刻を返す。

    audio_path の音はそのまま使う。伴奏が大きいと合わせにくいので、Demucs の vocals を渡すか
    どうかは呼び出し側（パイプライン）が決める。lyrics_text は行に分かれた歌詞で、空行は連の
    区切りとして読み飛ばす（結果には含めない）。各行は行頭末の空白だけ除いて AlignedLine.text
    になり、char_times は len(text) と同じ長さ。空白・記号と、合わせられなかった行は None。

    language は Whisper の言語コード（"ja"・"en" など）。None なら歌詞の文字から推定し、
    推定できなければ音から Whisper に判定させる。model_name を省くと環境変数
    CHORDGEN_WHISPER_MODEL、それもなければ large-v3。重みは models_dir()/whisper に
    初回にダウンロードする。

    位置合わせ自体の失敗は例外にせず、時刻を None にして返す。ライブラリ・モデル・音源を
    読み込めないとき（環境の問題）は例外を投げる。
    """
    lines = split_lyrics_lines(lyrics_text)
    if not lines:
        return []
    untimed = [AlignedLine(text=t, char_times=[None] * len(t)) for t in lines]

    import stable_whisper
    import torch

    from chordgen.paths import models_dir

    audio = _load_audio_16k(Path(audio_path))
    if audio.size == 0:
        return untimed

    name = resolve_model_name(model_name)
    download_root = models_dir() / "whisper"
    download_root.mkdir(parents=True, exist_ok=True)
    _use_cpu_timing_kernels()
    log.info("Whisper %s を読み込む（%s）", name, device)
    model = stable_whisper.load_model(name, device=device, download_root=str(download_root))
    fp16 = device != "cpu"

    try:
        lang = language or guess_language(lines)
        if lang is None:
            try:
                lang = _detect_language(model, audio)
            except Exception:
                log.warning("言語の判定に失敗した。en として扱う", exc_info=True)
            lang = lang or "en"
        log.info("歌詞の言語: %s", lang)

        best = untimed
        best_ratio = 0.0
        try:
            # align は fp32 で動く（fp16 の指定は受け付けない）
            res = model.align(audio, "\n".join(lines), language=lang, verbose=None)
            best = build_aligned_lines(lines, _words_of(res))
            best_ratio = _timed_ratio(best)
            log.info("align で時刻の取れた文字: %.0f%%", best_ratio * 100)
        except Exception:
            log.warning("stable-ts の align に失敗した。書き起こしで合わせる", exc_info=True)

        if best_ratio < FALLBACK_BELOW:
            try:
                res = model.transcribe(
                    audio,
                    language=lang,
                    word_timestamps=True,
                    regroup=False,
                    condition_on_previous_text=False,
                    verbose=None,
                    fp16=fp16,
                )
                cand = build_aligned_lines(lines, _words_of(res))
                cand_ratio = _timed_ratio(cand)
                log.info("書き起こしで時刻の取れた文字: %.0f%%", cand_ratio * 100)
                if cand_ratio > best_ratio:
                    best, best_ratio = cand, cand_ratio
            except Exception:
                log.warning("書き起こしに失敗した", exc_info=True)
        return best
    finally:
        del model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
