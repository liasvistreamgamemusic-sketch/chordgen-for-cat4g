"""拍と小節の 1 拍目（ダウンビート）の検出、BPM と拍子の推定。

手法は Beat This!（CPJKU、ISMIR 2024、MIT ライセンス、PyPI の ``beat-this``）。拍と
ダウンビートを同時に出し、DBN（madmom）なしの最小の後処理で使える。重みは初回に
JKU のクラウドから ``models_dir()/beat_this/`` へダウンロードする。
beat_this が入っていない・失敗したときは librosa の beat_track にフォールバックし、
ダウンビートは 4 拍ごとと仮定する（警告を出す）。

モデルの出力は ``build_beat_grid``（純粋関数）で整える。
  1. 並べ替え・範囲外と近すぎる拍の除去
  2. 途中で抜けた拍を局所のテンポで補う
  3. 先頭と末尾をテンポで外挿する（音源の長さの範囲で）
  4. ダウンビートを拍の添字にし、拍子の分子を最頻の小節の拍数から決める
  5. ダウンビートの抜けと乱れを、拍子の分子に合わせて補う
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from chordgen.types import BeatGrid

log = logging.getLogger(__name__)

BEAT_THIS_CHECKPOINT = "final0"  # 論文の主モデル（約 78 MB）。軽くしたいときは "small0"
DEFAULT_BPM = 120.0  # 拍がほとんど取れないときに仮に使うテンポ
DEFAULT_BEATS_PER_BAR = 4
_BAR_CANDIDATES = (4, 3, 6, 2, 5, 7)  # 拍子の分子の候補。同数のときはこの順に優先する


# ---------------------------------------------------------------------------
# 音声の読み込み（torchaudio の backend に頼らない）
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> str | None:
    """PATH か media-dl の置き場所（MEDIA_DL_BIN_DIR）から ffmpeg を探す。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    bin_dir = os.environ.get("MEDIA_DL_BIN_DIR")
    if bin_dir:
        return shutil.which("ffmpeg", path=bin_dir)
    return None


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """音源を float32 モノラルで読む。サンプリング周波数は元のまま。

    soundfile で読めない形式（m4a など）は ffmpeg で一時 wav に直してから読む。
    """
    import soundfile as sf

    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile が対応していない形式
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(f"音源を読めません（ffmpeg も見つかりません）: {path}") from e
        with tempfile.TemporaryDirectory(prefix="chordgen_") as tmp:
            wav = Path(tmp) / "audio.wav"
            cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(path), "-vn", "-ac", "1", "-c:a", "pcm_f32le", str(wav)]
            proc = subprocess.run(cmd, shell=False, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg で音源を変換できません: {path}\n{proc.stderr.strip()}") from e
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# 整形（純粋関数）
# ---------------------------------------------------------------------------


def _median_ioi(beats: np.ndarray) -> float | None:
    if len(beats) < 2:
        return None
    return float(np.median(np.diff(beats)))


def clean_beats(beat_times: Sequence[float], duration: float | None = None,
                min_ratio: float = 0.35) -> list[float]:
    """昇順に並べ、範囲外・重複・近すぎる拍（中央値の間隔の min_ratio 未満）を除く。"""
    b = np.asarray([t for t in beat_times if np.isfinite(t)], dtype=float)
    b = np.unique(b)  # 昇順で完全な重複なし
    b = b[b >= 0.0]
    if duration is not None:
        b = b[b <= duration]
    ioi = _median_ioi(b)
    if ioi is None or ioi <= 0:
        return b.tolist()
    kept = [b[0]]
    for t in b[1:]:
        if t - kept[-1] >= min_ratio * ioi:
            kept.append(t)
    return [float(t) for t in kept]


def fill_beat_gaps(beat_times: Sequence[float], max_ratio: float = 1.5,
                   window: int = 8) -> list[float]:
    """拍の間隔が周りの間隔の max_ratio 倍を超えるところに、等間隔で拍を足す。"""
    b = np.asarray(beat_times, dtype=float)
    if len(b) < 3:
        return b.tolist()
    d = np.diff(b)
    global_ioi = float(np.median(d))
    normal = d <= max_ratio * global_ioi  # 抜けのない間隔
    out = [float(b[0])]
    for i, gap in enumerate(d):
        if gap > max_ratio * global_ioi:
            lo, hi = max(0, i - window), min(len(d), i + window + 1)
            near = d[lo:hi][normal[lo:hi]]
            ref = float(np.median(near)) if len(near) else global_ioi
            n = max(1, int(round(gap / ref)))
            out.extend(float(b[i] + gap * k / n) for k in range(1, n))
        out.append(float(b[i + 1]))
    return out


def extrapolate_beats(beat_times: Sequence[float], duration: float,
                      window: int = 8, tol: float = 1e-6) -> list[float]:
    """先頭と末尾の拍のない部分を、端の付近のテンポで 0 秒〜duration 秒の範囲まで埋める。"""
    b = [float(t) for t in beat_times]
    if len(b) < 2:
        return b
    head = float(np.median(np.diff(b[: window + 1])))
    tail = float(np.median(np.diff(b[-(window + 1):])))
    front: list[float] = []
    t = b[0] - head
    while head > 0 and t >= -tol:
        front.append(max(t, 0.0))
        t -= head
    back: list[float] = []
    t = b[-1] + tail
    while tail > 0 and t <= duration + tol:
        back.append(min(t, duration))
        t += tail
    return front[::-1] + b + back


def downbeats_to_indices(beat_times: Sequence[float], downbeat_times: Sequence[float],
                         max_ratio: float = 0.5) -> list[int]:
    """ダウンビートの時刻を、いちばん近い拍の添字にする（離れすぎたものは捨てる）。"""
    b = np.asarray(beat_times, dtype=float)
    if len(b) == 0:
        return []
    ioi = _median_ioi(b) or 60.0 / DEFAULT_BPM
    idx: set[int] = set()
    for t in downbeat_times:
        if not np.isfinite(t):
            continue
        j = int(np.argmin(np.abs(b - t)))
        if abs(b[j] - t) <= max_ratio * ioi:
            idx.add(j)
    return sorted(idx)


def infer_beats_per_bar(downbeat_indices: Sequence[int],
                        default: int = DEFAULT_BEATS_PER_BAR) -> int:
    """隣り合うダウンビートの間の拍数の最頻値を拍子の分子にする。

    8 拍以上の間隔はダウンビートの抜けとみなして数えない。同数のときは 4、3、6、2 の順。
    """
    d = sorted(set(int(i) for i in downbeat_indices))
    counts = Counter(b - a for a, b in zip(d, d[1:]) if b - a in _BAR_CANDIDATES)
    if not counts:
        return default
    best = max(counts.values())
    return next(n for n in _BAR_CANDIDATES if counts.get(n) == best)


def regularize_downbeats(downbeat_indices: Sequence[int], n_beats: int,
                         beats_per_bar: int, min_run: int = 2) -> list[int]:
    """ダウンビートの抜けを補い、乱れを直す。

    各ダウンビートの「位相」（添字 mod 拍子の分子）を見て、同じ位相が min_run 回以上
    続く区間を正しい区間とし、その位相で beats_per_bar 拍ごとにダウンビートを置く。
    続かない位相（1 つだけずれたもの）は誤検出として捨てる。位相が変わって続く
    ところ（2/4 の小節が挟まるなど）だけ、つなぎ目の小節の拍数を変えて残す。
    つなぎ目の余りが 1 拍なら前の小節に含める。最初のダウンビートより前は
    同じ位相で埋め、それより前の拍は弱起とする。
    """
    n = int(beats_per_bar)
    if n_beats <= 0:
        return []
    if n < 1:
        raise ValueError("beats_per_bar は 1 以上")
    d = sorted(set(int(i) for i in downbeat_indices if 0 <= i < n_beats))
    if not d:
        return list(range(0, n_beats, n))

    runs: list[tuple[int, list[int]]] = []
    for i in d:
        if runs and runs[-1][0] == i % n:
            runs[-1][1].append(i)
        else:
            runs.append((i % n, [i]))
    stable = [r for r in runs if len(r[1]) >= min_run]
    if not stable:
        counts = Counter(i % n for i in d)
        best = max(counts.values())
        phase = next(i % n for i in d if counts[i % n] == best)  # 同数なら先に出た位相
        stable = [(phase, [i for i in d if i % n == phase])]
    segments: list[tuple[int, list[int]]] = []
    for ph, idx in stable:
        if segments and segments[-1][0] == ph:
            segments[-1][1].extend(idx)
        else:
            segments.append((ph, list(idx)))

    first = segments[0][1][0]
    out = list(range(first % n, first, n))
    for k, (_, idx) in enumerate(segments):
        start = idx[0]
        end = segments[k + 1][1][0] if k + 1 < len(segments) else n_beats
        fill = list(range(start, end, n))
        if k + 1 < len(segments) and len(fill) > 1 and end - fill[-1] < 2:
            fill.pop()  # 1 拍だけの小節は作らない
        out.extend(fill)
    return sorted(set(out))


def build_beat_grid(beat_times: Sequence[float], downbeat_times: Sequence[float] | None,
                    duration: float, beats_per_bar: int | None = None) -> BeatGrid:
    """検出した拍とダウンビートの時刻から、後段で使いやすい BeatGrid を作る。

    downbeat_times が None なら 4 拍ごと（beats_per_bar を渡せばその拍数ごと）と仮定する。
    """
    duration = max(0.0, float(duration))
    beats = clean_beats(beat_times, duration)
    if len(beats) < 2:
        warnings.warn(f"拍がほとんど検出できませんでした。{DEFAULT_BPM:g} BPM と仮定します。",
                      stacklevel=2)
        start = beats[0] if beats else 0.0
        step = 60.0 / DEFAULT_BPM
        beats = [start, start + step] if start + step <= duration else [start]
        if len(beats) == 1:
            return BeatGrid(beat_times=beats, downbeat_indices=[0], bpm=DEFAULT_BPM,
                            beats_per_bar=beats_per_bar or DEFAULT_BEATS_PER_BAR,
                            beat_unit=beat_unit_for(beats_per_bar or DEFAULT_BEATS_PER_BAR))
    beats = fill_beat_gaps(beats)
    beats = extrapolate_beats(beats, duration)
    beats = clean_beats(beats, duration)

    if downbeat_times is None:
        n = beats_per_bar or DEFAULT_BEATS_PER_BAR
        downs = list(range(0, len(beats), n))
    else:
        detected = downbeats_to_indices(beats, downbeat_times)
        if not detected:
            log.warning("ダウンビートが検出できなかったので先頭の拍から数えます")
        n = beats_per_bar or infer_beats_per_bar(detected)
        downs = regularize_downbeats(detected, len(beats), n)

    bpm = estimate_bpm(beats)
    return BeatGrid(beat_times=beats, downbeat_indices=downs, bpm=round(bpm, 2),
                    beats_per_bar=n, beat_unit=beat_unit_for(n))


def estimate_bpm(beat_times: Sequence[float]) -> float:
    """拍の間隔の中央値から BPM を出す。

    中央値の ±15% に入る間隔だけを平均して仕上げる。モデルの拍の時刻は 20 ms 刻み
    （50 fps）なので、中央値だけだと 90 BPM が 90.91 になるような偏りが出るため。
    """
    d = np.diff(np.asarray(beat_times, dtype=float))
    d = d[d > 0]
    if len(d) == 0:
        return DEFAULT_BPM
    med = float(np.median(d))
    near = d[np.abs(d - med) <= 0.15 * med]
    return 60.0 / float(near.mean() if len(near) else med)


def beat_unit_for(beats_per_bar: int) -> int:
    """拍子の分母。

    6・9・12 拍は複合拍子（6/8 など）とみなして 8 にする。拍検出が 1 小節に 6 拍を
    数えるのは、ゆっくりした 6/8 で 8 分音符を拍として取っているときがほとんどで、
    ポップスでは 6/4 より 6/8 の書き方が普通なため。BPM はそのとき 8 分音符の数になる。
    """
    return 8 if beats_per_bar in (6, 9, 12) else 4


# ---------------------------------------------------------------------------
# 検出
# ---------------------------------------------------------------------------


def _beat_this_checkpoint(name: str = BEAT_THIS_CHECKPOINT) -> Path:
    """重みを models_dir()/beat_this/ に置く（なければダウンロード）。"""
    import torch
    from beat_this.inference import CHECKPOINT_URL

    from chordgen.paths import models_dir

    dest = models_dir() / "beat_this" / f"beat_this-{name}.ckpt"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        log.info("Beat This! の重み %s をダウンロードします", name)
        torch.hub.download_url_to_file(f"{CHECKPOINT_URL}/{name}.ckpt", str(tmp), progress=False)
        tmp.replace(dest)
    return dest


@lru_cache(maxsize=2)
def _beat_this_model(device: str):
    from beat_this.inference import Audio2Beats

    return Audio2Beats(checkpoint_path=str(_beat_this_checkpoint()), device=device, dbn=False)


def _detect_with_beat_this(y: np.ndarray, sr: int, device: str) -> tuple[list[float], list[float]]:
    try:
        model = _beat_this_model(device)
        beats, downbeats = model(y, sr)
    except Exception:
        if device == "cpu":
            raise
        log.warning("Beat This! が %s で失敗したので CPU でやり直します", device, exc_info=True)
        model = _beat_this_model("cpu")
        beats, downbeats = model(y, sr)
    return [float(t) for t in beats], [float(t) for t in downbeats]


def _detect_with_librosa(y: np.ndarray, sr: int) -> list[float]:
    import librosa

    target = 22050
    if sr != target:
        y = librosa.resample(y, orig_sr=sr, target_sr=target)
    _, beats = librosa.beat.beat_track(y=y, sr=target, units="time")
    return [float(t) for t in np.atleast_1d(beats)]


def detect_beats(audio_path: Path, device: str) -> BeatGrid:
    """音源の拍・ダウンビート・BPM・拍子を推定する。"""
    y, sr = load_audio_mono(Path(audio_path))
    duration = len(y) / sr if sr else 0.0
    try:
        beats, downbeats = _detect_with_beat_this(y, sr, device)
        log.info("Beat This!: %d 拍、%d ダウンビート", len(beats), len(downbeats))
        return build_beat_grid(beats, downbeats, duration)
    except Exception as e:  # ImportError（未導入）や重みの取得失敗など
        msg = (f"Beat This! を使えないため librosa の beat_track で拍を検出します"
               f"（ダウンビートは 4 拍ごとと仮定）: {type(e).__name__}: {e}")
        log.warning(msg)
        warnings.warn(msg, stacklevel=2)
    try:
        beats = _detect_with_librosa(y, sr)
    except ImportError as e:
        raise RuntimeError("拍の検出には beat-this（と torch）か librosa が必要です") from e
    return build_beat_grid(beats, None, duration)
