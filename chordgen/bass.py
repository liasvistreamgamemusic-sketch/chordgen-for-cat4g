"""分離したベースの音から、拍ごとのベースのピッチクラスを求める。

numpy だけで動く YIN（自己相関の差分関数）でベースの音域の f0 を推定し、
拍の区間ごとに有声のフレームのピッチクラスを数えて最頻値を 1 音に決める。
オクターブの取り違え（倍・半分の周期）はピッチクラスが変わらないので気にしなくてよい。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from chordgen.types import BassNote, BeatGrid

# f0 を探す範囲（およそ E1〜G3 に半音ずつ余裕を持たせる）
FMIN_HZ = 38.89  # D#1
FMAX_HZ = 207.65  # G#3
ANALYSIS_SR = 11025  # 解析前にこのくらいまで間引く
LOWPASS_HZ = 1500.0  # 間引く前の低域通過（ベースの倍音の下のほうだけ残す）
WIN_SEC = 0.093  # YIN の窓（最低音の周期の 3.5 倍ほど）
HOP_SEC = 0.0116  # フレームの間隔

YIN_THRESHOLD = 0.15  # 差分関数がこれを下回る最初の谷を周期とする
VOICED_MAX_APERIODICITY = 0.35  # これ以下のフレームを有声とみなす
SILENCE_DB = -35.0  # 曲全体の大きい音（95 パーセンタイル）からこれ以上小さいフレームは無音
ABS_SILENCE_RMS = 1e-4

MIN_VOICED_RATIO = 0.3  # 拍の中の有声フレームがこれ未満なら None
MIN_AGREEMENT = 0.5  # 有声フレームのうち最多のピッチクラスの割合がこれ未満なら None


# ---------------------------------------------------------------- 前処理


def _fft_convolve_same(x: np.ndarray, h: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """奇数長のフィルタ h で x を畳み込み、x と同じ長さ（中心をそろえる）で返す。"""
    m = len(h)
    nfft = 1 << int(np.ceil(np.log2(block + m - 1)))
    hf = np.fft.rfft(h, nfft)
    full = np.zeros(len(x) + m - 1)
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * hf, nfft)[:len(seg) + m - 1]
        full[s:s + len(y)] += y
    d = (m - 1) // 2
    return full[d:d + len(x)]


def _downsample(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """低域通過のあと整数分の 1 に間引く。"""
    factor = max(1, int(sr // ANALYSIS_SR))
    cutoff = min(LOWPASS_HZ, 0.45 * sr / factor)
    taps = 16 * factor + 1
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff / sr
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
    h /= h.sum()
    y = _fft_convolve_same(x.astype(np.float64), h)
    return y[::factor], sr / factor


# ---------------------------------------------------------------- YIN


def _yin(x: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """フレームごとの f0（Hz）、非周期性（0〜1、小さいほど周期的）、RMS と、フレームの間隔（秒）。

    フレーム i の中心は i * hop。
    """
    win = int(round(WIN_SEC * sr))
    hop = max(1, int(round(HOP_SEC * sr)))
    tau_min = max(2, int(np.floor(sr / FMAX_HZ)))
    tau_max = int(np.ceil(sr / FMIN_HZ))
    frame_len = win + tau_max + 1
    pad = frame_len // 2
    xp = np.pad(x, (pad, pad + frame_len))
    n_frames = 1 + len(x) // hop
    nfft = 1 << int(np.ceil(np.log2(frame_len + win)))

    f0 = np.zeros(n_frames)
    aper = np.ones(n_frames)
    rms = np.zeros(n_frames)
    taus = np.arange(tau_max + 1)
    batch = 512
    for b0 in range(0, n_frames, batch):
        idx = np.arange(b0, min(n_frames, b0 + batch))
        starts = idx * hop
        frames = xp[starts[:, None] + np.arange(frame_len)[None, :]]
        rms[idx] = np.sqrt(np.mean(frames[:, :win + tau_max] ** 2, axis=1))

        # r(tau) = sum_{j<win} x[j] x[j+tau]
        a = np.fft.rfft(frames, nfft)
        b = np.fft.rfft(frames[:, :win][:, ::-1], nfft)
        acf = np.fft.irfft(a * b, nfft)[:, win - 1:win + tau_max]
        sq = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
        energy = sq[:, taus + win] - sq[:, taus]  # sum_{j=tau}^{tau+win-1} x[j]^2
        diff = np.maximum(energy[:, :1] + energy - 2 * acf, 0.0)

        # 累積平均で正規化した差分関数
        cmnd = np.ones_like(diff)
        csum = np.cumsum(diff[:, 1:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd[:, 1:] = diff[:, 1:] * taus[1:] / np.where(csum > 0, csum, np.inf)
        cmnd[~np.isfinite(cmnd)] = 1.0

        seg = cmnd[:, tau_min:tau_max]  # 谷の判定に右隣を使うので tau_max は含めない
        right = cmnd[:, tau_min + 1:tau_max + 1]
        left = cmnd[:, tau_min - 1:tau_max - 1]
        is_min = (seg <= left) & (seg <= right)
        below = (seg < YIN_THRESHOLD) & is_min
        first = np.where(below.any(axis=1), below.argmax(axis=1), -1)
        # しきい値を下回る谷がなければ、谷のうち最小のもの
        masked = np.where(is_min, seg, np.inf)
        best = masked.argmin(axis=1)
        k = np.where(first >= 0, first, best)
        rows = np.arange(len(idx))
        tau = (k + tau_min).astype(np.float64)
        val = seg[rows, k]
        ok = np.isfinite(masked[rows, k])

        # 放物線補間で周期を細かく求める
        t = (k + tau_min)
        y0 = cmnd[rows, t - 1]
        y1 = cmnd[rows, t]
        y2 = cmnd[rows, t + 1]
        denom = y0 - 2 * y1 + y2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
        tau = tau + np.clip(shift, -0.5, 0.5)

        f0[idx] = np.where(ok, sr / tau, 0.0)
        aper[idx] = np.where(ok, np.clip(val, 0.0, 1.0), 1.0)
    return f0, aper, rms, hop / sr


# ---------------------------------------------------------------- 拍ごとの集計


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def detect_bass(bass_path: Path, beats: BeatGrid) -> list[BassNote]:
    """拍ごと（beat_times の隣り合う拍の間）のベースのピッチクラスを返す。

    音が小さい拍・ピッチが定まらない拍は pitch_class=None。confidence は
    「有声フレームの割合 × 最多のピッチクラスの割合」（0〜1）。
    """
    times = list(beats.beat_times)
    if len(times) < 2:
        return []

    x, sr = _load_mono(Path(bass_path))
    y, sr2 = _downsample(x, sr)
    f0, aper, rms, hop_sec = _yin(y, sr2)
    frame_t = np.arange(len(f0)) * hop_sec

    # 無音の判定（曲全体の大きい音を基準にする）
    loud = rms[rms > ABS_SILENCE_RMS]
    ref = float(np.percentile(loud, 95)) if loud.size else 0.0
    gate = max(ABS_SILENCE_RMS, ref * 10 ** (SILENCE_DB / 20))
    voiced = (rms > gate) & (aper <= VOICED_MAX_APERIODICITY) & (f0 > 0)

    midi = np.zeros_like(f0)
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440.0)
    # 曲全体のチューニングのずれ（半音未満）を差し引く
    if voiced.any():
        dev = midi[voiced] - np.round(midi[voiced])
        tuning = float(np.angle(np.mean(np.exp(2j * np.pi * dev))) / (2 * np.pi))
        midi[voiced] -= tuning
    pcs = np.mod(np.round(midi).astype(int), 12)

    notes: list[BassNote] = []
    for start, end in zip(times[:-1], times[1:]):
        sel = (frame_t >= start) & (frame_t < end)
        n = int(sel.sum())
        if end <= start or n == 0:
            notes.append(BassNote(start=float(start), end=float(end), pitch_class=None, confidence=0.0))
            continue
        v = sel & voiced
        nv = int(v.sum())
        voiced_ratio = nv / n
        if nv == 0:
            notes.append(BassNote(start=float(start), end=float(end), pitch_class=None, confidence=0.0))
            continue
        counts = np.bincount(pcs[v], weights=1.0 - aper[v], minlength=12)
        pc = int(np.argmax(counts))
        agreement = float(counts[pc] / counts.sum()) if counts.sum() > 0 else 0.0
        conf = float(np.clip(voiced_ratio * agreement, 0.0, 1.0))
        if voiced_ratio < MIN_VOICED_RATIO or agreement < MIN_AGREEMENT:
            notes.append(BassNote(start=float(start), end=float(end), pitch_class=None, confidence=conf))
        else:
            notes.append(BassNote(start=float(start), end=float(end), pitch_class=pc, confidence=conf))
    return notes
