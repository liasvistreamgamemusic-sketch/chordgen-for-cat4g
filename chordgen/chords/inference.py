"""特徴量の計算と、窓に分けた推論。

ChordMini の src/evaluation/test.py（commit aa6e3a8d）の処理を関数にしたもの。README が BTC に勧める
次の指定と同じ動きにしている:
  --smooth_logits --use_overlap --use_gaussian --kernel_size 9 --vote_aggregation logit
  --min_segment_duration 0.5 --smooth_predictions
元の BTC（model="btc"）も、ChordMini が先生役を評価したときと同じく、この処理で動かす（重みと mean/std だけが違う）。

torch と librosa は関数の中で import する。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .labels import NUM_CLASSES

# config/ChordMini.yaml の mp3 / feature 節と同じ値
SAMPLE_RATE = 22050
HOP_LENGTH = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
FMIN_NOTE = "C1"
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # 約 0.0929 秒
SEQ_LEN = 108

# test.py の推論の指定（README の BTC の例）
OVERLAP_RATIO = 0.5  # --use_overlap の既定
KERNEL_SIZE = 9
USE_GAUSSIAN = True
BATCH_SIZE = 16
MIN_SEGMENT_DURATION = 0.5


def load_audio(audio_path: Path) -> np.ndarray:
    """22050 Hz のモノラルで読む（librosa.load。学習時と同じ読み方・リサンプル）。"""
    import librosa

    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def compute_features(audio: np.ndarray) -> np.ndarray:
    """log-CQT [フレーム数, 144]。ChordMini の extract_song_features と同じ。"""
    import librosa

    cqt = librosa.cqt(
        audio,
        sr=SAMPLE_RATE,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz(FMIN_NOTE),
    )
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


@dataclass
class LoadedModel:
    model: Any  # BTCModel
    mean: float
    std: float
    device: str


_cache: dict[tuple[str, str], LoadedModel] = {}
_lock = threading.Lock()


def get_model(model: str, device: str) -> LoadedModel:
    """重みを（なければダウンロードして）読み、eval にしてデバイスに載せる。同じ組み合わせは使い回す。"""
    import torch

    from .btc_model import BTCModel
    from .weights import checkpoint_state_and_stats, ensure_weights, load_checkpoint

    key = (model, device)
    with _lock:
        if key in _cache:
            return _cache[key]
        path = ensure_weights(model)
        state, mean, std = checkpoint_state_and_stats(load_checkpoint(path))
        net = BTCModel()
        net.load_state_dict(state, strict=True)
        net.eval()
        net.to(torch.device(device))
        loaded = LoadedModel(model=net, mean=mean, std=std, device=device)
        _cache[key] = loaded
        return loaded


def predict_logits(loaded: LoadedModel, features: np.ndarray) -> np.ndarray:
    """窓（108 フレーム、半分ずつ重ねる）に分けて推論し、フレームごとのロジットの平均 [T, 170] を返す。

    ChordMini の predict_sliding_windows（vote_aggregation="logit"、smooth_logits）と同じ手順:
    末尾を 0 で埋めて seq_len の倍数にし、正規化し、窓ごとにロジットをガウスでならしてから足し合わせる。
    test.py は足し合わせたロジットの argmax を取る。平均でも argmax は同じで、確信度の softmax に使える。
    """
    import torch

    from .btc_model import apply_temporal_smoothing

    feats = np.asarray(features, dtype=np.float32)
    n = int(feats.shape[0])
    if feats.ndim != 2 or n == 0:
        return np.zeros((0, NUM_CLASSES), dtype=np.float32)

    num_pad = (-n) % SEQ_LEN
    padded = np.pad(feats, ((0, num_pad), (0, 0)), mode="constant") if num_pad else feats
    stride = max(1, int(SEQ_LEN * (1.0 - OVERLAP_RATIO)))
    total = padded.shape[0]
    starts = [i * stride for i in range(max(1, (total - SEQ_LEN) // stride + 1))]

    acc = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    dev = torch.device(loaded.device)
    with torch.no_grad():
        for b in range(0, len(starts), BATCH_SIZE):
            batch_starts = [s for s in starts[b : b + BATCH_SIZE] if s + SEQ_LEN <= total and s < n]
            if not batch_starts:
                continue
            x = torch.from_numpy(np.stack([padded[s : s + SEQ_LEN] for s in batch_starts])).to(dev)
            x = (x - loaded.mean) / (loaded.std + 1e-8)
            logits = apply_temporal_smoothing(loaded.model(x), KERNEL_SIZE, USE_GAUSSIAN)
            out = logits.float().cpu().numpy()
            for k, s in enumerate(batch_starts):
                valid = min(SEQ_LEN, n - s)
                acc[s : s + valid] += out[k, :valid]
                counts[s : s + valid] += 1
    counts = np.maximum(counts, 1)
    return (acc / counts[:, None]).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
