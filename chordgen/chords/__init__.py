"""コード認識。ChordMini の BTC（CL full、既定）か元の BTC で、170 クラスの大語彙のコードを出す。

    recognize_chords(audio_path, device, model="chordmini") -> list[ChordSegment]

ラベルは Harte 記法（"C:maj"、"A:min7"、"N"、"X"）。重みは初回に models_dir() / "chords" へ
ダウンロードする（weights.py。重みは CC-BY-NC-SA 4.0 の非商用として扱う）。
torch と librosa はこのパッケージを import しただけでは読み込まない。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from chordgen.types import ChordSegment

from .labels import VOCAB, index_to_label, label_to_index
from .segments import frames_to_segments, majority_filter, merge_short_runs
from .weights import WEIGHTS, WeightsError, ensure_weights, get_spec

MODELS = tuple(WEIGHTS)  # ("chordmini", "btc")

__all__ = [
    "MODELS",
    "VOCAB",
    "WeightsError",
    "decode_logits",
    "ensure_weights",
    "index_to_label",
    "label_to_index",
    "recognize_chords",
]


def decode_logits(
    logits: np.ndarray,
    duration_sec: float | None = None,
    *,
    frame_duration: float | None = None,
    kernel_size: int | None = None,
    min_segment_duration: float | None = None,
) -> list[ChordSegment]:
    """フレームごとのロジット [T, 170] から区間を作る（torch なしで動く）。

    argmax → 多数決フィルタ（test.py の --smooth_predictions）→ 短い区間を隣に吸収 → 区間にまとめる。
    confidence は区間の各フレームでの softmax 確率（選んだコードの値）の平均。
    """
    from .inference import FRAME_DURATION, KERNEL_SIZE, MIN_SEGMENT_DURATION, softmax

    fd = FRAME_DURATION if frame_duration is None else frame_duration
    ks = KERNEL_SIZE if kernel_size is None else kernel_size
    msd = MIN_SEGMENT_DURATION if min_segment_duration is None else min_segment_duration

    logits = np.asarray(logits, dtype=np.float32)
    if duration_sec is not None:
        # test.py と同じく、曲の長さを超えるフレームを切る
        logits = logits[: max(0, int(math.floor(duration_sec / fd)))]
    if logits.shape[0] == 0:
        return []
    probs = softmax(logits)
    indices = majority_filter(probs.argmax(axis=1), ks)
    indices = merge_short_runs(indices, math.ceil(msd / fd - 1e-9), probs)
    return frames_to_segments(indices, fd, probs, total_duration=duration_sec)


def recognize_chords(audio_path: Path, device: str, model: str = "chordmini") -> list[ChordSegment]:
    """音源のコードを認識する。

    model: "chordmini"（ChordMini の BTC、CL full。既定）/ "btc"（元の BTC、比較用）
    device: "cuda" か "cpu"（ROCm 版の PyTorch も "cuda"）
    """
    get_spec(model)  # 名前の誤りは音源を読む前に知らせる
    from .inference import SAMPLE_RATE, compute_features, get_model, load_audio, predict_logits

    audio = load_audio(Path(audio_path))
    duration = len(audio) / SAMPLE_RATE
    if len(audio) == 0:
        return []
    loaded = get_model(model, device)
    logits = predict_logits(loaded, compute_features(audio))
    return decode_logits(logits, duration)
