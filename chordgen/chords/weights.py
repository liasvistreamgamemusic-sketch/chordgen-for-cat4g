"""コード認識モデルの重みの取得と読み込み。

重みは初回に chordgen.paths.models_dir() / "chords" へダウンロードする（リポジトリには入れない）。
取得元はコミットで固定し、SHA-256 を照合する。途中で切れたファイルは残さない
（一時ファイルに書いてから照合し、通ったら os.replace で置き換える）。

ライセンス: どちらの重みも CC-BY-NC-SA 4.0（非商用）として扱う。
  - 元の BTC（jayg996/BTC-ISMIR19）の重みは CC-BY-NC-SA 4.0
  - ChordMini の重みはライセンスが書かれていないが、元の BTC を先生役に作られているので同じく非商用とする
  商用には使わない。
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chordgen.paths import models_dir

_CHORDMINI_COMMIT = "aa6e3a8d7b017f082fd2aaff9329d5c26af49c03"
_BTC_COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"

# 特徴量とモデルの設定の出典（値は inference.py と btc_model.py に定数で持つ。PyYAML を要らなくするため）
CHORDMINI_CONFIG_URL = (
    f"https://raw.githubusercontent.com/ptnghia-j/ChordMini/{_CHORDMINI_COMMIT}/config/ChordMini.yaml"
)


@dataclass(frozen=True)
class WeightSpec:
    name: str  # recognize_chords の model 引数
    description: str
    filename: str
    url: str
    sha256: str
    size: int


WEIGHTS: dict[str, WeightSpec] = {
    "chordmini": WeightSpec(
        name="chordmini",
        description="ChordMini BTC（CL full）",
        filename="chordmini_btc_model_best.pth",
        url=f"https://raw.githubusercontent.com/ptnghia-j/ChordMini/{_CHORDMINI_COMMIT}/checkpoints/btc_model_best.pth",
        sha256="e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc",
        size=35942692,
    ),
    "btc": WeightSpec(
        name="btc",
        description="BTC（ISMIR19、大語彙）",
        filename="btc_model_large_voca.pt",
        url=f"https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/{_BTC_COMMIT}/test/btc_model_large_voca.pt",
        sha256="1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51",
        size=12229576,
    ),
}

ProgressCallback = Callable[[dict[str, Any]], None]


class WeightsError(RuntimeError):
    """重みの取得や照合に失敗した。"""


def weights_dir() -> Path:
    p = models_dir() / "chords"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_spec(model: str) -> WeightSpec:
    try:
        return WEIGHTS[model]
    except KeyError:
        raise ValueError(f"model は {sorted(WEIGHTS)} のどれか: {model!r}") from None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, spec: WeightSpec) -> bool:
    return path.is_file() and path.stat().st_size == spec.size and _sha256(path) == spec.sha256


def _stderr_progress(event: dict[str, Any]) -> None:
    """進捗を標準エラーに 1 行で出す。"""
    done, total = event["downloaded"], event["total"]
    mb = 1024 * 1024
    if total:
        msg = f"\r{event['description']} の重みをダウンロード中: {done / mb:.1f} / {total / mb:.1f} MB ({done * 100 // total}%)"
    else:
        msg = f"\r{event['description']} の重みをダウンロード中: {done / mb:.1f} MB"
    end = "\n" if event.get("finished") else ""
    print(msg, end=end, file=sys.stderr, flush=True)


def _download(spec: WeightSpec, dest: Path, on_progress: ProgressCallback | None) -> None:
    import urllib.request

    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    report = on_progress or _stderr_progress
    try:
        req = urllib.request.Request(spec.url, headers={"User-Agent": "chordgen"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
            total = int(resp.headers.get("Content-Length") or 0) or spec.size
            h = hashlib.sha256()
            done = 0
            last = 0.0
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last >= 0.5:
                    last = now
                    report({"event": "download", "model": spec.name, "description": spec.description,
                            "downloaded": done, "total": total, "finished": False})
        report({"event": "download", "model": spec.name, "description": spec.description,
                "downloaded": done, "total": total, "finished": True})
        if done != spec.size or h.hexdigest() != spec.sha256:
            raise WeightsError(
                f"{spec.description} の重みが壊れている（{done} バイト、SHA-256 不一致）: {spec.url}"
            )
        os.replace(tmp, dest)
    except WeightsError:
        raise
    except Exception as e:
        raise WeightsError(f"{spec.description} の重みをダウンロードできなかった: {spec.url}（{e}）") from e
    finally:
        tmp.unlink(missing_ok=True)


def ensure_weights(model: str = "chordmini", on_progress: ProgressCallback | None = None) -> Path:
    """重みのファイルのパスを返す。なければ（または壊れていれば）ダウンロードする。"""
    spec = get_spec(model)
    dest = weights_dir() / spec.filename
    if _is_valid(dest, spec):
        return dest
    _download(spec, dest, on_progress)
    return dest


def load_checkpoint(path: Path) -> dict:
    """torch.load で読む。まず weights_only=True で読み、numpy のスカラーだけ許可して再試行する。

    元の BTC の重みは std が numpy.float64 で入っていて、そのままでは weights_only で読めない。
    それでも読めない torch / numpy の組み合わせでは、SHA-256 を照合したうえで weights_only=False にする
    （取得元をコミットで固定しているため）。map_location="cpu" で読み、デバイスへは後で移す。
    """
    import pickle

    import numpy as np
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        pass
    try:
        core = getattr(np, "_core", None) or np.core
        allowed = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    known = {s.sha256 for s in WEIGHTS.values()}
    if _sha256(Path(path)) not in known:
        raise WeightsError(f"weights_only で読めず、既知の重みとも一致しない: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_state_and_stats(checkpoint: dict) -> tuple[dict, float, float]:
    """state_dict と正規化の mean / std を取り出す（ChordMini の checkpoint_utils と同じ規則）。

    ChordMini: {"model_state_dict", "normalization": {"mean", "std"}}
    元の BTC: {"model", "mean", "std"}
    """
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    if state and next(iter(state)).startswith("module."):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    mean = checkpoint.get("mean", 0.0)
    std = checkpoint.get("std", 1.0)
    norm = checkpoint.get("normalization")
    if isinstance(norm, dict):
        mean = norm.get("mean", mean)
        std = norm.get("std", std)
    mean = float(mean.item()) if hasattr(mean, "item") else float(mean)
    std = float(std.item()) if hasattr(std, "item") else float(std)
    return state, mean, max(std, 1e-8)
