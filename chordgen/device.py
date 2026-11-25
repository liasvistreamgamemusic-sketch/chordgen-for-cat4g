"""推論に使うデバイスを選ぶ。ROCm 版の PyTorch も torch.cuda として見える。"""

from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """"cuda" か "cpu" を返す。prefer に "cpu" / "cuda" を渡すと強制する。"""
    if prefer:
        return prefer
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
