"""推論に使うデバイスを選ぶ。ROCm 版の PyTorch も torch.cuda として見える。

GPU が複数見えるとき（CPU 内蔵の Radeon と単体の Radeon など）、0 番が使える GPU とは限らない。
AMD の PyTorch は GPU の種類（gfx1201 など）ごとのカーネルを別の wheel で入れるので、
カーネルのない GPU で計算すると "invalid kernel file" で落ちる。そこで、入っているカーネルに
合う GPU を選び、torch.cuda.set_device でその GPU を "cuda" の既定にする。
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 使う GPU の番号を明示する（torch.cuda の番号。0 始まり）
ENV_GPU_INDEX = "CHORDGEN_GPU_INDEX"

# AMD の wheel の名前から、カーネルが入っている GPU の種類を読む（例 amd-torch-device-gfx1201）
_DEVICE_PKG_RE = re.compile(r"^(?:amd-torch-device|rocm-sdk-device)-(gfx[0-9a-f]+)$")


@dataclass
class GpuInfo:
    index: int
    name: str
    arch: str | None  # AMD は gfx1201 など。NVIDIA は None
    integrated: bool
    total_memory: int


def installed_amd_archs() -> set[str]:
    """入っている AMD のデバイス用 wheel が対応する gfx の一覧。AMD 版でなければ空。"""
    archs: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        m = _DEVICE_PKG_RE.match(name)
        if m:
            archs.add(m.group(1))
    return archs


def list_gpus(torch: Any) -> list[GpuInfo]:
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        arch = getattr(p, "gcnArchName", None) or None
        if arch:
            arch = arch.split(":")[0]  # "gfx1201:sramecc-:xnack-" のような形もある
        gpus.append(GpuInfo(i, p.name, arch, bool(getattr(p, "is_integrated", False)), int(p.total_memory)))
    return gpus


def choose_gpu(gpus: list[GpuInfo], amd_archs: set[str], env_index: str | None = None) -> GpuInfo | None:
    """使う GPU を選ぶ。選べるものがなければ None。

    1. 環境変数 CHORDGEN_GPU_INDEX の番号
    2. AMD は、入っているカーネルに合う種類の GPU（内蔵でないものを先に）
    3. 内蔵でない GPU（内蔵 GPU は共有メモリのぶん total_memory が大きく見えるので、大きさでは選ばない）
    4. 0 番
    """
    if not gpus:
        return None
    if env_index not in (None, ""):
        try:
            idx = int(env_index)
        except ValueError:
            raise ValueError(f"{ENV_GPU_INDEX} は GPU の番号（0 始まりの整数）で指定してください: {env_index!r}")
        for g in gpus:
            if g.index == idx:
                return g
        raise ValueError(f"{ENV_GPU_INDEX}={idx} の GPU がありません（{len(gpus)} 台: 0〜{len(gpus) - 1}）")
    candidates = gpus
    if amd_archs:
        matched = [g for g in gpus if g.arch in amd_archs]
        if matched:
            candidates = matched
        else:
            logger.warning(
                "入っている GPU 用のカーネル（%s）に合う GPU が見つかりません: %s",
                ", ".join(sorted(amd_archs)),
                ", ".join(f"{g.index}: {g.name} ({g.arch})" for g in gpus),
            )
    discrete = [g for g in candidates if not g.integrated]
    return (discrete or candidates)[0]


def select_gpu() -> GpuInfo | None:
    """使う GPU を選んで torch の既定にする。GPU がなければ None。"""
    import torch

    if not torch.cuda.is_available():
        return None
    gpu = choose_gpu(list_gpus(torch), installed_amd_archs(), os.environ.get(ENV_GPU_INDEX))
    if gpu is not None:
        torch.cuda.set_device(gpu.index)
        if torch.cuda.device_count() > 1:
            logger.info("GPU %d（%s）を使います", gpu.index, gpu.name)
    return gpu


def get_device(prefer: str | None = None) -> str:
    """"cuda" か "cpu" を返す。prefer に "cpu" / "cuda" を渡すと強制する。

    "cuda" を返すときは、使う GPU を torch の既定にしてある（"cuda" がその GPU を指す）。
    """
    if prefer == "cpu":
        return "cpu"
    gpu = select_gpu()
    if prefer:
        return prefer
    return "cuda" if gpu is not None else "cpu"
