"""GPU の選び方（torch なしで確かめる）。"""

import pytest

from chordgen.device import GpuInfo, choose_gpu

IGPU = GpuInfo(0, "AMD Radeon(TM) Graphics", "gfx1036", True, 43 * 2**30)
DGPU = GpuInfo(1, "AMD Radeon RX 9070 XT", "gfx1201", False, 16 * 2**30)


def test_amd_picks_gpu_matching_installed_kernels():
    # 内蔵 GPU が 0 番でも、入っているカーネル（gfx1201）に合う単体 GPU を選ぶ
    assert choose_gpu([IGPU, DGPU], {"gfx1201"}) is DGPU


def test_prefers_discrete_when_no_arch_info():
    # NVIDIA など種類の情報がないときは内蔵でないものを選ぶ（メモリの大きさでは選ばない）
    a = GpuInfo(0, "iGPU", None, True, 64 * 2**30)
    b = GpuInfo(1, "dGPU", None, False, 8 * 2**30)
    assert choose_gpu([a, b], set()) is b


def test_falls_back_when_no_arch_matches():
    assert choose_gpu([IGPU, DGPU], {"gfx1100"}) is DGPU


def test_env_index_overrides():
    assert choose_gpu([IGPU, DGPU], {"gfx1201"}, "0") is IGPU


@pytest.mark.parametrize("value", ["x", "5"])
def test_env_index_invalid(value):
    with pytest.raises(ValueError):
        choose_gpu([IGPU, DGPU], {"gfx1201"}, value)


def test_no_gpu():
    assert choose_gpu([], {"gfx1201"}) is None
