"""chordgen.chords のテスト。

重みが要るテストは環境変数 CHORDGEN_TEST_MODELS=1 のときだけ走る（torch と librosa も要る）。
それ以外（ラベルの変換、区間のまとめ方、ダウンロードの扱い）は常に走る。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chordgen.chords import MODELS, VOCAB, decode_logits, index_to_label, label_to_index, recognize_chords
from chordgen.chords import weights as W
from chordgen.chords.segments import frames_to_segments, majority_filter, merge_short_runs

# Harte 記法（ルート:種類）か N / X
HARTE = re.compile(r"^(N|X|[A-G]#?:(maj|min|dim|aug|min6|maj6|min7|minmaj7|maj7|7|dim7|hdim7|sus2|sus4))$")


# ---- ラベル ----

def test_index_to_label_examples():
    assert index_to_label(0) == "C:min"
    assert index_to_label(1) == "C:maj"
    assert index_to_label(9 * 14 + 6) == "A:min7"
    assert index_to_label(1 * 14 + 1) == "C#:maj"
    assert index_to_label(7 * 14 + 9) == "G:7"
    assert index_to_label(11 * 14 + 11) == "B:hdim7"
    assert index_to_label(168) == "X"
    assert index_to_label(169) == "N"


def test_vocab_roundtrip_and_format():
    assert len(VOCAB) == 170 and len(set(VOCAB)) == 170
    for i, label in enumerate(VOCAB):
        assert HARTE.match(label), label
        assert label_to_index(label) == i
    # メジャーも "C:maj" と書き、ルートだけのラベルはない
    assert all(":" in label for label in VOCAB if label not in ("N", "X"))


@pytest.mark.parametrize("bad", [-1, 170, 1000])
def test_index_out_of_range(bad):
    with pytest.raises(ValueError):
        index_to_label(bad)


@pytest.mark.parametrize("bad", ["C", "Db:maj", "C:9", "H:min", ""])
def test_label_to_index_rejects(bad):
    with pytest.raises(ValueError):
        label_to_index(bad)


# ---- 区間のまとめ方 ----

def _reference_majority_filter(values, kernel_size):
    """ChordMini の majority_filter_indices をそのまま写したもの（比較用）。"""
    values = np.asarray(values, dtype=np.int64)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1 or values.size < kernel_size:
        return values.copy()
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    out = np.empty_like(values)
    for idx in range(values.size):
        labels, counts = np.unique(padded[idx : idx + kernel_size], return_counts=True)
        cands = labels[counts == counts.max()]
        out[idx] = values[idx] if values[idx] in cands else int(cands[0])
    return out


@pytest.mark.parametrize("seed", range(5))
def test_majority_filter_matches_chordmini(seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 4, size=200)
    for k in (1, 3, 8, 9):
        np.testing.assert_array_equal(majority_filter(values, k), _reference_majority_filter(values, k))


def test_majority_filter_removes_blip():
    values = np.array([1] * 10 + [5] + [1] * 10)
    assert set(majority_filter(values, 9).tolist()) == {1}


def test_merge_short_runs_uses_probability():
    idx = np.array([1] * 10 + [2] * 2 + [3] * 10)
    probs = np.full((22, 170), 0.0)
    probs[:, 1] = 0.1
    probs[:, 3] = 0.1
    probs[10:12, 3] = 0.5  # 短い区間では 3 のほうが確からしい
    out = merge_short_runs(idx, 6, probs)
    assert out.tolist() == [1] * 10 + [3] * 12


def test_merge_short_runs_without_probs_prefers_longer_and_keeps_single():
    idx = np.array([1] * 8 + [2] * 2 + [3] * 20)
    assert merge_short_runs(idx, 6).tolist() == [1] * 8 + [3] * 22
    assert merge_short_runs(np.array([4, 4]), 6).tolist() == [4, 4]
    assert merge_short_runs(np.array([], dtype=np.int64), 6).tolist() == []


def test_frames_to_segments_merges_and_is_contiguous():
    fd = 0.1
    idx = np.array([1, 1, 1, 169, 169, 9 * 14 + 6, 9 * 14 + 6])
    probs = np.zeros((7, 170))
    probs[[0, 1, 2], 1] = [0.6, 0.8, 1.0]
    probs[[3, 4], 169] = 0.5
    probs[[5, 6], 9 * 14 + 6] = 0.9
    segs = frames_to_segments(idx, fd, probs, total_duration=0.75)
    assert [s.label for s in segs] == ["C:maj", "N", "A:min7"]
    assert segs[0].start == 0.0
    for a, b in zip(segs, segs[1:]):
        assert a.end == pytest.approx(b.start)
    assert segs[-1].end == pytest.approx(0.75)
    assert segs[0].confidence == pytest.approx(0.8)
    assert segs[2].confidence == pytest.approx(0.9)
    assert frames_to_segments(np.array([], dtype=np.int64), fd) == []


def _planted_logits(labels_and_frames):
    rows = []
    for label, n in labels_and_frames:
        row = np.zeros(170, dtype=np.float32)
        row[label_to_index(label)] = 8.0
        rows += [row] * n
    return np.stack(rows)


def test_decode_logits_end_to_end():
    logits = _planted_logits([("C:maj", 30), ("G:maj", 2), ("C:maj", 3), ("A:min", 30), ("N", 20)])
    fd = 0.1
    segs = decode_logits(logits, duration_sec=8.55, frame_duration=fd)
    # 2 フレームの G は多数決フィルタで消え、C と A:min と N の 3 区間になる
    assert [s.label for s in segs] == ["C:maj", "A:min", "N"]
    assert segs[0].start == 0.0 and segs[-1].end == pytest.approx(8.55)
    assert segs[1].start == pytest.approx(3.5, abs=0.15)  # 境目は多数決で 1 フレームずれうる
    assert all(0.8 < s.confidence <= 1.0 for s in segs)  # 多数決で書き換えたフレームは確率が低い


def test_decode_logits_truncates_to_duration_and_handles_empty():
    logits = _planted_logits([("D:7", 50)])
    segs = decode_logits(logits, duration_sec=2.0, frame_duration=0.1)
    assert len(segs) == 1 and segs[0].end == pytest.approx(2.0)
    assert decode_logits(np.zeros((0, 170)), duration_sec=0.0) == []


def test_short_segments_leave_no_gap():
    # 0.5 秒（6 フレーム）未満の区間は隣にまとまり、隙間が残らない
    logits = _planted_logits([("C:maj", 20), ("F:maj", 4), ("G:maj", 20)])
    segs = decode_logits(logits, duration_sec=4.4, frame_duration=0.1, kernel_size=1)
    assert [s.label for s in segs] == ["C:maj", "G:maj"]
    assert segs[0].end == pytest.approx(segs[1].start)


# ---- import と引数 ----

def test_import_does_not_load_torch_or_librosa():
    code = "import sys, chordgen.chords; assert 'torch' not in sys.modules and 'librosa' not in sys.modules"
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, shell=False)


def test_unknown_model_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError):
        recognize_chords(tmp_path / "missing.wav", "cpu", model="nope")
    assert set(MODELS) == {"chordmini", "btc"}


# ---- 重みの取得 ----

def _fake_spec(tmp_path, payload: bytes, sha: str | None = None) -> W.WeightSpec:
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    return W.WeightSpec(
        name="fake", description="テスト", filename="fake.pth", url=src.as_uri(),
        sha256=sha or hashlib.sha256(payload).hexdigest(), size=len(payload),
    )


def test_ensure_weights_downloads_once(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORDGEN_HOME", str(tmp_path / "home"))
    spec = _fake_spec(tmp_path, b"x" * 200_000)
    monkeypatch.setitem(W.WEIGHTS, "fake", spec)
    events = []
    path = W.ensure_weights("fake", on_progress=events.append)
    assert path == tmp_path / "home" / "models" / "chords" / "fake.pth"
    assert path.read_bytes() == b"x" * 200_000
    assert events and events[-1]["finished"] and events[-1]["downloaded"] == 200_000
    events.clear()
    assert W.ensure_weights("fake", on_progress=events.append) == path
    assert events == []  # 2 回目はダウンロードしない
    assert [p.name for p in path.parent.iterdir()] == ["fake.pth"]


def test_ensure_weights_bad_hash_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORDGEN_HOME", str(tmp_path / "home"))
    spec = _fake_spec(tmp_path, b"abc", sha="0" * 64)
    monkeypatch.setitem(W.WEIGHTS, "fake", spec)
    with pytest.raises(W.WeightsError):
        W.ensure_weights("fake", on_progress=lambda e: None)
    assert list((tmp_path / "home" / "models" / "chords").iterdir()) == []


def test_ensure_weights_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORDGEN_HOME", str(tmp_path / "home"))
    spec = W.WeightSpec("fake", "テスト", "fake.pth", (tmp_path / "none.bin").as_uri(), "0" * 64, 1)
    monkeypatch.setitem(W.WEIGHTS, "fake", spec)
    with pytest.raises(W.WeightsError):
        W.ensure_weights("fake", on_progress=lambda e: None)
    assert list((tmp_path / "home" / "models" / "chords").iterdir()) == []


def test_weight_specs_pinned():
    for spec in W.WEIGHTS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256)
        assert re.search(r"/[0-9a-f]{40}/", spec.url), "取得元はコミットで固定する"


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def test_checkpoint_state_and_stats_formats():
    chordmini = {"model_state_dict": {"module.a": 1}, "normalization": {"mean": _Scalar(-2.37), "std": _Scalar(1.96)}}
    state, mean, std = W.checkpoint_state_and_stats(chordmini)
    assert state == {"a": 1} and mean == pytest.approx(-2.37) and std == pytest.approx(1.96)
    btc = {"model": {"b": 2}, "mean": -2.22, "std": np.float64(1.72)}
    state, mean, std = W.checkpoint_state_and_stats(btc)
    assert state == {"b": 2} and mean == pytest.approx(-2.22) and std == pytest.approx(1.72)


# ---- 実際の重みで推論（CHORDGEN_TEST_MODELS=1 のときだけ） ----

_HAS_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "librosa"))
needs_models = pytest.mark.skipif(
    os.environ.get("CHORDGEN_TEST_MODELS") != "1" or not _HAS_DEPS,
    reason="CHORDGEN_TEST_MODELS=1 と torch・librosa が要る（重みをダウンロードする）",
)

SR = 22050


def _chord_tone(midis, sec):
    t = np.arange(int(SR * sec)) / SR
    y = np.zeros_like(t)
    for m in midis:
        f = 440 * 2 ** ((m - 69) / 12)
        for h in range(1, 6):
            y += 0.6 ** (h - 1) * np.sin(2 * np.pi * f * h * t)
    return y * np.minimum(1, np.minimum(t / 0.02, (sec - t) / 0.05))


@needs_models
@pytest.mark.parametrize("model", ["chordmini", "btc"])
def test_recognize_synthetic_progression(tmp_path, model):
    import soundfile as sf

    prog = [("C:maj", [48, 60, 64, 67]), ("G:maj", [43, 59, 62, 67]), ("A:min", [45, 60, 64, 69]), ("F:maj", [41, 60, 65, 69])]
    y = np.concatenate([_chord_tone(notes, 3.0) for _, notes in prog])
    path = tmp_path / "prog.wav"
    sf.write(path, (0.3 * y / np.abs(y).max()).astype(np.float32), SR)

    segs = recognize_chords(path, "cpu", model=model)
    assert [s.label for s in segs] == [label for label, _ in prog]
    assert segs[0].start == 0.0 and segs[-1].end == pytest.approx(12.0, abs=0.01)
    for seg, expected_start in zip(segs, (0.0, 3.0, 6.0, 9.0)):
        assert seg.start == pytest.approx(expected_start, abs=0.25)
        assert 0.5 < seg.confidence <= 1.0
