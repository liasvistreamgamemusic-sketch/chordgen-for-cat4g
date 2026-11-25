"""Demucs（htdemucs）による音源分離と、コード認識に入れる音の組み立て。

Demucs は PyPI の 4.1.0 以降を使う（推論に torchaudio が要らない版）。重みは
HuggingFace Hub から取得し、置き場所は HF_HOME（run.bat がアプリのフォルダに向ける）に従う。
torch と demucs は separate() の中で読み込むので、import chordgen.separation だけなら要らない。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from chordgen.types import STEM_NAMES, ChordInputMode, Stems

# 分離の設定
DEMUCS_MODEL = "htdemucs"
# 曲を区切って順に処理する（VRAM は区切り 1 つぶんで済み、曲の長さによらない）。
# segment=None は htdemucs の上限（約 7.8 秒）を使う
DEMUCS_SPLIT = True
DEMUCS_SEGMENT: float | None = None
DEMUCS_OVERLAP = 0.25
DEMUCS_SHIFTS = 1  # demucs の CLI と同じ既定値

MANIFEST_NAME = "separation.json"
_MANIFEST_VERSION = 1

# コード認識に入れる音。値はステムごとの倍率（載っていないステムは使わない）。
# boost_other は 2025 年の報告（other だけを大きくして組み直すのがいちばん良かった）にならう。
# 倍率は報告に合わせて後で変えられるよう定数にしておく
OTHER_BOOST_GAIN = 2.0  # 約 +6 dB
MODE_GAINS: dict[str, dict[str, float]] = {
    "no_drums": {"vocals": 1.0, "bass": 1.0, "other": 1.0},
    "no_drums_vocals": {"bass": 1.0, "other": 1.0},
    "boost_other": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": OTHER_BOOST_GAIN},
}
# 組み直した音のピーク（-1 dBFS）。これに合わせて正規化し、クリップさせない
PEAK_TARGET = 10 ** (-1.0 / 20.0)

_separators: dict[str, object] = {}


class SeparationError(RuntimeError):
    """音源を読めない・分離できないとき。"""


# ---------------------------------------------------------------- 音声の読み込み


def _ffmpeg_to_wav(src: Path, dst: Path) -> None:
    """soundfile で読めない形式（mp3 / m4a など）を PATH の ffmpeg で wav に直す。"""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SeparationError(f"{src.name} を読むには ffmpeg が要ります（PATH に見つかりません）")
    cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-vn", "-c:a", "pcm_f32le", "-f", "wav", str(dst)]
    proc = subprocess.run(cmd, shell=False, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.exists():
        raise SeparationError(f"ffmpeg で {src.name} を変換できませんでした: {proc.stderr.strip()}")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """音声を (チャンネル, サンプル) の float32 と sample rate で返す。"""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.SoundFileError, RuntimeError):
        with tempfile.TemporaryDirectory(prefix="chordgen_") as tmp:
            wav = Path(tmp) / "decoded.wav"
            _ffmpeg_to_wav(path, wav)
            data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise SeparationError(f"{path.name} に音がありません")
    return np.ascontiguousarray(data.T), int(sr)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- 分離


def _settings() -> dict:
    return {
        "model": DEMUCS_MODEL,
        "split": DEMUCS_SPLIT,
        "segment": DEMUCS_SEGMENT,
        "overlap": DEMUCS_OVERLAP,
        "shifts": DEMUCS_SHIFTS,
    }


def _stem_paths(out_dir: Path) -> dict[str, Path]:
    return {name: out_dir / f"{name}.wav" for name in STEM_NAMES}


def _load_cached(out_dir: Path, source_hash: str) -> Stems | None:
    """同じ音源・同じ設定で分けた結果が out_dir にあれば返す。"""
    manifest = out_dir / MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (info.get("version") != _MANIFEST_VERSION or info.get("source_sha256") != source_hash
            or info.get("settings") != _settings()):
        return None
    paths = _stem_paths(out_dir)
    if not all(p.is_file() for p in paths.values()):
        return None
    return Stems(paths=paths, sample_rate=int(info["sample_rate"]))


def _get_separator(device: str):
    """モデルを読み込んだ Separator を返す（デバイスごとに使い回す）。"""
    sep = _separators.get(device)
    if sep is None:
        try:
            from demucs.api import Separator
        except ImportError as e:
            raise SeparationError("demucs（4.1.0 以降）が入っていません") from e
        sep = Separator(model=DEMUCS_MODEL, device=device, shifts=DEMUCS_SHIFTS,
                        overlap=DEMUCS_OVERLAP, split=DEMUCS_SPLIT, segment=DEMUCS_SEGMENT,
                        progress=False)
        _separators[device] = sep
    return sep


def separate(audio_path: Path, out_dir: Path, device: str) -> Stems:
    """htdemucs で vocals / drums / bass / other に分け、out_dir/<stem>.wav に書く。

    out_dir に同じ音源（内容のハッシュで比べる）を同じ設定で分けた結果があれば再利用する。
    out_dir は曲ごとに分ける想定（別の曲を分けると上書きする）。
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_hash = _file_sha256(audio_path)
    cached = _load_cached(out_dir, source_hash)
    if cached is not None:
        return cached

    import torch

    data, sr = read_audio(audio_path)
    sep = _get_separator(device)
    model_sr = int(sep.samplerate)
    channels = int(sep.audio_channels)

    # チャンネル数をモデルに合わせる（separate_tensor は sample rate が同じだと直さない）
    if data.shape[0] == 1 and channels > 1:
        data = np.repeat(data, channels, axis=0)
    elif data.shape[0] > channels:
        data = data[:channels]
    wav = torch.from_numpy(np.ascontiguousarray(data))

    with torch.no_grad():
        _, stems = sep.separate_tensor(wav, sr)

    manifest = out_dir / MANIFEST_NAME
    manifest.unlink(missing_ok=True)  # 書きかけの結果を再利用しないよう、先に消す
    paths = _stem_paths(out_dir)
    for name, path in paths.items():
        arr = stems[name].detach().cpu().numpy().T.astype(np.float32)
        tmp = path.with_name(path.stem + ".part.wav")
        # 分離した音は 1.0 を超えることがあるので float で書く
        sf.write(str(tmp), arr, model_sr, subtype="FLOAT")
        tmp.replace(path)
    manifest.write_text(json.dumps({
        "version": _MANIFEST_VERSION,
        "source": audio_path.name,
        "source_sha256": source_hash,
        "settings": _settings(),
        "sample_rate": model_sr,
        "stems": {k: v.name for k, v in paths.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return Stems(paths=paths, sample_rate=model_sr)


# ---------------------------------------------------------------- コード認識に入れる音


def make_chord_input(audio_path: Path, stems: Stems, mode: ChordInputMode, out_dir: Path) -> Path:
    """コード認識に入れる音を作り、そのパスを返す。

    original は audio_path をそのまま返す。ほかは MODE_GAINS の倍率でステムを足し、
    ピークを PEAK_TARGET にそろえて out_dir/chord_input_<mode>.wav（16 bit）に書く。
    """
    if mode == "original":
        return Path(audio_path)
    if mode not in MODE_GAINS:
        raise ValueError(f"未対応の mode です: {mode}")

    mix: np.ndarray | None = None
    for name, gain in MODE_GAINS[mode].items():
        data, sr = sf.read(str(stems.paths[name]), dtype="float32", always_2d=True)
        if sr != stems.sample_rate:
            raise SeparationError(f"{name} の sample rate が {sr} で、{stems.sample_rate} と違います")
        data = data.astype(np.float64) * gain
        if mix is None:
            mix = data
        else:
            n = min(len(mix), len(data))
            if mix.shape[1] != data.shape[1]:  # チャンネル数が違えばモノラルにそろえる
                mix = mix.mean(axis=1, keepdims=True)
                data = data.mean(axis=1, keepdims=True)
            mix = mix[:n] + data[:n]
    assert mix is not None

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-9:
        mix *= PEAK_TARGET / peak

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chord_input_{mode}.wav"
    tmp = out.with_name(out.stem + ".part.wav")
    sf.write(str(tmp), mix.astype(np.float32), stems.sample_rate, subtype="PCM_16")
    tmp.replace(out)
    return out
