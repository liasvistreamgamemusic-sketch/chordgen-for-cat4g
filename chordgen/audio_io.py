"""音源ファイルの情報の取得と読み込み、題名・アーティストの確定。

- 長さとサンプルレート: soundfile → mutagen → ffprobe の順に試す
- タグ: mutagen（ID3 / MP4 / Vorbis / APE / ASF、WAV の id3 チャンク）、WAV の LIST/INFO は自前で読む
- 読み込み: soundfile で読めない形式は ffmpeg で f32le に直して標準出力から受け取る
- リサンプル: scipy があれば resample_poly、なければ numpy の FFT（区間ごと）で行う

ffmpeg / ffprobe は media-dl と共有する。MEDIA_DL_FFMPEG_DIR → MEDIA_DL_BIN_DIR → PATH の順に探す。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from chordgen.types import AudioInfo, TrackMeta

DEFAULT_TITLE = "無題"


class AudioError(RuntimeError):
    """音源を読めない。"""


# ---- ffmpeg / ffprobe ----


def find_ffmpeg_tool(name: str) -> str | None:
    """ffmpeg / ffprobe の実行ファイルを探す。見つからなければ None。"""
    exe = f"{name}.exe" if os.name == "nt" else name
    for env in ("MEDIA_DL_FFMPEG_DIR", "MEDIA_DL_BIN_DIR"):
        d = os.environ.get(env, "").strip()
        if d and (Path(d) / exe).is_file():
            return str(Path(d) / exe)
    return shutil.which(name)


def _ffprobe(path: Path) -> tuple[float, int] | None:
    ffprobe = find_ffmpeg_tool("ffprobe")
    if not ffprobe:
        return None
    args = [
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,duration:format=duration",
        "-of", "json", str(path),
    ]  # fmt: skip
    try:
        res = subprocess.run(args, capture_output=True, check=False, shell=False, timeout=60)
        data = json.loads(res.stdout.decode("utf-8", errors="replace") or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    streams = data.get("streams") or []
    if res.returncode != 0 or not streams:
        return None
    try:
        sr = int(streams[0].get("sample_rate") or 0)
        dur = float(streams[0].get("duration") or (data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        return None
    return (dur, sr) if sr > 0 and dur > 0 else None


def _ffmpeg_decode(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    if not ffmpeg:
        raise AudioError(
            f"{path.name} は soundfile で読めない形式です。ffmpeg が見つからないので変換できません"
            "（`media-dl setup` で入れるか、PATH に追加してください）。"
        )
    args = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]  # fmt: skip
    try:
        res = subprocess.run(args, capture_output=True, check=False, shell=False)
    except OSError as e:
        raise AudioError(f"ffmpeg を起動できませんでした: {e}") from e
    if res.returncode != 0:
        msg = res.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg で {path.name} を読めませんでした: {msg}")
    n = len(res.stdout) // 4
    return np.frombuffer(res.stdout[: n * 4], dtype="<f4").astype(np.float32)


# ---- 長さ・サンプルレート ----


def _probe_stream(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / info.samplerate, int(info.samplerate)
    except Exception:  # libsndfile が読めない形式
        pass
    try:
        import mutagen

        mf = mutagen.File(str(path))
        if mf is not None and mf.info is not None:
            length = float(getattr(mf.info, "length", 0) or 0)
            sr = int(getattr(mf.info, "sample_rate", 0) or 0)
            if length > 0 and sr > 0:
                return length, sr
    except Exception:
        pass
    got = _ffprobe(path)
    if got:
        return got
    raise AudioError(f"音源の長さとサンプルレートが分かりません: {path}")


# ---- タグ ----


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).replace("\x00", "").strip()
    return s or None


def _tag_text(tags: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        try:
            v = tags.get(key)
        except Exception:
            v = None
        if v is None:
            continue
        items = v.text if hasattr(v, "text") else v if isinstance(v, list) else [v]
        texts = [t for t in (_clean(i) for i in items) if t]
        if texts:
            return ", ".join(dict.fromkeys(texts))
    return None


# 形式ごとのタグのキー（題名、アーティスト）
_TITLE_KEYS = ("TIT2", "\xa9nam", "title", "Title", "TITLE")
_ARTIST_KEYS = ("TPE1", "\xa9ART", "artist", "Artist", "ARTIST", "Author", "TPE2", "aART", "albumartist", "WM/AlbumArtist")


def _mutagen_tags(path: Path) -> tuple[str | None, str | None]:
    try:
        import mutagen

        mf = mutagen.File(str(path))
    except Exception:
        return None, None
    tags = getattr(mf, "tags", None) if mf is not None else None
    if not tags:
        return None, None
    return _tag_text(tags, _TITLE_KEYS), _tag_text(tags, _ARTIST_KEYS)


def _decode_info(raw: bytes) -> str | None:
    raw = raw.split(b"\x00", 1)[0]
    for enc in ("utf-8", "cp932"):
        try:
            return _clean(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return _clean(raw.decode("latin-1"))


def read_riff_info(path: Path) -> dict[str, str]:
    """WAV の LIST/INFO チャンク（INAM=題名、IART=アーティストなど）を読む。"""
    out: dict[str, str] = {}
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return out
            while True:
                h = f.read(8)
                if len(h) < 8:
                    break
                cid, size = h[:4], int.from_bytes(h[4:], "little")
                if cid == b"LIST" and size <= 1 << 20:
                    body = f.read(size + (size & 1))
                    if body[:4] == b"INFO":
                        pos = 4
                        while pos + 8 <= len(body):
                            sid = body[pos : pos + 4].decode("ascii", errors="replace")
                            ssize = int.from_bytes(body[pos + 4 : pos + 8], "little")
                            text = _decode_info(body[pos + 8 : pos + 8 + ssize])
                            if text:
                                out.setdefault(sid, text)
                            pos += 8 + ssize + (ssize & 1)
                else:
                    f.seek(size + (size & 1), os.SEEK_CUR)
    except OSError:
        pass
    return out


def read_tags(path: Path) -> TrackMeta:
    """音源のタグから題名とアーティストを取る。なければ None。"""
    path = Path(path)
    title, artist = _mutagen_tags(path)
    if not (title and artist):
        info = read_riff_info(path)
        title = title or info.get("INAM")
        artist = artist or info.get("IART")
    if not (title and artist):
        try:
            with sf.SoundFile(str(path)) as f:
                md = f.copy_metadata()
            title = title or _clean(md.get("title"))
            artist = artist or _clean(md.get("artist"))
        except Exception:
            pass
    return TrackMeta(title=title, artist=artist)


def meta_from_filename(path: Path) -> TrackMeta:
    """`アーティスト - 題名.ext` なら分け、そうでなければ拡張子を除いた名前を題名にする。"""
    stem = Path(path).stem.strip()
    if " - " in stem:
        artist, title = (s.strip() for s in stem.split(" - ", 1))
        if artist and title:
            return TrackMeta(title=title, artist=artist)
    return TrackMeta(title=stem or None)


# ---- 公開関数 ----


def probe_audio(path: Path) -> AudioInfo:
    """長さ・サンプルレートと、タグ（なければファイル名）の題名・アーティストを返す。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"音源ファイルが見つかりません: {path}")
    duration, sr = _probe_stream(path)
    tags = read_tags(path)
    fn = meta_from_filename(path)
    meta = TrackMeta(title=tags.title or fn.title, artist=tags.artist or fn.artist)
    return AudioInfo(path=path, duration_sec=float(duration), sample_rate=int(sr), meta=meta)


def _resample_fft(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """FFT で帯域を制限してリサンプルする（区間ごと、前後に余白を付けて継ぎ目を消す）。"""
    n_out = math.ceil(len(x) * up / down)
    k = max(1, math.ceil(480_000 / down))  # 1 区間の入力は down * k（48 kHz で約 10 秒）サンプル
    m = max(1, math.ceil(24_000 / down))  # 前後の余白は down * m（同じく約 0.5 秒）サンプル
    block, margin = down * k, down * m
    n_blocks = max(1, math.ceil(len(x) / block))
    padded = np.zeros(n_blocks * block + 2 * margin, dtype=np.float64)
    padded[margin : margin + len(x)] = x
    seg_in = block + 2 * margin
    seg_out = seg_in * up // down
    out = np.empty(n_blocks * up * k, dtype=np.float32)
    for i in range(n_blocks):
        seg = padded[i * block : i * block + seg_in]
        spec = np.fft.rfft(seg)
        bins = seg_out // 2 + 1
        res = np.zeros(bins, dtype=np.complex128)
        keep = min(bins, len(spec))
        res[:keep] = spec[:keep]
        y = np.fft.irfft(res, seg_out) * (seg_out / seg_in)
        out[i * up * k : (i + 1) * up * k] = y[up * m : up * m + up * k]
    return out[:n_out]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """モノラルの信号を sr_in から sr_out に直す。"""
    if sr_in == sr_out or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return _resample_fft(np.asarray(x, dtype=np.float64), up, down)
    return resample_poly(np.asarray(x, dtype=np.float64), up, down).astype(np.float32)


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """float32 のモノラルで読み、sample_rate に直す。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"音源ファイルが見つかりません: {path}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate が正しくありません: {sample_rate}")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return _ffmpeg_decode(path, sample_rate)
    mono = data.mean(axis=1, dtype=np.float32) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(resample(mono, int(sr), int(sample_rate)), dtype=np.float32)


def _first(*values: str | None) -> str | None:
    for v in values:
        s = _clean(v)
        if s:
            return s
    return None


def resolve_meta(*, user: TrackMeta, fetched: TrackMeta | None, audio: AudioInfo) -> TrackMeta:
    """入力 → URL から取得した情報 → 音源のタグ → ファイル名 の順で題名とアーティストを決める。

    項目ごとに最初に値のあるものを使う。題名は必ず 1 文字以上になる。
    """
    f = fetched or TrackMeta()
    fn = meta_from_filename(audio.path)
    title = _first(user.title, f.title, audio.meta.title, fn.title) or DEFAULT_TITLE
    artist = _first(user.artist, f.artist, audio.meta.artist, fn.artist)
    source_url = _first(user.source_url, f.source_url, audio.meta.source_url)
    return TrackMeta(title=title, artist=artist, source_url=source_url)
