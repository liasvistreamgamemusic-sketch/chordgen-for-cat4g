"""chordgen.audio_io のテスト。音源は numpy で作って tmp に書く。"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chordgen import audio_io
from chordgen.audio_io import load_mono, meta_from_filename, probe_audio, read_riff_info, resolve_meta
from chordgen.types import AudioInfo, TrackMeta


def sine(freq: float, sr: int, sec: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_16") -> Path:
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def peak_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * sr / len(x))


def add_riff_info(path: Path, fields: dict[bytes, bytes]) -> None:
    """WAV の末尾に LIST/INFO チャンクを足し、RIFF の長さを直す。"""
    body = b"INFO"
    for k, v in fields.items():
        v = v + b"\x00"
        body += k + struct.pack("<I", len(v)) + v + (b"\x00" if len(v) & 1 else b"")
    chunk = b"LIST" + struct.pack("<I", len(body)) + body
    raw = bytearray(path.read_bytes()) + chunk
    raw[4:8] = struct.pack("<I", len(raw) - 8)
    path.write_bytes(bytes(raw))


# ---- probe_audio ----


def test_probe_wav_duration_and_filename_title(tmp_path):
    p = write_wav(tmp_path / "テスト曲.wav", sine(440, 44100, 2.0), 44100)
    info = probe_audio(p)
    assert info.path == p
    assert info.sample_rate == 44100
    assert info.duration_sec == pytest.approx(2.0, abs=1e-3)
    assert info.meta.title == "テスト曲"
    assert info.meta.artist is None


def test_probe_filename_artist_title(tmp_path):
    p = write_wav(tmp_path / "歌手 - 題名 - Remix.wav", sine(440, 22050, 0.5), 22050)
    info = probe_audio(p)
    assert info.meta.artist == "歌手"
    assert info.meta.title == "題名 - Remix"


def test_meta_from_filename_edge_cases():
    assert meta_from_filename(Path("a.b.c.wav")) == TrackMeta(title="a.b.c")
    assert meta_from_filename(Path(" - 題名.wav")) == TrackMeta(title="- 題名")
    assert meta_from_filename(Path("A-B.wav")) == TrackMeta(title="A-B")


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_audio(tmp_path / "none.wav")


def test_probe_unreadable(tmp_path, monkeypatch):
    p = tmp_path / "x.wav"
    p.write_bytes(b"not audio at all" * 10)
    monkeypatch.setattr(audio_io, "find_ffmpeg_tool", lambda name: None)
    with pytest.raises(audio_io.AudioError):
        probe_audio(p)


def test_tags_flac_vorbis(tmp_path):
    from mutagen.flac import FLAC

    p = tmp_path / "file name.flac"
    sf.write(str(p), sine(440, 44100, 1.0), 44100, format="FLAC")
    f = FLAC(str(p))
    f["title"] = "タグの題名"
    f["artist"] = ["歌手A", "歌手B"]
    f.save()
    info = probe_audio(p)
    assert info.meta.title == "タグの題名"
    assert info.meta.artist == "歌手A, 歌手B"
    assert info.duration_sec == pytest.approx(1.0, abs=1e-3)


def test_tags_wav_id3_chunk(tmp_path):
    from mutagen.id3 import TIT2, TPE1
    from mutagen.wave import WAVE

    p = write_wav(tmp_path / "Artist - FileTitle.wav", sine(440, 44100, 1.0), 44100)
    w = WAVE(str(p))
    w.add_tags()
    w.tags.add(TIT2(encoding=3, text="ID3 の題名"))
    w.save()
    info = probe_audio(p)
    assert info.meta.title == "ID3 の題名"
    assert info.meta.artist == "Artist"  # タグになければファイル名から
    w = WAVE(str(p))
    w.tags.add(TPE1(encoding=3, text="ID3 の歌手"))
    w.save()
    assert probe_audio(p).meta.artist == "ID3 の歌手"


def test_tags_wav_riff_info(tmp_path):
    p = write_wav(tmp_path / "x.wav", sine(440, 44100, 0.5), 44100)
    add_riff_info(p, {b"INAM": "日本語の題名".encode(), b"IART": "歌手".encode("cp932")})
    assert read_riff_info(p) == {"INAM": "日本語の題名", "IART": "歌手"}
    info = probe_audio(p)
    assert info.meta.title == "日本語の題名"
    assert info.meta.artist == "歌手"
    assert info.duration_sec == pytest.approx(0.5, abs=1e-3)


# ---- load_mono / リサンプル ----


def test_load_mono_downmix_same_rate(tmp_path):
    left = sine(440, 44100, 1.0, 0.4)
    right = sine(440, 44100, 1.0, 0.2)
    p = write_wav(tmp_path / "st.wav", np.stack([left, right], axis=1), 44100, subtype="FLOAT")
    y = load_mono(p, 44100)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) == 44100
    np.testing.assert_allclose(y, (left + right) / 2, atol=1e-6)


@pytest.mark.parametrize("use_scipy", [True, False])
@pytest.mark.parametrize("sr_in,sr_out", [(44100, 22050), (44100, 16000), (22050, 44100), (48000, 44100)])
def test_load_mono_resample(tmp_path, monkeypatch, use_scipy, sr_in, sr_out):
    if use_scipy:
        pytest.importorskip("scipy")
    else:
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
    sec = 3.0
    p = write_wav(tmp_path / "s.wav", sine(1000, sr_in, sec), sr_in, subtype="FLOAT")
    y = load_mono(p, sr_out)
    assert y.dtype == np.float32
    assert abs(len(y) - sr_out * sec) <= 1
    assert peak_freq(y, sr_out) == pytest.approx(1000, abs=2)
    # 振幅が保たれ、継ぎ目に大きな誤差がない
    mid = y[sr_out // 10 : -sr_out // 10]
    assert np.max(np.abs(mid)) == pytest.approx(0.5, abs=0.01)
    t = np.arange(len(y)) / sr_out
    ref = 0.5 * np.sin(2 * np.pi * 1000 * t)
    err = np.abs(y - ref)[sr_out // 10 : -sr_out // 10]
    assert err.max() < 0.01


def test_resample_fft_removes_alias():
    """ダウンサンプルで新しいナイキストより上の成分を落とす。"""
    sr_in, sr_out = 44100, 16000
    x = sine(12000, sr_in, 2.0) + sine(1000, sr_in, 2.0)
    g = np.gcd(sr_in, sr_out)
    y = audio_io._resample_fft(x.astype(np.float64), sr_out // g, sr_in // g)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr_out)
    alias = spec[(freqs > 3500) & (freqs < 4500)].max()  # 12 kHz は 16 kHz で 4 kHz に折り返す
    assert alias < spec.max() * 1e-3


def test_load_mono_bad_rate(tmp_path):
    p = write_wav(tmp_path / "s.wav", sine(440, 8000, 0.1), 8000)
    with pytest.raises(ValueError):
        load_mono(p, 0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_load_mono_via_ffmpeg(tmp_path):
    """soundfile で読めない形式（m4a）は ffmpeg で読む。"""
    src = write_wav(tmp_path / "src.wav", sine(440, 44100, 2.0), 44100)
    m4a = tmp_path / "歌手 - 曲.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", "-b:a", "128k", str(m4a)],
        check=True,
    )
    info = probe_audio(m4a)
    assert info.sample_rate == 44100
    assert info.duration_sec == pytest.approx(2.0, abs=0.1)
    assert info.meta == TrackMeta(title="曲", artist="歌手")
    y = load_mono(m4a, 22050)
    assert y.dtype == np.float32
    assert abs(len(y) - 2.0 * 22050) < 2048
    assert peak_freq(y, 22050) == pytest.approx(440, abs=3)


def test_load_mono_without_ffmpeg(tmp_path, monkeypatch):
    p = tmp_path / "x.m4a"
    p.write_bytes(b"\x00" * 100)
    monkeypatch.setattr(audio_io, "find_ffmpeg_tool", lambda name: None)
    with pytest.raises(audio_io.AudioError, match="ffmpeg"):
        load_mono(p, 22050)


def test_find_ffmpeg_tool_prefers_media_dl_dir(tmp_path, monkeypatch):
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    (tmp_path / exe).write_text("")
    monkeypatch.delenv("MEDIA_DL_FFMPEG_DIR", raising=False)
    monkeypatch.setenv("MEDIA_DL_BIN_DIR", str(tmp_path))
    assert audio_io.find_ffmpeg_tool("ffmpeg") == str(tmp_path / exe)


# ---- resolve_meta ----


def audio(path: str = "歌手F - 題名F.wav", title: str | None = "題名T", artist: str | None = "歌手T") -> AudioInfo:
    return AudioInfo(path=Path(path), duration_sec=1.0, sample_rate=44100, meta=TrackMeta(title, artist))


def test_resolve_meta_user_first():
    m = resolve_meta(
        user=TrackMeta("題名U", "歌手U"),
        fetched=TrackMeta("題名W", "歌手W", "https://e.com"),
        audio=audio(),
    )
    assert m == TrackMeta("題名U", "歌手U", "https://e.com")


def test_resolve_meta_fetched_then_tags_then_filename():
    a = audio()
    assert resolve_meta(user=TrackMeta(), fetched=TrackMeta("題名W", "歌手W"), audio=a) == TrackMeta("題名W", "歌手W")
    assert resolve_meta(user=TrackMeta(), fetched=None, audio=a) == TrackMeta("題名T", "歌手T")
    assert resolve_meta(user=TrackMeta(), fetched=None, audio=audio(title=None, artist=None)) == TrackMeta(
        "題名F", "歌手F"
    )


def test_resolve_meta_per_field_and_blank():
    m = resolve_meta(
        user=TrackMeta(title="  ", artist="歌手U"),
        fetched=TrackMeta(title="", artist=None),
        audio=audio(title=None, artist=None),
    )
    assert m.title == "題名F"
    assert m.artist == "歌手U"


def test_resolve_meta_title_never_empty():
    m = resolve_meta(user=TrackMeta(), fetched=None, audio=audio(path=".wav", title=None, artist=None))
    assert m.title and len(m.title) >= 1
    m = resolve_meta(user=TrackMeta(), fetched=TrackMeta(), audio=audio(path=" .wav", title=" ", artist=None))
    assert m.title == audio_io.DEFAULT_TITLE
    assert m.artist is None
