"""パイプラインのテスト用の偽の段（テストはここには置かない）。

各段のモジュールは monkeypatch.setitem で sys.modules に差し込むので、テストが終われば
元の状態（本物が import 済みなら本物、無ければ無し）に戻る。本物のモジュールの属性は触らない。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from chordgen.types import (
    AlignedLine,
    AudioInfo,
    BassNote,
    BeatGrid,
    ChordSegment,
    FetchResult,
    KeyEstimate,
    Stems,
    TrackMeta,
)

STAGE_MODULES = (
    "audio_io", "fetch", "separation", "beats", "chords", "key", "postprocess", "bass", "lyrics", "songformat",
)


class FakeFetchError(Exception):
    """chordgen.fetch.FetchError の偽物。"""

    def __init__(self, code: str, user_message: str, exit_code: int = 1):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.exit_code = exit_code


class FakeStages:
    """呼ばれた順を calls に記録する偽の段。fail に関数名と例外を入れるとその関数が例外を出す。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.fail: dict[str, BaseException] = {}
        self.progress_events: list[dict] = [{"type": "progress", "percent": 50.0}]
        self.written: list[Path] = []

    # ---- 記録 ----
    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def args_of(self, name: str) -> tuple:
        for n, a in self.calls:
            if n == name:
                return a
        raise AssertionError(f"{name} は呼ばれていません: {self.names()}")

    def _hit(self, name: str, *args) -> None:
        self.calls.append((name, args))
        if name in self.fail:
            raise self.fail[name]

    # ---- 偽のモジュールを作る ----
    def modules(self) -> dict[str, types.ModuleType]:
        f = self
        mods: dict[str, types.ModuleType] = {}

        def mod(name: str, **attrs) -> None:
            m = types.ModuleType(f"chordgen.{name}")
            m.__dict__.update(attrs)
            m.__fake__ = True
            mods[name] = m

        def probe_audio(path):
            f._hit("probe_audio", path)
            return AudioInfo(path=Path(path), duration_sec=8.0, sample_rate=44100, meta=TrackMeta(title=Path(path).stem))

        def resolve_meta(*, user, fetched, audio):
            f._hit("resolve_meta", user, fetched, audio)
            srcs = [user, fetched or TrackMeta(), audio.meta]
            pick = lambda attr: next((getattr(s, attr) for s in srcs if getattr(s, attr)), None)  # noqa: E731
            return TrackMeta(title=pick("title"), artist=pick("artist"), source_url=pick("source_url"))

        mod("audio_io", probe_audio=probe_audio, resolve_meta=resolve_meta)

        def is_url(s):
            return s.startswith(("http://", "https://"))

        def fetch_info(url):
            f._hit("fetch_info", url)
            return TrackMeta(title="Fake/Song", artist="Fake Artist", source_url=url)

        def fetch_audio(url, dest_dir, on_progress=None):
            f._hit("fetch_audio", url, dest_dir)
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            p = dest_dir / "fake.wav"
            p.write_bytes(b"RIFF")
            for ev in f.progress_events:
                if on_progress:
                    on_progress(ev)
            return FetchResult(audio_path=p, meta=TrackMeta(title="Fake/Song"))

        mod("fetch", is_url=is_url, fetch_info=fetch_info, fetch_audio=fetch_audio, FetchError=FakeFetchError)

        def separate(audio_path, out_dir, device):
            f._hit("separate", audio_path, out_dir, device)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for n in ("vocals", "drums", "bass", "other"):
                paths[n] = out_dir / f"{n}.wav"
                paths[n].write_bytes(b"RIFF")
            return Stems(paths=paths, sample_rate=44100)

        def make_chord_input(audio_path, stems, mode, out_dir):
            f._hit("make_chord_input", audio_path, stems, mode, out_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return Path(out_dir) / f"{mode}.wav"

        mod("separation", separate=separate, make_chord_input=make_chord_input)

        def detect_beats(audio_path, device):
            f._hit("detect_beats", audio_path, device)
            times = [i * 0.5 for i in range(16)]
            return BeatGrid(beat_times=times, downbeat_indices=[0, 4, 8, 12], bpm=120.0, beats_per_bar=4)

        mod("beats", detect_beats=detect_beats)

        def recognize_chords(audio_path, device, model="chordmini"):
            f._hit("recognize_chords", audio_path, device, model)
            return [
                ChordSegment(0.0, 0.1, "N"),
                ChordSegment(0.1, 2.1, "C:maj", 0.9),
                ChordSegment(2.1, 4.0, "A:min7", 0.8),
                ChordSegment(4.0, 8.0, "G:maj", 0.7),
            ]

        mod("chords", recognize_chords=recognize_chords)

        def estimate_key(audio_path, chords=None):
            f._hit("estimate_key", audio_path, chords)
            return KeyEstimate("C", "major", 0.9)

        mod("key", estimate_key=estimate_key)

        def snap_to_beats(chords, beats):
            f._hit("snap_to_beats", chords, beats)
            return [ChordSegment(round(c.start * 2) / 2, round(c.end * 2) / 2, c.label, c.confidence,
                                 {**c.extras, "snapped": True}) for c in chords]

        def apply_bass(chords, bass, mode):
            f._hit("apply_bass", chords, bass, mode)
            return [ChordSegment(c.start, c.end, c.label, c.confidence, {**c.extras, "bass": mode}) for c in chords]

        mod("postprocess", snap_to_beats=snap_to_beats, apply_bass=apply_bass)

        def detect_bass(bass_path, beats):
            f._hit("detect_bass", bass_path, beats)
            return [BassNote(t, t + 0.5, 0, 0.9) for t in beats.beat_times]

        mod("bass", detect_bass=detect_bass)

        def align_lyrics(audio_path, lyrics_text, device, language=None):
            f._hit("align_lyrics", audio_path, lyrics_text, device, language)
            lines = [ln for ln in lyrics_text.splitlines() if ln.strip()]
            return [AlignedLine(ln, [float(i) for i in range(len(ln))]) for ln in lines]

        mod("lyrics", align_lyrics=align_lyrics)

        def build_document(result, *, bars_per_line=4):
            f._hit("build_document", result, bars_per_line)
            return {
                "format": "cat4g-song", "version": 1,
                "song": {"title": result.meta.title, "extensions": result.extras},
                "chords": [c.label for c in result.chords],
            }

        def validate_document(doc):
            f._hit("validate_document", doc)

        def write_document(doc, path):
            f._hit("write_document", doc, path)
            Path(path).write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            f.written.append(Path(path))

        mod("songformat", build_document=build_document, validate_document=validate_document,
            write_document=write_document)
        return mods

    def install(self, monkeypatch: pytest.MonkeyPatch, exclude: tuple[str, ...] = ()) -> "FakeStages":
        for name, m in self.modules().items():
            if name not in exclude:
                monkeypatch.setitem(sys.modules, f"chordgen.{name}", m)
        return self


@pytest.fixture
def app_home(tmp_path, monkeypatch) -> Path:
    """work / output をテスト用の一時フォルダに向ける。"""
    home = tmp_path / "home"
    monkeypatch.setenv("CHORDGEN_HOME", str(home))
    return home


@pytest.fixture
def make_fakes(monkeypatch, app_home):
    """make_fakes(exclude=(...)) で偽の段を差し込む。exclude の段は本物（あれば）を使う。"""

    def factory(exclude: tuple[str, ...] = ()) -> FakeStages:
        return FakeStages().install(monkeypatch, exclude)

    return factory


@pytest.fixture
def fakes(make_fakes) -> FakeStages:
    return make_fakes()


@pytest.fixture
def audio_file(tmp_path) -> Path:
    """中身は使わない（probe_audio などは偽物）。名前だけの音源ファイル。"""
    p = tmp_path / "in" / "my song.wav"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"RIFF")
    return p
