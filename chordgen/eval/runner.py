"""データセットの読み込み、重い処理の実行とキャッシュ、条件ごとの採点。

重い処理（分離・ビート・コード認識・ベース・キー）は条件の間で使い回し、曲ごとに
work_dir()/eval/cache/<曲名>-<ハッシュ>/ へ JSON で保存する。音源の大きさか更新時刻が
変わったら作り直す。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from chordgen.eval.conditions import Condition, apply_condition
from chordgen.eval.labfile import read_lab
from chordgen.eval.metrics import SongScore, score_song
from chordgen.types import BassNote, BeatGrid, ChordSegment, KeyEstimate, Stems

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus")
CACHE_VERSION = 1


@dataclass
class Song:
    name: str
    audio: Path
    lab: Path


def find_songs(dataset: Path) -> tuple[list[Song], list[str]]:
    """フォルダの中の「曲名.<音源>」と「曲名.lab」の組。返り値の 2 つ目は警告。"""
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"フォルダがありません: {dataset}")
    warnings: list[str] = []
    audios: dict[str, Path] = {}
    for p in sorted(dataset.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            if p.stem in audios:
                warnings.append(f"同じ名前の音源が複数あります。{audios[p.stem].name} を使います: {p.name}")
                continue
            audios[p.stem] = p
    labs = {p.stem: p for p in sorted(dataset.iterdir()) if p.is_file() and p.suffix.lower() == ".lab"}
    songs = [Song(name, audios[name], labs[name]) for name in sorted(audios) if name in labs]
    for name in sorted(set(audios) - set(labs)):
        warnings.append(f"正解の .lab がないので飛ばします: {audios[name].name}")
    for name in sorted(set(labs) - set(audios)):
        warnings.append(f"音源がないので飛ばします: {labs[name].name}")
    return songs, warnings


# ---------------------------------------------------------------------------
# JSON との変換
# ---------------------------------------------------------------------------


def _dump(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    return obj


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_dump(data), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chords_to_json(chords: list[ChordSegment]) -> list[dict]:
    return [asdict(c) for c in chords]


def chords_from_json(data: list[dict]) -> list[ChordSegment]:
    return [ChordSegment(**d) for d in data]


# ---------------------------------------------------------------------------
# 曲ごとのキャッシュ
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:60] or "song"


class SongCache:
    """1 曲ぶんの重い処理の結果。必要になったときに計算し、JSON に保存する。"""

    def __init__(self, song: Song, root: Path, device: str, log: Callable[[str], None] = print):
        self.song = song
        self.device = device
        self.log = log
        h = hashlib.sha1(str(song.audio.resolve()).encode("utf-8")).hexdigest()[:8]
        self.dir = Path(root) / f"{_slug(song.name)}-{h}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Any] = {}
        self._check_manifest()

    def _check_manifest(self) -> None:
        st = self.song.audio.stat()
        manifest = {"version": CACHE_VERSION, "audio": str(self.song.audio.resolve()),
                    "size": st.st_size, "mtime": st.st_mtime}
        path = self.dir / "manifest.json"
        if path.exists():
            try:
                if _read_json(path) == manifest:
                    return
            except (OSError, ValueError):
                pass
            self.log(f"  [{self.song.name}] 音源が変わったのでキャッシュを作り直します")
        self.clear()
        _write_json(path, manifest)

    def clear(self) -> None:
        for p in self.dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

    def _cached(self, name: str, compute: Callable[[], Any], to_json, from_json) -> Any:
        if name in self._mem:
            return self._mem[name]
        path = self.dir / f"{name}.json"
        if path.exists():
            try:
                value = from_json(_read_json(path))
                self._mem[name] = value
                return value
            except (OSError, ValueError, TypeError, KeyError):
                self.log(f"  [{self.song.name}] キャッシュが読めないので作り直します: {path.name}")
        self.log(f"  [{self.song.name}] {name} を計算します")
        value = compute()
        _write_json(path, to_json(value))
        self._mem[name] = value
        return value

    def stems(self) -> Stems:
        def compute() -> Stems:
            from chordgen.separation import separate

            return separate(self.song.audio, self.dir / "stems", self.device)

        def from_json(d: dict) -> Stems:
            s = Stems(paths={k: Path(v) for k, v in d["paths"].items()}, sample_rate=int(d["sample_rate"]))
            if not all(p.exists() for p in s.paths.values()):
                raise ValueError("分離した音のファイルがありません")
            return s

        return self._cached("stems", compute, asdict, from_json)

    def chord_input(self, mode: str) -> Path:
        # 原音はそのまま使う（分離を待たずに済むように make_chord_input を通さない）
        if mode == "original":
            return self.song.audio

        def compute() -> dict:
            from chordgen.separation import make_chord_input

            return {"path": make_chord_input(self.song.audio, self.stems(), mode, self.dir / "inputs")}

        def from_json(d: dict) -> dict:
            if not Path(d["path"]).exists():
                raise ValueError("コード認識に入れる音のファイルがありません")
            return {"path": Path(d["path"])}

        return Path(self._cached(f"input_{mode}", compute, lambda d: d, from_json)["path"])

    def beats(self) -> BeatGrid:
        def compute() -> BeatGrid:
            from chordgen.beats import detect_beats

            return detect_beats(self.song.audio, self.device)

        return self._cached("beats", compute, asdict, lambda d: BeatGrid(**d))

    def chords(self, mode: str, model: str) -> list[ChordSegment]:
        def compute() -> list[ChordSegment]:
            from chordgen.chords import recognize_chords

            return recognize_chords(self.chord_input(mode), self.device, model=model)

        return self._cached(f"chords_{model}_{mode}", compute, chords_to_json, chords_from_json)

    def bass(self) -> list[BassNote]:
        def compute() -> list[BassNote]:
            from chordgen.bass import detect_bass

            return detect_bass(self.stems().paths["bass"], self.beats())

        return self._cached("bass", compute, lambda v: [asdict(b) for b in v],
                            lambda d: [BassNote(**b) for b in d])

    def key(self, mode: str, model: str) -> KeyEstimate:
        # キーは原音から、そのモデル・入力で認識したコードも渡して推定する
        def compute() -> KeyEstimate:
            from chordgen.key import estimate_key

            return estimate_key(self.song.audio, self.chords(mode, model))

        return self._cached(f"key_{model}_{mode}", compute, asdict, lambda d: KeyEstimate(**d))


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------


@dataclass
class RunOutput:
    results: dict[Condition, dict[str, SongScore]]
    predictions: dict[Condition, dict[str, list[ChordSegment]]]
    failures: dict[Condition, str]  # 条件 → 失敗の理由（どれかの曲で失敗した条件は結果から除く）


def run(
    songs: list[Song],
    conditions: list[Condition],
    cache_root: Path,
    device: str,
    log: Callable[[str], None] = print,
) -> RunOutput:
    results: dict[Condition, dict[str, SongScore]] = {c: {} for c in conditions}
    predictions: dict[Condition, dict[str, list[ChordSegment]]] = {c: {} for c in conditions}
    failures: dict[Condition, str] = {}
    for song in songs:
        log(f"[{song.name}]")
        reference = read_lab(song.lab)
        cache = SongCache(song, cache_root, device, log)
        # 重い処理の失敗は（入力, モデル）などの単位で覚えて、同じ原因で何度も試さない
        errors: dict[str, str] = {}

        def get(name: str, fn: Callable[[], Any]) -> Any:
            if name in errors:
                raise RuntimeError(errors[name])
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - 1 つの処理の失敗で全体を止めない
                errors[name] = f"{name}: {type(e).__name__}: {e}"
                raise RuntimeError(errors[name]) from e

        for cond in conditions:
            if cond in failures:
                continue
            try:
                chords = get(f"chords_{cond.model}_{cond.input}", lambda: cache.chords(cond.input, cond.model))
                beats = get("beats", cache.beats) if cond.snap != "off" else None
                bass = get("bass", cache.bass) if cond.bass != "off" else None
                key = (get(f"key_{cond.model}_{cond.input}", lambda: cache.key(cond.input, cond.model))
                       if cond.key != "off" else None)
                pred = apply_condition(cond, chords, beats, bass, key)
                results[cond][song.name] = score_song(reference, pred)
                predictions[cond][song.name] = pred
            except Exception as e:  # noqa: BLE001
                reason = f"[{song.name}] {e}"
                if reason not in failures.values():  # 同じ原因は 1 度だけ出す
                    log(f"  失敗: {e}（この処理を使う条件は結果から除きます）")
                failures[cond] = reason
    for cond in failures:
        results.pop(cond, None)
        predictions.pop(cond, None)
    return RunOutput(results=results, predictions=predictions, failures=failures)


def stderr_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
