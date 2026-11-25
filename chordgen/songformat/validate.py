"""CaT4G Song Format v1 の文書を検証して書き出す。

JSON Schema（chordgen.paths.schema_path()）に加えて、仕様 14 章の「アプリだけが検査すること」
のうち作る側で守れるもの（extensions の入れ子 64 段、exportedAt が日時として読めること、
1 曲ファイル 5 MB まで）と、NaN・Infinity・JSON にない型、作る側が書かない cat4g. のキーを調べる。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from chordgen.paths import schema_path

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_EXTENSION_DEPTH = 64


class SongFormatError(ValueError):
    """文書が形式に合わない。errors に「場所: 理由」の一覧を持つ。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        shown = "\n".join(f"- {e}" for e in errors[:20])
        more = f"\n…ほか {len(errors) - 20} 件" if len(errors) > 20 else ""
        super().__init__(f"CaT4G Song Format v1 の検証に通りません（{len(errors)} 件）\n{shown}{more}")


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft7Validator

    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def _path_str(parts) -> str:
    out = ""
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}" if out else str(p)
    return out or "(最上位)"


def _walk_json(doc: Any) -> Iterator[str]:
    """JSON にない値（NaN・Infinity・文字列でないキー・dict/list 以外の入れ物など）を探す。"""
    stack: list[tuple[Any, tuple]] = [(doc, ())]
    while stack:
        value, path = stack.pop()
        if value is None or isinstance(value, (bool, str, int)):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                yield f"{_path_str(path)}: 有限の数ではありません（{value}）"
            continue
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    yield f"{_path_str(path)}: キーが文字列ではありません（{k!r}）"
                    continue
                stack.append((v, path + (k,)))
            continue
        if isinstance(value, list):
            stack.extend((v, path + (i,)) for i, v in enumerate(value))
            continue
        yield f"{_path_str(path)}: JSON の値ではありません（{type(value).__name__}）"


def _depth(value: Any) -> int:
    """オブジェクト・配列の入れ子の段数（value 自体を 1 段目とする）。"""
    best = 0
    stack = [(value, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, dict):
            best = max(best, d)
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, list):
            best = max(best, d)
            stack.extend((x, d + 1) for x in v)
    return best


def _extensions(doc: dict) -> Iterator[tuple[tuple, Any]]:
    song = doc.get("song")
    if not isinstance(song, dict):
        return
    if "extensions" in song:
        yield ("song", "extensions"), song["extensions"]
    for si, sec in enumerate(song.get("sections") or []):
        if not isinstance(sec, dict):
            continue
        base = ("song", "sections", si)
        if "extensions" in sec:
            yield base + ("extensions",), sec["extensions"]
        for li, line in enumerate(sec.get("lines") or []):
            if not isinstance(line, dict):
                continue
            if "extensions" in line:
                yield base + ("lines", li, "extensions"), line["extensions"]
            for ci, chord in enumerate(line.get("chords") or []):
                if isinstance(chord, dict) and "extensions" in chord:
                    yield base + ("lines", li, "chords", ci, "extensions"), chord["extensions"]


def _app_checks(doc: dict) -> Iterator[str]:
    exported = doc.get("exportedAt")
    if isinstance(exported, str):
        try:
            datetime.fromisoformat(exported.replace("Z", "+00:00"))
        except ValueError:
            yield f"exportedAt: 日時として読めません（{exported}）"
    for path, ext in _extensions(doc):
        depth = _depth(ext)
        if depth > MAX_EXTENSION_DEPTH:
            yield f"{_path_str(path)}: 入れ子が {MAX_EXTENSION_DEPTH} 段を超えています（{depth} 段）"
        if isinstance(ext, dict):
            for k in ext:
                if isinstance(k, str) and k.startswith("cat4g."):
                    yield f"{_path_str(path + (k,))}: cat4g. で始まるキーはアプリが使うので書けません"


def _encode(doc: dict) -> bytes:
    text = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2)
    return (text + "\n").encode("utf-8")


def validate_document(doc: dict) -> None:
    """文書を検証する。通らなければ SongFormatError。"""
    if not isinstance(doc, dict):
        raise SongFormatError(["(最上位): オブジェクトではありません"])
    errors = list(_walk_json(doc))
    if errors:  # JSON にならない値があると、以降の検査は意味がない
        raise SongFormatError(errors)
    for err in sorted(_validator().iter_errors(doc), key=lambda e: list(map(str, e.absolute_path))):
        errors.append(f"{_path_str(err.absolute_path)}: {err.message}")
    errors.extend(_app_checks(doc))
    if not errors:
        size = len(_encode(doc))
        if size > MAX_FILE_BYTES:
            errors.append(f"(ファイル): {size} バイトで、1 曲ファイルの上限 5 MB を超えます")
    if errors:
        raise SongFormatError(errors)


def write_document(doc: dict, path: Path) -> None:
    """検証してから UTF-8 の JSON で書く。通らなければ何も書かない。"""
    validate_document(doc)
    data = _encode(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 途中で失敗しても壊れたファイルを残さないよう、同じフォルダに書いてから置き換える
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
