"""output フォルダの .cat4g.json の一覧と、名前からパスへの安全な解決。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from chordgen.gui.jobs import now_iso

SUFFIX = ".cat4g.json"
MAX_LIST_BYTES = 16 * 1024 * 1024  # これより大きいファイルは一覧で読まない（1 曲ファイルの上限は 5 MB）


class BadName(ValueError):
    pass


def resolve_output(out_dir: Path, name: str) -> Path:
    """ファイル名だけを受け付け、out_dir の中に解決されることを確かめる。

    区切り文字・ドライブ・: （NTFS の代替データストリーム）・. で始まる名前（.. を含む）を弾く。
    在るかどうかは見ない。
    """
    if not isinstance(name, str) or not name or len(name) > 255:
        raise BadName(name)
    if any(c in name for c in ("/", "\\", ":", "\x00")) or name.startswith(".") or name in (".", ".."):
        raise BadName(name)
    if any(ord(c) < 0x20 for c in name):
        raise BadName(name)
    if not name.lower().endswith(SUFFIX) or name.lower() == SUFFIX:
        raise BadName(name)
    base = out_dir.resolve()
    p = (base / name).resolve()
    if p.parent != base or p.name != name:
        raise BadName(name)
    return p


def _chord_count(song: dict[str, Any]) -> int:
    n = 0
    for sec in song.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        for line in sec.get("lines") or []:
            if isinstance(line, dict) and isinstance(line.get("chords"), list):
                n += len(line["chords"])
    return n


def summarize(path: Path) -> dict[str, Any] | None:
    """一覧の 1 件。読めないファイルは None。"""
    try:
        st = path.stat()
        if st.st_size > MAX_LIST_BYTES:
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("song"), dict):
        return None
    song = doc["song"]
    audio = song.get("audio") if isinstance(song.get("audio"), dict) else {}
    return {
        "name": path.name,
        "title": song.get("title"),
        "artist": song.get("artist"),
        "originalKey": song.get("originalKey"),
        "bpm": song.get("bpm"),
        "durationSec": audio.get("durationSec"),
        "chordCount": _chord_count(song),
        "modifiedAt": now_iso(st.st_mtime),
        "size": st.st_size,
    }


def list_outputs(out_dir: Path) -> list[dict[str, Any]]:
    """新しい順。"""
    items: list[tuple[float, dict[str, Any]]] = []
    try:
        paths = [p for p in out_dir.iterdir() if p.name.lower().endswith(SUFFIX) and p.is_file()]
    except OSError:
        return []
    for p in paths:
        s = summarize(p)
        if s is None:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        items.append((mtime, s))
    items.sort(key=lambda x: (x[0], x[1]["name"]), reverse=True)
    return [s for _, s in items]


def content_disposition(name: str, kind: str = "attachment") -> str:
    """RFC 6266 / 5987。ASCII でない名前は filename*（UTF-8）で渡し、filename には置き換えた名前を入れる。"""
    fallback = "".join(c if 0x20 <= ord(c) < 0x7F and c not in '"\\;%' else "_" for c in name)
    value = f'{kind}; filename="{fallback}"'
    if fallback != name:
        value += f"; filename*=UTF-8''{quote(name, safe='')}"
    return value
