""".lab ファイルの読み書きと、cat4g の表記から Harte 記法への変換（純粋関数）。

.lab は 1 行 1 区間で「開始秒 終了秒 ラベル」（空白かタブで区切る）。空行と # で始まる行は読み飛ばす。
ラベルは Harte 記法（"A:min7"、"C:maj/3"、"N"）でも cat4g の表記（"Am7"、"C/E"、"N.C."）でもよい。
"""

from __future__ import annotations

import re
from pathlib import Path

from chordgen.postprocess import degree_to_semitone, note_to_pc, parse_harte
from chordgen.types import ChordSegment

# cat4g の種類の書き方 → Harte の種類。長いものから順に照合する
_CAT4G_QUALITIES: dict[str, str] = {
    "": "maj",
    "M": "maj",
    "maj": "maj",
    "m": "min",
    "min": "min",
    "-": "min",
    "dim": "dim",
    "°": "dim",
    "o": "dim",
    "m-5": "dim",
    "mb5": "dim",
    "m(b5)": "dim",
    "m(-5)": "dim",
    "aug": "aug",
    "+": "aug",
    "+5": "aug",
    "(#5)": "aug",
    "7": "7",
    "M7": "maj7",
    "maj7": "maj7",
    "Maj7": "maj7",
    "△7": "maj7",
    "Δ7": "maj7",
    "△": "maj7",
    "m7": "min7",
    "min7": "min7",
    "-7": "min7",
    "mM7": "minmaj7",
    "mmaj7": "minmaj7",
    "mMaj7": "minmaj7",
    "m(maj7)": "minmaj7",
    "m(M7)": "minmaj7",
    "m△7": "minmaj7",
    "dim7": "dim7",
    "°7": "dim7",
    "o7": "dim7",
    "m7-5": "hdim7",
    "m7b5": "hdim7",
    "m7(b5)": "hdim7",
    "m7(-5)": "hdim7",
    "ø": "hdim7",
    "ø7": "hdim7",
    "6": "maj6",
    "M6": "maj6",
    "m6": "min6",
    "9": "9",
    "M9": "maj9",
    "maj9": "maj9",
    "m9": "min9",
    "11": "11",
    "m11": "min11",
    "13": "13",
    "M13": "maj13",
    "maj13": "maj13",
    "m13": "min13",
    "sus4": "sus4",
    "sus": "sus4",
    "sus2": "sus2",
    "7sus4": "sus4(b7)",
    "7sus": "sus4(b7)",
    "add9": "maj(9)",
    "add2": "maj(2)",
    "madd9": "min(9)",
    "5": "5",
    "aug7": "aug(b7)",
    "7+5": "aug(b7)",
    "7#5": "aug(b7)",
    "7-5": "7(*5,b5)",
    "7b5": "7(*5,b5)",
}
_CAT4G_RE = re.compile(r"^([A-G][#b♯♭]?)(.*?)(?:/([A-G][#b♯♭]?))?$")
_TENSION_RE = re.compile(r"^(.*?)\(([^()]*)\)$")
_HARTE_LIKE_RE = re.compile(r"^[A-G](?:b*|#*)(?:/(?:b*|#*)\d+)?$")
_NO_CHORD = {"N", "N.C.", "NC", "N.C", "-"}


def _interval_degree(semi: int) -> str:
    return ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")[semi % 12]


def _tension_degree(text: str) -> str:
    """cat4g のテンション（"b9"、"-9"、"+11"、"#11"、"13"）→ Harte の度数。"""
    t = text.strip().replace("♭", "b").replace("♯", "#")
    if t.startswith("-"):
        t = "b" + t[1:]
    elif t.startswith("+"):
        t = "#" + t[1:]
    degree_to_semitone(t)  # 形の確認
    return t


def cat4g_to_harte(name: str) -> str:
    """cat4g の表記（"Am7"、"C/E"、"F#m7-5"、"G7(b9)"）を Harte 記法にする。

    "N.C." など無音は "N"、"X" は "X"。読めなければ ValueError。
    """
    s = name.strip().replace("♯", "#").replace("♭", "b")
    if s in _NO_CHORD:
        return "N"
    if s == "X":
        return "X"
    m = _CAT4G_RE.match(s)
    if not m:
        raise ValueError(f"コード名として読めません: {name!r}")
    root, suffix, bass = m.group(1), m.group(2), m.group(3)
    if suffix in _CAT4G_QUALITIES:
        quality = _CAT4G_QUALITIES[suffix]
    else:
        tm = _TENSION_RE.match(suffix)
        if not tm or tm.group(1) not in _CAT4G_QUALITIES or "(" in _CAT4G_QUALITIES[tm.group(1)]:
            raise ValueError(f"コードの種類が読めません: {name!r}")
        tensions = [_tension_degree(t) for t in tm.group(2).split(",") if t.strip()]
        quality = f"{_CAT4G_QUALITIES[tm.group(1)]}({','.join(tensions)})"
    label = f"{root}:{quality}"
    if bass is not None:
        semi = (note_to_pc(bass) - note_to_pc(root)) % 12
        if semi != 0:
            chord = parse_harte(label)
            # 構成音の度数があればその書き方（"#5"、"bb7" など）を使う
            deg = next((d for d in chord.degrees if degree_to_semitone(d) == semi), _interval_degree(semi))
            label += f"/{deg}"
    parse_harte(label)  # 形の確認
    return label


def to_harte(label: str) -> str:
    """Harte でも cat4g の表記でも受け取り、Harte 記法にする。"""
    s = label.strip()
    if s in ("N", "X"):
        return s
    if ":" in s or _HARTE_LIKE_RE.match(s):
        parse_harte(s)
        return s
    return cat4g_to_harte(s)


def parse_lab(text: str) -> list[ChordSegment]:
    """.lab の中身を区間の並びにする（開始秒の順）。"""
    segs: list[ChordSegment] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"{lineno} 行目: 「開始秒 終了秒 ラベル」の形ではありません: {raw!r}")
        try:
            start, end = float(parts[0]), float(parts[1])
            label = to_harte(parts[2])
        except ValueError as e:
            raise ValueError(f"{lineno} 行目: {e}") from e
        if end < start:
            raise ValueError(f"{lineno} 行目: 終了秒が開始秒より前です: {raw!r}")
        if end > start:
            segs.append(ChordSegment(start=start, end=end, label=label))
    segs.sort(key=lambda s: s.start)
    return segs


def read_lab(path: Path) -> list[ChordSegment]:
    """.lab ファイルを読む（UTF-8。BOM があってもよい）。"""
    return parse_lab(Path(path).read_text(encoding="utf-8-sig"))


def format_lab(segments: list[ChordSegment]) -> str:
    return "".join(f"{s.start:.6f}\t{s.end:.6f}\t{s.label}\n" for s in segments)


def write_lab(path: Path, segments: list[ChordSegment]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_lab(segments), encoding="utf-8")
