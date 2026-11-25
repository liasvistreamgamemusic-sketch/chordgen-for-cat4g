"""Harte 記法のコード名を cat4g の表記（仕様書 8 章）に直す。

cat4g の表記: ルート音 + 種類 + 任意の分数ベース（例 "Am7"、"C/E"、"F#m7b5"）。
臨時記号は ASCII の "#" と "b" で書く。
"""

from __future__ import annotations

import re

from chordgen.types import KeyEstimate

# Harte の種類 → cat4g の種類。上の 14 行が BTC の大語彙（170 クラス）の全種類
QUALITY_MAP: dict[str, str] = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "6",
    "min7": "m7",
    "minmaj7": "mmaj7",  # 仕様の種類の書き方（m + maj7）に合わせる
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
    # BTC は出さないが、Harte の略記として出てきうるもの
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "min11": "m11",
    "13": "13",
    "maj13": "maj13",
    "min13": "m13",
    "5": "5",
}

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# 度数（1〜13）→ 半音
_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b♯♭]*)$")
_DEGREE_RE = re.compile(r"^([#b♯♭]*)(\d{1,2})$")

# 調の主音の綴り（ピッチクラス → 名前）
_MAJOR_TONIC = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MINOR_TONIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
# 主音からの半音 → 度数の綴り（ポップスでよく使う書き方: bIII、bVI、bVII、#IV など）
_SCALE_DEGREES = [(1, 0), (2, -1), (2, 0), (3, -1), (3, 0), (4, 0), (4, 1), (5, 0), (6, -1), (6, 0), (7, -1), (7, 0)]
_RAISED_DEGREES = {1: (1, 1), 3: (2, 1), 8: (5, 1)}
_DIMINISHED = {"dim", "dim7", "hdim7"}
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_AWKWARD = {"E#", "B#", "Cb", "Fb"}


def _parse_note(text: str) -> tuple[str, int] | None:
    """音名を (文字, 臨時記号の数。# は +、b は -) に分ける。"""
    m = _NOTE_RE.match(text)
    if not m:
        return None
    acc = m.group(2).replace("♯", "#").replace("♭", "b")
    if "#" in acc and "b" in acc:
        return None
    return m.group(1).upper(), acc.count("#") - acc.count("b")


def _note_pc(letter: str, acc: int) -> int:
    return (_LETTER_PC[letter] + acc) % 12


def _name(letter: str, acc: int) -> str | None:
    """文字と臨時記号から綴りを作る。ダブルシャープ・ダブルフラットや E# などの読みにくい綴りは None。"""
    if acc not in (-1, 0, 1):
        return None
    name = letter + ("#" if acc == 1 else "b" if acc == -1 else "")
    return None if name in _AWKWARD else name


def _spell_interval(letter: str, acc: int, degree: int, shift: int) -> tuple[str, int]:
    """letter+acc から degree 度（shift で半音ずらす）上の音を、度数どおりの文字で返す。"""
    steps = (degree - 1) % 7
    new_letter = _LETTERS[(_LETTERS.index(letter) + steps) % 7]
    target = (_note_pc(letter, acc) + _DEGREE_SEMITONES[steps + 1] + shift) % 12
    diff = (target - _LETTER_PC[new_letter]) % 12
    new_acc = diff if diff <= 6 else diff - 12
    return new_letter, new_acc


def _tonic(key: KeyEstimate) -> tuple[str, int] | None:
    parsed = _parse_note(key.tonic.strip())
    if parsed is None:
        return None
    pc = _note_pc(*parsed)
    table = _MINOR_TONIC if key.mode == "minor" else _MAJOR_TONIC
    return _parse_note(table[pc])


def _prefers_flats(key: KeyEstimate) -> bool:
    tonic = _tonic(key)
    if tonic is None:
        return False
    letter, acc = tonic
    if acc < 0 or (letter == "F" and acc == 0 and key.mode == "major"):
        return True
    if key.mode == "minor" and acc == 0 and letter in "DGCF":
        return True
    return False


def spell_pc_in_key(pc: int, key: KeyEstimate, *, raised: bool = False) -> str:
    """ピッチクラスをキーに合った綴りにする（主音からの度数で文字を決める）。

    raised は減三和音などの導音的なコード用で、#I・#II・#V の綴りを選ぶ（C で C#dim）。
    """
    tonic = _tonic(key)
    if tonic is not None:
        semis = (pc - _note_pc(*tonic)) % 12
        degree, shift = _RAISED_DEGREES.get(semis, _SCALE_DEGREES[semis]) if raised else _SCALE_DEGREES[semis]
        name = _name(*_spell_interval(tonic[0], tonic[1], degree, shift))
        if name is not None:
            return name
    return (_FLAT_NAMES if _prefers_flats(key) else _SHARP_NAMES)[pc % 12]


def key_to_cat4g(key: KeyEstimate | None) -> str | None:
    """KeyEstimate を originalKey の書き方（"A"、"F#m"、"Bb" など）にする。"""
    if key is None:
        return None
    tonic = _tonic(key)
    if tonic is None:
        return None
    name = (_MINOR_TONIC if key.mode == "minor" else _MAJOR_TONIC)[_note_pc(*tonic)]
    return name + ("m" if key.mode == "minor" else "")


def harte_to_cat4g(label: str, key: KeyEstimate | None = None) -> str | None:
    """Harte 記法のラベルを cat4g の表記にする。N（無音）・X（不明）は None。

    分数ベースは度数（"C:maj/3"）と音名（"C/E"）の両方を受け付け、"C/E" の形で書く。
    ベースがルートと同じ音なら分数にしない。読めないラベルはそのまま返す。
    key があればルートとベースの綴りをキーに合わせる（例 F メジャーの A# → Bb）。
    """
    text = label.strip()
    if text in ("", "N", "X"):
        return None
    body, _, bass_text = text.partition("/")
    root_text, sep, quality = body.partition(":")
    if not sep:
        quality = "maj"
    root = _parse_note(root_text)
    if root is None:
        return text
    quality = quality.strip()
    if quality not in QUALITY_MAP:
        return text
    suffix = QUALITY_MAP[quality]
    root_pc = _note_pc(*root)
    raised = quality in _DIMINISHED
    root_name = spell_pc_in_key(root_pc, key, raised=raised) if key else (_name(*root) or _SHARP_NAMES[root_pc])

    bass_name = None
    if bass_text:
        bass_name = _bass_name(bass_text.strip(), root, root_pc, key)
        if bass_name is False:
            return text
    return root_name + suffix + (f"/{bass_name}" if bass_name else "")


def _bass_name(bass_text: str, root: tuple[str, int], root_pc: int, key: KeyEstimate | None):
    """分数ベースの綴り。ルートと同じなら None、読めなければ False。"""
    deg = _DEGREE_RE.match(bass_text)
    if deg:
        acc_text = deg.group(1).replace("♯", "#").replace("♭", "b")
        degree = int(deg.group(2))
        if degree < 1:
            return False
        shift = acc_text.count("#") - acc_text.count("b")
        letter, acc = _spell_interval(root[0], root[1], degree, shift)
    else:
        note = _parse_note(bass_text)
        if note is None:
            return False
        letter, acc = note
    pc = _note_pc(letter, acc)
    if pc == root_pc:
        return None
    if key is not None:
        return spell_pc_in_key(pc, key)
    name = _name(letter, acc)
    if name is None:  # ダブルシャープなどはルートの臨時記号に合わせて単純な綴りへ
        name = (_FLAT_NAMES if root[1] < 0 else _SHARP_NAMES)[pc]
    return name
