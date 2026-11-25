"""コード認識の結果の後処理（純粋関数。numpy だけを使う）。

- snap_to_beats: コードの切り替えを拍にそろえる
- apply_bass: ベースの認識結果でルートを直す / 分数コードにする
- apply_key: キーに合わない確信度の低いコードを近いダイアトニックコードに寄せる（比較用）

Harte 記法の小さな解析器もここに持つ（chordgen.songformat には依存しない）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from chordgen.types import BassMode, BassNote, BeatGrid, ChordSegment, KeyEstimate

# ---------------------------------------------------------------------------
# Harte 記法
# ---------------------------------------------------------------------------

PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# 度数の数字 → 長音階での半音数
_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 12, 9: 14, 10: 16, 11: 17, 12: 19, 13: 21}

# 種類（shorthand）→ 構成音の度数。mir_eval / Harte の定義に合わせる
QUALITY_DEGREES: dict[str, tuple[str, ...]] = {
    "maj": ("1", "3", "5"),
    "min": ("1", "b3", "5"),
    "dim": ("1", "b3", "b5"),
    "aug": ("1", "3", "#5"),
    "maj7": ("1", "3", "5", "7"),
    "min7": ("1", "b3", "5", "b7"),
    "7": ("1", "3", "5", "b7"),
    "dim7": ("1", "b3", "b5", "bb7"),
    "hdim7": ("1", "b3", "b5", "b7"),
    "minmaj7": ("1", "b3", "5", "7"),
    "maj6": ("1", "3", "5", "6"),
    "min6": ("1", "b3", "5", "6"),
    "9": ("1", "3", "5", "b7", "9"),
    "maj9": ("1", "3", "5", "7", "9"),
    "min9": ("1", "b3", "5", "b7", "9"),
    "11": ("1", "3", "5", "b7", "9", "11"),
    "maj11": ("1", "3", "5", "7", "9", "11"),
    "min11": ("1", "b3", "5", "b7", "9", "11"),
    "13": ("1", "3", "5", "b7", "9", "11", "13"),
    "maj13": ("1", "3", "5", "7", "9", "11", "13"),
    "min13": ("1", "b3", "5", "b7", "9", "11", "13"),
    "sus2": ("1", "2", "5"),
    "sus4": ("1", "4", "5"),
    "1": ("1",),
    "5": ("1", "5"),
    "": (),
}

_HARTE_RE = re.compile(r"^([A-G](?:b*|#*))(?::([^/]*))?(?:/(.+))?$")
_DEGREE_RE = re.compile(r"^(b*|#*)(\d+)$")
_QUALITY_RE = re.compile(r"^([^()]*)(?:\(([^()]*)\))?$")


@dataclass(frozen=True)
class HarteChord:
    """Harte のラベルを分けたもの。"""

    root: str  # 書かれたままの音名（"Db" など）
    root_pc: int
    quality: str  # ":" の後ろ（拡張の括弧を含む）。省略時は "maj"
    shorthand: str  # 括弧の前の部分（"min7" など）
    degrees: tuple[str, ...]  # 構成音の度数（拡張を反映）
    bass: str | None  # "/" の後ろの度数。なければ None

    @property
    def intervals(self) -> frozenset[int]:
        """ルートからの半音数（0〜11）の集合。"""
        return frozenset(degree_to_semitone(d) for d in self.degrees)

    @property
    def pitch_classes(self) -> frozenset[int]:
        return frozenset((self.root_pc + i) % 12 for i in self.intervals)

    @property
    def bass_pc(self) -> int:
        """ベースのピッチクラス（分数でなければルート）。"""
        if self.bass is None:
            return self.root_pc
        return (self.root_pc + degree_to_semitone(self.bass)) % 12


def note_to_pc(name: str) -> int:
    """音名（"C#"、"Bb"、"Cbb" など）→ ピッチクラス。"""
    if not name or name[0] not in _NATURAL_PC:
        raise ValueError(f"音名ではありません: {name!r}")
    pc = _NATURAL_PC[name[0]]
    for ch in name[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            raise ValueError(f"音名ではありません: {name!r}")
    return pc % 12


def degree_to_semitone(degree: str) -> int:
    """度数（"b3"、"#5"、"bb7"、"9"）→ ルートからの半音数（0〜11）。"""
    m = _DEGREE_RE.match(degree)
    if not m or int(m.group(2)) not in _DEGREE_SEMITONES:
        raise ValueError(f"度数ではありません: {degree!r}")
    acc = m.group(1)
    shift = len(acc) if acc.startswith("#") else -len(acc)
    return (_DEGREE_SEMITONES[int(m.group(2))] + shift) % 12


def _degree_number(degree: str) -> int:
    m = _DEGREE_RE.match(degree)
    if not m:
        raise ValueError(f"度数ではありません: {degree!r}")
    return int(m.group(2))


def parse_harte(label: str) -> HarteChord | None:
    """Harte のラベルを解析する。N / X は None。解析できなければ ValueError。"""
    label = label.strip()
    if label in ("N", "X"):
        return None
    m = _HARTE_RE.match(label)
    if not m:
        raise ValueError(f"Harte 記法ではありません: {label!r}")
    root, quality, bass = m.group(1), m.group(2), m.group(3)
    if quality is None:
        quality = "maj"
    qm = _QUALITY_RE.match(quality)
    if not qm:
        raise ValueError(f"Harte 記法ではありません: {label!r}")
    shorthand = qm.group(1)
    if shorthand not in QUALITY_DEGREES:
        raise ValueError(f"知らない種類です: {label!r}")
    degrees = list(QUALITY_DEGREES[shorthand])
    if qm.group(2) is not None:
        for item in (s.strip() for s in qm.group(2).split(",")):
            if not item:
                continue
            if item.startswith("*"):
                semi = degree_to_semitone(item[1:])
                degrees = [d for d in degrees if degree_to_semitone(d) != semi]
            else:
                degree_to_semitone(item)  # 形の確認
                degrees.append(item)
    if bass is not None:
        degree_to_semitone(bass)
    return HarteChord(
        root=root,
        root_pc=note_to_pc(root),
        quality=quality,
        shorthand=shorthand,
        degrees=tuple(degrees),
        bass=bass,
    )


def harte_label(root: str, quality: str, bass: str | None = None) -> str:
    """ルート・種類・ベースの度数から Harte のラベルを作る。"""
    label = f"{root}:{quality or 'maj'}"
    if bass is not None and degree_to_semitone(bass) != 0:
        label += f"/{bass}"
    return label


def _try_parse(label: str) -> HarteChord | None:
    """解析できないラベルは None（触らない）にする。"""
    try:
        return parse_harte(label)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 共通の小道具
# ---------------------------------------------------------------------------


def _copy(seg: ChordSegment, **changes) -> ChordSegment:
    out = replace(seg, **changes)
    if "extras" not in changes:
        out.extras = dict(seg.extras)
    return out


def _weighted_conf(pairs: Iterable[tuple[float | None, float]]) -> float | None:
    """(確信度, 長さ) の組から長さで重みづけした平均。確信度がなければ None。"""
    num = 0.0
    den = 0.0
    for conf, dur in pairs:
        if conf is None or dur <= 0:
            continue
        num += conf * dur
        den += dur
    return num / den if den > 0 else None


def _merge_adjacent(segments: list[ChordSegment], eps: float = 1e-6) -> list[ChordSegment]:
    """隣り合う（すき間のない）同じラベルの区間をまとめる。"""
    out: list[ChordSegment] = []
    for seg in segments:
        if seg.end - seg.start <= eps:
            continue
        if out and out[-1].label == seg.label and abs(out[-1].end - seg.start) <= eps:
            prev = out[-1]
            prev_dur = prev.end - prev.start
            seg_dur = seg.end - seg.start
            conf = _weighted_conf([(prev.confidence, prev_dur), (seg.confidence, seg_dur)])
            extras = prev.extras if prev_dur >= seg_dur else dict(seg.extras)
            out[-1] = ChordSegment(prev.start, seg.end, prev.label, conf, extras)
        else:
            out.append(_copy(seg))
    return out


# ---------------------------------------------------------------------------
# snap_to_beats
# ---------------------------------------------------------------------------

# 1 拍だけの孤立した区間を前後に吸収するかの既定。拍ごとの多数決を通っても残る 1 拍の区間は、
# BTC の揺れ（A:min → C:maj → A:min など）であることが多く、下書きでは消す手間のほうが重いので吸収する。
# 吸収しない場合との差は chordgen.eval の条件 snap=beat / snap=beat_absorb で比べられる。
ABSORB_ISOLATED_DEFAULT = True
# 拍の区間のうち、どのコードも鳴っていない部分がこの割合を超えたら空き（コードなし）にする
_MIN_BEAT_COVERAGE = 0.5


def snap_to_beats(
    chords: list[ChordSegment],
    beats: BeatGrid,
    *,
    absorb_isolated: bool = ABSORB_ISOLATED_DEFAULT,
) -> list[ChordSegment]:
    """コードの切り替えを拍にそろえる。

    隣り合う拍の間（拍の区間）ごとに、いちばん長く鳴っているラベル（確信度があれば
    長さ × 確信度）をその拍のコードとし、同じラベルの続く拍をまとめ直す。
    最初の拍より前・最後の拍より後は元の区間を残す。
    absorb_isolated が真なら、前後を別のコードに挟まれた 1 拍だけの区間を、その拍の中で
    票の多いほうの隣に吸収する（前後が同じならそのコード。N には吸収しない）。
    確信度は、まとめた区間に含まれる元の区間（同じラベル）の長さで重みづけした平均。
    """
    segs = sorted((s for s in chords if s.end > s.start), key=lambda s: s.start)
    times = np.asarray(sorted(beats.beat_times), dtype=float)
    if len(times) >= 2:
        times = times[np.concatenate(([True], np.diff(times) > 1e-9))]
    if not segs or len(times) < 2:
        return [_copy(s) for s in segs]

    first, last = float(times[0]), float(times[-1])
    n_iv = len(times) - 1
    # 拍の区間ごとの {ラベル: [票, 長さ, Σ確信度×長さ, 確信度のある長さ, {元の添字: 長さ}]}
    votes: list[dict[str, list]] = [dict() for _ in range(n_iv)]
    for idx, seg in enumerate(segs):
        lo = max(seg.start, first)
        hi = min(seg.end, last)
        if hi <= lo:
            continue
        i0 = max(int(np.searchsorted(times, lo, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(times, hi, side="left")), n_iv)
        for i in range(i0, i1):
            ov = min(hi, times[i + 1]) - max(lo, times[i])
            if ov <= 0:
                continue
            w = ov * (seg.confidence if seg.confidence is not None else 1.0)
            entry = votes[i].setdefault(seg.label, [0.0, 0.0, 0.0, 0.0, {}])
            entry[0] += w + 1e-9 * ov  # 確信度がすべて 0 のときは長さで決める
            entry[1] += ov
            if seg.confidence is not None:
                entry[2] += seg.confidence * ov
                entry[3] += ov
            entry[4][idx] = entry[4].get(idx, 0.0) + ov

    labels: list[str | None] = []
    for i, v in enumerate(votes):
        covered = sum(e[1] for e in v.values())
        if not v or covered < _MIN_BEAT_COVERAGE * (times[i + 1] - times[i]):
            labels.append(None)
        else:
            labels.append(max(v.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0])

    if absorb_isolated:
        labels = _absorb_isolated(labels, votes)

    # 同じラベルの続く拍をまとめる
    out: list[ChordSegment] = []
    # 範囲の前
    for seg in segs:
        if seg.start < first:
            out.append(_copy(seg, end=min(seg.end, first)))
    i = 0
    while i < n_iv:
        lab = labels[i]
        j = i
        while j + 1 < n_iv and labels[j + 1] == lab:
            j += 1
        if lab is not None:
            num = den = 0.0
            per_src: dict[int, float] = {}
            for k in range(i, j + 1):
                e = votes[k].get(lab)
                if e is None:
                    continue
                num += e[2]
                den += e[3]
                for src, ov in e[4].items():
                    per_src[src] = per_src.get(src, 0.0) + ov
            extras = dict(segs[max(per_src, key=per_src.get)].extras) if per_src else {}
            out.append(
                ChordSegment(
                    start=float(times[i]),
                    end=float(times[j + 1]),
                    label=lab,
                    confidence=(num / den) if den > 0 else None,
                    extras=extras,
                )
            )
        i = j + 1
    # 範囲の後
    for seg in segs:
        if seg.end > last:
            out.append(_copy(seg, start=max(seg.start, last)))

    out.sort(key=lambda s: s.start)
    return _merge_adjacent(out)


def _absorb_isolated(labels: list[str | None], votes: list[dict[str, list]]) -> list[str | None]:
    """前後を別のラベルに挟まれた 1 拍だけのラベルを隣に寄せる（元の並びで判断する）。"""
    out = list(labels)
    n = len(labels)
    for i in range(1, n - 1):
        lab, prev, nxt = labels[i], labels[i - 1], labels[i + 1]
        if lab is None or prev is None or nxt is None or prev == lab or nxt == lab:
            continue
        cands = [c for c in (prev, nxt) if c != "N"]
        if not cands:
            continue
        if prev == nxt:
            out[i] = prev
            continue
        # その拍の中で票の多いほう。同じなら前
        out[i] = max(cands, key=lambda c: (votes[i].get(c, [0.0])[0], c == prev))
    return out


# ---------------------------------------------------------------------------
# apply_bass
# ---------------------------------------------------------------------------

# この確信度より低いベースの音は数えない（confidence が None の音は数える）
BASS_MIN_CONFIDENCE = 0.5
# コードの区間のうち、いちばん多いベースの音（確信度がしきい値以上）が占める割合の下限
BASS_MIN_OCCUPANCY = 0.6
# 分数にしてよいベースの度数（3 度・5 度・7 度）。それ以外は経過音とみて変えない
SLASH_DEGREE_NUMBERS = frozenset({3, 5, 7})
# ルートを直すときの候補の種類（元の種類は必ず候補に入れる）
ROOT_FIX_QUALITIES = ("maj", "min", "7", "maj7", "min7")
# ルートを直すとき、候補が元のコードと共有していなければならない構成音の数
ROOT_FIX_MIN_SHARED = 2


def _dominant_bass(seg: ChordSegment, bass: list[BassNote]) -> tuple[int, float] | None:
    """区間でいちばん長く鳴っているベースの音と、その占有率。しきい値に届かなければ None。"""
    dur = seg.end - seg.start
    if dur <= 0:
        return None
    acc = np.zeros(12)
    for note in bass:
        if note.pitch_class is None:
            continue
        if note.confidence is not None and note.confidence < BASS_MIN_CONFIDENCE:
            continue
        ov = min(seg.end, note.end) - max(seg.start, note.start)
        if ov > 0:
            acc[note.pitch_class % 12] += ov
    pc = int(np.argmax(acc))
    occ = float(acc[pc] / dur)
    if acc[pc] <= 0 or occ < BASS_MIN_OCCUPANCY:
        return None
    return pc, occ


def _root_fix_candidate(chord: HarteChord, bass_pc: int) -> tuple[str, str] | None:
    """ベースの音をルートにしたほうが妥当なコード（ルート名, 種類）。なければ None。

    - ベースが元のコードの構成音なら転回形とみて、構成音がまったく同じ候補（A:min7 ↔ C:maj6 など）だけを採る
    - 構成音でなければ、元のコードと共有する構成音が最も多い候補（同数なら余分な音が少ない、元と同じ種類の順）
      を採り、共有が ROOT_FIX_MIN_SHARED 未満なら直さない
    """
    orig = chord.pitch_classes

    def pcs_of(q: str) -> frozenset[int]:
        return frozenset((bass_pc + degree_to_semitone(d)) % 12 for d in QUALITY_DEGREES[q])

    if bass_pc in orig:
        same = [q for q in QUALITY_DEGREES if QUALITY_DEGREES[q] and pcs_of(q) == orig]
        return (PC_NAMES[bass_pc], same[0]) if same else None

    qualities = [chord.shorthand] + [q for q in ROOT_FIX_QUALITIES if q != chord.shorthand]
    best: tuple[tuple, str] | None = None
    for q in qualities:
        if not QUALITY_DEGREES.get(q):
            continue
        pcs = pcs_of(q)
        shared = len(pcs & orig)
        if shared < ROOT_FIX_MIN_SHARED:
            continue
        key = (shared, -len(pcs - orig), q == chord.shorthand)
        if best is None or key > best[0]:
            best = (key, q)
    return (PC_NAMES[bass_pc], best[1]) if best else None


def _slash_degree(chord: HarteChord, bass_pc: int) -> str | None:
    """ベースが 3 度・5 度・7 度の構成音なら、その度数の表記。違えば None。"""
    interval = (bass_pc - chord.root_pc) % 12
    for d in chord.degrees:
        if degree_to_semitone(d) == interval and _degree_number(d) in SLASH_DEGREE_NUMBERS:
            return d
    return None


def apply_bass(chords: list[ChordSegment], bass: list[BassNote], mode: BassMode) -> list[ChordSegment]:
    """ベースの認識結果でコードを直す。

    off: 何もしない。
    root_fix: 区間で安定して鳴るベースの音がルートと違い、その音をルートにしたコードのほうが
        妥当なとき（_root_fix_candidate）にルートと種類を直す。
    slash: ベースの音がルートと違い 3 度・5 度・7 度の構成音のとき、分数ベース（"C:maj/3"）にする。
    ベースは確信度 BASS_MIN_CONFIDENCE 以上の音だけを数え、占有率 BASS_MIN_OCCUPANCY 以上で採る。
    N / X、すでに分数のコード、解析できないラベルは触らない。直した区間の extras に
    bassPitchClass・bassOccupancy と rootFixedFrom / slashFrom を残す。
    """
    if mode not in ("off", "root_fix", "slash"):
        raise ValueError(f"知らない mode です: {mode!r}")
    if mode == "off":
        return [_copy(s) for s in chords]
    notes = sorted(bass, key=lambda b: b.start)
    starts = np.asarray([b.start for b in notes], dtype=float)
    max_len = max((b.end - b.start for b in notes), default=0.0)

    out: list[ChordSegment] = []
    for seg in chords:
        chord = _try_parse(seg.label)
        if chord is None or chord.bass is not None:
            out.append(_copy(seg))
            continue
        # 区間に重なりうる音だけを渡す
        lo = int(np.searchsorted(starts, seg.start - max_len, side="left"))
        hi = int(np.searchsorted(starts, seg.end, side="left"))
        dom = _dominant_bass(seg, notes[lo:hi])
        if dom is None or dom[0] == chord.root_pc:
            out.append(_copy(seg))
            continue
        bass_pc, occ = dom
        new_label = None
        record: dict[str, object] = {"bassPitchClass": bass_pc, "bassOccupancy": round(occ, 3)}
        if mode == "root_fix":
            cand = _root_fix_candidate(chord, bass_pc)
            if cand is not None:
                new_label = harte_label(cand[0], cand[1])
                record["rootFixedFrom"] = seg.label
        else:
            deg = _slash_degree(chord, bass_pc)
            if deg is not None:
                new_label = harte_label(chord.root, chord.quality, deg)
                record["slashFrom"] = seg.label
        if new_label is None or new_label == seg.label:
            out.append(_copy(seg))
            continue
        out.append(_copy(seg, label=new_label, extras={**seg.extras, **record}))
    return out


# ---------------------------------------------------------------------------
# apply_key（比較用。CONTRACT にはない追加の関数）
# ---------------------------------------------------------------------------

# この確信度より低いコードだけを寄せる（確信度が None のコードは触らない）
KEY_FIX_MAX_CONFIDENCE = 0.5
# 寄せる先が元のコードと共有していなければならない構成音の数
KEY_FIX_MIN_SHARED = 2

_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
# (度数の半音, 種類)。短調は自然短音階に和声的短音階の V・V7・vii° を足す
_DIATONIC = {
    "major": [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim"),
              (0, "maj7"), (2, "min7"), (4, "min7"), (5, "maj7"), (7, "7"), (9, "min7"), (11, "hdim7")],
    "minor": [(0, "min"), (2, "dim"), (3, "maj"), (5, "min"), (7, "min"), (7, "maj"), (8, "maj"),
              (10, "maj"), (11, "dim"),
              (0, "min7"), (2, "hdim7"), (3, "maj7"), (5, "min7"), (7, "min7"), (7, "7"), (8, "maj7"),
              (10, "7")],
}


def _key_scale(key: KeyEstimate) -> frozenset[int]:
    tonic = note_to_pc(key.tonic)
    if key.mode == "major":
        return frozenset((tonic + i) % 12 for i in _MAJOR_SCALE)
    # 短調は導音（長 7 度）も音階の内とみなす
    return frozenset((tonic + i) % 12 for i in _NATURAL_MINOR_SCALE) | {(tonic + 11) % 12}


def apply_key(
    chords: list[ChordSegment],
    key: KeyEstimate | None,
    *,
    max_confidence: float = KEY_FIX_MAX_CONFIDENCE,
) -> list[ChordSegment]:
    """キーの音階に合わず確信度の低いコードを、構成音が最も近いダイアトニックコードに寄せる。

    寄せる先は元と共有する構成音が KEY_FIX_MIN_SHARED 以上のものに限り、同点なら
    ルートが同じもの、余分な音が少ないもの、元と同じ種類（三和音 / 四和音）の順に選ぶ。
    N / X、分数コード、確信度が None のコードは触らない。寄せた記録を extras["keyFixedFrom"] に残す。
    """
    if key is None:
        return [_copy(s) for s in chords]
    scale = _key_scale(key)
    tonic = note_to_pc(key.tonic)
    diatonic = [((tonic + off) % 12, q) for off, q in _DIATONIC[key.mode]]
    out: list[ChordSegment] = []
    for seg in chords:
        chord = _try_parse(seg.label)
        if (
            chord is None
            or chord.bass is not None
            or seg.confidence is None
            or seg.confidence >= max_confidence
            or chord.pitch_classes <= scale
        ):
            out.append(_copy(seg))
            continue
        orig = chord.pitch_classes
        is_tetrad = len(orig) >= 4
        best: tuple[tuple, int, str] | None = None
        for root_pc, q in diatonic:
            pcs = frozenset((root_pc + degree_to_semitone(d)) % 12 for d in QUALITY_DEGREES[q])
            shared = len(pcs & orig)
            if shared < KEY_FIX_MIN_SHARED:
                continue
            rank = (shared, root_pc == chord.root_pc, -len(pcs - orig), (len(pcs) >= 4) == is_tetrad)
            if best is None or rank > best[0]:
                best = (rank, root_pc, q)
        if best is None:
            out.append(_copy(seg))
            continue
        new_label = harte_label(PC_NAMES[best[1]], best[2])
        out.append(_copy(seg, label=new_label, extras={**seg.extras, "keyFixedFrom": seg.label}))
    return out
