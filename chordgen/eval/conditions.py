"""比べる条件の組み合わせと、条件ごとの後処理（純粋関数）。

条件の軸（--conditions の名前と値）:
    input  コード認識に入れる音    original / no_drums / no_drums_vocals / boost_other
    snap   拍にそろえる処理        off / beat（1 拍の孤立区間を残す）/ beat_absorb（吸収する）
    bass   ベースの使い方          off / root_fix / slash
    model  コード認識のモデル      chordmini / btc
    key    キー推定の使い方        off / diatonic（キーに合わない確信度の低いコードを寄せる）

後処理の順は key → snap → bass（キーの補正は区間ごとの確信度を使うので拍にそろえる前、
ベースの判定は拍にそろえた区間で行う）。
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

from chordgen.postprocess import apply_bass, apply_key, snap_to_beats
from chordgen.types import BassNote, BeatGrid, ChordSegment, KeyEstimate

AXES: dict[str, tuple[str, ...]] = {
    "input": ("original", "no_drums", "no_drums_vocals", "boost_other"),
    "snap": ("off", "beat", "beat_absorb"),
    "bass": ("off", "root_fix", "slash"),
    "model": ("chordmini", "btc"),
    "key": ("off", "diatonic"),
}
BASELINE = {"input": "original", "snap": "off", "bass": "off", "model": "chordmini", "key": "off"}


@dataclass(frozen=True, order=True)
class Condition:
    input: str
    snap: str
    bass: str
    model: str
    key: str

    @property
    def id(self) -> str:
        return f"{self.input}/{self.snap}/{self.bass}/{self.model}/{self.key}"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def needs_stems(self) -> bool:
        return self.input != "original" or self.bass != "off"


def parse_filters(specs: list[str] | None) -> dict[str, tuple[str, ...]]:
    """["input=original,no_drums", "model=btc"] → 軸ごとの値。書かない軸はすべての値。"""
    chosen = dict(AXES)
    for spec in specs or []:
        for part in spec.replace(";", " ").split():
            if "=" not in part:
                raise ValueError(f"条件は 軸=値,値 の形で書いてください: {part!r}")
            axis, _, values = part.partition("=")
            axis = axis.strip()
            if axis not in AXES:
                raise ValueError(f"知らない軸です: {axis!r}（{', '.join(AXES)}）")
            vals = tuple(v.strip() for v in values.split(",") if v.strip())
            bad = [v for v in vals if v not in AXES[axis]]
            if bad or not vals:
                raise ValueError(f"{axis} の値は {', '.join(AXES[axis])} から選んでください: {values!r}")
            chosen[axis] = vals
    return chosen


def enumerate_conditions(filters: dict[str, tuple[str, ...]] | None = None) -> list[Condition]:
    f = filters or AXES
    return [
        Condition(*vals)
        for vals in itertools.product(*(f[a] for a in ("input", "snap", "bass", "model", "key")))
    ]


def baseline(conditions: list[Condition]) -> Condition | None:
    """差を見る基準。BASELINE の条件が比べる中にあればそれ、なければ先頭。"""
    if not conditions:
        return None
    base = Condition(**BASELINE)
    return base if base in conditions else conditions[0]


def apply_condition(
    cond: Condition,
    chords: list[ChordSegment],
    beats: BeatGrid | None,
    bass: list[BassNote] | None,
    key: KeyEstimate | None,
) -> list[ChordSegment]:
    """認識したままのコードに、条件の後処理（key → snap → bass）をかける。"""
    out = list(chords)
    if cond.key == "diatonic":
        out = apply_key(out, key)
    if cond.snap != "off":
        if beats is None:
            raise ValueError("拍の検出結果がありません")
        out = snap_to_beats(out, beats, absorb_isolated=(cond.snap == "beat_absorb"))
    if cond.bass != "off":
        if bass is None:
            raise ValueError("ベースの検出結果がありません")
        out = apply_bass(out, bass, cond.bass)  # type: ignore[arg-type]
    return out
