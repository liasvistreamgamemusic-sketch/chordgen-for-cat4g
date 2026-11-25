"""chordgen.eval の純粋な部分のテスト（合成したデータ。モデルも mir_eval も使わない）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chordgen.eval import conditions as cond_mod
from chordgen.eval.conditions import Condition, enumerate_conditions, parse_filters
from chordgen.eval.labfile import cat4g_to_harte, format_lab, parse_lab, read_lab, to_harte, write_lab
from chordgen.eval.metrics import METRICS, SongScore, aggregate, fill_gaps
from chordgen.types import ChordSegment

REF = [
    ChordSegment(0.0, 1.0, "N"),
    ChordSegment(1.0, 3.0, "C:maj"),
    ChordSegment(3.0, 5.0, "A:min7"),
    ChordSegment(5.0, 7.0, "F:maj/3"),
]


# --- .lab と表記 ----------------------------------------------------------


def test_lab_roundtrip(tmp_path: Path):
    path = tmp_path / "sub" / "曲.lab"
    write_lab(path, REF)
    back = read_lab(path)
    assert [(s.start, s.end, s.label) for s in back] == [(s.start, s.end, s.label) for s in REF]
    assert parse_lab(format_lab(back))[3].label == "F:maj/3"


def test_parse_lab_formats():
    text = "﻿# コメント\n\n0 1.5 N.C.\n1.5\t3.0\tAm7\n3.0 4.0 C/E\n4.0 4.0 G\n4.0 5.0 B:hdim7\n"
    segs = parse_lab(text.lstrip("﻿"))
    assert [s.label for s in segs] == ["N", "A:min7", "C:maj/3", "B:hdim7"]  # 長さ 0 の行は捨てる
    with pytest.raises(ValueError, match="3 行目"):
        parse_lab("0 1 C\n1 2 G\n2 x Am\n")
    with pytest.raises(ValueError):
        parse_lab("0 1 Hm\n")
    with pytest.raises(ValueError):
        parse_lab("2 1 C\n")


@pytest.mark.parametrize(
    "name, harte",
    [
        ("C", "C:maj"),
        ("Am", "A:min"),
        ("Am7", "A:min7"),
        ("C/E", "C:maj/3"),
        ("C/G", "C:maj/5"),
        ("C/Bb", "C:maj/b7"),
        ("Am/G", "A:min/b7"),
        ("Am7/G", "A:min7/b7"),
        ("F#m7-5", "F#:hdim7"),
        ("Bbmaj7", "Bb:maj7"),
        ("EbM7", "Eb:maj7"),
        ("C#dim", "C#:dim"),
        ("Cdim7", "C:dim7"),
        ("Caug", "C:aug"),
        ("Dsus4", "D:sus4"),
        ("E7sus4", "E:sus4(b7)"),
        ("G7(b9)", "G:7(b9)"),
        ("Cadd9", "C:maj(9)"),
        ("CmM7", "C:minmaj7"),
        ("C6", "C:maj6"),
        ("Am6", "A:min6"),
        ("D9", "D:9"),
        ("N.C.", "N"),
        ("X", "X"),
    ],
)
def test_cat4g_to_harte(name, harte):
    assert cat4g_to_harte(name) == harte


def test_to_harte_accepts_both():
    assert to_harte("A:min7") == "A:min7"
    assert to_harte("C") == "C"
    assert to_harte("C/3") == "C/3"
    assert to_harte("Am7") == "A:min7"
    with pytest.raises(ValueError):
        to_harte("C:bogus")


# --- 集計・条件 ---------------------------------------------------------


def test_aggregate():
    a = SongScore({m: 1.0 for m in METRICS}, {m: 10.0 for m in METRICS})
    b = SongScore({m: 0.5 for m in METRICS}, {m: 30.0 for m in METRICS})
    agg = aggregate([a, b])
    assert agg.mean["root"] == pytest.approx(0.75)
    assert agg.weighted["root"] == pytest.approx((10 + 15) / 40)
    assert agg.minimum["root"] == pytest.approx(0.5)


def test_fill_gaps():
    out = fill_gaps([ChordSegment(1.0, 2.0, "C:maj")], 0.0, 3.0)
    assert [(s.start, s.end, s.label) for s in out] == [(0.0, 1.0, "N"), (1.0, 2.0, "C:maj"), (2.0, 3.0, "N")]


def test_conditions_enumerate_and_filter():
    all_conds = enumerate_conditions()
    assert len(all_conds) == 4 * 3 * 3 * 2 * 2
    f = parse_filters(["input=original,no_drums", "model=btc snap=off"])
    conds = enumerate_conditions(f)
    assert len(conds) == 2 * 1 * 3 * 1 * 2
    assert {c.input for c in conds} == {"original", "no_drums"}
    for bad in (["foo=1"], ["input=xxx"], ["input"]):
        with pytest.raises(ValueError):
            parse_filters(bad)
    assert cond_mod.baseline(all_conds) == Condition("original", "off", "off", "chordmini", "off")
