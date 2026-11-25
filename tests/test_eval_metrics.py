"""chordgen.eval の採点・表のテスト（mir_eval が要る。なければ skip）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mir_eval")

from chordgen.eval.conditions import Condition, apply_condition  # noqa: E402
from chordgen.eval.metrics import METRICS, score_song  # noqa: E402
from chordgen.types import BassNote, BeatGrid, ChordSegment  # noqa: E402

REF = [
    ChordSegment(0.0, 1.0, "N"),
    ChordSegment(1.0, 3.0, "C:maj"),
    ChordSegment(3.0, 5.0, "A:min7"),
    ChordSegment(5.0, 7.0, "F:maj/3"),
]


def test_perfect_match_is_one():
    sc = score_song(REF, REF)
    assert all(sc.scores[m] == pytest.approx(1.0) for m in METRICS)
    assert sc.durations["root"] == pytest.approx(7.0)


def test_root_only_match():
    est = [ChordSegment(0.0, 1.0, "N"), ChordSegment(1.0, 3.0, "C:min"), ChordSegment(3.0, 5.0, "A:maj"),
           ChordSegment(5.0, 7.0, "F:min/b3")]
    sc = score_song(REF, est)
    assert sc.scores["root"] == pytest.approx(1.0)
    assert sc.scores["majmin"] < 1.0
    assert sc.scores["thirds"] < 1.0


def test_partial_and_gaps():
    # 予測が 3〜5 秒を欠く → N とみなして外れ
    est = [ChordSegment(0.0, 1.0, "N"), ChordSegment(1.0, 3.0, "C:maj"), ChordSegment(5.0, 7.0, "F:maj/3")]
    sc = score_song(REF, est)
    assert sc.scores["root"] == pytest.approx(5.0 / 7.0)
    # 正解の範囲の外の予測は数えない
    sc2 = score_song(REF, est + [ChordSegment(7.0, 20.0, "G:maj")])
    assert sc2.scores["root"] == pytest.approx(5.0 / 7.0)


def test_sevenths_vs_majmin():
    # A:min7 を A:min と予測 → majmin は合い、sevenths は外れ
    est = [ChordSegment(0.0, 1.0, "N"), ChordSegment(1.0, 3.0, "C:maj"), ChordSegment(3.0, 5.0, "A:min"),
           ChordSegment(5.0, 7.0, "F:maj/3")]
    sc = score_song(REF, est)
    assert sc.scores["majmin"] == pytest.approx(1.0)
    assert sc.scores["sevenths"] == pytest.approx(5.0 / 7.0)


def test_slash_bass_matters_only_for_inversion_metrics():
    est = [s if s.label != "F:maj/3" else ChordSegment(5.0, 7.0, "F:maj") for s in REF]
    sc = score_song(REF, est)
    for m in ("root", "majmin", "thirds", "triads", "sevenths", "tetrads", "mirex"):
        assert sc.scores[m] == pytest.approx(1.0), m  # この 7 つは分数ベースを見ない
    assert sc.scores["majmin_inv"] == pytest.approx(5.0 / 7.0)
    assert sc.scores["sevenths_inv"] == pytest.approx(5.0 / 7.0)


def test_unknown_reference_is_excluded():
    ref = [ChordSegment(0.0, 2.0, "C:maj"), ChordSegment(2.0, 4.0, "X")]
    est = [ChordSegment(0.0, 4.0, "C:maj")]
    sc = score_song(ref, est)
    assert sc.scores["majmin"] == pytest.approx(1.0) and sc.durations["majmin"] == pytest.approx(2.0)


def test_bad_label_raises():
    with pytest.raises(Exception):
        score_song(REF, [ChordSegment(0.0, 7.0, "C:nonsense")])



# --- 条件 -----------------------------------------------------------------


def test_apply_condition_improves_synthetic_prediction():
    beats = BeatGrid([1.0 + 0.5 * i for i in range(13)], [0, 4, 8, 12], 120.0, 4)
    raw = [ChordSegment(0.0, 1.0, "N"), ChordSegment(1.0, 3.08, "C:maj"), ChordSegment(3.08, 4.93, "A:min7"),
           ChordSegment(4.93, 7.0, "F:maj")]
    bass = [BassNote(5.0 + 0.5 * i, 5.5 + 0.5 * i, 9, 0.9) for i in range(4)]
    plain = score_song(REF, apply_condition(Condition("original", "off", "off", "chordmini", "off"),
                                            raw, beats, bass, None))
    snapped = apply_condition(Condition("original", "beat", "slash", "chordmini", "off"), raw, beats, bass, None)
    assert [s.label for s in snapped] == ["N", "C:maj", "A:min7", "F:maj/3"]
    full = score_song(REF, snapped)
    assert full.scores["triads"] == pytest.approx(1.0)
    assert plain.scores["triads"] < 1.0


def test_report_tables_and_files(tmp_path: Path):
    from chordgen.eval.report import format_tables, write_reports

    c1 = Condition("original", "off", "off", "chordmini", "off")
    c2 = Condition("no_drums", "beat", "off", "chordmini", "off")
    results = {
        c1: {"曲A": score_song(REF, REF), "曲B": score_song(REF, REF[:2])},
        c2: {"曲A": score_song(REF, REF[:3]), "曲B": score_song(REF, REF)},
    }
    text = format_tables(results, ["曲A", "曲B"], ("root",))
    assert "S1=曲A" in text and "== root ==" in text and c2.id in text
    csv_path, json_path = write_reports(results, ["曲A", "曲B"], tmp_path, {"dataset": "x"})
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["songs"] == ["曲A", "曲B"] and len(data["conditions"]) == 2
    assert data["conditions"][0]["perSong"]["曲A"]["scores"]["root"] is not None
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("condition,input,snap,bass,model,key,song,metric,score,duration")
    assert len(lines) == 1 + 2 * (2 + 3) * len(METRICS)
