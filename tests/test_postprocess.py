"""chordgen.postprocess のテスト（合成したデータ）。"""

from __future__ import annotations

import pytest

from chordgen import postprocess as pp
from chordgen.postprocess import apply_bass, apply_key, parse_harte, snap_to_beats
from chordgen.types import BassNote, BeatGrid, ChordSegment, KeyEstimate


def grid(times: list[float]) -> BeatGrid:
    return BeatGrid(beat_times=times, downbeat_indices=list(range(0, len(times), 4)), bpm=120.0, beats_per_bar=4)


def seg(start, end, label, conf=None, **extras) -> ChordSegment:
    return ChordSegment(start, end, label, conf, dict(extras))


def spans(segs):
    return [(round(s.start, 6), round(s.end, 6), s.label) for s in segs]


# --- Harte ---------------------------------------------------------------


def test_parse_harte_basic():
    c = parse_harte("A:min7")
    assert c.root_pc == 9 and c.shorthand == "min7" and c.bass is None
    assert c.pitch_classes == frozenset({9, 0, 4, 7})
    assert parse_harte("C").quality == "maj"
    assert parse_harte("C:maj/3").bass_pc == 4
    assert parse_harte("Db:dim7/bb7").bass_pc == (1 + 9) % 12
    assert parse_harte("C:maj(9)").pitch_classes == frozenset({0, 2, 4, 7})
    assert parse_harte("C:7(*5)").pitch_classes == frozenset({0, 4, 10})
    assert parse_harte("N") is None and parse_harte("X") is None
    for bad in ("H:maj", "C:foo", "C/E", "Am7"):
        with pytest.raises(ValueError):
            parse_harte(bad)


# --- snap_to_beats --------------------------------------------------------

BEATS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def test_snap_aligns_slightly_off_boundaries():
    chords = [seg(0.0, 2.07, "C:maj"), seg(2.07, 3.46, "A:min"), seg(3.46, 6.0, "F:maj")]
    out = snap_to_beats(chords, grid(BEATS))
    assert spans(out) == [(0.0, 2.0, "C:maj"), (2.0, 3.5, "A:min"), (3.5, 6.0, "F:maj")]


def test_snap_keeps_outside_range():
    # 最初の拍より前・最後の拍より後の区間は元のまま（境界も動かさない）
    chords = [seg(0.0, 0.3, "N"), seg(0.3, 1.2, "G:maj"), seg(1.2, 4.8, "C:maj"), seg(4.8, 5.2, "D:min"),
              seg(5.2, 6.0, "N")]
    out = snap_to_beats(chords, grid(BEATS))
    assert spans(out) == [(0.0, 0.3, "N"), (0.3, 1.0, "G:maj"), (1.0, 5.0, "C:maj"), (5.0, 5.2, "D:min"),
                          (5.2, 6.0, "N")]


def test_snap_confidence_weighting_picks_label():
    # 拍の区間 [1.0, 1.5) で G は 0.3 秒・確信度 0.2、C は 0.2 秒・確信度 0.9 → C が勝つ
    chords = [seg(0.9, 1.3, "G:maj", 0.2), seg(1.3, 3.0, "C:maj", 0.9)]
    out = snap_to_beats(chords, grid([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert spans(out) == [(0.9, 1.0, "G:maj"), (1.0, 3.0, "C:maj")]


def test_snap_confidence_is_length_weighted_mean():
    chords = [seg(1.0, 2.0, "C:maj", 0.9), seg(2.0, 3.0, "C:maj", 0.5)]
    out = snap_to_beats(chords, grid([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(0.7)
    chords = [seg(1.0, 1.5, "C:maj", 0.9), seg(1.5, 3.0, "C:maj", 0.3)]
    out = snap_to_beats(chords, grid([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert out[0].confidence == pytest.approx((0.9 * 0.5 + 0.3 * 1.5) / 2.0)
    # 確信度がなければ None
    out = snap_to_beats([seg(1.0, 3.0, "C:maj")], grid([1.0, 2.0, 3.0]))
    assert out[0].confidence is None


def test_snap_isolated_beat_absorbed_or_kept():
    chords = [seg(1.0, 2.5, "C:maj", 0.8), seg(2.5, 3.0, "G:maj", 0.4), seg(3.0, 5.0, "C:maj", 0.8)]
    absorbed = snap_to_beats(chords, grid(BEATS), absorb_isolated=True)
    assert spans(absorbed) == [(1.0, 5.0, "C:maj")]
    # 吸収した拍は確信度の平均に入れない（C の区間だけの平均）
    assert absorbed[0].confidence == pytest.approx(0.8)
    kept = snap_to_beats(chords, grid(BEATS), absorb_isolated=False)
    assert spans(kept) == [(1.0, 2.5, "C:maj"), (2.5, 3.0, "G:maj"), (3.0, 5.0, "C:maj")]
    assert pp.ABSORB_ISOLATED_DEFAULT is True
    assert spans(snap_to_beats(chords, grid(BEATS))) == spans(absorbed)


def test_snap_isolated_between_different_chords_goes_to_stronger_neighbor():
    # [2.5, 3.0) は D が主だが、その中で次の F が前の C より長い → F に吸収
    chords = [seg(1.0, 2.5, "C:maj"), seg(2.5, 2.8, "D:min"), seg(2.8, 5.0, "F:maj")]
    out = snap_to_beats(chords, grid(BEATS))
    assert spans(out) == [(1.0, 2.5, "C:maj"), (2.5, 5.0, "F:maj")]


def test_snap_isolated_not_absorbed_into_no_chord():
    chords = [seg(1.0, 2.5, "N"), seg(2.5, 3.0, "C:maj"), seg(3.0, 5.0, "N")]
    assert spans(snap_to_beats(chords, grid(BEATS))) == spans(chords)


def test_snap_two_beat_run_is_not_isolated():
    chords = [seg(1.0, 2.0, "C:maj"), seg(2.0, 3.0, "G:maj"), seg(3.0, 5.0, "C:maj")]
    assert spans(snap_to_beats(chords, grid(BEATS))) == spans(chords)


def test_snap_gap_and_edge_cases():
    # 拍の大半が空きならコードを置かない（[3.0, 3.5) は G が 0.1 秒だけ → 空き）
    chords = [seg(1.0, 2.0, "C:maj"), seg(3.4, 5.0, "G:maj")]
    out = snap_to_beats(chords, grid(BEATS), absorb_isolated=False)
    assert spans(out) == [(1.0, 2.0, "C:maj"), (3.5, 5.0, "G:maj")]
    # 半分以上鳴っていればその拍のコードにする
    chords = [seg(1.0, 2.0, "C:maj"), seg(3.2, 5.0, "G:maj")]
    out = snap_to_beats(chords, grid(BEATS), absorb_isolated=False)
    assert spans(out) == [(1.0, 2.0, "C:maj"), (3.0, 5.0, "G:maj")]
    # 拍が 2 つ未満なら何もしない
    chords = [seg(0.0, 1.03, "C:maj"), seg(1.03, 2.0, "G:maj")]
    assert spans(snap_to_beats(chords, grid([1.0]))) == spans(chords)
    assert snap_to_beats([], grid(BEATS)) == []


def test_snap_keeps_extras_and_does_not_mutate_input():
    chords = [seg(1.0, 2.07, "C:maj", 0.9, note="a"), seg(2.07, 3.0, "G:maj", 0.9)]
    out = snap_to_beats(chords, grid([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert out[0].extras == {"note": "a"}
    out[0].extras["x"] = 1
    assert chords[0].extras == {"note": "a"} and chords[0].end == 2.07


# --- apply_bass -----------------------------------------------------------


def beat_bass(pcs, start=0.0, step=0.5, conf=0.9):
    return [BassNote(start + i * step, start + (i + 1) * step, pc, conf) for i, pc in enumerate(pcs)]


def test_bass_off_is_identity():
    chords = [seg(0.0, 2.0, "C:maj")]
    out = apply_bass(chords, beat_bass([4, 4, 4, 4]), "off")
    assert spans(out) == spans(chords) and out[0] is not chords[0]


def test_bass_slash_third_fifth_seventh():
    chords = [seg(0.0, 2.0, "C:maj"), seg(2.0, 4.0, "C:maj"), seg(4.0, 6.0, "C:7"), seg(6.0, 8.0, "A:min")]
    bass = beat_bass([4] * 4 + [7] * 4 + [10] * 4 + [0] * 4)
    out = apply_bass(chords, bass, "slash")
    assert [s.label for s in out] == ["C:maj/3", "C:maj/5", "C:7/b7", "A:min/b3"]
    assert out[0].extras == {"bassPitchClass": 4, "bassOccupancy": 1.0, "slashFrom": "C:maj"}


def test_bass_slash_ignores_non_chord_tone_and_root():
    chords = [seg(0.0, 2.0, "C:maj"), seg(2.0, 4.0, "C:maj")]
    out = apply_bass(chords, beat_bass([2] * 4 + [0] * 4), "slash")  # D は経過音、C はルート
    assert [s.label for s in out] == ["C:maj", "C:maj"]
    assert out[0].extras == {} and out[1].extras == {}


def test_bass_thresholds():
    chords = [seg(0.0, 2.0, "C:maj")]
    # 占有率 0.5（< 0.6）では変えない
    assert apply_bass(chords, beat_bass([4, 4, 0, 0]), "slash")[0].label == "C:maj"
    # 占有率 0.75 なら変える
    assert apply_bass(chords, beat_bass([4, 4, 4, 0]), "slash")[0].label == "C:maj/3"
    # 確信度が低い音は数えない
    low = beat_bass([4, 4, 4, 4], conf=pp.BASS_MIN_CONFIDENCE - 0.01)
    assert apply_bass(chords, low, "slash")[0].label == "C:maj"
    # 確信度が None の音は数える
    assert apply_bass(chords, beat_bass([4, 4, 4, 4], conf=None), "slash")[0].label == "C:maj/3"
    # 音なし（None）は数えない
    assert apply_bass(chords, beat_bass([4, None, None, None]), "slash")[0].label == "C:maj"


def test_bass_root_fix():
    chords = [
        seg(0.0, 2.0, "C:maj"),   # ベース A（構成音でない）→ A:min7（C・E・G を含む）
        seg(2.0, 4.0, "A:min7"),  # ベース C（構成音）で構成音が同じ → C:maj6
        seg(4.0, 6.0, "C:maj"),   # ベース E（構成音。転回形）→ 変えない
        seg(6.0, 8.0, "C:maj"),   # ベース D（共有する音が少ない）→ 変えない
        seg(8.0, 10.0, "A:min"),  # ベース F → F:maj7
    ]
    bass = beat_bass([9] * 4 + [0] * 4 + [4] * 4 + [2] * 4 + [5] * 4)
    out = apply_bass(chords, bass, "root_fix")
    assert [s.label for s in out] == ["A:min7", "C:maj6", "C:maj", "C:maj", "F:maj7"]
    assert out[0].extras == {"bassPitchClass": 9, "bassOccupancy": 1.0, "rootFixedFrom": "C:maj"}
    assert out[2].extras == {}


def test_bass_leaves_no_chord_unknown_and_slash():
    chords = [seg(0.0, 2.0, "N"), seg(2.0, 4.0, "X"), seg(4.0, 6.0, "C:maj/5"), seg(6.0, 8.0, "weird")]
    bass = beat_bass([4] * 16)
    for mode in ("root_fix", "slash"):
        assert [s.label for s in apply_bass(chords, bass, mode)] == ["N", "X", "C:maj/5", "weird"]


def test_bass_works_on_unaligned_segments_and_keeps_confidence():
    chords = [seg(0.2, 1.7, "G:maj", 0.6, src="m")]
    out = apply_bass(chords, beat_bass([11, 11, 11, 11]), "slash")
    assert out[0].label == "G:maj/3" and out[0].confidence == 0.6
    assert out[0].extras["src"] == "m" and out[0].extras["slashFrom"] == "G:maj"


def test_bass_bad_mode():
    with pytest.raises(ValueError):
        apply_bass([], [], "bogus")  # type: ignore[arg-type]


# --- apply_key ------------------------------------------------------------


def test_apply_key():
    key = KeyEstimate("C", "major")
    chords = [
        seg(0, 1, "E:maj", 0.3),    # キー外・低確信 → E:min（ルートが同じものを優先）
        seg(1, 2, "E:maj", 0.9),    # 高確信は触らない
        seg(2, 3, "C#:min", 0.3),   # 共有する音が 1 つ（E）だけ → 触らない
        seg(3, 4, "D:min", 0.1),    # キー内は触らない
        seg(4, 5, "E:maj", None),   # 確信度なしは触らない
        seg(5, 6, "D:7", 0.2),      # D:7（D F# A C）→ D:min7
        seg(6, 7, "N", 0.0),
    ]
    out = apply_key(chords, key)
    assert out[0].extras == {"keyFixedFrom": "E:maj"}
    assert [s.label for s in out] == ["E:min", "E:maj", "C#:min", "D:min", "E:maj", "D:min7", "N"]
    assert apply_key(chords, None)[0].label == "E:maj"
    # 短調では V（E:maj）はキー内
    assert apply_key([seg(0, 1, "E:maj", 0.1)], KeyEstimate("A", "minor"))[0].label == "E:maj"
