"""合成した AnalysisResult から文書を作り、検証と中身を確かめる。"""

import math
from pathlib import Path

import pytest

from chordgen.songformat import build_document, validate_document
from chordgen.types import (
    AlignedLine,
    AnalysisResult,
    AudioInfo,
    BeatGrid,
    ChordSegment,
    KeyEstimate,
    TrackMeta,
)


def make_grid(start=1.0, period=0.5, n=64, bpb=4, first_down=0, unit=4):
    beats = [start + i * period for i in range(n)]
    return BeatGrid(
        beat_times=beats,
        downbeat_indices=list(range(first_down, n, bpb)),
        bpm=60.0 / period,
        beats_per_bar=bpb,
        beat_unit=unit,
    )


def make_result(chords, beats, *, lyrics=None, duration=40.0, key=None, meta=None, extras=None):
    return AnalysisResult(
        audio=AudioInfo(path=Path("work") / "demo.wav", duration_sec=duration, sample_rate=44100),
        meta=meta or TrackMeta(title="合成テスト", artist="テスト"),
        beats=beats,
        key=key,
        chords=[ChordSegment(s, e, lab, conf) for s, e, lab, conf in chords],
        lyrics=lyrics,
        extras=extras or {},
    )


def all_chords(doc):
    return [c for sec in doc["song"]["sections"] for line in sec["lines"] for c in line["chords"]]


def check_invariants(doc, beats_per_bar):
    """bar と行の measures の累計、position の刻み、時刻の順を確かめる。"""
    validate_document(doc)
    offset = doc["song"]["audio"].get("offsetSec", 0)
    last_time = -1.0
    for sec in doc["song"]["sections"]:
        assert len(sec["lines"]) <= 2000
        bar0 = 1
        for line in sec["lines"]:
            assert 1 <= line["measures"] <= 64
            assert len(line["chords"]) <= 200
            prev_pos = -1
            for c in line["chords"]:
                assert bar0 <= c["bar"] < bar0 + line["measures"], (c, bar0, line["measures"])
                assert 1 <= c["beat"] < beats_per_bar + 1
                assert (c["position"] * 2) == int(c["position"] * 2)
                assert c["position"] > prev_pos
                prev_pos = c["position"]
                assert c["duration"] > 0
                assert c["time"] + offset >= last_time - 1e-9
                last_time = c["time"] + offset
                assert "method" not in c and "strokePattern" not in c and "arpeggioOrder" not in c
            bar0 += line["measures"]


def test_no_lyrics_basic():
    chords = [
        (1.0, 3.0, "C:maj", 0.9),
        (3.0, 4.0, "A:min7", 0.8),
        (4.0, 5.0, "F:maj/3", 0.7),
        (5.0, 9.0, "G:7", 0.6),
        (9.0, 11.0, "C:maj", 0.5),
    ]
    doc = build_document(make_result(chords, make_grid(), key=KeyEstimate("C", "major", 0.8)))
    check_invariants(doc, 4)
    song = doc["song"]
    assert doc["format"] == "cat4g-song" and doc["version"] == 1
    assert doc["appVersion"].startswith("chordgen-for-cat4g ")
    assert song["capo"] == 0 and song["transpose"] == 0
    assert song["bpm"] == 120 and song["timeSignature"] == {"beats": 4, "unit": 4}
    assert song["originalKey"] == "C"
    assert song["audio"]["offsetSec"] == 1 and song["audio"]["durationSec"] == 40
    assert song["extensions"]["analyzer.keyConfidence"] == 0.8
    assert len(song["sections"]) == 1
    lines = song["sections"][0]["lines"]
    assert [ln["measures"] for ln in lines] == [4, 1]
    assert all(ln["lyrics"] == "" for ln in lines)
    first = lines[0]["chords"]
    assert [c["chord"] for c in first] == ["C", "Am7", "F/A", "G7"]
    assert [c["position"] for c in first] == [0, 4, 6, 8]
    assert [(c["bar"], c["beat"]) for c in first] == [(1, 1), (2, 1), (2, 3), (3, 1)]
    assert [c["duration"] for c in first] == [4, 2, 2, 8]
    assert first[0]["time"] == 0 and first[1]["time"] == 2
    assert first[0]["extensions"] == {"analyzer.confidence": 0.9}


def test_n_and_x_are_dropped_and_same_chords_merge():
    chords = [
        (1.0, 2.0, "C:maj", 0.8),
        (2.0, 3.0, "C:maj", 0.6),  # まとめる
        (3.0, 4.0, "N", None),
        (4.0, 5.0, "X", None),
        (5.0, 7.0, "C:maj", 0.9),  # N をはさむので別のコード
        (7.0, 9.0, "D:min", 0.9),
    ]
    doc = build_document(make_result(chords, make_grid()))
    check_invariants(doc, 4)
    got = [(c["chord"], c["bar"], c["beat"], c["duration"]) for c in all_chords(doc)]
    assert got == [("C", 1, 1, 4), ("C", 3, 1, 4), ("Dm", 4, 1, 4)]
    assert all_chords(doc)[0]["extensions"]["analyzer.confidence"] == 0.7


def test_anacrusis_goes_into_a_pickup_bar():
    # 最初の 1 拍目は 3 拍目（添字 3）。その前の 3 拍に弱起のコードがある
    grid = make_grid(start=0.5, first_down=3)
    chords = [(1.0, 2.0, "G:maj", 0.9), (2.0, 6.0, "C:maj", 0.9), (6.0, 10.0, "F:maj", 0.9)]
    doc = build_document(make_result(chords, grid))
    check_invariants(doc, 4)
    lines = doc["song"]["sections"][0]["lines"]
    assert lines[0]["measures"] == 1  # 弱起の小節だけの行
    g = lines[0]["chords"][0]
    assert (g["chord"], g["bar"], g["beat"], g["duration"]) == ("G", 1, 3, 2)
    c = lines[1]["chords"][0]
    assert (c["chord"], c["bar"], c["beat"]) == ("C", 2, 1)
    # 仮の小節の 1 拍目は 0.5 − 0.5 = 0 秒
    assert doc["song"]["audio"]["offsetSec"] == 0
    assert g["time"] == 1


def test_chords_outside_beat_range():
    # 拍は 2〜6 秒にしかないが、コードは 0.2 秒から 12 秒まで
    grid = BeatGrid(beat_times=[2.0 + 0.5 * i for i in range(9)], downbeat_indices=[0, 4, 8], bpm=120, beats_per_bar=4)
    chords = [(0.2, 2.0, "E:min", 0.5), (2.0, 4.0, "C:maj", 0.5), (4.0, 12.0, "D:maj", 0.5)]
    doc = build_document(make_result(chords, grid, duration=12.0))
    check_invariants(doc, 4)
    got = [(c["chord"], c["bar"], c["beat"]) for c in all_chords(doc)]
    assert got[0] == ("Em", 1, 1.4)
    assert got[1] == ("C", 2, 1)
    assert got[2] == ("D", 3, 1)
    total_beats = sum(c["duration"] for c in all_chords(doc))
    assert total_beats == pytest.approx((12.0 - 0.2) / 0.5)


def test_three_four_time():
    grid = make_grid(bpb=3, n=60)
    chords = [(1.0 + 1.5 * i, 1.0 + 1.5 * (i + 1), lab, 0.5) for i, lab in enumerate(["C:maj", "G:maj", "A:min", "F:maj"] * 3)]
    doc = build_document(make_result(chords, grid))
    check_invariants(doc, 3)
    assert doc["song"]["timeSignature"] == {"beats": 3, "unit": 4}
    lines = doc["song"]["sections"][0]["lines"]
    assert [ln["measures"] for ln in lines] == [4, 4, 4]
    assert [c["position"] for c in lines[0]["chords"]] == [0, 3, 6, 9]
    assert all(c["duration"] == 3 and c["beat"] == 1 for c in all_chords(doc))


def test_six_eight_unit():
    grid = make_grid(bpb=6, unit=8, period=0.25)
    doc = build_document(make_result([(1.0, 4.0, "D:maj", 0.5)], grid))
    check_invariants(doc, 6)
    assert doc["song"]["timeSignature"] == {"beats": 6, "unit": 8}
    assert all_chords(doc)[0]["duration"] == 12


def test_long_chord_is_restated_at_line_start():
    doc = build_document(make_result([(1.0, 17.0, "E:min7", 0.5)], make_grid()))
    check_invariants(doc, 4)
    lines = doc["song"]["sections"][0]["lines"]
    assert [ln["measures"] for ln in lines] == [4, 4]
    a, b = lines[0]["chords"][0], lines[1]["chords"][0]
    assert a["tieToNext"] is True and a["duration"] == 16
    assert (b["chord"], b["bar"], b["position"], b["time"]) == ("Em7", 5, 0, 8)
    assert b["extensions"]["analyzer.continued"] is True


def test_positions_do_not_collide():
    # 1 拍ごとに長いコード名が並んでも、前の名前に重ならない
    chords = [(1.0 + 0.5 * i, 1.5 + 0.5 * i, lab, None) for i, lab in enumerate(["F#:min7", "B:7", "C#:hdim7", "E:maj"])]
    doc = build_document(make_result(chords, make_grid()))
    check_invariants(doc, 4)
    pos = [c["position"] for c in all_chords(doc)]
    names = [c["chord"] for c in all_chords(doc)]
    for (p1, n1), p2 in zip(zip(pos, names), pos[1:]):
        assert p2 - p1 >= len(n1) * 0.5


def test_bars_per_line_option_and_limits():
    doc = build_document(make_result([(1.0, 9.0, "C:maj", 0.5)], make_grid()), bars_per_line=2)
    assert [ln["measures"] for ln in doc["song"]["sections"][0]["lines"]] == [2, 2]
    with pytest.raises(ValueError):
        build_document(make_result([], make_grid()), bars_per_line=0)


def test_more_than_200_chords_in_a_line_is_split():
    grid = make_grid(n=300, period=0.25)
    chords = [(1.0 + 0.25 * i, 1.25 + 0.25 * i, ["C:maj", "D:min"][i % 2], None) for i in range(256)]
    doc = build_document(make_result(chords, grid, duration=80), bars_per_line=64)
    check_invariants(doc, 4)
    lines = doc["song"]["sections"][0]["lines"]
    assert [ln["measures"] for ln in lines] == [32, 32]
    assert sum(len(ln["chords"]) for ln in lines) == 256


def test_more_than_2000_lines_is_split_into_sections():
    n_bars = 2100
    grid = make_grid(n=n_bars * 4 + 4, period=0.25)
    chords = [(1.0 + i, 2.0 + i, ["C:maj", "G:maj"][i % 2], None) for i in range(n_bars)]
    doc = build_document(make_result(chords, grid, duration=n_bars + 10), bars_per_line=1)
    check_invariants(doc, 4)
    secs = doc["song"]["sections"]
    assert [len(s["lines"]) for s in secs] == [2000, 100]
    assert secs[0]["name"] == "全体 (1/2)"
    assert secs[1]["lines"][0]["chords"][0]["bar"] == 1


def test_empty_result_still_validates():
    doc = build_document(make_result([(0.0, 5.0, "N", None)], make_grid()))
    validate_document(doc)
    assert doc["song"]["sections"][0]["lines"] == [{"lyrics": "", "measures": 1, "memo": None, "chords": []}]


def test_no_beats_at_all():
    grid = BeatGrid(beat_times=[], downbeat_indices=[], bpm=float("nan"), beats_per_bar=4)
    doc = build_document(make_result([(0.0, 4.0, "C:maj", None), (4.0, 8.0, "G:maj", None)], grid))
    check_invariants(doc, 4)
    assert doc["song"]["bpm"] is None


def test_meta_fallback_and_extensions():
    meta = TrackMeta(title=None, artist=None, source_url="https://example.com/watch?v=1")
    res = make_result([(1.0, 3.0, "C:maj", math.nan)], make_grid(), meta=meta, extras={"chordInputMode": "original", "bad": math.inf})
    doc = build_document(res)
    validate_document(doc)
    song = doc["song"]
    assert song["title"] == "demo"
    assert song["sourceUrl"] == "https://example.com/watch?v=1"
    assert song["audio"]["url"] == "https://example.com/watch?v=1"
    ext = song["extensions"]
    assert ext["analyzer.sourceUrl"] == "https://example.com/watch?v=1"
    assert ext["analyzer.chordInputMode"] == "original"
    assert ext["analyzer.bad"] is None
    assert "extensions" not in all_chords(doc)[0]


# ---- 歌詞あり ----


def lyric(text, times):
    """times は「何文字目 → 時刻」。書いていない文字は時刻なし。"""
    return AlignedLine(text=text, char_times=[times.get(i) for i in range(len(text))])


def lyric_result():
    # 1 小節 = 2 秒（120 BPM、4/4）。小節 k は 1 + 2k 秒から
    lines = [
        lyric("🌂かさをさそう", {1: 3.0, 2: 3.5, 3: 4.0, 4: 5.0, 5: 5.5, 6: 6.0}),
        lyric("👨‍👩‍👧いえにかえろう", {5: 9.0, 6: 9.5, 7: 10.0, 8: 11.0, 9: 11.5, 10: 12.0, 11: 12.5}),
    ]
    chords = [
        (1.0, 3.0, "C:maj", 0.9),  # 前奏（1 小節）
        (3.0, 5.0, "G:maj", 0.8),  # 「か」
        (5.0, 9.0, "A:min", 0.7),  # 「さ」（5 文字目）
        (9.0, 11.0, "F:maj", 0.6),  # 「い」
        (11.0, 13.0, "G:7", 0.6),  # 「か」
        (13.0, 14.0, "N", None),
        (14.0, 21.0, "C:maj", 0.5),  # 後奏
    ]
    return make_result(chords, make_grid(), lyrics=lines)


def test_lyrics_lines_and_positions():
    doc = build_document(lyric_result())
    check_invariants(doc, 4)
    lines = doc["song"]["sections"][0]["lines"]
    assert [ln["lyrics"] for ln in lines] == ["", "🌂かさをさそう", "👨‍👩‍👧いえにかえろう", ""]
    assert [ln["measures"] for ln in lines] == [1, 3, 2, 4]
    assert [(c["chord"], c["position"], c["bar"]) for c in lines[0]["chords"]] == [("C", 0, 1)]
    # 🌂 は 1 コードポイントなので「か」は 1
    assert [(c["chord"], c["position"], c["bar"]) for c in lines[1]["chords"]] == [("G", 1, 2), ("Am", 4, 3)]
    # 👨‍👩‍👧 は 5 コードポイント（人 3 つ + ZWJ 2 つ）なので「い」は 5、「か」は 8
    assert [(c["chord"], c["position"], c["bar"]) for c in lines[2]["chords"]] == [("F", 5, 5), ("G7", 8, 6)]
    # 後奏は歌詞のない行（小節 7〜10）
    assert [(c["chord"], c["bar"], c["beat"]) for c in lines[3]["chords"]] == [("C", 7, 3)]


def test_lyric_timings():
    doc = build_document(lyric_result())
    line = doc["song"]["sections"][0]["lines"][1]
    timings = line["lyricTimings"]
    assert [t["at"] for t in timings] == [1, 2, 3, 4, 5, 6]
    assert timings[0] == {"at": 1, "bar": 2, "beat": 1, "time": 2}
    assert timings[1] == {"at": 2, "bar": 2, "beat": 2, "time": 2.5}
    assert "lyricTimings" not in doc["song"]["sections"][0]["lines"][0]


def test_lyric_chord_after_singing_goes_past_text():
    lines = [lyric("あいう", {0: 1.0, 1: 1.5, 2: 2.0})]
    chords = [(1.0, 4.0, "C:maj", None), (4.0, 5.0, "G:maj", None)]
    doc = build_document(make_result(chords, make_grid(), lyrics=lines))
    check_invariants(doc, 4)
    ln = doc["song"]["sections"][0]["lines"][0]
    assert ln["measures"] == 2
    # G は歌い終わり（2.0 秒）から 4 拍後なので、歌詞の長さ 3 + 3
    assert [c["position"] for c in ln["chords"]] == [0, 6]


def test_lyric_lines_in_same_bar_and_untimed_line():
    lines = [
        lyric("あい", {0: 1.0, 1: 1.5}),
        lyric("うえ", {0: 2.0, 1: 2.5}),  # 前の行と同じ小節で歌い出す
        lyric("おか", {}),  # 時刻が分からない
        lyric("きく", {0: 9.0, 1: 9.5}),
    ]
    chords = [(1.0 + 2 * i, 3.0 + 2 * i, lab, None) for i, lab in enumerate(["C:maj", "D:min", "E:min", "F:maj", "G:maj"])]
    doc = build_document(make_result(chords, make_grid(), lyrics=lines))
    check_invariants(doc, 4)
    got = [(ln["lyrics"], ln["measures"]) for ln in doc["song"]["sections"][0]["lines"]]
    assert [g[0] for g in got] == ["あい", "うえ", "おか", "きく"]
    assert all(m >= 1 for _, m in got)
    assert sum(m for _, m in got) == 5


def test_lyric_line_before_first_chord():
    lines = [lyric("らら", {0: 1.2, 1: 1.6})]
    chords = [(3.0, 5.0, "A:maj", None)]
    doc = build_document(make_result(chords, make_grid(), lyrics=lines))
    check_invariants(doc, 4)
    ln = doc["song"]["sections"][0]["lines"]
    assert ln[0]["lyrics"] == "らら" and ln[0]["measures"] == 2
    assert ln[0]["chords"][0]["bar"] == 2
