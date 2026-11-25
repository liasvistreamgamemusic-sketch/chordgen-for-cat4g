"""検証と書き出し。"""

import copy
import json
import math
from pathlib import Path

import pytest

from chordgen.paths import schema_path
from chordgen.songformat import SongFormatError, validate_document, write_document

SAMPLE = schema_path().parent / "samples" / "analyzed.cat4g.json"


@pytest.fixture
def sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_is_valid(sample):
    validate_document(sample)


def first_chord(doc):
    return doc["song"]["sections"][1]["lines"][0]["chords"][0]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(sample, bad):
    first_chord(sample)["time"] = bad
    with pytest.raises(SongFormatError) as ei:
        validate_document(sample)
    assert "song.sections[1].lines[0].chords[0].time" in str(ei.value)


def test_non_finite_inside_extensions_is_rejected(sample):
    sample["song"]["extensions"]["analyzer.x"] = [1.0, math.nan]
    with pytest.raises(SongFormatError):
        validate_document(sample)


def test_unknown_property_is_rejected(sample):
    first_chord(sample)["confidence"] = 0.9
    with pytest.raises(SongFormatError) as ei:
        validate_document(sample)
    assert "confidence" in str(ei.value)


def test_schema_ranges(sample):
    for mutate in (
        lambda d: first_chord(d).update(position=-1),
        lambda d: first_chord(d).update(bar=0),
        lambda d: d["song"]["sections"][0]["lines"][0].update(measures=65),
        lambda d: d["song"].update(title=""),
        lambda d: d.update(version=2),
        lambda d: d["song"]["timeSignature"].update(unit=3),
    ):
        doc = copy.deepcopy(sample)
        mutate(doc)
        with pytest.raises(SongFormatError):
            validate_document(doc)


def test_too_many_chords_in_a_line(sample):
    line = sample["song"]["sections"][0]["lines"][0]
    line["chords"] = [{"chord": "C", "position": i} for i in range(201)]
    with pytest.raises(SongFormatError):
        validate_document(sample)


def test_extensions_depth_limit(sample):
    def nested(levels):  # 配列を levels 段。extensions 自体を足すと levels + 1 段
        v = 1
        for _ in range(levels):
            v = [v]
        return v

    sample["song"]["extensions"] = {"analyzer.deep": nested(63)}
    validate_document(sample)
    sample["song"]["extensions"] = {"analyzer.deep": nested(64)}
    with pytest.raises(SongFormatError) as ei:
        validate_document(sample)
    assert "64" in str(ei.value)


def test_forbidden_and_reserved_keys(sample):
    doc = copy.deepcopy(sample)
    doc["song"]["extensions"]["analyzer.x"] = {"__proto__": 1}
    with pytest.raises(SongFormatError):
        validate_document(doc)
    doc = copy.deepcopy(sample)
    doc["song"]["extensions"]["cat4g.viewMode"] = "compact"
    with pytest.raises(SongFormatError):
        validate_document(doc)


@pytest.mark.parametrize("value", ["2026-13-45T25:61:61Z", "2026-02-30T00:00:00Z", "2026-11-11 09:00:00Z"])
def test_bad_dates(sample, value):
    sample["exportedAt"] = value
    with pytest.raises(SongFormatError):
        validate_document(sample)


@pytest.mark.parametrize("value", ["2026-02-03T04:05:06.123456+00:00", "2026-11-11T18:00:00+09:00"])
def test_good_dates(sample, value):
    sample["exportedAt"] = value
    validate_document(sample)


def test_non_json_types_are_rejected(sample):
    sample["song"]["tags"] = ("a",)
    with pytest.raises(SongFormatError):
        validate_document(sample)


def test_write_document(tmp_path, sample):
    out = tmp_path / "sub" / "曲.cat4g.json"
    write_document(sample, out)
    raw = out.read_bytes()
    assert "しろいくも".encode("utf-8") in raw  # ensure_ascii=False
    assert json.loads(raw.decode("utf-8")) == sample
    assert [p.name for p in out.parent.iterdir()] == [out.name]  # 一時ファイルが残らない


def test_write_document_refuses_invalid(tmp_path, sample):
    out = tmp_path / "bad.cat4g.json"
    first_chord(sample)["position"] = math.nan
    with pytest.raises(SongFormatError):
        write_document(sample, out)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_does_not_overwrite_on_failure(tmp_path, sample):
    out = tmp_path / "keep.cat4g.json"
    out.write_text("old", encoding="utf-8")
    sample["version"] = 2
    with pytest.raises(SongFormatError):
        write_document(sample, out)
    assert out.read_text(encoding="utf-8") == "old"


def test_write_built_document(tmp_path):
    from chordgen.songformat import build_document
    from chordgen.types import AnalysisResult, AudioInfo, BeatGrid, ChordSegment, TrackMeta

    res = AnalysisResult(
        audio=AudioInfo(Path("a.wav"), 10.0, 44100),
        meta=TrackMeta(title="かきだし"),
        beats=BeatGrid([0.5 * i for i in range(20)], list(range(0, 20, 4)), 120.0, 4),
        key=None,
        chords=[ChordSegment(0.0, 4.0, "A:min7", 0.5), ChordSegment(4.0, 8.0, "D:7", 0.5)],
    )
    out = tmp_path / "x.cat4g.json"
    write_document(build_document(res), out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_document(doc)
    assert doc["song"]["title"] == "かきだし"
