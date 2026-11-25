"""Harte 記法 → cat4g の表記の変換。"""

import pytest

from chordgen.songformat import harte_to_cat4g, key_to_cat4g
from chordgen.types import KeyEstimate

# BTC の大語彙の 14 種類
ALL_QUALITIES = [
    ("maj", "A"),
    ("min", "Am"),
    ("dim", "Adim"),
    ("aug", "Aaug"),
    ("min6", "Am6"),
    ("maj6", "A6"),
    ("min7", "Am7"),
    ("minmaj7", "Ammaj7"),
    ("maj7", "Amaj7"),
    ("7", "A7"),
    ("dim7", "Adim7"),
    ("hdim7", "Am7b5"),
    ("sus2", "Asus2"),
    ("sus4", "Asus4"),
]


@pytest.mark.parametrize("quality,expected", ALL_QUALITIES)
def test_all_btc_qualities(quality, expected):
    assert harte_to_cat4g(f"A:{quality}") == expected


def test_quality_count_matches_btc():
    assert len(ALL_QUALITIES) == 14


@pytest.mark.parametrize("label", ["N", "X", " N ", ""])
def test_no_chord_is_none(label):
    assert harte_to_cat4g(label) is None


@pytest.mark.parametrize(
    "label,expected",
    [
        ("C:maj/3", "C/E"),
        ("C/E", "C/E"),
        ("C:maj/5", "C/G"),
        ("C:maj/b7", "C/Bb"),
        ("C:min/b3", "Cm/Eb"),
        ("A:min7/b7", "Am7/G"),
        ("F#:maj/3", "F#/A#"),
        ("Bb:maj/3", "Bb/D"),
        ("D:7/3", "D7/F#"),
        ("G:maj/1", "G"),  # ベースがルートと同じなら分数にしない
        ("C:maj/9", "C/D"),
    ],
)
def test_slash_chords(label, expected):
    assert harte_to_cat4g(label) == expected


def test_root_only_is_major():
    assert harte_to_cat4g("G") == "G"
    assert harte_to_cat4g("Db") == "Db"


def test_unknown_label_is_returned_as_is():
    assert harte_to_cat4g("C:(1,3,5)") == "C:(1,3,5)"
    assert harte_to_cat4g("H:maj") == "H:maj"


def test_spelling_follows_key():
    f_major = KeyEstimate("F", "major")
    assert harte_to_cat4g("A#:maj", f_major) == "Bb"
    assert harte_to_cat4g("A#:maj/3", f_major) == "Bb/D"
    assert harte_to_cat4g("D#:maj7", KeyEstimate("A#", "major")) == "Ebmaj7"
    e_major = KeyEstimate("E", "major")
    assert harte_to_cat4g("Ab:min", e_major) == "G#m"
    assert harte_to_cat4g("C:maj/b7", KeyEstimate("G", "major")) == "C/Bb"
    # G メジャーの借用コード（bVII・bIII）はフラットで書く
    g_major = KeyEstimate("G", "major")
    assert harte_to_cat4g("F:maj", g_major) == "F"
    assert harte_to_cat4g("A#:maj", g_major) == "Bb"
    # 減三和音は導音として #I・#V で書く
    c_major = KeyEstimate("C", "major")
    assert harte_to_cat4g("C#:dim7", c_major) == "C#dim7"
    assert harte_to_cat4g("G#:dim", c_major) == "G#dim"
    assert harte_to_cat4g("Db:maj", c_major) == "Db"
    # マイナーキー
    assert harte_to_cat4g("G#:dim", KeyEstimate("A", "minor")) == "G#dim"
    assert harte_to_cat4g("A#:maj", KeyEstimate("D", "minor")) == "Bb"
    assert harte_to_cat4g("D#:min", KeyEstimate("C#", "minor")) == "D#m"


def test_without_key_keeps_spelling():
    assert harte_to_cat4g("A#:maj") == "A#"
    assert harte_to_cat4g("Bb:min7") == "Bbm7"


def test_key_to_cat4g():
    assert key_to_cat4g(KeyEstimate("A", "major")) == "A"
    assert key_to_cat4g(KeyEstimate("F#", "minor")) == "F#m"
    assert key_to_cat4g(KeyEstimate("A#", "major")) == "Bb"
    assert key_to_cat4g(KeyEstimate("C#", "major")) == "Db"
    assert key_to_cat4g(KeyEstimate("G#", "minor")) == "G#m"
    assert key_to_cat4g(None) is None
