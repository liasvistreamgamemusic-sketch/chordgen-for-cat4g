"""chordgen.lyrics のモデルに依存しない部分のテスト。歌詞は自作の短い文。"""

from __future__ import annotations

import pytest

from chordgen.lyrics import (
    TimedWord,
    _keep_monotonic,
    _lcs_pairs,
    build_aligned_lines,
    guess_language,
    resolve_model_name,
    split_lyrics_lines,
)


def _chars(words: list[tuple[str, float, float]]) -> list[TimedWord]:
    return [TimedWord(t, s, e) for t, s, e in words]


def _check_shape(lines, result):
    assert [r.text for r in result] == list(lines)
    for r in result:
        assert len(r.char_times) == len(r.text)


def _flat(result):
    return [t for r in result for t in r.char_times if t is not None]


# --- 行の分け方 --------------------------------------------------------------


def test_split_lines_keeps_text_and_drops_blank_lines():
    text = "\ufeff  朝の光が　窓を照らす  \r\n\n\t\n Hello,  world! \n　\n最後の行"
    assert split_lyrics_lines(text) == ["朝の光が　窓を照らす", "Hello,  world!", "最後の行"]


def test_split_lines_empty():
    assert split_lyrics_lines("") == []
    assert split_lyrics_lines("\n \n　\n") == []


# --- 言語の推定 --------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["あさのひかり"], "ja"),
        (["ネコがいる"], "ja"),
        (["ｱｲｳ"], "ja"),
        (["Hello", "朝日"], "ja"),
        (["안녕하세요"], "ko"),
        (["Hello world", "Good night"], None),
        (["♪ 123 !!"], None),
    ],
)
def test_guess_language(lines, expected):
    assert guess_language(lines) == expected


def test_resolve_model_name(monkeypatch):
    monkeypatch.delenv("CHORDGEN_WHISPER_MODEL", raising=False)
    assert resolve_model_name() == "large-v3"
    monkeypatch.setenv("CHORDGEN_WHISPER_MODEL", "base")
    assert resolve_model_name() == "base"
    assert resolve_model_name("tiny") == "tiny"


# --- 照合の部品 --------------------------------------------------------------


def test_lcs_pairs():
    assert _lcs_pairs(list("abc"), list("abc")) == [(0, 0), (1, 1), (2, 2)]
    assert _lcs_pairs(list("abcd"), list("xbzd")) == [(1, 1), (3, 3)]
    assert _lcs_pairs([], list("abc")) == []
    pairs = _lcs_pairs(list("ab_cd_ef"), list("abcdef"))
    assert [i for i, _ in pairs] == [0, 1, 3, 4, 6, 7]


def test_keep_monotonic_drops_outliers():
    times = [1.0, 2.0, 50.0, 3.0, None, 4.0, 0.5, 5.0]
    assert _keep_monotonic(times) == [1.0, 2.0, None, 3.0, None, 4.0, None, 5.0]
    assert _keep_monotonic([None, None]) == [None, None]


# --- AlignedLine を作る ------------------------------------------------------


def test_japanese_tokens_and_interpolation():
    lines = ["朝の光", "小さな声で"]
    words = _chars(
        [
            (" 朝", 1.0, 1.2),
            ("の", 1.2, 1.4),
            ("光", 1.4, 1.8),
            (" 小", 3.0, 3.2),
            ("さな", 3.2, 3.6),  # 2 文字のトークンは中で線形補間
            ("声", 3.6, 3.8),
            ("で", 3.8, 4.0),
        ]
    )
    result = build_aligned_lines(lines, words)
    _check_shape(lines, result)
    assert result[0].char_times == [1.0, 1.2, 1.4]
    assert result[1].char_times == [3.0, 3.2, 3.4, 3.6, 3.8]


def test_english_words_spaces_and_punctuation():
    lines = ["Hello, world!", "Good night"]
    words = _chars([(" Hello,", 0.0, 1.0), (" world!", 1.0, 2.0), (" Good", 3.0, 3.4), (" night", 3.4, 4.4)])
    result = build_aligned_lines(lines, words)
    _check_shape(lines, result)
    h = result[0].char_times
    # "Hello" の 5 文字が 0.0〜0.8、記号と空白は None
    assert h[:5] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert h[5] is None and h[6] is None and h[12] is None
    assert h[7:12] == [1.0, 1.2, 1.4, 1.6, 1.8]
    assert result[1].char_times[4] is None
    assert result[1].char_times[0] == 3.0
    assert result[1].char_times[5] == 3.4


def test_case_width_and_kana_differences_still_match():
    # 書き起こしは大文字小文字・全角半角・カタカナとひらがなが歌詞と違うことがある
    lines = ["ＨＥＬＬＯ ココロ"]
    words = _chars([(" hello", 0.0, 0.5), (" こころ", 1.0, 1.3)])
    result = build_aligned_lines(lines, words)
    assert result[0].char_times == [0.0, 0.1, 0.2, 0.3, 0.4, None, 1.0, 1.1, 1.2]


def test_emoji_and_symbols_are_none_and_text_is_unchanged():
    lines = ["🎵 ラララ ♪ (la la) 🐱✨", "★☆★"]
    words = _chars([("ラ", 1.0, 1.1), ("ラ", 1.1, 1.2), ("ラ", 1.2, 1.3), (" la", 2.0, 2.2), (" la", 2.2, 2.4)])
    result = build_aligned_lines(lines, words)
    _check_shape(lines, result)
    line = result[0]
    for ch, t in zip(line.text, line.char_times):
        if ch in "ラla":
            assert t is not None
        else:
            assert t is None, ch
    assert line.char_times[line.text.index("ラ")] == 1.0
    # 記号だけの行は全部 None
    assert result[1].char_times == [None, None, None]


def test_surrogate_pairs_count_as_one_codepoint():
    lines = ["𠮷野家🍚です"]
    words = _chars([("𠮷", 0.0, 0.3), ("野", 0.3, 0.6), ("家", 0.6, 0.9), ("です", 1.0, 1.4)])
    result = build_aligned_lines(lines, words)
    assert len(result[0].char_times) == 6
    assert result[0].char_times == [0.0, 0.3, 0.6, None, 1.0, 1.2]


def test_monotonic_across_lines():
    lines = ["あいう", "えお", "かき"]
    # "え" の時刻が前の行より前に戻っている（Whisper の誤り）
    words = _chars(
        [("あ", 1.0, 1.1), ("い", 1.1, 1.2), ("う", 1.2, 1.3), ("え", 0.2, 0.3), ("お", 2.0, 2.1),
         ("か", 3.0, 3.1), ("き", 3.1, 3.2)]
    )
    result = build_aligned_lines(lines, words)
    flat = _flat(result)
    assert flat == sorted(flat)
    assert result[1].char_times == [None, 2.0]  # 行頭の外れは補間せず None
    assert result[2].char_times == [3.0, 3.1]


def test_overlapping_words_inside_a_line_are_fixed_by_interpolation():
    lines = ["さくらさく"]
    words = _chars([("さ", 1.0, 1.2), ("く", 1.2, 1.4), ("ら", 0.9, 1.0), ("さ", 1.6, 1.8), ("く", 1.8, 2.0)])
    result = build_aligned_lines(lines, words)
    times = result[0].char_times
    assert times[2] == pytest.approx(1.4)  # 前後の文字から補間
    assert times == sorted(times)


def test_line_without_any_timing_is_all_none():
    lines = ["一行目です", "歌われない行", "三行目です"]
    words = _chars([("一", 1.0, 1.1), ("行", 1.1, 1.2), ("目", 1.2, 1.3), ("で", 1.3, 1.4), ("す", 1.4, 1.5),
                    ("三", 5.0, 5.1), ("行", 5.1, 5.2), ("目", 5.2, 5.3), ("で", 5.3, 5.4), ("す", 5.4, 5.5)])
    result = build_aligned_lines(lines, words)
    _check_shape(lines, result)
    assert all(t is None for t in result[1].char_times)
    assert result[0].char_times[0] == 1.0
    assert result[2].char_times[0] == 5.0


def test_line_with_few_accidental_matches_is_all_none():
    # 書き起こしに 1 文字だけ偶然一致した長い行は、合わせ損ねとして扱う
    lines = ["あいうえおかきくけこ", "さしすせそ"]
    words = _chars([("か", 1.0, 1.1), ("さ", 2.0, 2.1), ("し", 2.1, 2.2), ("す", 2.2, 2.3), ("せ", 2.3, 2.4),
                    ("そ", 2.4, 2.5)])
    result = build_aligned_lines(lines, words)
    assert result[0].char_times == [None] * 10
    assert result[1].char_times == [2.0, 2.1, 2.2, 2.3, 2.4]


def test_zero_duration_and_nan_words_are_ignored():
    lines = ["Morning light"]
    words = _chars([(" Morning", 10.0, 10.0), (" light", float("nan"), float("nan"))])
    result = build_aligned_lines(lines, words)
    assert result[0].char_times == [None] * len("Morning light")


def test_no_words_at_all():
    lines = ["歌詞はある", "でも音がない"]
    result = build_aligned_lines(lines, [])
    _check_shape(lines, result)
    assert _flat(result) == []


def test_transcript_with_different_kanji_still_maps_most_characters():
    # 書き起こしが漢字とかなで違うとき、一致した文字の間を補間する
    lines = ["君の声が聞こえる"]
    words = _chars([("きみ", 1.0, 1.4), ("の", 1.4, 1.6), ("声", 1.6, 1.8), ("が", 1.8, 2.0), ("きこえる", 2.0, 2.8)])
    result = build_aligned_lines(lines, words)
    times = result[0].char_times
    assert times[0] is None  # 行頭の「君」は一致しない
    assert times[1:4] == [1.4, 1.6, 1.8]
    assert times[4] == pytest.approx(2.0)  # 「聞」は「が」(1.8) と「こ」(2.2) の間で補間
    assert times[5:] == [2.2, 2.4, 2.6]
    flat = _flat(result)
    assert flat == sorted(flat)


def test_repeated_chorus_sung_twice():
    lines = ["ラララ", "ルルル", "ラララ"]
    words = _chars([("ラララ", 1.0, 1.6), ("ルルル", 2.0, 2.6), ("ラララ", 3.0, 3.6)])
    result = build_aligned_lines(lines, words)
    assert result[0].char_times == [1.0, 1.2, 1.4]
    assert result[1].char_times == [2.0, 2.2, 2.4]
    assert result[2].char_times == [3.0, 3.2, 3.4]


def test_unsung_line_does_not_push_out_the_next_line():
    # 歌われない注記の行に、align が次の行より遅い時刻を少しだけ詰め込んだとき
    lines = ["（ここは歌わない行です）", "風に乗って"]
    words = _chars([("ここ", 11.10, 11.22), ("は", 11.22, 11.3), (" 風", 11.0, 11.1), ("に", 11.34, 11.4),
                    ("乗", 11.42, 11.5), ("って", 11.58, 11.74)])
    result = build_aligned_lines(lines, words)
    assert all(t is None for t in result[0].char_times)
    assert result[1].char_times == [11.0, 11.34, 11.42, 11.58, 11.66]
