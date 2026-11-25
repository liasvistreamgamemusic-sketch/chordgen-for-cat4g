"""モジュールの間で受け渡すデータ型。

時刻はすべて音源の先頭からの秒（float）。コードのラベルは Harte 記法
（例 "A:min7"、"C:maj/3"、無音 "N"、不明 "X"）で持ち、cat4g の表記への変換は
chordgen.songformat だけが行う。音名はシャープ表記（"C#"）を基本とする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

STEM_NAMES = ("vocals", "drums", "bass", "other")

# コード認識に入れる音（要件「決めていないこと」。進め方の 4 で比べて選ぶ）
ChordInputMode = Literal["original", "no_drums", "no_drums_vocals", "boost_other"]
# ベースの認識結果の使い方
BassMode = Literal["off", "root_fix", "slash"]


@dataclass
class TrackMeta:
    """題名など。None は「分からない」。"""

    title: str | None = None
    artist: str | None = None
    source_url: str | None = None


@dataclass
class AudioInfo:
    path: Path
    duration_sec: float
    sample_rate: int
    meta: TrackMeta = field(default_factory=TrackMeta)  # 音源のタグ・ファイル名から取った値


@dataclass
class FetchResult:
    """media-dl で取得した結果。"""

    audio_path: Path
    meta: TrackMeta


@dataclass
class Stems:
    """Demucs の出力。paths のキーは STEM_NAMES。"""

    paths: dict[str, Path]
    sample_rate: int


@dataclass
class BeatGrid:
    beat_times: list[float]  # 各拍の時刻（昇順）
    downbeat_indices: list[int]  # beat_times のうち小節の 1 拍目の添字（昇順）
    bpm: float
    beats_per_bar: int  # 拍子の分子（例 4）
    beat_unit: int = 4  # 拍子の分母（1/2/4/8/16）


@dataclass
class KeyEstimate:
    tonic: str  # "A"、"F#" など
    mode: Literal["major", "minor"]
    confidence: float | None = None


@dataclass
class ChordSegment:
    start: float
    end: float
    label: str  # Harte 記法。"N" = 無音、"X" = 不明
    confidence: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)  # 補正の記録など（extensions に出す）


@dataclass
class BassNote:
    """拍や区間ごとのベースの音。pitch_class は 0=C … 11=B、None は音なし。"""

    start: float
    end: float
    pitch_class: int | None
    confidence: float | None = None


@dataclass
class AlignedLine:
    """歌詞 1 行と、各コードポイントの時刻。

    char_times は len(text) と同じ長さ。時刻が分からない文字（空白・記号など）は None。
    """

    text: str
    char_times: list[float | None]


@dataclass
class AnalysisResult:
    """パイプラインの結果。chordgen.songformat.build_document に渡す。"""

    audio: AudioInfo
    meta: TrackMeta  # 入力 → URL の情報 → タグ → ファイル名の順で確定した値
    beats: BeatGrid
    key: KeyEstimate | None
    chords: list[ChordSegment]  # 拍にそろえた後の最終結果
    lyrics: list[AlignedLine] | None = None
    extras: dict[str, Any] = field(default_factory=dict)  # song.extensions に出す解析独自の値
