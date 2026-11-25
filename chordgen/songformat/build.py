"""AnalysisResult から CaT4G Song Format v1 の文書（dict）を組み立てる。

決めたこと（要件の「決めていないこと」への答え）:
- 区間は曲全体で 1 つ（名前は「全体」）。行が 2000 を超えるときだけ「全体 (1/2)」のように分ける。
- 小節 1 は、最初のコード（N・X を除く）か最初に歌う文字を含む小節。最初に検出した 1 拍目より
  前は拍子どおりの仮の小節を前に足すので、弱起のコードは仮の小節（1 小節目）の途中に入る。
  拍の検出範囲の外のコードは、拍の間隔の中央値で延ばした仮の拍で小節・拍を数える。
- audio.offsetSec は 1 小節目の 1 拍目の時刻（負になるときは 0）。time は音源の時刻 − offsetSec。
- 歌詞のない行: bars_per_line 小節ごと。区切りは検出した最初の 1 拍目（歌詞があるときは歌い出しの
  小節）にそろえ、その前の余りの小節は短い行にする。position は 1 拍 = 1（コードポイント）で、
  前のコード名が重ならないように少し右へずらす。
- 歌詞のある行: 歌い出しの小節から始め、次の行の始まりまで。歌い終わりから次の行まで 2 小節以上
  空くときは、その間を歌詞のない行にする。
- 行をまたぐコードは、次の行の先頭で同じコードを書き直し、前の側に tieToNext を付ける。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from chordgen import APP_NAME, __version__
from chordgen.songformat.chordname import harte_to_cat4g, key_to_cat4g
from chordgen.songformat.grid import Grid, build_grid
from chordgen.types import AlignedLine, AnalysisResult

EXT_PREFIX = "analyzer."
SECTION_NAME = "全体"
MAX_LINES = 2000
MAX_CHORDS_PER_LINE = 200
MAX_MEASURES = 64
MERGE_GAP_SEC = 0.1  # 同じコードの間がこれ以下ならまとめる
INSTRUMENTAL_GAP_BARS = 2  # 歌い終わりから次の行までこれ以上空けば歌詞のない行にする
POS_PER_BEAT = 1.0  # 歌詞のない行で 1 拍あたりの position
_FORBIDDEN_KEYS = {"__proto__", "constructor", "prototype"}


@dataclass
class _Chord:
    start: float
    end: float
    name: str
    confidence: float | None
    extras: dict[str, Any] = field(default_factory=dict)
    p_start: float = 0.0  # 拍の位置
    p_end: float = 0.0


@dataclass
class _LineSpec:
    b0: int  # 格子の小節番号（0 始まり、この行の最初の小節）
    b1: int  # この行の次の小節
    aligned: AlignedLine | None = None


# ---- 小さな道具 ----


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _num(x: float, ndigits: int = 3) -> int | float:
    v = round(x, ndigits)
    if v == 0:
        return 0
    return int(v) if v == int(v) else v


def _half(x: float) -> float:
    return math.floor(x * 2 + 0.5) / 2


def _name_gap(name: str) -> float:
    """歌詞のない行で、コード名 name の次のコードを置けるまでの幅（全角 1 文字 = 1）。"""
    return max(1.0, math.ceil((len(name) * 0.5 + 0.5) * 2) / 2)


def _json_safe(value: Any, depth: int = 0) -> Any:
    """extensions に入れられる値（有限の数・文字列・真偽値・null・配列・オブジェクト）にする。"""
    if depth > 60:
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v, depth + 1)
            for k, v in value.items()
            if str(k) not in _FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, depth + 1) for v in value]
    item = getattr(value, "item", None)  # numpy のスカラー
    if callable(item):
        try:
            return _json_safe(item(), depth + 1)
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)  # numpy の配列
    if callable(tolist):
        return _json_safe(tolist(), depth + 1)
    return str(value)


def _ext(values: dict[str, Any]) -> dict[str, Any]:
    """解析独自の値を、名前空間 analyzer. をつけて extensions にする。"""
    out: dict[str, Any] = {}
    for k, v in values.items():
        key = str(k)
        if key in _FORBIDDEN_KEYS:
            continue
        if not key.startswith(EXT_PREFIX):
            key = EXT_PREFIX + key
        out[key] = _json_safe(v, 1)
    return out


def _text(s: str | None, limit: int | None = None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return s[:limit] if limit else s


# ---- コードの準備 ----


def _prepare_chords(result: AnalysisResult) -> list[_Chord]:
    """N・X を除いて cat4g の表記にし、同じコードが続くところをまとめる。"""
    duration = _finite(result.audio.duration_sec)
    items: list[_Chord] = []
    for seg in sorted(result.chords, key=lambda s: (s.start, s.end)):
        start, end = _finite(seg.start), _finite(seg.end)
        if start is None or end is None:
            continue
        start = max(0.0, start)
        if duration is not None and duration > start:
            end = min(end, duration)
        if end <= start:
            continue
        name = harte_to_cat4g(seg.label, result.key)
        if name is None:
            continue
        items.append(_Chord(start, end, name[:200], _finite(seg.confidence), dict(seg.extras)))

    merged: list[_Chord] = []
    for c in items:
        if merged:
            prev = merged[-1]
            if c.start < prev.end:  # 重なりは後ろのコードを優先する
                prev.end = c.start
            if prev.name == c.name and c.start - prev.end <= MERGE_GAP_SEC:
                prev.confidence = _weighted_conf(prev, c)
                prev.end = max(prev.end, c.end)
                continue
            if prev.end <= prev.start:
                merged.pop()
        merged.append(c)
    return merged


def _weighted_conf(a: _Chord, b: _Chord) -> float | None:
    if a.confidence is None or b.confidence is None:
        return a.confidence if b.confidence is None else b.confidence
    wa, wb = a.end - a.start, b.end - b.start
    return (a.confidence * wa + b.confidence * wb) / (wa + wb) if wa + wb > 0 else a.confidence


def _prepare_lyrics(lines: list[AlignedLine] | None) -> list[AlignedLine]:
    out: list[AlignedLine] = []
    for line in lines or []:
        text = line.text.replace("\r", "").replace("\n", " ")
        if not text.strip():
            continue
        text = text[:10000]
        times: list[float | None] = []
        for i in range(len(text)):
            t = _finite(line.char_times[i]) if i < len(line.char_times) else None
            times.append(t if t is not None and t >= 0 else None)
        out.append(AlignedLine(text=text, char_times=times))
    return out


def _timed(line: AlignedLine) -> list[tuple[int, float]]:
    return [(i, t) for i, t in enumerate(line.char_times) if t is not None]


# ---- 行の割り付け ----


def _chunk(a: int, b: int, size: int, phase: int) -> list[_LineSpec]:
    """小節 a〜b-1 を、(小節 − phase) が size の倍数になるところで区切る。"""
    out: list[_LineSpec] = []
    s = a
    while s < b:
        nxt = s + ((phase - s) % size or size)
        nxt = min(nxt, b, s + MAX_MEASURES)
        out.append(_LineSpec(s, nxt))
        s = nxt
    return out


def _lyric_specs(grid: Grid, lyrics: list[AlignedLine], first_bar: int, end_bar: int, bpl: int) -> list[_LineSpec]:
    starts: list[int] = []
    sung_end: list[int] = []  # 歌い終わりの小節の次
    for line in lyrics:
        timed = _timed(line)
        floor = starts[-1] + 1 if starts else first_bar
        if timed:
            first = min(t for _, t in timed)
            last = max(t for _, t in timed)
            s = max(grid.bar_of_time(first), floor)
            e = max(grid.bar_of_time(last) + 1, s + 1)
        else:  # 時刻が分からない行は前の行のすぐ後ろに置く
            s = floor
            e = s + 1
        starts.append(s)
        sung_end.append(e)
    total_end = max(end_bar, starts[-1] + 1)

    specs = _chunk(first_bar, starts[0], bpl, phase=starts[0])  # 前奏
    for i, line in enumerate(lyrics):
        nxt = starts[i + 1] if i + 1 < len(lyrics) else total_end
        end = nxt
        if nxt - sung_end[i] >= INSTRUMENTAL_GAP_BARS:
            end = sung_end[i]
        lyric_end = min(end, starts[i] + MAX_MEASURES)
        specs.append(_LineSpec(starts[i], lyric_end, line))
        specs.extend(_chunk(lyric_end, nxt, bpl, phase=lyric_end))  # 間奏・後奏
    return specs


# ---- コードと歌詞の位置 ----


def _pieces(spec: _LineSpec, grid: Grid, chords: list[_Chord]) -> list[tuple[_Chord, float, float, bool, bool]]:
    """行に入るコードの部分を (コード, 始まり, 終わり, 前の行から続く, 次の行へ続く) で返す。

    chords は始まりの順で重なりがない（_prepare_chords の結果）ので、終わりも昇順。
    """
    p0, p1 = grid.bar_start_pos(spec.b0), grid.bar_start_pos(spec.b1)
    out = []
    for c in chords[bisect_right(chords, p0, key=lambda c: c.p_end) :]:
        if c.p_start >= p1:
            break
        s, e = max(c.p_start, p0), min(c.p_end, p1)
        if e - s <= 0:
            continue
        out.append((c, s, e, c.p_start < p0, c.p_end > p1))
    return out


def _lyric_position(line: AlignedLine, t: float, grid: Grid) -> float:
    timed = _timed(line)
    if not timed:
        return 0.0
    last_t = max(ct for _, ct in timed)
    after = grid.beat_pos(t) - grid.beat_pos(last_t)
    if after > 1:  # 歌い終わりより 1 拍を超えて後ろのコードは歌詞の後ろに置く
        return len(line.text) + _half(max(0.0, after - 1) * POS_PER_BEAT)
    idx, _ = min(timed, key=lambda it: (abs(it[1] - t), it[0]))
    return float(idx)


def _line_dict(
    spec: _LineSpec, grid: Grid, chords: list[_Chord], sec_first_bar: int, offset: float
) -> dict[str, Any]:
    p0 = grid.bar_start_pos(spec.b0)
    pieces = _pieces(spec, grid, chords)
    if len(pieces) > MAX_CHORDS_PER_LINE:  # 1 小節に 200 を超えるときだけ。長いものを残す
        keep = sorted(pieces, key=lambda p: p[2] - p[1], reverse=True)[:MAX_CHORDS_PER_LINE]
        pieces = sorted(keep, key=lambda p: p[1])
    out_chords: list[dict[str, Any]] = []
    prev_pos: float | None = None
    prev_name = ""
    for c, s, e, continued, tie in pieces:
        t_audio = grid.time_at(s) if continued else c.start
        if spec.aligned is None:
            pos = _half((s - p0) * POS_PER_BEAT)
            if prev_pos is not None:
                pos = max(pos, prev_pos + _name_gap(prev_name))
        else:
            pos = _lyric_position(spec.aligned, t_audio, grid)
            if prev_pos is not None:
                pos = max(pos, prev_pos + 1)
        prev_pos, prev_name = pos, c.name
        bar = grid.bar_of_pos(s)
        item: dict[str, Any] = {
            "chord": c.name,
            "position": _num(pos, 1),
            "bar": bar - sec_first_bar + 1,
            "beat": _num(s - grid.bar_start_pos(bar) + 1, 2),
            "duration": _num(e - s, 2),
            "time": _num(max(0.0, t_audio - offset), 3),
        }
        if tie:
            item["tieToNext"] = True
        ext: dict[str, Any] = {}
        if c.confidence is not None:
            ext["confidence"] = round(c.confidence, 3)
        if continued:
            ext["continued"] = True
        ext.update(c.extras)
        if ext:
            item["extensions"] = _ext(ext)
        out_chords.append(item)

    line: dict[str, Any] = {
        "lyrics": spec.aligned.text if spec.aligned else "",
        "measures": spec.b1 - spec.b0,
        "memo": None,
        "chords": out_chords,
    }
    if spec.aligned is not None:
        timings = []
        for idx, ct in _timed(spec.aligned):
            pos = round(grid.beat_pos(ct), 2)
            bar = grid.bar_of_pos(pos)
            entry: dict[str, Any] = {"at": idx}
            if bar >= sec_first_bar:
                entry["bar"] = bar - sec_first_bar + 1
                entry["beat"] = _num(max(1.0, pos - grid.bar_start_pos(bar) + 1), 2)
            if ct >= offset:
                entry["time"] = _num(ct - offset, 3)
            timings.append(entry)
        if timings:
            line["lyricTimings"] = timings
    return line


def _fit_chord_limit(specs: list[_LineSpec], grid: Grid, chords: list[_Chord]) -> list[_LineSpec]:
    """コードが 200 を超える行を、小節の半分で分ける（後ろ半分は歌詞なしの行）。"""
    out: list[_LineSpec] = []
    todo = list(specs)
    while todo:
        spec = todo.pop(0)
        if spec.b1 - spec.b0 > 1 and len(_pieces(spec, grid, chords)) > MAX_CHORDS_PER_LINE:
            mid = (spec.b0 + spec.b1) // 2
            todo[:0] = [_LineSpec(spec.b0, mid, spec.aligned), _LineSpec(mid, spec.b1)]
            continue
        out.append(spec)
    return out


# ---- 文書 ----


def _section(name: str, lines: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "name": name,
        "repeatCount": 1,
        "capo": None,
        "transpose": None,
        "bpm": None,
        "playbackSpeed": None,
        "timeSignature": None,
        "tempoChange": None,
        "repeatMark": None,
        "lines": lines,
        "extensions": {EXT_PREFIX + "sectionLabel": label},
    }


def _exported_at() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def build_document(result: AnalysisResult, *, bars_per_line: int = 4) -> dict:
    """AnalysisResult から封筒つきの文書を作る（検証は validate_document）。"""
    if not isinstance(bars_per_line, int) or not 1 <= bars_per_line <= MAX_MEASURES:
        raise ValueError(f"bars_per_line は 1〜{MAX_MEASURES} の整数: {bars_per_line!r}")
    bg = result.beats
    lyrics = _prepare_lyrics(result.lyrics)
    chords = _prepare_chords(result)

    content: list[float] = [c.start for c in chords] + [c.end for c in chords]
    content += [t for line in lyrics for _, t in _timed(line)]
    sections: list[dict[str, Any]] = []
    offset = 0.0
    if content:
        t_first, t_last = min(content), max(content)
        grid = build_grid(bg, t_first, t_last)
        for c in chords:
            c.p_start = round(grid.beat_pos(c.start), 2)
            c.p_end = round(grid.beat_pos(c.end), 2)
        chords = [c for c in chords if c.p_end > c.p_start]
        first_bar = grid.bar_of_time(t_first)
        last_chord = max((c.p_end for c in chords), default=None)
        end_bar = first_bar + 1
        if last_chord is not None:
            end_bar = max(end_bar, grid.bar_of_pos(last_chord - 0.01) + 1)
        lyric_times = [t for line in lyrics for _, t in _timed(line)]
        if lyric_times:
            end_bar = max(end_bar, grid.bar_of_time(max(lyric_times)) + 1)
        offset = max(0.0, round(grid.bar_start_time(first_bar), 3))

        if lyrics:
            specs = _lyric_specs(grid, lyrics, first_bar, end_bar, bars_per_line)
            need = max(s.b1 for s in specs)
            if need > grid.n_bars:  # 時刻の分からない行で後ろへ押し出したぶん、格子を延ばす
                extra = (need - grid.n_bars + 1) * max(1, bg.beats_per_bar or 4) * grid.period
                grid = build_grid(bg, t_first, t_last + extra)
        else:
            phase = grid.first_detected_bar if grid.first_detected_bar >= first_bar else first_bar
            specs = _chunk(first_bar, end_bar, bars_per_line, phase)
        specs = _fit_chord_limit(specs, grid, chords)

        groups = [specs[i : i + MAX_LINES] for i in range(0, len(specs), MAX_LINES)]
        for gi, group in enumerate(groups):
            sec_first = group[0].b0
            lines = [_line_dict(s, grid, chords, sec_first, offset) for s in group]
            name = SECTION_NAME if len(groups) == 1 else f"{SECTION_NAME} ({gi + 1}/{len(groups)})"
            sections.append(_section(name, lines, "whole"))
    else:
        empty = {"lyrics": "", "measures": 1, "memo": None, "chords": []}
        sections.append(_section(SECTION_NAME, [empty], "whole"))

    return {
        "format": "cat4g-song",
        "version": 1,
        "exportedAt": _exported_at(),
        "appVersion": f"{APP_NAME} {__version__}",
        "song": _song(result, sections, offset),
    }


def _song(result: AnalysisResult, sections: list[dict[str, Any]], offset: float) -> dict[str, Any]:
    meta, audio, bg = result.meta, result.audio, result.beats
    title = (
        _text(meta.title, 1000)
        or _text(audio.meta.title, 1000)
        or _text(Path(audio.path).stem, 1000)
        or "無題"
    )
    artist = _text(meta.artist) or _text(audio.meta.artist)
    source_url = _text(meta.source_url) or _text(audio.meta.source_url)
    bpm = _finite(bg.bpm)
    bpm_value = _num(bpm, 1) if bpm is not None and 1 <= round(bpm, 1) <= 999 else None
    beats = bg.beats_per_bar if isinstance(bg.beats_per_bar, int) and 1 <= bg.beats_per_bar <= 32 else 4
    unit = bg.beat_unit if bg.beat_unit in (1, 2, 4, 8, 16) else 4

    audio_obj: dict[str, Any] = {}
    audio_title = _text(audio.meta.title) or _text(Path(audio.path).name)
    if audio_title:
        audio_obj["title"] = audio_title
    if source_url:
        audio_obj["url"] = source_url
    dur = _finite(audio.duration_sec)
    if dur is not None and dur >= 0:
        audio_obj["durationSec"] = _num(dur, 3)
    audio_obj["offsetSec"] = _num(offset, 3)

    ext: dict[str, Any] = {"name": APP_NAME, "version": __version__}
    if result.key is not None:
        ext["key"] = {"tonic": result.key.tonic, "mode": result.key.mode}
        if _finite(result.key.confidence) is not None:
            ext["keyConfidence"] = round(float(result.key.confidence), 3)
    if source_url:
        ext["sourceUrl"] = source_url
    if bpm is not None:
        ext["bpmEstimate"] = round(bpm, 3)
    ext.update(result.extras)

    return {
        "title": title,
        "artist": artist,
        "originalKey": key_to_cat4g(result.key),
        "bpm": bpm_value,
        "timeSignature": {"beats": beats, "unit": unit},
        "capo": 0,
        "transpose": 0,
        "playbackSpeed": 1,
        "tuning": "standard",
        "difficulty": None,
        "sourceUrl": source_url,
        "notes": None,
        "tags": [],
        "voicings": {},
        "audio": audio_obj,
        "sections": sections,
        "extensions": _ext(ext),
    }
