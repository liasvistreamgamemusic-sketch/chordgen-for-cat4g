// .cat4g.json（CaT4G Song Format v1）を cat4g の画面に近い形で見せる。
// position は Unicode のコードポイント単位（仕様 9 章）。表示は書記素ごとにまとめ、
// 書記素の中のどのコードポイントかで横の位置を割り振る（絵文字の並びを崩さないため）。

import { el, formatDuration, formatTime } from "./util.js";

// 確信度の区切り。これより低いものを色で示す。
export const CONFIDENCE_LOW = 0.5;
export const CONFIDENCE_MID = 0.7;

export function confidenceOf(chord) {
  const ext = chord && chord.extensions;
  if (!ext || typeof ext !== "object") return null;
  const v = ext["analyzer.confidence"];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function confidenceLevel(conf) {
  if (conf === null) return "none";
  if (conf < CONFIDENCE_LOW) return "low";
  if (conf < CONFIDENCE_MID) return "mid";
  return "ok";
}

/** 文字列を書記素に分け、各書記素が何コードポイント目から何個かを返す。 */
export function splitGraphemes(text) {
  const out = [];
  let cp = 0;
  const push = (s) => {
    const n = Array.from(s).length;
    out.push({ text: s, start: cp, length: n });
    cp += n;
  };
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    const seg = new Intl.Segmenter("ja", { granularity: "grapheme" });
    for (const { segment } of seg.segment(text)) push(segment);
  } else {
    for (const ch of Array.from(text)) push(ch);
  }
  return out;
}

/**
 * コードの position を、書記素の何番目の中のどの割合の位置かに直す。
 * 歌詞の後ろ（position が長さ以上）は、長さを超えた分だけ 1 文字分の空きを足した位置にする。
 */
export function locate(graphemes, position, cpLength) {
  const p = Math.max(0, Number(position) || 0);
  if (p >= cpLength) {
    const extra = p - cpLength;
    return { index: graphemes.length + Math.floor(extra), frac: extra - Math.floor(extra) };
  }
  for (let i = 0; i < graphemes.length; i++) {
    const g = graphemes[i];
    if (p < g.start + g.length) return { index: i, frac: (p - g.start) / g.length };
  }
  return { index: graphemes.length, frac: 0 };
}

function chordTitle(chord, conf) {
  const parts = [];
  if (conf !== null) parts.push(`確信度 ${conf.toFixed(2)}`);
  else parts.push("確信度なし");
  if (Number.isFinite(chord.bar)) {
    parts.push(Number.isFinite(chord.beat) ? `${chord.bar} 小節目 ${trimNum(chord.beat)} 拍目` : `${chord.bar} 小節目`);
  }
  if (Number.isFinite(chord.time)) parts.push(formatTime(chord.time));
  if (Number.isFinite(chord.duration)) parts.push(`${trimNum(chord.duration)} 拍`);
  return parts.join("、");
}

function trimNum(n) {
  return String(Math.round(n * 100) / 100);
}

function chordLabel(chord) {
  const conf = confidenceOf(chord);
  const level = confidenceLevel(conf);
  const name = chord.chord === "" ? "（空）" : String(chord.chord);
  const node = el("span", { class: `chord conf-${level}`, title: chordTitle(chord, conf), "data-position": chord.position }, name);
  if (level === "low" || level === "mid") {
    node.append(el("span", { class: "visually-hidden" }, level === "low" ? "（確信度が低い）" : "（確信度がやや低い）"));
  }
  return { node, level };
}

function renderLyricLine(line, stats) {
  const text = String(line.lyrics);
  const graphemes = splitGraphemes(text);
  const cpLength = graphemes.length ? graphemes.at(-1).start + graphemes.at(-1).length : 0;
  const chords = [...(line.chords || [])].sort((a, b) => a.position - b.position);

  const anchors = new Map();
  let needed = graphemes.length;
  for (const c of chords) {
    const loc = locate(graphemes, c.position, cpLength);
    needed = Math.max(needed, loc.index + 1);
    if (!anchors.has(loc.index)) anchors.set(loc.index, []);
    anchors.get(loc.index).push({ chord: c, frac: loc.frac });
  }

  const row = el("div", { class: "line line-lyric" });
  for (let i = 0; i < needed; i++) {
    const g = graphemes[i];
    const cell = el("span", { class: g ? "g" : "g g-pad", "data-cp": g ? g.start : cpLength + i - graphemes.length }, g ? g.text : "\u3000");
    if (g && /^\s+$/.test(g.text)) cell.classList.add("g-space");
    for (const a of anchors.get(i) || []) {
      const { node, level } = chordLabel(a.chord);
      stats[level]++;
      node.style.left = `${(a.frac * 100).toFixed(1)}%`;
      cell.append(node);
    }
    row.append(cell);
  }
  return row;
}

/** 歌詞のない行。小節の枠を描き、拍の位置にコードを置く。 */
function renderChordLine(line, beatsPerBar, stats) {
  const chords = [...(line.chords || [])].sort((a, b) => a.position - b.position);
  const measures = Math.max(1, line.measures | 0);
  const hasGrid = chords.length > 0 && chords.every((c) => Number.isFinite(c.bar) && Number.isFinite(c.beat));
  const row = el("div", { class: "line line-bars" });
  row.style.setProperty("--beats", String(beatsPerBar));
  if (hasGrid) {
    const firstBar = Math.min(...chords.map((c) => c.bar));
    for (let m = 0; m < measures; m++) {
      const bar = el("span", { class: "bar" });
      for (const c of chords.filter((x) => x.bar - firstBar === m)) {
        const { node, level } = chordLabel(c);
        stats[level]++;
        const frac = Math.min(Math.max((c.beat - 1) / beatsPerBar, 0), 0.999);
        node.style.left = `${(frac * 100).toFixed(1)}%`;
        bar.append(node);
      }
      row.append(bar);
    }
    // 行の小節数より後ろの小節を指すコードは、最後の小節に寄せずに後ろに足す
    const extra = chords.filter((x) => x.bar - firstBar >= measures);
    if (extra.length) {
      const bar = el("span", { class: "bar bar-extra" });
      for (const c of extra) {
        const { node, level } = chordLabel(c);
        stats[level]++;
        bar.append(node);
      }
      row.append(bar);
    }
  } else {
    // bar・beat がなければ position を 1 拍として並べる
    row.classList.add("line-free");
    for (const c of chords) {
      const { node, level } = chordLabel(c);
      stats[level]++;
      node.style.left = `calc(${Number(c.position) || 0} * var(--beat-w))`;
      row.append(node);
    }
    const span = chords.length ? Math.max(...chords.map((c) => Number(c.position) || 0)) + 2 : 1;
    row.style.setProperty("--span", String(span));
  }
  return row;
}

function metaItem(label, value) {
  return el("div", { class: "meta-item" }, el("dt", {}, label), el("dd", {}, value));
}

/** 文書全体を描く。戻り値は { node, stats }。 */
export function renderSong(doc) {
  const song = doc && doc.song;
  if (!song || !Array.isArray(song.sections)) {
    return { node: el("p", { class: "form-error" }, "このファイルは CaT4G Song Format として読めません。"), stats: null };
  }
  const stats = { ok: 0, mid: 0, low: 0, none: 0 };
  const ts = song.timeSignature || { beats: 4, unit: 4 };
  const wrap = el("article", { class: "song" });

  const head = el("header", { class: "song-head" });
  head.append(el("h3", { class: "song-title" }, song.title || "（題名なし）"));
  head.append(el("p", { class: "song-artist" }, song.artist || "アーティスト不明"));
  const meta = el("dl", { class: "song-meta" });
  meta.append(metaItem("キー", song.originalKey || "—"));
  meta.append(metaItem("BPM", song.bpm != null ? String(Math.round(song.bpm * 10) / 10) : "—"));
  meta.append(metaItem("拍子", `${ts.beats}/${ts.unit}`));
  const dur = song.audio && song.audio.durationSec;
  meta.append(metaItem("長さ", Number.isFinite(dur) ? formatDuration(dur) : "—"));
  if (song.capo) meta.append(metaItem("カポ", String(song.capo)));
  head.append(meta);
  wrap.append(head);

  const legend = el("div", { class: "legend" });
  wrap.append(legend);

  const body = el("div", { class: "song-body" });
  for (const sec of song.sections) {
    const beatsPerBar = (sec.timeSignature && sec.timeSignature.beats) || ts.beats || 4;
    const s = el("section", { class: "sec" });
    const name = el("h4", { class: "sec-name" }, sec.name || "（区間名なし）");
    if (sec.repeatCount > 1) name.append(el("span", { class: "sec-repeat" }, `×${sec.repeatCount}`));
    s.append(name);
    for (const line of sec.lines || []) {
      const hasLyrics = typeof line.lyrics === "string" && line.lyrics.length > 0;
      s.append(hasLyrics ? renderLyricLine(line, stats) : renderChordLine(line, beatsPerBar, stats));
    }
    body.append(s);
  }
  wrap.append(body);

  const total = stats.ok + stats.mid + stats.low + stats.none;
  legend.append(
    el("span", { class: "legend-title" }, "確信度"),
    el("span", { class: "legend-item" }, el("span", { class: "chord conf-ok swatch" }, "C"), ` ${CONFIDENCE_MID} 以上`),
    el("span", { class: "legend-item" }, el("span", { class: "chord conf-mid swatch" }, "C"), ` ${CONFIDENCE_LOW}〜${CONFIDENCE_MID}（${stats.mid}）`),
    el("span", { class: "legend-item" }, el("span", { class: "chord conf-low swatch" }, "C"), ` ${CONFIDENCE_LOW} 未満（${stats.low}）`),
  );
  if (stats.none) legend.append(el("span", { class: "legend-item" }, el("span", { class: "chord conf-none swatch" }, "C"), ` 値なし（${stats.none}）`));
  legend.append(el("span", { class: "legend-note" }, `コード ${total} 個。色の付いたコードから聞き直してください。コード名は実際に鳴る音です。`));
  return { node: wrap, stats };
}

/**
 * 描いた後に、同じ行で重なったコード名を右へずらす（cat4g の画面と同じく、近いコードは詰めて読めるようにする）。
 */
export function spreadChords(root) {
  for (const line of root.querySelectorAll(".line-lyric, .line-free, .line-bars")) {
    const nodes = Array.from(line.querySelectorAll(".chord"));
    for (const n of nodes) n.style.marginLeft = "";
    let prevRight = -Infinity;
    let prevTop = null;
    for (const n of nodes) {
      const r = n.getBoundingClientRect();
      if (prevTop !== null && Math.abs(r.top - prevTop) > 4) prevRight = -Infinity;
      const gap = 6;
      if (r.left < prevRight + gap) {
        const shift = prevRight + gap - r.left;
        n.style.marginLeft = `${shift}px`;
        prevRight = r.right + shift;
      } else {
        prevRight = r.right;
      }
      prevTop = r.top;
    }
  }
}
