// 画面のあちこちで使う小さな道具。

/** 要素を作る。attrs の on* は addEventListener、それ以外は属性。子は文字列か Node。 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** 秒を「3:05」や「1:02:03」に。 */
export function formatDuration(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "—";
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
}

/** 経過時間を「12 秒」「3 分 05 秒」に。 */
export function formatElapsed(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "";
  const s = Math.floor(sec);
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  const r = String(s % 60).padStart(2, "0");
  if (m < 60) return `${m} 分 ${r} 秒`;
  return `${Math.floor(m / 60)} 時間 ${m % 60} 分`;
}

/** 曲の中の時刻（秒）を「0:12.3」に。 */
export function formatTime(sec) {
  if (!Number.isFinite(sec)) return "";
  const m = Math.floor(sec / 60);
  const s = (sec - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

const dateFmt = new Intl.DateTimeFormat("ja-JP", {
  year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
});

/** ISO 8601 を手元の時刻で「2026/11/26 14:05」に。 */
export function formatDate(iso) {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? "" : dateFmt.format(new Date(t));
}

export function formatBytes(n) {
  if (!Number.isFinite(n)) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/** ISO 8601 の秒。読めなければ null。 */
export function parseTime(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t / 1000;
}

/** media-dl の progress から割合（0〜100）を取り出す（cli.py の _percent と同じ考え方）。 */
export function percentOf(data) {
  if (!data || typeof data !== "object") return null;
  for (const k of ["percent", "percentage", "progress"]) {
    const v = data[k];
    if (typeof v === "number" && Number.isFinite(v)) return k === "progress" && v <= 1 ? v * 100 : v;
    if (typeof v === "string") {
      const f = parseFloat(v.trim().replace(/%$/, ""));
      if (Number.isFinite(f)) return f;
    }
  }
  const done = data.downloadedBytes ?? data.downloaded;
  const total = data.totalBytes ?? data.total;
  if (typeof done === "number" && typeof total === "number" && total > 0) return (100 * done) / total;
  return null;
}
