// chordgen の画面。入力 → ジョブ（SSE で進捗）→ 出力のプレビュー。

import { api, submitJob, setHandlers, ApiError } from "./api.js";
import { renderSong, spreadChords } from "./preview.js";
import { el, formatBytes, formatDate, formatElapsed, parseTime, percentOf } from "./util.js";

const $ = (id) => document.getElementById(id);

const TERMINAL = new Set(["done", "failed", "cancelled"]);
const STATUS_TEXT = { queued: "待ち", running: "実行中", done: "完了", failed: "失敗", cancelled: "中止" };
const STAGE_TEXT = {
  waiting: "待ち", running: "実行中", done: "済み", skipped: "飛ばした", failed: "失敗",
  unused: "使わない", notrun: "—", cancelled: "中止",
};
const FINISHED_SHOWN = 5;
// ブラウザは同じオリジンへの接続を 6 本までしか張らないので、SSE は実行中のジョブと、待ちのジョブの一部だけに張る
// （残りの待ちのジョブは一覧を数秒ごとに取り直して見る）
const MAX_SSE = 3;
const LOG_LIMIT = 3000;

// 既定の段（/api/info が読めないときの予備。pipeline.STAGE_LABELS と同じ並び）
const FALLBACK_STAGES = [
  ["fetch", "音源の取得"], ["probe", "音源の読み込み"], ["separate", "音源の分離"], ["beats", "ビート検出"],
  ["chords", "コード認識"], ["key", "キー推定"], ["snap", "拍にそろえる"], ["bass", "ベースで補正"],
  ["lyrics", "歌詞の位置合わせ"], ["write", "書き出し"],
].map(([stage, label]) => ({ stage, label }));

const state = {
  info: null,
  stages: FALLBACK_STAGES,
  jobs: new Map(), // id → JobView
  outputs: [],
  selected: null, // 選んでいる出力の name
  selectedDoc: null,
  file: null,
  autoPreview: new Set(), // 終わったら下書きを開くジョブの id
  shutDown: false,
};

// ---- 画面上部の案内 ---------------------------------------------------------

let bannerKind = null;
function showBanner(kind, text) {
  if (bannerKind === "forbidden" && kind !== "forbidden") return; // 403 の案内を優先
  bannerKind = kind;
  const b = $("banner");
  b.textContent = text;
  b.dataset.kind = kind;
  b.hidden = false;
}
function clearBanner(kind) {
  if (bannerKind !== kind) return;
  bannerKind = null;
  $("banner").hidden = true;
}

setHandlers({
  forbidden: () => showBanner("forbidden", "この画面を開く権限がありません。コンソールに出ている URL から開き直してください。"),
  unreachable: () => {
    if (!state.shutDown) showBanner("unreachable", "サーバーにつながりません。run.bat のコンソールが開いたままか確かめてください。つながると自動で戻ります。");
  },
  reachable: () => clearBanner("unreachable"),
});

let liveTimer = 0;
function announce(text) {
  const live = $("live");
  live.textContent = "";
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => { live.textContent = text; }, 50);
}

function showError(node, err) {
  if (!err) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.textContent = err instanceof Error ? err.message : String(err);
  node.hidden = false;
}

// ---- ヘッダ（版・デバイス・終了） ------------------------------------------

let infoTimer = 0;
async function loadInfo() {
  clearTimeout(infoTimer);
  try {
    const info = await api.info();
    const first = !state.info;
    state.info = info;
    if (Array.isArray(info.stages) && info.stages.length) state.stages = info.stages;
    renderHeader();
    if (first) {
      renderAdvanced();
      renderUrlAvailability();
      for (const v of state.jobs.values()) v.render();
    }
    if (!info.device || info.device.kind === "unknown") infoTimer = setTimeout(loadInfo, 3000);
  } catch (e) {
    $("device-text").textContent = "サーバーの情報を読めません";
    $("device").dataset.kind = "error";
    if (!state.info) {
      $("advanced-body").replaceChildren(el("p", { class: "form-error" }, `設定の選択肢を読めませんでした: ${e.message}`));
    }
    if (!state.shutDown && !(e instanceof ApiError && e.status === 403)) infoTimer = setTimeout(loadInfo, 5000);
  }
}

function renderHeader() {
  const info = state.info;
  $("app-version").textContent = info.version ? `版 ${info.version}` : "";
  const d = info.device || { kind: "unknown" };
  const box = $("device");
  box.dataset.kind = d.kind;
  let text;
  if (d.kind === "cuda") text = `GPU: ${d.name || "名前不明"}`;
  else if (d.kind === "cpu") {
    text = `CPU${d.name ? `: ${d.name}` : ""}` + (d.torchAvailable === false ? "（PyTorch が見つかりません）" : "（GPU が使えないので時間がかかります）");
  }
  else text = "使うデバイスを調べています…";
  $("device-text").textContent = text;
  box.title = d.kind === "cpu" ? `${text}。GPU が使えないので、解析にとても時間がかかります。` : text;
}

function setupShutdown() {
  const btn = $("shutdown-btn");
  btn.addEventListener("click", () => {
    const active = [...state.jobs.values()].filter((v) => !TERMINAL.has(v.job.status));
    if (active.length) {
      openInlineConfirm(btn, `解析中のジョブ（${active.length} 件）があります。終了すると止まります。`, "終了する", doShutdown);
    } else {
      doShutdown();
    }
  });
}

async function doShutdown() {
  try {
    await api.shutdown();
  } catch (e) {
    if (!(e instanceof ApiError && e.code === "NETWORK")) {
      showBanner("error", `終了できませんでした: ${e.message}`);
      return;
    }
  }
  state.shutDown = true;
  clearTimeout(infoTimer);
  for (const v of state.jobs.values()) v.close();
  $("banner").hidden = true;
  $("shutdown-screen").hidden = false;
  document.title = "chordgen（終了しました）";
}

/** ボタンの横に「〜しますか？ [する] [やめる]」を出す。ブラウザの確認ダイアログの代わり。 */
function openInlineConfirm(anchor, message, okText, onOk) {
  const prev = anchor.parentElement.querySelector(".inline-confirm");
  if (prev) prev.remove();
  const cancel = el("button", { type: "button", class: "btn btn-small" }, "やめる");
  const ok = el("button", { type: "button", class: "btn btn-small btn-danger" }, okText);
  const box = el("span", { class: "inline-confirm", role: "group", "aria-label": message },
    el("span", { class: "inline-confirm-text" }, message), ok, cancel);
  const close = () => {
    box.remove();
    anchor.hidden = false;
    anchor.focus();
  };
  cancel.addEventListener("click", close);
  ok.addEventListener("click", async () => {
    ok.disabled = cancel.disabled = true;
    try {
      await onOk();
    } finally {
      if (box.isConnected) {
        box.remove();
        anchor.hidden = false;
      }
    }
  });
  box.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      close();
    }
  });
  anchor.hidden = true;
  anchor.after(box);
  cancel.focus();
}

// ---- 入力 -------------------------------------------------------------------

const AUDIO_EXT = /\.(wav|mp3|flac|m4a|aac|ogg|oga|opus|wma|aiff?|alac|mp4|m4v|webm|mkv|mov)$/i;

function setFile(file) {
  state.file = file;
  $("drop-empty").hidden = !!file;
  $("drop-chosen").hidden = !file;
  $("drop").classList.toggle("has-file", !!file);
  if (file) {
    $("file-name").textContent = file.name;
    const odd = !(file.type.startsWith("audio/") || file.type.startsWith("video/") || AUDIO_EXT.test(file.name));
    $("file-size").textContent = formatBytes(file.size) + (odd ? "（音源ではないかもしれません）" : "");
    if ($("url-input").value) $("url-input").value = "";
  } else {
    $("file-input").value = "";
  }
  showError($("form-error"), null);
}

function setupInput() {
  const drop = $("drop");
  const fileInput = $("file-input");
  $("file-choose").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });
  $("file-clear").addEventListener("click", () => {
    setFile(null);
    $("file-choose").focus();
  });

  // 画面のどこに落としてもブラウザがファイルを開かないようにし、置き場所の上だけで受け取る
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  window.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener("drop", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  drop.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    depth++;
    drop.classList.add("is-over");
  });
  drop.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) drop.classList.remove("is-over");
  });
  drop.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  drop.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    drop.classList.remove("is-over");
    const files = e.dataTransfer.files;
    if (files && files.length) {
      setFile(files[0]);
      if (files.length > 1) announce(`ファイルは 1 つずつ解析します。「${files[0].name}」を使います。`);
      else announce(`「${files[0].name}」を選びました。`);
    }
  });

  $("lyrics-input").addEventListener("input", () => clearFieldError($("lyrics-input")));
  $("url-input").addEventListener("input", () => {
    if ($("url-input").value.trim() && state.file) setFile(null);
    clearFieldError($("url-input"));
  });

  const lyricsFile = $("lyrics-file");
  $("lyrics-file-btn").addEventListener("click", () => lyricsFile.click());
  lyricsFile.addEventListener("change", async () => {
    const f = lyricsFile.files && lyricsFile.files[0];
    if (!f) return;
    try {
      const text = new TextDecoder("utf-8", { fatal: true }).decode(await f.arrayBuffer());
      $("lyrics-input").value = text.replace(/^﻿/, "").replace(/\r\n?/g, "\n");
      announce(`歌詞を「${f.name}」から読みました。`);
    } catch {
      showError($("form-error"), `「${f.name}」を UTF-8 のテキストとして読めません。UTF-8 で保存し直してください。`);
    }
    lyricsFile.value = "";
  });

  $("job-form").addEventListener("submit", onSubmit);
}

function renderUrlAvailability() {
  const ok = state.info ? state.info.mediaDlAvailable !== false : true;
  const input = $("url-input");
  input.disabled = !ok;
  const note = $("url-note");
  if (ok) {
    note.textContent = "ファイルと URL はどちらか一方を使います。";
    note.classList.remove("hint-warn");
  } else {
    input.placeholder = "使えません";
    note.textContent = "URL からの取得に使う media-dl が見つかりません。アプリのフォルダで setup.bat を実行して入れてください。ファイルの入力は使えます。";
    note.classList.add("hint-warn");
  }
}

function radioGroup(name, legend, items, def, help) {
  const fs = el("fieldset", { class: "choice" }, el("legend", {}, legend));
  if (help) fs.append(el("p", { class: "hint" }, help));
  for (const it of items) {
    const id = `opt-${name}-${it.value}`;
    const input = el("input", { type: "radio", name, id, value: it.value, checked: it.value === def });
    const label = el("label", { for: id, class: "choice-item" }, input,
      el("span", { class: "choice-text" },
        el("span", { class: "choice-label" }, it.label || it.value, it.value === def ? el("span", { class: "default-tag" }, "既定") : null),
        it.description ? el("span", { class: "choice-desc" }, it.description) : null));
    fs.append(label);
  }
  return fs;
}

function renderAdvanced() {
  const info = state.info || {};
  const d = info.defaults || {};
  const ch = info.choices || {};
  const body = $("advanced-body");
  body.replaceChildren();
  if (Array.isArray(ch.chordInput) && ch.chordInput.length) {
    body.append(radioGroup("chordInput", "コード認識に入れる音", ch.chordInput, d.chordInput,
      "原音以外は音源の分離（Demucs）を使うので時間が延びます。"));
  }
  if (Array.isArray(ch.bass) && ch.bass.length) {
    body.append(radioGroup("bass", "ベースの使い方", ch.bass, d.bass));
  }
  if (Array.isArray(ch.model) && ch.model.length) {
    body.append(radioGroup("model", "コード認識のモデル", ch.model, d.model));
  }
  const snapDefault = d.snap !== false;
  body.append(el("div", { class: "field" },
    el("label", { class: "check" },
      el("input", { type: "checkbox", id: "opt-snap", checked: snapDefault }),
      el("span", {}, "コードの切り替えを拍にそろえる", snapDefault ? el("span", { class: "default-tag" }, "既定") : null))));
  body.append(el("div", { class: "field-pair" },
    el("div", { class: "field" },
      el("label", { class: "label", for: "opt-language" }, "歌詞の言語"),
      el("input", { type: "text", id: "opt-language", autocomplete: "off", spellcheck: "false", placeholder: "自動", value: d.language || "", "aria-describedby": "opt-language-note", maxlength: "16" }),
      el("p", { class: "hint", id: "opt-language-note" }, "空なら自動で判定します。例 ja、en")),
    el("div", { class: "field" },
      el("label", { class: "label", for: "opt-bars" }, "1 行の小節数"),
      el("input", { type: "number", id: "opt-bars", min: "1", max: "64", step: "1", value: String(d.barsPerLine ?? 4), inputmode: "numeric", "aria-describedby": "opt-bars-note" }),
      el("p", { class: "hint", id: "opt-bars-note" }, "歌詞がないときだけ使います。1〜64"))));
  body.append(el("div", { class: "field" },
    el("label", { class: "check" },
      el("input", { type: "checkbox", id: "opt-keep", "aria-describedby": "opt-keep-note" }),
      el("span", {}, "作業ファイルを残す")),
    el("p", { class: "hint", id: "opt-keep-note" }, "取得した音源や分離した音を work フォルダに残します。")));
  body.append(el("div", { class: "advanced-foot" },
    el("button", { type: "button", class: "btn btn-small btn-quiet", id: "opt-reset", onclick: () => { renderAdvanced(); announce("詳細な設定を既定に戻しました。"); } }, "既定に戻す")));
  $("opt-bars").addEventListener("input", () => clearFieldError($("opt-bars")));
}

function fieldError(input, message) {
  input.setAttribute("aria-invalid", "true");
  let msg = input.parentElement.querySelector(".field-error");
  if (!msg) {
    msg = el("p", { class: "field-error", id: `${input.id}-error` });
    input.after(msg);
  }
  msg.textContent = message;
  const desc = new Set((input.getAttribute("aria-describedby") || "").split(" ").filter(Boolean));
  desc.add(msg.id);
  input.setAttribute("aria-describedby", [...desc].join(" "));
}

function clearFieldError(input) {
  input.removeAttribute("aria-invalid");
  const msg = input.parentElement.querySelector(".field-error");
  if (msg) msg.remove();
}

function checkedValue(name) {
  const r = document.querySelector(`input[name="${name}"]:checked`);
  return r ? r.value : null;
}

/** 誤りの code から、近くに出す欄を選ぶ（分からなければ送信ボタンの下）。 */
function fieldForError(err) {
  const c = String(err.code || "").toUpperCase();
  if (c.includes("URL") || c.includes("MEDIA")) return $("url-input");
  if (c.includes("BAR")) return $("opt-bars");
  if (c.includes("LYRICS")) return $("lyrics-input");
  const m = String(err.message || "");
  if (c === "INVALID_OPTION" && m.includes("小節")) return $("opt-bars");
  if (c === "INVALID_OPTION" && m.includes("言語")) return $("opt-language");
  if (c.includes("LANG")) return $("opt-language");
  return null;
}

async function onSubmit(e) {
  e.preventDefault();
  const formError = $("form-error");
  showError(formError, null);
  for (const n of document.querySelectorAll("#job-form [aria-invalid]")) clearFieldError(n);

  const url = $("url-input").value.trim();
  const file = state.file;
  if (!file && !url) {
    showError(formError, "音源のファイルを選ぶか、URL を入れてください。");
    $("file-choose").focus();
    return;
  }
  if (!file && url && !/^https?:\/\//i.test(url)) {
    fieldError($("url-input"), "URL は http:// か https:// から書いてください。");
    $("url-input").focus();
    return;
  }
  const limits = (state.info && state.info.limits) || {};
  if (file && limits.maxUploadBytes && file.size > limits.maxUploadBytes) {
    showError(formError, `音源ファイルが大きすぎます（${formatBytes(file.size)}。${formatBytes(limits.maxUploadBytes)} まで）。`);
    $("file-choose").focus();
    return;
  }
  if (limits.maxLyricsChars && $("lyrics-input").value.length > limits.maxLyricsChars) {
    fieldError($("lyrics-input"), `歌詞が長すぎます（${limits.maxLyricsChars} 文字まで）。`);
    $("lyrics-input").focus();
    return;
  }
  const bars = $("opt-bars");
  const maxBars = limits.maxBarsPerLine || 64;
  if (bars) {
    const n = Number(bars.value);
    if (!Number.isInteger(n) || n < 1 || n > maxBars) {
      $("advanced").open = true;
      fieldError(bars, `1〜${maxBars} の整数にしてください。`);
      bars.focus();
      return;
    }
  }

  const fd = new FormData();
  if (file) fd.append("file", file, file.name);
  else fd.append("url", url);
  const lyrics = $("lyrics-input").value;
  if (lyrics.trim()) fd.append("lyrics", lyrics);
  const title = $("title-input").value.trim();
  const artist = $("artist-input").value.trim();
  if (title) fd.append("title", title);
  if (artist) fd.append("artist", artist);
  for (const name of ["chordInput", "bass", "model"]) {
    const v = checkedValue(name);
    if (v) fd.append(name, v);
  }
  if ($("opt-snap")) fd.append("snap", $("opt-snap").checked ? "true" : "false");
  const lang = $("opt-language") ? $("opt-language").value.trim() : "";
  if (lang) fd.append("language", lang);
  if (bars) fd.append("barsPerLine", String(Number(bars.value)));
  if ($("opt-keep")) fd.append("keepWork", $("opt-keep").checked ? "true" : "false");

  const btn = $("submit-btn");
  const status = $("submit-status");
  const bar = $("upload-bar");
  btn.disabled = true;
  status.textContent = file ? "送っています…" : "受け付けています…";
  if (file) {
    bar.hidden = false;
    bar.firstElementChild.style.width = "0%";
  }
  let lastAnnounced = -1;
  try {
    const res = await submitJob(fd, (ratio) => {
      const pct = Math.floor(ratio * 100);
      bar.firstElementChild.style.width = `${pct}%`;
      status.textContent = `送っています… ${pct}%`;
      if (pct >= lastAnnounced + 25) {
        lastAnnounced = pct;
        announce(`ファイルを送っています。${pct}%`);
      }
    });
    status.textContent = "受け付けました。";
    announce("解析を受け付けました。");
    state.autoPreview.add(res.id);
    // 入力は次の曲のために空ける（詳細な設定は残す）
    setFile(null);
    $("url-input").value = "";
    $("lyrics-input").value = "";
    $("title-input").value = "";
    $("artist-input").value = "";
    await addJobById(res.id);
    setTimeout(() => { if (status.textContent === "受け付けました。") status.textContent = ""; }, 4000);
  } catch (err) {
    status.textContent = "";
    const field = err instanceof ApiError ? fieldForError(err) : null;
    if (field) {
      if (field.closest("details")) field.closest("details").open = true;
      fieldError(field, err.message);
      field.focus();
    } else {
      showError(formError, err);
    }
  } finally {
    btn.disabled = false;
    bar.hidden = true;
  }
}

// ---- ジョブ -----------------------------------------------------------------

function truthy(v) {
  return v === true || v === "true" || v === 1 || v === "1";
}

/** pipeline.plan_stages と同じ規則で、走る段の候補を作る。total（書き出しを含む段の数）に合うものを選ぶ。 */
function planStages(job, total) {
  const o = job.options || {};
  const d = (state.info && state.info.defaults) || {};
  const chordInput = o.chordInput ?? d.chordInput ?? "original";
  const bass = o.bass ?? d.bass ?? "off";
  const snap = o.snap === undefined ? d.snap !== false : truthy(o.snap);
  const known = o.hasLyrics ?? o.lyrics;
  const lyricCases = typeof known === "boolean" ? [known] : typeof known === "string" ? [known.length > 0] : [false, true];
  const plans = lyricCases.map((hasLyrics) => {
    const s = job.sourceKind === "url" ? ["fetch"] : [];
    s.push("probe");
    if (chordInput !== "original" || bass !== "off" || hasLyrics) s.push("separate");
    s.push("beats", "chords", "key");
    if (snap) s.push("snap");
    if (bass !== "off") s.push("bass");
    if (hasLyrics) s.push("lyrics");
    s.push("write");
    return s;
  });
  const knownStages = new Set(state.stages.map((s) => s.stage));
  const fit = plans.filter((p) => p.every((s) => knownStages.has(s)) && (!total || p.length === total));
  if (!fit.length) return null;
  // 歌詞の有無が分からず候補が 2 つ残るときは、どちらにもない段だけを「使わない」にする
  return new Set(fit.flat());
}

class JobView {
  constructor(job) {
    this.job = job;
    this.stageState = new Map(); // stage → {status, elapsed, message, percent, detail}
    this.logs = [];
    this.warnings = [];
    this.total = job.currentStage ? job.currentStage.total : null;
    this.es = null;
    this.seen = new Set();
    this.resetOnNext = false;
    this.retry = 0;
    this.retryTimer = 0;
    this.cancelling = false;
    this.lastLogRender = 0;
    this.build();
  }

  build() {
    const id = this.job.id;
    this.root = el("article", { class: "job", "aria-labelledby": `job-${id}-name` });
    this.nameEl = el("h3", { class: "job-name", id: `job-${id}-name` });
    this.statusEl = el("span", { class: "badge" });
    this.elapsedEl = el("span", { class: "job-elapsed" });
    this.subEl = el("p", { class: "job-sub" });
    this.stagesEl = el("ol", { class: "stages", "aria-label": "解析の段" });
    this.errorEl = el("div", { class: "job-error", role: "alert", hidden: true });
    this.warnEl = el("ul", { class: "job-warnings", "aria-label": "警告", hidden: true });
    this.actionsEl = el("div", { class: "job-actions" });
    this.logCount = el("span", { class: "log-count" });
    this.logPre = el("pre", { class: "log", tabindex: "0", "aria-label": "子プロセスのログ" });
    this.logEl = el("details", { class: "job-log" }, el("summary", {}, "ログ", this.logCount), this.logPre);
    this.logEl.addEventListener("toggle", () => { if (this.logEl.open) this.renderLog(true); });
    this.root.append(
      el("header", { class: "job-head" }, this.nameEl, this.statusEl, this.elapsedEl),
      this.subEl, this.stagesEl, this.errorEl, this.warnEl, this.actionsEl, this.logEl,
    );
    this.render();
  }

  get finished() {
    return TERMINAL.has(this.job.status);
  }

  // -- SSE --
  /** SSE に繋ぐ。終わったジョブも、記録（段・ログ）を読むために一度だけ繋げる（最後の status で閉じる）。 */
  connect({ history = false } = {}) {
    if (this.es || state.shutDown || (this.finished && !history)) return;
    this.historyOnly = this.finished;
    const es = new EventSource(api.eventsUrl(this.job.id), { withCredentials: true });
    this.es = es;
    es.onopen = () => {
      this.retry = 0;
      this.resetOnNext = true;
    };
    for (const type of ["progress", "warning", "status", "log"]) {
      es.addEventListener(type, (e) => this.onMessage(type, e));
    }
    es.onerror = () => {
      if (this.es !== es) return;
      if (this.finished && this.historyOnly) {
        // 終わったジョブの記録を読みに行っただけなので、繋ぎ直さない
        this.close();
        return;
      }
      if (es.readyState === EventSource.CLOSED) {
        // ブラウザが繋ぎ直しをあきらめた（403 など）。少し待ってから状態を確かめて繋ぎ直す
        this.es = null;
        es.close();
        this.scheduleReconnect();
      }
      // CONNECTING のあいだはブラウザが自動で繋ぎ直す（繋がると最初から流し直される）
    };
  }

  scheduleReconnect() {
    clearTimeout(this.retryTimer);
    const wait = Math.min(15000, 1000 * 2 ** this.retry);
    this.retry++;
    this.retryTimer = setTimeout(async () => {
      try {
        const job = await api.job(this.job.id);
        this.applyJob(job);
        // 繋ぎ直すと最初から流し直される。id で重複を除く（id がなければ最初の 1 件で描き直す）
        this.connect({ history: true });
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          if (this.finished) return;
          this.job.status = "cancelled";
          this.render();
          return;
        }
        if (!(e instanceof ApiError && e.status === 403)) this.scheduleReconnect();
      }
    }, wait);
  }

  close() {
    clearTimeout(this.retryTimer);
    if (this.es) {
      this.es.close();
      this.es = null;
    }
    this.historyDone();
  }

  /** 終わったジョブの記録を SSE の流し直しで読む。読み終わる（か 5 秒たつ）と解決する。 */
  loadHistory() {
    return new Promise((resolve) => {
      this.onHistoryDone = resolve;
      setTimeout(() => this.historyDone(), 5000);
      this.connect({ history: true });
      if (!this.es) this.historyDone();
    });
  }

  historyDone() {
    const f = this.onHistoryDone;
    this.onHistoryDone = null;
    if (f) f();
  }

  reset() {
    this.stageState.clear();
    this.logs = [];
    this.warnings = [];
    this.seen.clear();
  }

  onMessage(type, e) {
    // status は今の状態そのものなので、同じ id でも必ず当てはめる（サーバーは流し直しの最後に、
    // 最後の記録と同じ id で今の status を送る）。それ以外は id で重複を除き、id がなければ流し直しの最初で描き直す
    if (type === "status") {
      // 重複を除かない
    } else if (e.lastEventId) {
      if (this.seen.has(e.lastEventId)) return;
      this.seen.add(e.lastEventId);
      this.resetOnNext = false;
    } else if (this.resetOnNext) {
      this.resetOnNext = false;
      this.reset();
    }
    let data;
    try {
      data = JSON.parse(e.data);
    } catch {
      data = e.data;
    }
    this.handle(type, data, true);
  }

  handle(type, data, live) {
    if (type === "progress") this.onProgress(data, live);
    else if (type === "warning") {
      const msg = typeof data === "string" ? data : data && (data.message ?? data.text);
      if (msg) this.warnings.push(String(msg));
      if (live) this.render();
    } else if (type === "log") {
      const lines = typeof data === "string" ? data : data && (data.line ?? data.message ?? data.text ?? JSON.stringify(data));
      for (const l of String(lines ?? "").split("\n")) this.logs.push(l);
      if (this.logs.length > LOG_LIMIT) this.logs.splice(0, this.logs.length - LOG_LIMIT);
      if (live) this.renderLog();
    } else if (type === "status") {
      if (data && typeof data === "object" && data.id) this.applyJob(data, live);
    }
  }

  onProgress(p, live) {
    if (!p || typeof p !== "object" || !p.stage) return;
    if (Number.isFinite(p.total)) this.total = p.total;
    const cur = this.stageState.get(p.stage) || {};
    if (p.status === "progress") {
      const pct = percentOf(p.data);
      if (pct !== null) cur.percent = Math.max(0, Math.min(100, pct));
      if (p.message) cur.detail = p.message;
      if (!cur.status) cur.status = "start";
    } else {
      cur.status = p.status;
      if (Number.isFinite(p.elapsed)) cur.elapsed = p.elapsed;
      if (p.message) cur.message = p.message;
      if (p.status === "start") cur.startedAt = Date.now() / 1000;
    }
    this.stageState.set(p.stage, cur);
    if (p.status === "start" && !this.finished) {
      this.job.currentStage = { stage: p.stage, label: p.label, index: p.index, total: p.total };
      if (live) announce(`${this.displayName()}: ${p.label}（${p.index}/${p.total}）`);
    }
    if (live) this.render();
  }

  applyJob(job, live = false) {
    const before = this.job.status;
    const events = this.job.events;
    this.job = { ...job };
    if (!this.job.events && events) this.job.events = events;
    if (this.job.currentStage && Number.isFinite(this.job.currentStage.total)) this.total = this.job.currentStage.total;
    this.render();
    if (this.finished) {
      this.close();
      if (before !== job.status) onJobFinished(this, live);
    }
    renderJobsList();
  }

  displayName() {
    return this.job.title || this.job.sourceName || "（名前なし）";
  }

  // -- 描画 --
  stageRows() {
    const plan = planStages(this.job, this.total);
    const job = this.job;
    return state.stages.map(({ stage, label }) => {
      const s = this.stageState.get(stage);
      let kind;
      if (s && s.status) {
        kind = { start: "running", progress: "running", done: "done", skipped: "skipped", failed: "failed" }[s.status] || "running";
        if (kind === "running" && this.finished) kind = job.status === "cancelled" ? "cancelled" : job.status === "failed" ? "failed" : "done";
      } else if (plan && !plan.has(stage)) kind = "unused";
      else if (this.finished) kind = job.status === "done" ? "unused" : "notrun";
      else kind = "waiting";
      return { stage, label, kind, s: s || {} };
    });
  }

  render() {
    const job = this.job;
    const r = this.root;
    r.dataset.status = job.status;
    this.nameEl.textContent = this.displayName();
    this.nameEl.title = job.sourceName || "";
    this.statusEl.textContent = STATUS_TEXT[job.status] || job.status;
    this.statusEl.dataset.status = job.status;

    const sub = [];
    if (job.title && job.sourceName && job.title !== job.sourceName) sub.push(job.sourceName);
    if (job.artist) sub.push(job.artist);
    sub.push(job.sourceKind === "url" ? "URL" : "ファイル");
    if (job.status === "queued") sub.push(job.queuePosition ? `${job.queuePosition} 番目に待っています` : "順番を待っています");
    if (job.createdAt) sub.push(`${formatDate(job.createdAt)} に受け付け`);
    this.subEl.textContent = sub.join("、");
    this.updateElapsed();

    // 段
    const rows = this.stageRows();
    this.stagesEl.replaceChildren(...rows.map(({ stage, label, kind, s }) => {
      const li = el("li", { class: `stage stage-${kind}`, "data-stage": stage },
        el("span", { class: "stage-mark", "aria-hidden": "true" }),
        el("span", { class: "stage-label" }, label),
        el("span", { class: "stage-state" }, stageStateText(kind, s)));
      if (kind === "running" && s.percent !== undefined) {
        li.append(el("span", { class: "stage-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(s.percent)), "aria-label": `${label}の進み具合` },
          el("span", { style: `width:${s.percent.toFixed(1)}%` })));
      }
      if ((kind === "skipped" || kind === "failed") && s.message) li.append(el("span", { class: "stage-msg" }, s.message));
      else if (kind === "unused") li.title = "この条件では使いません";
      return li;
    }));

    // 誤り
    if (job.status === "failed") {
      const err = job.error || {};
      const stageLabel = err.stage ? (state.stages.find((x) => x.stage === err.stage) || {}).label || err.stage : null;
      const msgHasStage = stageLabel && String(err.message || "").includes(stageLabel);
      this.errorEl.replaceChildren(
        el("p", { class: "job-error-title" }, stageLabel && !msgHasStage ? `${stageLabel}で失敗しました` : "解析に失敗しました"),
        el("p", { class: "job-error-msg" }, err.message || "理由が分かりません。ログを見てください。"),
        err.code ? el("p", { class: "job-error-code" }, `コード: ${err.code}`) : null,
      );
      this.errorEl.hidden = false;
    } else {
      this.errorEl.hidden = true;
    }

    // 警告
    const warns = [...new Set(this.warnings)];
    this.warnEl.replaceChildren(...warns.map((w) => el("li", {}, w)));
    this.warnEl.hidden = !warns.length;

    // 操作
    const actions = [];
    if (!this.finished) {
      const b = el("button", { type: "button", class: "btn btn-small", disabled: this.cancelling }, this.cancelling ? "中止しています…" : "中止");
      b.addEventListener("click", () => this.cancel(b));
      actions.push(b);
    }
    if (job.status === "done" && job.outputName) {
      const open = el("button", { type: "button", class: "btn btn-small" }, state.selected === job.outputName ? "表示しています" : "下書きを見る");
      open.disabled = state.selected === job.outputName;
      open.addEventListener("click", () => selectOutput(job.outputName, { focus: true }));
      actions.push(open);
    }
    if (this.finished && job.options && truthy(job.options.keepWork)) {
      const w = el("button", { type: "button", class: "btn btn-small btn-quiet" }, "作業フォルダを開く");
      w.addEventListener("click", () => openFolder("work", null, this.actionsEl));
      actions.push(w);
    }
    if (this.finished) {
      const retry = el("button", { type: "button", class: "btn btn-small btn-quiet" }, "閉じる");
      retry.setAttribute("aria-label", `${this.displayName()}の表示を閉じる`);
      retry.addEventListener("click", () => {
        hiddenJobs.add(job.id);
        renderJobsList();
      });
      actions.push(retry);
    }
    // 中止の確認中は描き直さない
    if (!this.actionsEl.querySelector(".inline-confirm")) {
      const focused = this.actionsEl.contains(document.activeElement) ? document.activeElement.textContent : null;
      this.actionsEl.replaceChildren(...actions);
      if (focused) {
        const again = actions.find((a) => a.textContent === focused && !a.disabled) || actions.find((a) => !a.disabled);
        if (again) again.focus();
      }
    }
    this.renderLog();
  }

  updateElapsed() {
    const job = this.job;
    const start = parseTime(job.startedAt);
    let text = "";
    if (start !== null) {
      const end = parseTime(job.finishedAt) ?? (this.finished ? null : Date.now() / 1000);
      if (end !== null) text = formatElapsed(end - start);
    } else if (job.status === "queued") {
      const c = parseTime(job.createdAt);
      if (c !== null) text = `待ち ${formatElapsed(Date.now() / 1000 - c)}`;
    }
    this.elapsedEl.textContent = text;
    if (!this.finished) {
      for (const li of this.stagesEl.querySelectorAll(".stage-running")) {
        const s = this.stageState.get(li.dataset.stage);
        const st = li.querySelector(".stage-state");
        if (s && st) st.textContent = stageStateText("running", s);
      }
    }
  }

  renderLog(force = false) {
    this.logCount.textContent = this.logs.length ? `（${this.logs.length} 行）` : "（まだありません）";
    if (!this.logEl.open && !force) return;
    const now = Date.now();
    if (!force && now - this.lastLogRender < 250) {
      clearTimeout(this.logTimer);
      this.logTimer = setTimeout(() => this.renderLog(true), 260);
      return;
    }
    this.lastLogRender = now;
    const pre = this.logPre;
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
    pre.textContent = this.logs.join("\n");
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }

  async cancel(btn) {
    this.cancelling = true;
    btn.disabled = true;
    btn.textContent = "中止しています…";
    try {
      await api.cancel(this.job.id);
      announce(`${this.displayName()}を中止します。`);
      // 状態は SSE の status で届く。届かないときのために取り直す
      setTimeout(async () => {
        if (this.finished) return;
        try {
          this.applyJob(await api.job(this.job.id), true);
        } catch { /* 次の SSE を待つ */ }
      }, 3000);
    } catch (e) {
      this.cancelling = false;
      this.render();
      const p = el("p", { class: "field-error", role: "alert" }, `中止できませんでした: ${e.message}`);
      this.actionsEl.append(p);
    }
  }
}

function stageStateText(kind, s) {
  if (kind === "running") {
    const parts = ["実行中"];
    if (s.percent !== undefined) parts.push(`${Math.floor(s.percent)}%`);
    if (s.startedAt) parts.push(formatElapsed(Date.now() / 1000 - s.startedAt));
    return parts.join(" ");
  }
  if (kind === "done" && Number.isFinite(s.elapsed)) return `済み ${s.elapsed < 10 ? s.elapsed.toFixed(1) : Math.round(s.elapsed)} 秒`;
  return STAGE_TEXT[kind];
}

const hiddenJobs = new Set();

function renderJobsList() {
  const box = $("jobs");
  const all = [...state.jobs.values()].sort((a, b) => (Date.parse(b.job.createdAt) || 0) - (Date.parse(a.job.createdAt) || 0));
  const active = all.filter((v) => !v.finished).reverse(); // 実行中を上、待ちをその下（古い順）
  active.sort((a, b) => (a.job.status === "running" ? 0 : 1) - (b.job.status === "running" ? 0 : 1) || (a.job.queuePosition || 0) - (b.job.queuePosition || 0));
  const finished = all.filter((v) => v.finished && !hiddenJobs.has(v.job.id)).slice(0, FINISHED_SHOWN);
  const want = [...active, ...finished].map((v) => v.root);
  // 並びが同じなら触らない（details の開閉やフォーカスを保つ）
  const cur = Array.from(box.children);
  if (cur.length !== want.length || cur.some((n, i) => n !== want[i])) {
    const focused = document.activeElement;
    box.replaceChildren(...want);
    if (focused && box.contains(focused)) focused.focus();
  }
  $("jobs-empty").hidden = want.length > 0;
  updateTitle();
}

function updateTitle() {
  if (state.shutDown) return;
  const running = [...state.jobs.values()].find((v) => v.job.status === "running");
  if (running && running.job.currentStage) {
    const c = running.job.currentStage;
    document.title = `(${c.index}/${c.total}) ${c.label} - chordgen`;
  } else if (running) {
    document.title = "(解析中) chordgen";
  } else {
    document.title = "chordgen";
  }
}

async function addJobById(id) {
  let job;
  try {
    job = await api.job(id);
  } catch {
    job = { id, status: "queued", sourceName: "", sourceKind: "file", options: {}, createdAt: new Date().toISOString() };
  }
  upsertJob(job, { connect: true });
}

function upsertJob(job, { connect = false } = {}) {
  let v = state.jobs.get(job.id);
  if (!v) {
    v = new JobView(job);
    state.jobs.set(job.id, v);
  } else {
    v.applyJob(job);
  }
  if (connect && !v.finished && (v.job.status === "running" || openStreams() < MAX_SSE)) v.connect();
  renderJobsList();
  return v;
}

function openStreams() {
  let n = 0;
  for (const v of state.jobs.values()) if (v.es) n++;
  return n;
}

/** SSE を張っていない待ちのジョブがあるあいだ、一覧を取り直す。 */
async function pollWaiting() {
  if (state.shutDown) return;
  const waiting = [...state.jobs.values()].some((v) => !v.finished && !v.es);
  if (!waiting) return;
  try {
    const jobs = await api.jobs();
    for (const job of Array.isArray(jobs) ? jobs : []) upsertJob(job, { connect: true });
  } catch { /* 次の回に取り直す */ }
}

async function loadJobs() {
  try {
    const jobs = await api.jobs();
    if (!Array.isArray(jobs)) return;
    for (const job of jobs) {
      const v = upsertJob(job, { connect: true });
      if (!v.finished) state.autoPreview.add(job.id);
    }
    // 終わったジョブは、出す分だけ SSE の流し直しで段の様子とログを戻す
    const finished = jobs.filter((j) => TERMINAL.has(j.status)).slice(0, FINISHED_SHOWN);
    for (const j of finished) {
      const v = state.jobs.get(j.id);
      if (v) await v.loadHistory(); // 接続の数を食わないよう 1 つずつ
    }
  } catch (e) {
    showError($("form-error"), `ジョブの一覧を読めませんでした: ${e.message}`);
  }
}

function onJobFinished(v, live) {
  const job = v.job;
  const name = v.displayName();
  const open = state.autoPreview.has(job.id);
  if (job.status === "done") {
    announce(`${name}の解析が終わりました。`);
    loadOutputs().then(() => {
      if (job.outputName && open) selectOutput(job.outputName);
    });
  } else if (job.status === "failed") {
    announce(`${name}の解析に失敗しました。${(job.error && job.error.message) || ""}`);
  } else if (job.status === "cancelled") {
    announce(`${name}の解析を中止しました。`);
  }
  state.autoPreview.delete(job.id);
  v.cancelling = false;
  v.render();
  void live;
}

// ---- 出力の一覧とプレビュー --------------------------------------------------

async function loadOutputs() {
  const box = $("outputs");
  try {
    const list = await api.outputs();
    state.outputs = Array.isArray(list) ? list : [];
    renderOutputs();
  } catch (e) {
    box.replaceChildren(el("p", { class: "form-error", role: "alert" }, `一覧を読めませんでした: ${e.message}`));
  }
}

function renderOutputs() {
  const box = $("outputs");
  if (!state.outputs.length) {
    box.replaceChildren(el("p", { class: "empty" }, "まだありません。解析が終わると output フォルダに書き出され、ここに並びます。"));
    return;
  }
  const head = el("div", { class: "out-row out-head", "aria-hidden": "true" },
    el("span", {}, "題名"), el("span", {}, "アーティスト"), el("span", { class: "num" }, "キー"), el("span", { class: "num" }, "BPM"), el("span", { class: "num" }, "日時"));
  const items = state.outputs.map((o) => {
    const btn = el("button", { type: "button", class: "out-row", "data-name": o.name, "aria-current": state.selected === o.name ? "true" : null, title: o.name },
      el("span", { class: "out-title" }, o.title || o.name),
      el("span", { class: "out-artist" }, o.artist || "—"),
      el("span", { class: "num" }, o.originalKey || "—"),
      el("span", { class: "num" }, o.bpm != null ? String(Math.round(o.bpm)) : "—"),
      el("span", { class: "num out-date" }, formatDate(o.modifiedAt)));
    btn.setAttribute("aria-label", `${o.title || o.name}、${o.artist || "アーティスト不明"}、キー ${o.originalKey || "不明"}、BPM ${o.bpm != null ? Math.round(o.bpm) : "不明"}、${formatDate(o.modifiedAt)}`);
    btn.addEventListener("click", () => selectOutput(o.name, { focus: true }));
    return el("li", {}, btn);
  });
  box.replaceChildren(head, el("ul", { class: "out-list" }, items));
}

let previewSeq = 0;
async function selectOutput(name, { focus = false } = {}) {
  const seq = ++previewSeq;
  state.selected = name;
  renderOutputs();
  for (const v of state.jobs.values()) if (v.finished) v.render();
  const preview = $("preview");
  showError($("result-error"), null);
  preview.setAttribute("aria-busy", "true");
  try {
    const doc = await api.output(name);
    if (seq !== previewSeq) return;
    state.selectedDoc = doc;
    const { node } = renderSong(doc);
    preview.replaceChildren(node);
    requestAnimationFrame(() => spreadChords(preview));
    setActions(name, doc);
    if (focus) $("result").focus({ preventScroll: false });
  } catch (e) {
    if (seq !== previewSeq) return;
    preview.replaceChildren();
    $("result-actions").hidden = true;
    showError($("result-error"), e instanceof ApiError && e.status === 404 ? `「${name}」が見つかりません。消されたかもしれません。` : `開けませんでした: ${e.message}`);
    loadOutputs();
  } finally {
    preview.removeAttribute("aria-busy");
  }
}

function setActions(name, doc) {
  $("result-actions").hidden = false;
  const dl = $("download-btn");
  dl.href = api.downloadUrl(name);
  dl.setAttribute("download", name);
  const wrap = $("delete-wrap");
  const old = wrap.querySelector(".inline-confirm");
  if (old) old.remove();
  $("delete-btn").hidden = false;
  $("result-heading").textContent = "下書き";
  $("result-heading").title = name;
  void doc;
}

function clearPreview() {
  state.selected = null;
  state.selectedDoc = null;
  $("result-actions").hidden = true;
  $("preview").replaceChildren(el("p", { class: "empty" }, "解析が終わると、ここにコード譜の下書きが出ます。下の一覧から前の結果も開けます。"));
}

async function openFolder(target, name, near) {
  try {
    await api.openFolder(target, name);
    announce(target === "work" ? "作業フォルダを開きました。" : "フォルダを開きました。");
  } catch (e) {
    if (near === $("result-actions")) showError($("result-error"), `フォルダを開けませんでした: ${e.message}`);
    else near.append(el("p", { class: "field-error", role: "alert" }, `フォルダを開けませんでした: ${e.message}`));
  }
}

function setupResult() {
  $("open-folder-btn").addEventListener("click", () => openFolder("output", state.selected, $("result-actions")));
  $("delete-btn").addEventListener("click", () => {
    const name = state.selected;
    if (!name) return;
    const title = (state.selectedDoc && state.selectedDoc.song && state.selectedDoc.song.title) || name;
    openInlineConfirm($("delete-btn"), `「${title}」のファイルを削除しますか？元に戻せません。`, "削除する", async () => {
      try {
        await api.deleteOutput(name);
        announce(`「${title}」を削除しました。`);
        clearPreview();
        await loadOutputs();
        $("outputs-reload").focus();
      } catch (e) {
        showError($("result-error"), `削除できませんでした: ${e.message}`);
      }
    });
  });
  $("outputs-reload").addEventListener("click", loadOutputs);
  let t = 0;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => spreadChords($("preview")), 120);
  });
}

// ---- 起動 -------------------------------------------------------------------

function tick() {
  for (const v of state.jobs.values()) if (!v.finished) v.updateElapsed();
}

async function main() {
  setupShutdown();
  setupInput();
  setupResult();
  setInterval(tick, 1000);
  setInterval(pollWaiting, 3000);
  await loadInfo();
  await Promise.all([loadJobs(), loadOutputs()]);
  // 画面を開き直したとき、表示する下書きがなければ最新の結果を開く
  if (!state.selected && state.outputs.length) {
    const lastDone = [...state.jobs.values()].find((v) => v.job.status === "done" && v.job.outputName);
    const name = lastDone && state.outputs.some((o) => o.name === lastDone.job.outputName) ? lastDone.job.outputName : null;
    if (name) selectOutput(name);
  }
}

main();
