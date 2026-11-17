"use strict";

const STAGES = ["download", "probe", "layout", "transcribe", "captions", "render"];
const LAYOUTS = ["auto", "fullcam", "stacked", "face", "content"];
const LOW_CONFIDENCE = 0.6;
const TOKEN_KEY = "clipshorts_token";

const clips = new Map(); // id -> clip
const cards = new Map(); // id -> {root, ...element refs}
const $ = (id) => document.getElementById(id);

// -- token ------------------------------------------------------------------
function getToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}
function setToken(t) {
  try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode: lives until reload */ }
}
function withToken(url) {
  const t = getToken();
  if (!t) return url;
  return url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(t);
}
function needToken() { $("token-form").hidden = false; }

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    ...opts, headers, body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });
  if (res.status === 401) { needToken(); throw new Error("unauthorized"); }
  if (!res.ok) {
    let msg = res.status + " " + res.statusText;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

// -- tiny DOM helper -----------------------------------------------------------
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n[k] = v;
  }
  for (const kid of kids) n.append(kid);
  return n;
}

// -- cards ---------------------------------------------------------------------
const isActive = (c) => c.status === "queued" || c.status === "running";

function buildCard(id) {
  const video = el("video", { controls: true, preload: "metadata", playsInline: true });
  const placeholder = el("span", {}, "no output yet");
  const preview = el("div", { class: "preview" }, placeholder);
  const title = el("div", { class: "title" });
  const meta = el("div", { class: "meta" });
  const badges = el("div", { class: "badges" });
  const stages = el("div", { class: "stages" },
    ...STAGES.map((s) => el("span", { class: "stage", textContent: s })));
  const bar = el("progress", { max: 1, value: 0 });
  const error = el("p", { class: "error", hidden: true });
  const select = el("select", {
    onchange: () => rerender(id, select.value),
  }, ...LAYOUTS.map((l) => el("option", { value: l, textContent: l })));
  const layoutLabel = el("label", {}, "Layout", select);
  const download = el("a", { class: "btn", textContent: "Download", hidden: true });
  const retry = el("button", { class: "secondary", textContent: "Retry", hidden: true,
    onclick: () => rerender(id, null) });
  const del = el("button", { class: "danger", textContent: "Delete", onclick: () => remove(id) });
  const process = el("button", { textContent: "Process", hidden: true,
    title: "Run layout, captions and render (uses CPU)", onclick: () => processClip(id) });
  const actions = el("div", { class: "actions" }, process, layoutLabel, download, retry, del);
  const captions = buildCaptions(id, error);
  const hook = buildHook(id, error);
  const post = buildPost(id, error);
  const body = el("div", {}, title, meta, badges, stages, bar, error, actions, hook.root,
    captions.root, post.root);
  const root = el("article", { class: "card clip" }, preview, body);
  return { root, video, placeholder, preview, title, meta, badges, stages, bar, error, select,
    layoutLabel, process, download, retry, del, captions, hook, post, videoSrc: "" };
}

// -- caption text: edit words by hand, then re-render ----------------------------------------
function buildCaptions(id, errorEl) {
  const fail = (e) => { errorEl.hidden = false; errorEl.textContent = String(e.message || e); };
  const grid = el("div", { class: "words" });
  const note = el("p", { class: "muted" });
  let words = []; // [{start, end, text}] as loaded; the inputs hold the edits

  const input = (w) => {
    const i = el("input", { type: "text", value: w.text, title: w.start.toFixed(2) + " s",
      size: Math.max(2, w.text.length) });
    i.addEventListener("input", () => { i.size = Math.max(2, i.value.length); });
    return i;
  };

  function show(res) {
    words = res.words;
    grid.replaceChildren(...words.map(input));
    note.textContent = res.edited
      ? "edited by hand" : "Whisper text. Empty = remove word, a space splits a word in two.";
    reset.hidden = !res.edited;
  }

  const collect = () => words.map((w, i) => ({ ...w, text: grid.children[i].value }));

  const save = el("button", { textContent: "Save and re-render", onclick: async () => {
    errorEl.hidden = true;
    try {
      upsert(await api(`/api/clips/${id}/captions`, { method: "PUT", body: { words: collect() } }));
      show(await api(`/api/clips/${id}/captions`));
    } catch (e) { fail(e); }
  } });
  const reset = el("button", { class: "secondary", textContent: "Reset to Whisper text",
    hidden: true, onclick: async () => {
      errorEl.hidden = true;
      try {
        upsert(await api(`/api/clips/${id}/captions`, { method: "DELETE" }));
        show(await api(`/api/clips/${id}/captions`));
      } catch (e) { fail(e); }
    } });

  const root = el("details", { class: "post" },
    el("summary", {}, "Caption text"), grid, note,
    el("div", { class: "actions" }, save, reset));
  root.addEventListener("toggle", async () => {
    if (!root.open || root.dataset.loaded) return;
    root.dataset.loaded = "1";
    try { show(await api(`/api/clips/${id}/captions`)); } catch (e) { fail(e); }
  });
  return { root, save, reset };
}

// -- hook: attention line on the first seconds of the video ------------------------------------
function buildHook(id, errorEl) {
  const text = el("input", { type: "text", maxLength: 100, class: "hook-input",
    placeholder: "e.g. Wait for the end..." });
  const note = el("p", { class: "muted",
    textContent: "Shown for the first 3 s in the upper third, clear of TikTok's UI. Empty removes it." });
  const save = el("button", { textContent: "Save and re-render", onclick: async () => {
    errorEl.hidden = true;
    try {
      upsert(await api(`/api/clips/${id}/hook`, { method: "PUT", body: { text: text.value } }));
    } catch (e) { errorEl.hidden = false; errorEl.textContent = String(e.message || e); }
  } });
  const root = el("details", { class: "post" }, el("summary", {}, "Hook text"), text, note,
    el("div", { class: "actions" }, save));
  // follow the server value, but never overwrite what is being typed
  const sync = (c) => {
    if (c.hook != null && document.activeElement !== text) text.value = c.hook;
  };
  return { root, save, sync };
}

// -- post: metadata, export, publications (phase 4) -------------------------------------
const PLATFORMS = ["tiktok", "youtube", "instagram", "other"];
const RATINGS = ["", "good", "ok", "flop"];

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const old = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = old; }, 1200);
  } catch { window.prompt("Copy:", text); }
}

function buildPost(id, errorEl) {
  const fail = (e) => { errorEl.hidden = false; errorEl.textContent = String(e.message || e); };
  const title = el("input", { type: "text", maxLength: 90, placeholder: "title / hook" });
  const caption = el("textarea", { rows: 3, maxLength: 300, placeholder: "caption" });
  const tags = el("input", { type: "text", placeholder: "hashtags, comma separated" });
  const note = el("p", { class: "muted" });
  const copies = el("div", { class: "actions" });
  const pubs = el("div", { class: "pubs" });

  const setMeta = (m) => {
    title.value = m.title;
    caption.value = m.caption;
    tags.value = m.hashtags.join(", ");
    note.textContent = m.source === "auto"
      ? "default: clip title, creator, game and fixed hashtags" : "edited by hand";
  };
  const readTags = () => tags.value.split(/[\s,]+/).map((t) => t.replace(/^#/, "")).filter(Boolean);

  const save = async () => {
    try {
      setMeta(await api(`/api/clips/${id}/metadata`, {
        method: "PUT", body: { title: title.value, caption: caption.value, hashtags: readTags() },
      }));
      await loadExport();
    } catch (e) { fail(e); }
  };
  const saveBtn = el("button", { class: "secondary", textContent: "Save", onclick: save });

  async function loadExport() {
    copies.replaceChildren();
    let ex;
    try { ex = await api(`/api/clips/${id}/export`); } catch { return; }
    const p = ex.platforms;
    for (const [label, text] of [
      ["Copy TikTok / Reels caption", p.tiktok.caption],
      ["Copy YT title", p.youtube.title],
      ["Copy YT description", p.youtube.description],
    ]) {
      const b = el("button", { class: "secondary", textContent: label });
      b.addEventListener("click", () => copyText(text, b));
      copies.append(b);
    }
  }

  // publications
  const platform = el("select", {}, ...PLATFORMS.map((p) => el("option", { value: p, textContent: p })));
  const url = el("input", { type: "url", placeholder: "post URL (optional)" });
  const markBtn = el("button", { class: "secondary", textContent: "Mark as published",
    onclick: async () => {
      try {
        await api(`/api/clips/${id}/publications`, {
          method: "POST", body: { platform: platform.value, url: url.value.trim() || null },
        });
        url.value = "";
        await loadPubs();
      } catch (e) { fail(e); }
    } });

  function pubRow(p) {
    const views = el("input", { type: "number", min: 0, placeholder: "views 24 h",
      value: p.views_24h ?? "" });
    const rating = el("select", {}, ...RATINGS.map((r) => el("option", { value: r,
      textContent: r || "rating", selected: (p.rating || "") === r })));
    const pnote = el("input", { type: "text", placeholder: "note", value: p.note || "" });
    const send = async () => {
      try {
        await api(`/api/publications/${p.id}`, { method: "PATCH", body: {
          views_24h: views.value === "" ? null : Number(views.value),
          rating: rating.value || null, note: pnote.value.trim() || null,
        } });
      } catch (e) { fail(e); }
    };
    for (const n of [views, rating, pnote]) n.addEventListener("change", send);
    const head = p.url ? el("a", { href: p.url, target: "_blank", rel: "noopener",
      textContent: p.platform }) : el("span", { textContent: p.platform });
    const rm = el("button", { class: "danger", textContent: "×", title: "Remove",
      onclick: async () => {
        try { await api(`/api/publications/${p.id}`, { method: "DELETE" }); await loadPubs(); }
        catch (e) { fail(e); }
      } });
    return el("div", { class: "pub" }, head, views, rating, pnote, rm);
  }

  async function loadPubs() {
    try {
      const list = await api(`/api/clips/${id}/publications`);
      pubs.replaceChildren(...list.map(pubRow));
    } catch (e) { fail(e); }
  }

  const root = el("details", { class: "post" },
    el("summary", {}, "Post text and publishing"),
    title, caption, tags, note,
    el("div", { class: "actions" }, saveBtn),
    copies,
    el("h3", {}, "Published"),
    pubs,
    el("div", { class: "actions" }, platform, url, markBtn));

  root.addEventListener("toggle", async () => {
    if (!root.open || root.dataset.loaded) return;
    root.dataset.loaded = "1";
    try { setMeta(await api(`/api/clips/${id}/metadata`)); await loadExport(); } catch { /* none yet */ }
    await loadPubs();
  });
  return { root };
}

function badge(text, cls = "") {
  return el("span", { class: "badge " + cls, textContent: text });
}

function updateCard(c) {
  let card = cards.get(c.id);
  if (!card) {
    card = buildCard(c.id);
    cards.set(c.id, card);
    $("clips").prepend(card.root);
  }
  const active = isActive(c);

  card.title.textContent = c.title || c.slug;
  const bits = [c.channel, c.game, c.duration_s ? c.duration_s.toFixed(0) + " s" : null]
    .filter(Boolean);
  card.meta.textContent = bits.join(" · ");

  card.badges.replaceChildren();
  const stCls = c.status === "done" ? "ok" : c.status === "failed" ? "bad" : "";
  card.badges.append(badge(c.status === "fetched" ? "raw, not processed" : c.status, stCls));
  if (c.layout_type) {
    const conf = c.layout_confidence;
    const pct = conf == null ? "" : " " + Math.round(conf * 100) + "%";
    const manual = c.layout_override ? " (manual)" : "";
    card.badges.append(badge("layout: " + c.layout_type + pct + manual));
    if (!c.layout_override && conf != null && conf < LOW_CONFIDENCE) {
      card.badges.append(badge("low confidence, check layout", "warn"));
    }
  }
  const lang = c.detected_language || (c.language !== "auto" ? c.language : null);
  if (lang) card.badges.append(badge("language: " + lang));
  if (c.status === "done" && c.review_status && c.review_status !== "pending") {
    card.badges.append(badge(c.review_status, c.review_status === "approved" ? "ok" : "bad"));
  }

  // stage chips: done before the current stage, active on it
  const cur = STAGES.indexOf(c.stage);
  [...card.stages.children].forEach((chip, i) => {
    let cls = "stage";
    if (c.status === "done") cls += " done";
    else if (c.status === "fetched") { if (i < 2) cls += " done"; }
    else if (c.status === "failed" && i === cur) cls += " failed";
    else if (cur >= 0 && i < cur) cls += " done";
    else if (active && i === cur) cls += " active";
    chip.className = cls;
  });
  card.bar.hidden = !active;
  card.bar.value = c.progress || 0;

  card.error.hidden = !c.error;
  card.error.textContent = c.error || "";

  card.select.value = c.layout_override || "auto";
  card.select.disabled = active;
  card.layoutLabel.hidden = c.status === "fetched"; // changing layout would start processing
  card.process.hidden = c.status !== "fetched";
  card.retry.hidden = c.status !== "failed";
  card.del.disabled = active;
  card.post.root.hidden = c.status !== "done";
  card.captions.root.hidden = c.status !== "done";
  card.captions.save.disabled = active;
  card.captions.reset.disabled = active;
  card.hook.root.hidden = c.status !== "done";
  card.hook.save.disabled = active;
  card.hook.sync(c);

  // video: swap src only when the file changed (updated_at moves on every re-render).
  // Unprocessed clips show the raw download.
  const videoUrl = c.output_url || c.raw_url;
  if (videoUrl) {
    const src = withToken(videoUrl + (c.status === "done" || c.status === "fetched" ? "?v=" + encodeURIComponent(c.updated_at) : ""));
    if (card.videoSrc !== src && !active) {
      card.videoSrc = src;
      card.video.src = src;
      card.preview.replaceChildren(card.video);
    }
  }
  if (c.output_url) {
    card.download.href = withToken(c.output_url + "?download=1");
    card.download.hidden = active;
  } else {
    card.download.hidden = true;
  }
  card.preview.classList.toggle("stale", active && !!card.videoSrc);
  $("empty").hidden = clips.size > 0;
}

function removeCard(id) {
  clips.delete(id);
  const card = cards.get(id);
  if (card) { card.root.remove(); cards.delete(id); }
  $("empty").hidden = clips.size > 0;
  applyFilters();
  renderInbox();
}

// -- filters (day + creator) ---------------------------------------------------------------
const localDay = (iso) => {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
};

function matchesFilters(c) {
  const day = $("filter-day").value;
  const creator = $("filter-creator").value;
  return (!day || localDay(c.created_at) === day) && (!creator || c.channel === creator);
}

function refreshCreators() {
  const sel = $("filter-creator");
  const keep = sel.value;
  const names = [...new Set([...clips.values()].map((c) => c.channel).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
  if (keep && !names.includes(keep)) names.push(keep); // keep a selected creator visible
  // progress events call this constantly; rebuilding would close an open dropdown
  if (names.join("\n") === [...sel.options].slice(1).map((o) => o.value).join("\n")) return;
  sel.replaceChildren(el("option", { value: "", textContent: "all" }),
    ...names.map((n) => el("option", { value: n, textContent: n })));
  sel.value = keep;
}

function applyFilters() {
  refreshCreators();
  const active = !!($("filter-day").value || $("filter-creator").value);
  let shown = 0;
  for (const [id, card] of cards) {
    const ok = matchesFilters(clips.get(id));
    card.root.hidden = !ok;
    if (ok) shown++;
  }
  $("filter-clear").hidden = !active;
  $("no-match").hidden = !(active && clips.size > 0 && shown === 0);
}

// -- actions ---------------------------------------------------------------------
function upsert(clip) {
  clips.set(clip.id, clip);
  updateCard(clip);
  applyFilters();
  renderInbox();
}

async function rerender(id, layout) {
  const card = cards.get(id);
  card.error.hidden = true;
  try {
    upsert(await api(`/api/clips/${id}/rerender`, {
      method: "POST", body: layout === null ? {} : { layout },
    }));
  } catch (e) {
    card.error.hidden = false;
    card.error.textContent = String(e.message || e);
    updateCard(clips.get(id));
  }
}

async function processClip(id) {
  const card = cards.get(id);
  card.error.hidden = true;
  try {
    upsert(await api(`/api/clips/${id}/process`, { method: "POST", body: {} }));
  } catch (e) {
    card.error.hidden = false;
    card.error.textContent = String(e.message || e);
  }
}

async function remove(id) {
  if (!confirm("Delete this clip and its files?")) return;
  try {
    await api(`/api/clips/${id}`, { method: "DELETE" });
    removeCard(id);
  } catch (e) {
    const card = cards.get(id);
    card.error.hidden = false;
    card.error.textContent = String(e.message || e);
  }
}

async function loadClips() {
  const list = await api("/api/clips");
  const seen = new Set(list.map((c) => c.id));
  for (const id of [...clips.keys()]) if (!seen.has(id)) removeCard(id);
  // list is newest first; prepend() in reverse so the order matches
  [...list].reverse().forEach(upsert);
  $("empty").hidden = clips.size > 0;
}

// -- review inbox ---------------------------------------------------------------------------
const inboxCards = new Map(); // id -> {root, video, ...}
let inboxCurrent = null; // id of the highlighted card
let lastDecision = null; // id of the clip decided last (for undo)
let inboxSig = ""; // what the inbox DOM currently shows
let view = "clips";

const isPending = (c) => c.status === "done" && c.review_status === "pending" && !!c.output_url;
const inboxIds = () => [...clips.values()].filter(isPending).sort((a, b) => b.id - a.id)
  .map((c) => c.id);

function buildInboxCard(c) {
  const video = el("video", { controls: true, preload: "metadata", playsInline: true, loop: true });
  const info = el("div", { class: "meta" });
  const badges = el("div", { class: "badges" });
  const err = el("p", { class: "error", hidden: true });
  const approve = el("button", { class: "approve", textContent: "Approve (A)",
    onclick: () => decide(c.id, "approve") });
  const reject = el("button", { class: "reject", textContent: "Reject (R)",
    onclick: () => decide(c.id, "reject") });
  const root = el("article", { class: "card inbox-card", onclick: () => setCurrent(c.id, false) },
    el("div", { class: "preview" }, video),
    el("div", {}, el("div", { class: "title" }), info, badges, err,
      el("div", { class: "actions" }, approve, reject)));
  return { root, video, title: root.querySelector(".title"), info, badges, err, approve, reject,
    src: "" };
}

function updateInboxCard(c) {
  let card = inboxCards.get(c.id);
  if (!card) { card = buildInboxCard(c); inboxCards.set(c.id, card); }
  card.title.textContent = c.title || c.slug;
  const lang = c.detected_language || (c.language !== "auto" ? c.language : null);
  card.info.textContent = [c.channel, c.game, c.duration_s ? c.duration_s.toFixed(0) + " s" : null]
    .filter(Boolean).join(" · ");
  card.badges.replaceChildren(
    ...[c.layout_type ? "layout: " + c.layout_type : null, lang ? "language: " + lang : null]
      .filter(Boolean).map((t) => badge(t)));
  const src = withToken(c.output_url + "?v=" + encodeURIComponent(c.updated_at));
  if (card.src !== src) { card.src = src; card.video.src = src; }
  return card;
}

function renderInbox() {
  const ids = inboxIds();
  const n = ids.length;
  $("inbox-count").textContent = n;
  $("inbox-count").hidden = n === 0;
  $("inbox-empty").hidden = n > 0;
  for (const [id, card] of [...inboxCards]) {
    if (!ids.includes(id)) { card.video.pause(); card.root.remove(); inboxCards.delete(id); }
  }
  if (view !== "inbox") { inboxSig = ""; return; } // only build players while the tab is open
  // progress events of other clips fire constantly; re-attaching a <video> would stop playback
  const sig = ids.map((id) => id + ":" + clips.get(id).updated_at).join(",");
  if (sig === inboxSig) { highlightInbox(false); return; }
  inboxSig = sig;
  const order = ids.map((id) => updateInboxCard(clips.get(id)).root);
  $("inbox").replaceChildren(...order);
  if (!ids.includes(inboxCurrent)) inboxCurrent = ids[0] ?? null;
  highlightInbox(false);
}

function highlightInbox(scroll) {
  for (const [id, card] of inboxCards) card.root.classList.toggle("current", id === inboxCurrent);
  const cur = inboxCards.get(inboxCurrent);
  if (cur && scroll) cur.root.scrollIntoView({ block: "center", behavior: "smooth" });
}

function setCurrent(id, scroll = true) {
  if (id === inboxCurrent) return;
  const prev = inboxCards.get(inboxCurrent);
  if (prev) prev.video.pause();
  inboxCurrent = id;
  highlightInbox(scroll);
}

function moveCurrent(step) {
  const ids = inboxIds();
  if (!ids.length) return;
  const i = ids.indexOf(inboxCurrent);
  setCurrent(ids[Math.min(ids.length - 1, Math.max(0, i + step))]);
}

async function decide(id, action) {
  const card = inboxCards.get(id);
  const ids = inboxIds();
  const i = ids.indexOf(id);
  const next = ids[i + 1] ?? ids[i - 1] ?? null; // advance, or step back from the last card
  const wasCurrent = inboxCurrent === id;
  // move first: the SSE event can re-render the inbox before the response arrives and would
  // otherwise reset the selection to the first card
  if (wasCurrent) inboxCurrent = next;
  try {
    const clip = await api(`/api/clips/${id}/${action}`, { method: "POST" });
    if (card) card.video.pause();
    lastDecision = id;
    upsert(clip);
    highlightInbox(true);
  } catch (e) {
    if (wasCurrent) inboxCurrent = id;
    if (card) { card.err.hidden = false; card.err.textContent = String(e.message || e); }
  }
}

async function undoDecision() {
  if (lastDecision === null) return;
  const id = lastDecision;
  lastDecision = null;
  try {
    inboxCurrent = id;
    upsert(await api(`/api/clips/${id}/review/reset`, { method: "POST" }));
    highlightInbox(true);
  } catch { /* clip may be gone */ }
}

function toggleCurrentPlayback() {
  const card = inboxCards.get(inboxCurrent);
  if (!card) return;
  if (card.video.paused) card.video.play().catch(() => {}); else card.video.pause();
}

function showView(name) {
  view = name;
  $("view-clips").hidden = name !== "clips";
  $("view-inbox").hidden = name !== "inbox";
  for (const b of document.querySelectorAll(".tab")) {
    b.classList.toggle("active", b.dataset.view === name);
  }
  if (name !== "inbox") for (const card of inboxCards.values()) card.video.pause();
  renderInbox();
  if (name === "inbox") highlightInbox(false);
}

for (const b of document.querySelectorAll(".tab")) {
  b.addEventListener("click", () => showView(b.dataset.view));
}

document.addEventListener("keydown", (e) => {
  if (view !== "inbox" || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName))) return;
  const k = e.key.toLowerCase();
  let handled = true;
  if (k === "a") { if (inboxCurrent !== null) decide(inboxCurrent, "approve"); }
  else if (k === "r") { if (inboxCurrent !== null) decide(inboxCurrent, "reject"); }
  else if (k === "j" || k === "arrowdown" || k === "arrowright") moveCurrent(1);
  else if (k === "k" || k === "arrowup" || k === "arrowleft") moveCurrent(-1);
  else if (k === " ") toggleCurrentPlayback();
  else if (k === "u") undoDecision();
  else handled = false;
  if (handled) e.preventDefault();
});

// -- followed channels (auto-sync) ------------------------------------------------------
function ago(iso) {
  if (!iso) return "never";
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + " min ago";
  if (s < 129600) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " d ago";
}

function channelRow(ch) {
  const info = el("span", { class: "info" });
  const showInfo = (c) => {
    info.classList.toggle("bad", !!c.last_error);
    info.textContent = c.last_error
      ? "sync failed: " + c.last_error
      : `synced ${ago(c.last_sync_at)} · ${c.total_queued} queued` +
        (c.last_sync_at ? "" : " (first check pending)");
  };
  showInfo(ch);
  const enabled = el("input", { type: "checkbox", checked: ch.enabled });
  const min = el("input", { type: "number", min: 0, value: ch.min_views, title: "min views" });
  const lang = el("select", {}, ...["auto", "de", "en"].map((l) =>
    el("option", { value: l, textContent: l, selected: ch.language === l })));
  const patch = async (body) => {
    try { showInfo(await api(`/api/channels/${ch.id}`, { method: "PATCH", body })); }
    catch (e) { info.classList.add("bad"); info.textContent = String(e.message || e); }
  };
  enabled.addEventListener("change", () => patch({ enabled: enabled.checked }));
  min.addEventListener("change", () => patch({ min_views: Number(min.value) || 0 }));
  lang.addEventListener("change", () => patch({ language: lang.value }));
  const sync = el("button", { class: "secondary", textContent: "Check now", onclick: async () => {
    sync.disabled = true;
    try { showInfo(await api(`/api/channels/${ch.id}/sync`, { method: "POST" })); }
    catch (e) { info.classList.add("bad"); info.textContent = String(e.message || e); }
    finally { sync.disabled = false; }
  } });
  const rm = el("button", { class: "danger", textContent: "Unfollow", onclick: async () => {
    if (!confirm(`Stop following ${ch.name}? Existing clips stay.`)) return;
    try { await api(`/api/channels/${ch.id}`, { method: "DELETE" }); await loadChannels(); }
    catch (e) { info.classList.add("bad"); info.textContent = String(e.message || e); }
  } });
  return el("div", { class: "channel" },
    el("span", { class: "name", textContent: ch.name }),
    el("label", {}, enabled, "auto"),
    el("label", {}, "min views", min), lang, sync, rm, info);
}

async function loadChannels() {
  const list = await api("/api/channels");
  const box = $("channels");
  box.replaceChildren(...(list.length
    ? list.map(channelRow)
    : [el("p", { class: "muted", textContent: "No channels followed yet." })]));
  const active = list.filter((c) => c.enabled).length;
  $("channels-count").textContent = list.length ? `(${active} of ${list.length} active)` : "";
}

// -- live updates (SSE) -------------------------------------------------------------
let es = null;
let retryMs = 1000;
let retryTimer = null;

function setConn(live) {
  const n = $("conn");
  n.textContent = live ? "live" : "reconnecting";
  n.classList.toggle("live", live);
}

async function onClipEvent(ev) {
  let data;
  try { data = JSON.parse(ev.data); } catch { return; }
  const prev = clips.get(data.id);
  if (!prev) {
    try { upsert(await api(`/api/clips/${data.id}`)); } catch { /* ignore */ }
    return;
  }
  const merged = { ...prev, ...data };
  upsert(merged);
  // full row (output_url, detected_language) once a run ends
  if (data.status && data.status !== prev.status && (data.status === "done" || data.status === "failed" || data.status === "fetched")) {
    try { upsert(await api(`/api/clips/${data.id}`)); } catch { /* ignore */ }
  }
}

function connect() {
  clearTimeout(retryTimer);
  if (es) es.close();
  es = new EventSource(withToken("/api/events"));
  es.addEventListener("open", () => {
    retryMs = 1000;
    setConn(true);
    loadClips().catch(() => {}); // resync whatever was missed while disconnected
  });
  es.addEventListener("clip", onClipEvent);
  es.addEventListener("deleted", (ev) => {
    try { removeCard(JSON.parse(ev.data).id); } catch { /* ignore */ }
  });
  es.addEventListener("error", () => {
    setConn(false);
    if (es.readyState === EventSource.CLOSED) { // browser gave up (e.g. 401): retry ourselves
      es.close();
      retryTimer = setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 2, 15000);
    }
  });
}

// -- forms ---------------------------------------------------------------------------
$("filter-day").addEventListener("input", applyFilters);
$("filter-creator").addEventListener("change", applyFilters);
$("filter-clear").addEventListener("click", () => {
  $("filter-day").value = "";
  $("filter-creator").value = "";
  applyFilters();
});

$("token-form").addEventListener("submit", (e) => {
  e.preventDefault();
  setToken($("token").value.trim());
  $("token-form").hidden = true;
  loadClips().catch(() => {});
  loadChannels().catch(() => {});
  connect();
});

$("new-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("form-error");
  err.hidden = true;
  $("submit").disabled = true;
  try {
    const clip = await api("/api/clips", {
      method: "POST",
      body: { url: $("url").value.trim(), language: $("language").value, layout: $("layout").value },
    });
    upsert(clip);
    $("url").value = "";
  } catch (ex) {
    err.hidden = false;
    err.textContent = String(ex.message || ex);
  } finally {
    $("submit").disabled = false;
  }
});

$("channel-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("channel-error");
  err.hidden = true;
  try {
    await api("/api/channels", { method: "POST", body: {
      name: $("channel-name").value.trim(),
      min_views: Number($("channel-min").value) || 0,
      language: $("channel-language").value,
    } });
    $("channel-name").value = "";
    await loadChannels();
  } catch (ex) {
    err.hidden = false;
    err.textContent = String(ex.message || ex);
  }
});

// keep "synced x ago" fresh, but never rebuild rows while the user is editing one
setInterval(() => {
  if ($("channels-box").open && !$("channels").contains(document.activeElement)) {
    loadChannels().catch(() => {});
  }
}, 60000);

loadClips().catch(() => {});
loadChannels().catch(() => {});
connect();
