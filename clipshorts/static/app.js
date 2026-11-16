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
  const actions = el("div", { class: "actions" }, layoutLabel, download, retry, del);
  const post = buildPost(id, error);
  const body = el("div", {}, title, meta, badges, stages, bar, error, actions, post.root);
  const root = el("article", { class: "card clip" }, preview, body);
  return { root, video, placeholder, preview, title, meta, badges, stages, bar, error, select,
    download, retry, del, post, videoSrc: "" };
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
  card.badges.append(badge(c.status, stCls));
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

  // stage chips: done before the current stage, active on it
  const cur = STAGES.indexOf(c.stage);
  [...card.stages.children].forEach((chip, i) => {
    let cls = "stage";
    if (c.status === "done") cls += " done";
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
  card.retry.hidden = c.status !== "failed";
  card.del.disabled = active;
  card.post.root.hidden = c.status !== "done";

  // video: swap src only when the file changed (updated_at moves on every re-render)
  if (c.output_url) {
    const src = withToken(c.output_url + (c.status === "done" ? "?v=" + encodeURIComponent(c.updated_at) : ""));
    if (card.videoSrc !== src && !active) {
      card.videoSrc = src;
      card.video.src = src;
      card.preview.replaceChildren(card.video);
    }
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
}

// -- actions ---------------------------------------------------------------------
function upsert(clip) {
  clips.set(clip.id, clip);
  updateCard(clip);
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
  if (data.status && data.status !== prev.status && (data.status === "done" || data.status === "failed")) {
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
$("token-form").addEventListener("submit", (e) => {
  e.preventDefault();
  setToken($("token").value.trim());
  $("token-form").hidden = true;
  loadClips().catch(() => {});
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

loadClips().catch(() => {});
connect();
