/* Hybrid Vision GUI — single-page app.

   Architecture: one selected sequence; frame data + tracks fetched per
   sequence; every tab registers a draw(idx) — image loads and UI events
   invalidate the current frame, which re-runs the active draw. */

"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  sequences: [],
  selected: null,
  frames: null,       // /api/frames payload (+ computed per-frame status)
  tracks: null,
  metrics: null,
  idx: 0,
  playing: false,
  loop: false,
  fps: 30,
  wipe: 0.5,
  detView: "split",   // split | rgb | event
  lens: true,
  trackModality: "rgb",
  metricsView: "true",
  tab: "preview",
  polling: null,
  draw: null,         // active tab's draw(idx)
  show: null,         // active transport's show(idx)
};

const W = 1280, H = 720;
const TRACK_COLORS = ["#62b6d9", "#e0a050", "#a08cf0", "#e5645a", "#46c08a",
                      "#d8c66f", "#d86fb8"];
const ICONS = {
  play: '<svg viewBox="0 0 16 16"><path d="M4 2l10 6-10 6z"/></svg>',
  pause: '<svg viewBox="0 0 16 16"><path d="M3 2h4v12H3zM9 2h4v12H9z"/></svg>',
  prev: '<svg viewBox="0 0 16 16"><path d="M11 2v12L4 8zM3 2h2v12H3z"/></svg>',
  next: '<svg viewBox="0 0 16 16"><path d="M5 2v12l7-6zM11 2h2v12h-2z"/></svg>',
  loop: '<svg viewBox="0 0 16 16"><path d="M8 2a6 6 0 016 6h-2a4 4 0 00-4-4V2zM8 14a6 6 0 01-6-6h2a4 4 0 004 4v2z"/><path d="M14 5l-3 3h6zM2 11l3-3H-1z" transform="translate(0,0) scale(.8) translate(1.5,1.5)"/></svg>',
};

/* ---------------- image cache with redraw-on-load ---------------- */
const cache = new Map();
let redrawQueued = false;
function invalidate() {
  if (redrawQueued || !state.draw) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    if (state.draw) state.draw(state.idx);
  });
}
function frameImage(modality, stem) {
  const key = `${modality}/${stem}`;
  let img = cache.get(key);
  if (!img) {
    img = new Image();
    img.onload = invalidate;
    img.src = `/img/${state.selected}/${modality}/${stem}`;
    cache.set(key, img);
    if (cache.size > 280) {
      for (const k of cache.keys()) {
        if (cache.size <= 220) break;
        cache.delete(k);
      }
    }
  }
  return img;
}
function preload(modality, from, n) {
  const frames = state.frames.frames;
  for (let k = from; k < Math.min(from + n, frames.length); k++)
    frameImage(modality, frames[k].stem);
}

/* ---------------- geometry ---------------- */
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!inter) return 0;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / ua;
}

/* Per-frame verdicts for the timeline + HUD. */
function classifyFrames(frames) {
  for (const fr of frames) {
    const kept = fr.dets.filter((d) => d.kept);
    fr.nKept = kept.length;
    fr.nRej = fr.dets.length - kept.length;
    fr.miss = fr.gt.some((g) => !kept.some((d) => iou(g, d.box) >= 0.5));
    fr.fp = kept.some((d) => !fr.gt.some((g) => iou(g, d.box) >= 0.5));
    fr.ok = fr.gt.length > 0 && !fr.miss;
  }
}

/* ---------------- sidebar ---------------- */
async function refreshSequences() {
  state.sequences = await (await fetch("/api/sequences")).json();
  const list = $("#seq-list");
  list.innerHTML = "";
  for (const s of state.sequences) {
    const b = document.createElement("button");
    b.className = "seq-item" + (s.name === state.selected ? " selected" : "");
    const dot = s.running ? "running" : s.results ? "results"
              : s.aligned ? "aligned" : "";
    const stateTxt = s.running ? "running" : s.results ? "ready"
                   : s.aligned ? "aligned" : "raw";
    b.innerHTML = `<span class="dot ${dot}"></span>
                   <span class="name">seq ${s.name}</span>
                   <span class="seq-state">${stateTxt}</span>`;
    b.onclick = () => selectSequence(s.name);
    list.appendChild(b);
  }
  $("#run-btn").disabled = !state.selected ||
    state.sequences.some((s) => s.running);
}

function updateChips() {
  const el = $("#chips");
  if (!state.frames) { el.innerHTML = ""; return; }
  const frames = state.frames.frames;
  const missed = frames.filter((f) => f.miss).length;
  const withGT = frames.filter((f) => f.gt.length).length;
  el.innerHTML = `
    <span class="chip"><b>seq ${state.selected}</b></span>
    <span class="chip"><b>${frames.length.toLocaleString()}</b> frames</span>
    <span class="chip"><b>${withGT ? (100 * (1 - missed / withGT)).toFixed(1) : "—"}%</b> frames w/ drone found</span>`;
}

async function selectSequence(name) {
  state.selected = name;
  state.frames = state.tracks = state.metrics = null;
  state.idx = 0;
  state.playing = false;
  state.draw = null;
  cache.clear();
  logLine(`selected seq ${name}`);
  await refreshSequences();
  const info = state.sequences.find((s) => s.name === name);
  if (info && info.results) {
    state.frames = await (await fetch(`/api/frames/${name}`)).json();
    classifyFrames(state.frames.frames);
    logLine(`${state.frames.frames.length} frames loaded`);
  }
  updateChips();
  render();
}

/* ---------------- upload ---------------- */
function setupUpload() {
  const drop = $("#drop"), input = $("#file-input");
  drop.onclick = () => input.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("armed"); };
  drop.ondragleave = () => drop.classList.remove("armed");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("armed");
    if (e.dataTransfer.files[0]) uploadZip(e.dataTransfer.files[0]);
  };
  input.onchange = () => { if (input.files[0]) uploadZip(input.files[0]); };
}
async function uploadZip(file) {
  logLine(`uploading ${file.name} (${(file.size / 1e9).toFixed(2)} GB)…`);
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const out = await res.json();
    if (!res.ok) throw new Error(out.detail || res.statusText);
    logLine(`extracted → dataset/${out.sequence}` +
            (out.note ? ` (${out.note})` : ""));
    await refreshSequences();
    selectSequence(out.sequence);
  } catch (err) {
    logLine(`upload failed: ${err.message}`, true);
  }
}

/* ---------------- pipeline run ---------------- */
function logLine(msg, err = false) {
  const c = $("#console");
  const line = document.createElement("div");
  if (err) line.className = "err";
  line.textContent = msg;
  c.appendChild(line);
  while (c.childNodes.length > 300) c.removeChild(c.firstChild);
  c.scrollTop = c.scrollHeight;
}

async function runPipeline() {
  if (!state.selected) return;
  const overwrite = $("#overwrite").checked;
  const res = await fetch(
    `/api/run/${state.selected}?overwrite=${overwrite}`, { method: "POST" });
  if (!res.ok) {
    logLine((await res.json()).detail || "run refused", true);
    return;
  }
  logLine(`pipeline started on seq ${state.selected}`);
  $("#run-btn").disabled = true;
  pollStatus(state.selected);
}

function pollStatus(seq) {
  clearInterval(state.polling);
  let seen = 0;
  state.polling = setInterval(async () => {
    const st = await (await fetch(`/api/status/${seq}`)).json();
    if (st.log.length && st.log.length !== seen) {
      seen = st.log.length;
      logLine(st.log[st.log.length - 1].slice(0, 160));
    }
    if (st.state === "done" || st.state === "error") {
      clearInterval(state.polling);
      if (st.state === "error") logLine(st.error, true);
      else logLine("pipeline finished ✓");
      await refreshSequences();
      if (st.state === "done" && seq === state.selected) selectSequence(seq);
    }
  }, 1200);
}

/* ---------------- tabs & rendering ---------------- */
function render() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === state.tab));
  const c = $("#content");
  state.playing = false;
  state.draw = null;

  if (state.tab === "reconstruction") {  // static tab, needs no sequence
    renderReconstruction(c);
    return;
  }
  if (!state.selected) {
    c.innerHTML = `<div class="placeholder">Select or upload a sequence in
      the sidebar, then run the pipeline.<br>Playback, tracking and metrics
      appear here.</div>`;
    return;
  }
  if (!state.frames) {
    c.innerHTML = `<div class="placeholder">No pipeline results for
      <b>seq ${state.selected}</b> yet — press <b>Run pipeline</b>.<br>
      The full run (two detectors, two verifiers, fusion) takes a few
      minutes on GPU.</div>`;
    return;
  }
  ({ preview: renderPreview, detections: renderDetections,
     tracking: renderTracking, metrics: renderMetrics }[state.tab])(c);
}

/* ---------------- player scaffolding ---------------- */
function playerHTML({ head, stageClass, canvases, side }) {
  const cvs = canvases.map((id) =>
    `<canvas id="${id}" width="${W}" height="${H}"></canvas>`).join("");
  return `
  <div class="player">
    <div class="player-head">${head}
      <div class="spacer"></div><span class="hud" id="hud"></span></div>
    <div class="stage ${stageClass || ""}">
      ${canvases.length ? `<div class="frames">${cvs}</div>` : ""}${side || ""}</div>
    <div class="transport-wrap">
      <canvas id="timeline" width="1200" height="16"></canvas>
      <div class="transport">
        <button class="tbtn" id="pp" title="play/pause (space)">${ICONS.play}</button>
        <button class="tbtn" id="step-b" title="previous frame (←)">${ICONS.prev}</button>
        <button class="tbtn" id="step-f" title="next frame (→)">${ICONS.next}</button>
        <button class="tbtn ${state.loop ? "on" : ""}" id="loop" title="loop">${ICONS.loop}</button>
        <input type="range" id="seek" min="0"
               max="${state.frames.frames.length - 1}" value="${state.idx}">
        <select id="fps">
          ${[5, 15, 30, 60].map((f) =>
            `<option value="${f}" ${f === state.fps ? "selected" : ""}>${f} fps</option>`).join("")}
        </select>
        <span class="timecode" id="timecode"></span>
      </div>
      <div class="kbd-hint" style="margin-top:6px">
        <kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> frame ·
        <kbd>shift</kbd>+<kbd>←→</kbd> ×10 · timeline:
        <span style="color:var(--keep)">■</span> found
        <span style="color:var(--drop)">■</span> missed
        <span style="color:var(--rgb)">■</span> false positive
      </div>
    </div>
  </div>`;
}

function drawTimeline() {
  const cv = $("#timeline");
  if (!cv) return;
  const frames = state.frames.frames;
  const ctx = cv.getContext("2d");
  const n = frames.length, w = cv.width, h = cv.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#22282d";
  ctx.fillRect(0, 0, w, h);
  const bw = w / n;
  for (let i = 0; i < n; i++) {
    const f = frames[i];
    let col = null;
    if (f.miss) col = "#e5645a";
    else if (f.ok) col = "#2f7f5c";
    if (col) {
      ctx.fillStyle = col;
      ctx.fillRect(i * bw, 3, Math.max(bw, 1), h - 6);
    }
    if (f.fp) {
      ctx.fillStyle = "#e0a050";
      ctx.fillRect(i * bw, 0, Math.max(bw, 1), 3);
    }
  }
  ctx.fillStyle = "#e8edf0";
  ctx.fillRect((state.idx / n) * w - 1, 0, 2.5, h);
}

function bindTransport(draw) {
  const frames = state.frames.frames;
  const seek = $("#seek"), pp = $("#pp"), tc = $("#timecode");
  state.draw = draw;

  function show(i) {
    state.idx = Math.max(0, Math.min(frames.length - 1, i));
    seek.value = state.idx;
    const fr = frames[state.idx];
    tc.textContent =
      `${String(Math.floor(fr.t / 60)).padStart(2, "0")}:` +
      `${(fr.t % 60).toFixed(2).padStart(5, "0")} · ` +
      `${state.idx + 1}/${frames.length}`;
    const hud = $("#hud");
    if (hud) hud.innerHTML =
      `GT <b>${fr.gt.length}</b> · kept <b>${fr.nKept}</b> · ` +
      `rejected <b>${fr.nRej}</b>` +
      (fr.miss ? ' · <b style="color:var(--drop)">MISS</b>' : "");
    drawTimeline();
    draw(state.idx);
  }
  state.show = show;

  pp.onclick = () => togglePlay();
  function togglePlay() {
    state.playing = !state.playing;
    pp.innerHTML = state.playing ? ICONS.pause : ICONS.play;
    if (state.playing) requestAnimationFrame(loop);
  }
  state.togglePlay = togglePlay;

  $("#step-b").onclick = () => { state.playing = false; pp.innerHTML = ICONS.play; show(state.idx - 1); };
  $("#step-f").onclick = () => { state.playing = false; pp.innerHTML = ICONS.play; show(state.idx + 1); };
  $("#loop").onclick = (e) => {
    state.loop = !state.loop;
    e.currentTarget.classList.toggle("on", state.loop);
  };
  seek.oninput = () => { state.playing = false; pp.innerHTML = ICONS.play; show(+seek.value); };
  $("#fps").onchange = (e) => { state.fps = +e.target.value; };
  $("#timeline").onclick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    show(Math.round(((e.clientX - r.left) / r.width) * (frames.length - 1)));
  };

  let last = 0;
  function loop(ts) {
    if (!state.playing) return;
    if (!last || ts - last >= 1000 / state.fps) {
      last = ts;
      if (state.idx >= frames.length - 1) {
        if (state.loop) show(0);
        else {
          state.playing = false;
          pp.innerHTML = ICONS.play;
          return;
        }
      } else show(state.idx + 1);
    }
    requestAnimationFrame(loop);
  }
  show(state.idx);
}

function drawImg(ctx, img) {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, W, H);
}

function drawBox(ctx, box, color, width = 3, dash = []) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
  ctx.setLineDash([]);
}
function drawLabel(ctx, x, y, text, color) {
  ctx.font = "600 17px ui-monospace, monospace";
  ctx.fillStyle = color;
  ctx.fillText(text, x, Math.max(18, y));
}

/* Lens target: the frame's most interesting box. */
function lensTarget(fr) {
  return fr.dets.find((d) => d.kept)?.box || fr.gt[0] || fr.dets[0]?.box || null;
}

/* Draw a magnified crop around `target` into a dedicated square canvas. */
function drawLensPane(ctx, fr, img, target) {
  const L = ctx.canvas.width;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, L, L);
  if (!target || !(img.complete && img.naturalWidth)) return 0;
  const cx = (target[0] + target[2]) / 2, cy = (target[1] + target[3]) / 2;
  const half = Math.max(48, (target[2] - target[0]), (target[3] - target[1]));
  const sx = Math.max(0, Math.min(W - 2 * half, cx - half));
  const sy = Math.max(0, Math.min(H - 2 * half, cy - half));
  ctx.imageSmoothingEnabled = half < 90;  // crisp pixels at high zoom
  ctx.drawImage(img, sx, sy, 2 * half, 2 * half, 0, 0, L, L);
  const scale = L / (2 * half);
  for (const g of fr.gt)
    drawBox(ctx, [(g[0] - sx) * scale, (g[1] - sy) * scale,
                  (g[2] - sx) * scale, (g[3] - sy) * scale], "#58d68d", 2);
  for (const d of fr.dets) {
    if (!d.kept) continue;
    const b = d.box;
    drawBox(ctx, [(b[0] - sx) * scale, (b[1] - sy) * scale,
                  (b[2] - sx) * scale, (b[3] - sy) * scale],
            d.src === "rgb" ? "#e0a050" : "#f4d03f", 2);
  }
  return scale;
}

/* ---------------- Preview (wipe) ---------------- */
function renderPreview(c) {
  c.innerHTML = playerHTML({
    head: `<div class="legend">
        <span><span class="swatch" style="background:var(--rgb)"></span>RGB (left)</span>
        <span><span class="swatch" style="background:var(--event)"></span>Event (right)</span>
        <span>drag on the image to move the divider</span></div>`,
    canvases: ["cv"],
  });
  const cv = $("#cv"), ctx = cv.getContext("2d");

  function draw(i) {
    const fr = state.frames.frames[i];
    preload("rgb", i + 1, 30);
    preload("event", i + 1, 30);
    drawImg(ctx, frameImage("rgb", fr.stem));
    const x = state.wipe * W;
    const ev = frameImage("event", fr.stem);
    if (ev.complete && ev.naturalWidth) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, 0, W - x, H);
      ctx.clip();
      ctx.drawImage(ev, 0, 0, W, H);
      ctx.restore();
    }
    ctx.fillStyle = "rgba(232,237,240,.9)";
    ctx.fillRect(x - 1.5, 0, 3, H);
    ctx.beginPath(); ctx.arc(x, H / 2, 13, 0, 7);
    ctx.fillStyle = "#e8edf0"; ctx.fill();
    ctx.fillStyle = "#131719";           // grip: two short vertical bars
    ctx.fillRect(x - 5, H / 2 - 5, 3, 10);
    ctx.fillRect(x + 2, H / 2 - 5, 3, 10);
  }
  bindTransport(draw);

  let dragging = false;
  const setWipe = (e) => {
    const r = cv.getBoundingClientRect();
    state.wipe = Math.max(0.02, Math.min(0.98, (e.clientX - r.left) / r.width));
    invalidate();
  };
  cv.style.cursor = "ew-resize";
  cv.onpointerdown = (e) => { dragging = true; cv.setPointerCapture(e.pointerId); setWipe(e); };
  cv.onpointermove = (e) => { if (dragging) setWipe(e); };
  cv.onpointerup = () => { dragging = false; };
}

/* ---------------- Detections ---------------- */
function overlayDetections(ctx, fr) {
  for (const g of fr.gt) drawBox(ctx, g, "#58d68d", 3);
  for (const d of fr.dets) {
    if (d.kept) {
      const col = d.src === "rgb" ? "#e0a050" : "#f4d03f";
      drawBox(ctx, d.box, col, 3, d.src === "rgb" ? [10, 6] : []);
      drawLabel(ctx, d.box[0], d.box[1] - 6,
                `${d.cls || "drone"} ${d.fus.toFixed(2)}`, col);
    } else {
      drawBox(ctx, d.box, "#e5645a", 2, [4, 4]);
    }
  }
}

function renderDetections(c) {
  const seg = (id, val, label) =>
    `<button id="${id}" class="${state.detView === val ? "on" : ""}">${label}</button>`;
  const mods = state.detView === "split" ? ["rgb", "event"]
             : [state.detView === "event" ? "event" : "rgb"];
  const col = (m) => `
    <div class="modcol">
      <canvas class="video" id="cv-${m}" width="${W}" height="${H}"></canvas>
      ${state.lens ? `
      <div class="lensbox"><div class="cap"><span>${m} · zoom</span>
        <span id="zoom-${m}"></span></div>
        <canvas id="lens-${m}" width="320" height="320"></canvas></div>` : ""}
    </div>`;
  c.innerHTML = playerHTML({
    head: `
      <div class="seg">${seg("v-split", "split", "Side by side")}
        ${seg("v-rgb", "rgb", "RGB")}${seg("v-evt", "event", "Event")}</div>
      <button class="tbtn ${state.lens ? "on" : ""}" id="lens" title="zoom view"
        style="width:auto;padding:0 10px;font:600 11px var(--mono)">zoom</button>
      <div class="legend">
        <span><span class="swatch" style="background:#58d68d"></span>ground truth</span>
        <span><span class="swatch" style="background:#f4d03f"></span>kept</span>
        <span><span class="swatch dash" style="color:#e0a050"></span>kept · RGB-sourced</span>
        <span><span class="swatch dash" style="color:#e5645a"></span>rejected</span>
      </div>`,
    canvases: [],
    side: mods.map(col).join(""),
  });
  const vctx = Object.fromEntries(
    mods.map((m) => [m, $(`#cv-${m}`).getContext("2d")]));
  const lctx = state.lens ? Object.fromEntries(
    mods.map((m) => [m, $(`#lens-${m}`).getContext("2d")])) : null;

  $("#v-split").onclick = () => { state.detView = "split"; render(); };
  $("#v-rgb").onclick = () => { state.detView = "rgb"; render(); };
  $("#v-evt").onclick = () => { state.detView = "event"; render(); };
  $("#lens").onclick = () => { state.lens = !state.lens; render(); };

  bindTransport((i) => {
    const fr = state.frames.frames[i];
    preload("rgb", i + 1, 25);
    preload("event", i + 1, 25);
    const target = state.lens ? lensTarget(fr) : null;
    let zoom = 0;
    for (const m of mods) {
      const img = frameImage(m, fr.stem);
      drawImg(vctx[m], img);
      overlayDetections(vctx[m], fr);
      if (lctx) {
        zoom = drawLensPane(lctx[m], fr, img, target) || zoom;
        $(`#zoom-${m}`).textContent = zoom ? "×" + zoom.toFixed(1) : "—";
      }
    }
    if (state.lens) {
      const best = fr.dets.find((d) => d.kept);
      const hud = $("#hud");
      if (hud && target) {
        hud.querySelectorAll(".tgt").forEach((n) => n.remove());
        const span = document.createElement("span");
        span.className = "tgt";
        span.textContent =
          ` · target ${Math.round(target[2] - target[0])}×` +
          `${Math.round(target[3] - target[1])}px` +
          (best ? ` · fusion ${best.fus.toFixed(3)} · ${best.src}` : " · GT only");
        hud.appendChild(span);
      }
    }
  });
}

/* ---------------- Tracking ---------------- */
async function renderTracking(c) {
  if (!state.tracks) {
    c.innerHTML = `<div class="placeholder">Computing Kalman tracks…</div>`;
    state.tracks = await (await fetch(`/api/tracks/${state.selected}`)).json();
    if (state.tab !== "tracking") return;
  }
  c.innerHTML = playerHTML({
    head: `
      <div class="seg">
        <button id="m-rgb" class="${state.trackModality === "rgb" ? "on" : ""}">RGB</button>
        <button id="m-evt" class="${state.trackModality === "event" ? "on" : ""}">Event</button>
      </div>
      <div class="legend">
        <span><span class="swatch" style="background:#58d68d"></span>GT now + next second</span>
        <span><span class="swatch" style="background:#62b6d9"></span>track + prediction</span>
        <span><span class="swatch" style="background:#8d979f"></span>tentative</span>
        <span><span class="swatch" style="background:#f4d03f"></span>kept detection</span>
      </div>`,
    canvases: ["cv"],
  });
  const ctx = $("#cv").getContext("2d");
  $("#m-rgb").onclick = () => { state.trackModality = "rgb"; render(); };
  $("#m-evt").onclick = () => { state.trackModality = "event"; render(); };

  function path(pts, color) {
    if (!pts.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    for (const p of pts) {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fill();
    }
  }

  bindTransport((i) => {
    const fr = state.frames.frames[i];
    const tk = state.tracks.frames[i];
    preload(state.trackModality, i + 1, 30);
    drawImg(ctx, frameImage(state.trackModality, fr.stem));
    if (!tk) return;
    for (const b of tk.dets) drawBox(ctx, b, "#f4d03f", 2);
    tk.gt.forEach((g, gi) => {
      ctx.fillStyle = "#58d68d";
      ctx.beginPath(); ctx.arc(g[0], g[1], 5, 0, 7); ctx.fill();
      if (tk.gt_future[gi]) path(tk.gt_future[gi], "#58d68d");
    });
    for (const p of tk.tentative) {
      ctx.fillStyle = "#8d979f";
      ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill();
    }
    for (const tr of tk.tracks) {
      const col = TRACK_COLORS[tr.id % TRACK_COLORS.length];
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(tr.pos[0], tr.pos[1], 7, 0, 7); ctx.fill();
      path(tr.future, col);
      drawLabel(ctx, tr.pos[0] + 10, tr.pos[1] - 10,
                `T${tr.id} ${tr.speed.toFixed(0)}px/s`, col);
    }
  });
}

/* ---------------- Metrics ---------------- */
async function renderMetrics(c) {
  if (!state.metrics) {
    c.innerHTML = `<div class="placeholder">Loading metrics…</div>`;
    state.metrics = await (await fetch(`/api/metrics/${state.selected}`)).json();
    if (state.tab !== "metrics") return;
  }
  const man = state.metrics.manifest;
  const stats = state.metrics.merge_stats;
  const source = state.metricsView === "true"
    ? man.metrics.all : man.metrics_conditional.all;
  const NAMES = {
    "event_yolo_conf0.25": "Event YOLO · conf ≥ 0.25",
    "event_yolo_conf0.50": "Event YOLO · conf ≥ 0.50",
    "hybrid_yolo_conf0.25": "Hybrid YOLO · conf ≥ 0.25",
    "fusion_v4": `Fusion · λ=${man.fusion_lambda}, τ=${man.fusion_threshold.toFixed(3)}`,
  };
  const bar = (k, v, col) => `
    <div class="bar-row"><span class="k">${k}</span>
      <div class="bar"><i style="width:${(v * 100).toFixed(1)}%;background:${col}"></i></div>
      <span class="v">${(v * 100).toFixed(1)}%</span></div>`;

  const cards = Object.entries(source).map(([key, m]) => `
    <div class="card">
      <h3>${NAMES[key] || key}</h3>
      <div class="cm">
        <div></div><div class="axis">Pred drone</div><div class="axis">Pred bg</div>
        <div class="axis">GT drone</div>
        <div class="cell tp"><span class="n">${m.TP.toLocaleString()}</span><span class="lbl">TP</span></div>
        <div class="cell fn"><span class="n">${m.FN.toLocaleString()}</span><span class="lbl">FN</span></div>
        <div class="axis">GT bg</div>
        <div class="cell fp"><span class="n">${m.FP.toLocaleString()}</span><span class="lbl">FP</span></div>
        <div class="cell tn"><span class="n">—</span><span class="lbl">TN n/a</span></div>
      </div>
      <div class="bars">
        ${bar("P", m.precision, "var(--accent)")}
        ${bar("R", m.recall, "var(--keep)")}
        ${bar("F1", m.f1, "var(--rgb)")}
      </div>
    </div>`).join("");

  const ceiling = stats ? `
    <div class="statgrid">
      <div class="stat"><div class="v">${stats.n_gt_boxes.toLocaleString()}</div><div class="k">GT boxes</div></div>
      <div class="stat"><div class="v">${(stats.proposal_ceiling_recall * 100).toFixed(1)}%</div><div class="k">proposal ceiling</div></div>
      <div class="stat"><div class="v">${(stats.event_only_ceiling_recall * 100).toFixed(1)}%</div><div class="k">event-only ceiling</div></div>
      <div class="stat"><div class="v">${stats.by_source.rgb || 0}</div><div class="k">RGB-only proposals</div></div>
      <div class="stat"><div class="v">${stats.by_source.both || 0}</div><div class="k">both detectors</div></div>
    </div>` : "";

  c.innerHTML = `<div class="metrics-scroll">
    <div class="metrics-head">
      <h2>seq ${man.name}</h2>
      <span class="meta">${man.n_frames_total.toLocaleString()} frames ·
        ${man.n_detections_total.toLocaleString()} detections ·
        TN undefined for open-set detection</span>
    </div>
    <div class="pill-row">
      <button class="pill ${state.metricsView === "true" ? "on" : ""}" id="pv-true">
        True — misses counted</button>
      <button class="pill ${state.metricsView === "cond" ? "on" : ""}" id="pv-cond">
        Proposal-conditional — legacy</button>
    </div>
    ${ceiling}
    <div class="cards">${cards}</div>
    <p class="note-strip"><strong>Reading these numbers:</strong> the true view
    counts every ground-truth drone — one the detectors never proposed is a
    false negative. The legacy view only scores proposals and overstates
    recall; it is shown for comparison with older reports. No fusion threshold
    can recover a drone above the proposal ceiling.</p></div>`;

  $("#pv-true").onclick = () => { state.metricsView = "true"; render(); };
  $("#pv-cond").onclick = () => { state.metricsView = "cond"; render(); };
}

/* ---------------- Reconstruction (static info tab) ---------------- */
function renderReconstruction(c) {
  c.innerHTML = `<div class="metrics-scroll"><div class="recon">
    <h2>Event-based video reconstruction — the comparison study</h2>
    <p>The project description asks us to compare our hybrid detector with a
    model trained on <b>intensity images reconstructed from the event
    stream</b>, using at least two reconstruction methods. That study lives in
    its own branch — it is an offline comparison for the report, not part of
    this detection pipeline, and it needs extra machinery this app doesn't
    (the raw <span class="mono">events.raw</span> stream, OpenEB/Metavision
    decoding, and per-method model environments).</p>

    <a class="btn primary recon-link" target="_blank" rel="noopener"
       href="https://github.com/NoMoreNameToUse/AMI-Group7-SoSe26/tree/video_reconstruction_spacezhang">
       Open the reconstruction branch on GitHub ↗</a>

    <h3>What the branch does</h3>
    <div class="recon-flow mono">
FRED raw event stream (events.raw)
  │  decode (OpenEB) + 33 ms windows aligned to the RGB/label clock
  ▼
intensity-video reconstruction
  ├─ E2VID          (baseline from the project description)
  ├─ E2VID++ / V2V  (newer method)
  └─ optional: HyperE2VID, ET-Net
  ▼
YOLO11 trained on the reconstructed intensity frames
  ▼
detection metrics vs this hybrid pipeline → report</div>

    <h3>Why it stays separate</h3>
    <ul>
      <li>Reconstruction runs on the <b>raw event stream</b>, while this
      pipeline consumes the FRED event <i>frames</i> — different inputs,
      different preprocessing.</li>
      <li>Each reconstruction method ships its own checkpoints and
      environment (E2VID, V2V, ET-Net); bundling them here would bloat the
      deliverable without adding runtime functionality.</li>
      <li>The outcome is a set of <b>trained detectors and comparison
      tables</b> for the final report, not an interactive feature.</li>
    </ul>

    <p class="muted">Both studies share the same FRED sequences and the same
    time-alignment convention, so the comparison in the report is
    apples-to-apples per frame.</p>
  </div></div>`;
}

/* ---------------- keyboard ---------------- */
document.addEventListener("keydown", (e) => {
  if (!state.show || e.target.tagName === "INPUT" || e.target.tagName === "SELECT")
    return;
  if (e.code === "Space") {
    e.preventDefault();
    if (state.togglePlay) state.togglePlay();
  } else if (e.code === "ArrowLeft") {
    state.playing = false;
    state.show(state.idx - (e.shiftKey ? 10 : 1));
  } else if (e.code === "ArrowRight") {
    state.playing = false;
    state.show(state.idx + (e.shiftKey ? 10 : 1));
  }
});

/* ---------------- boot ---------------- */
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => { state.tab = t.dataset.tab; render(); };
});
$("#run-btn").onclick = runPipeline;
setupUpload();
refreshSequences().then(() => {
  const ready = state.sequences.find((s) => s.results);
  if (ready) selectSequence(ready.name);
  else render();
});
