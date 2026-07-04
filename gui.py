import streamlit as st
import tempfile
from streamlit_image_comparison import image_comparison
import json
import cv2
import os
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from src.preprocessing.preprocess_zip import preprocess_zip

GT_COLOR   = (0,255,0)
KEPT_COLOR = (0,255,255)
REJ_COLOR  = (255,0,0)

# --- HELPER FUNCTIONS ---
def gt_norm_to_px(cx, cy, w, h, img_w=1280, img_h=720):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

def draw_boxes(frame, detections: list[dict], gt_boxes_norm: list[list[float]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if det["kept"]:
            cv2.putText(
                frame,
                f"DRONE {det['fusion_score']:.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEPT_COLOR, 1, cv2.LINE_AA,
            )
    for box in gt_boxes_norm:
        cx, cy, bw, bh = box
        x1, y1, x2, y2 = gt_norm_to_px(cx, cy, bw, bh)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)

def generate_video(jsonl_data, image_key, video_name, fps=30):
    """
    Creates a video from image paths stored in JSONL records.

    Args:
        jsonl_data (tuple | list): Sequence of dictionaries parsed from a JSONL file.
        image_key (str): Key containing the image path.
        video_name (str): Output video filename.
        fps (int | float): Frames per second.
    """
    # Extract image paths
    image_paths = [
        record[image_key]
        for record in jsonl_data
        if image_key in record and os.path.exists(record[image_key])
    ]

    if not image_paths:
        raise ValueError("No valid image paths found.")

    # Read first image to determine frame size
    frame = cv2.imread(image_paths[0])
    if frame is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = frame.shape[:2]

    # Create video writer
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        fps,
        (width, height),
    )

    # Add images to the video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        # Resize if necessary
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

PROCESSED_DIR = "processed"
DATASET_DIR = "dataset"

def dataset_frame_paths(dataset, stream, processed_dir=PROCESSED_DIR):
    """Sorted aligned frame paths for one processed dataset. stream is "rgb" or "event"."""
    images_root = os.path.join(processed_dir, dataset, "matched", stream, "images")
    if not os.path.isdir(images_root):
        return []
    paths = []
    for split in sorted(os.listdir(images_root)):
        split_dir = os.path.join(images_root, split)
        if not os.path.isdir(split_dir):
            continue
        paths.extend(
            os.path.join(split_dir, f)
            for f in sorted(os.listdir(split_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    return paths

# --- DATASET PLAYER (Preview tab) ---
# Frames are served to the browser by a tiny HTTP server on this port, because
# the player is a client-side component that loads frames by URL (Streamlit's
# own static serving cannot reach processed/ outside its ./static root).
FRAME_SERVER_PORT = 8765

class _QuietFrameHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # keep the terminal clean
        pass

@st.cache_resource
def start_frame_server(port=FRAME_SERVER_PORT):
    handler = partial(_QuietFrameHandler, directory=os.path.abspath(PROCESSED_DIR))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    except OSError:
        return None  # port already taken, presumably by a previous app instance
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

# The players share the controls row (< > frame-number play progress fps) and
# the playback JS; they differ only in the viewer markup and a per-frame hook.
PLAYER_CSS = """
<style>
  /* mid-gray text stays readable in both light and dark mode */
  body { margin:0; background:transparent; font-family:"Source Sans Pro",sans-serif; color:#8a8f98; }
  #controls { display:flex; align-items:center; gap:6px; width:__VW__px; margin-bottom:8px; }
  #controls button { background:#ff4b4b; color:#fff; border:none; border-radius:4px;
                     width:34px; height:30px; font-size:14px; cursor:pointer; }
  #controls button:hover { filter:brightness(1.15); }
  #controls input[type=number] { background:#262730; color:#fafafa; border:1px solid #555;
                                 border-radius:4px; height:26px; text-align:center; }
  #num { width:70px; }
  #fps { width:48px; }
  #progress { flex:1; accent-color:#ff4b4b; }
  #label { font-size:13px; white-space:nowrap; }
  .dim { font-size:12px; }
  #viewer { position:relative; width:__VW__px; height:__VH__px; user-select:none; }
  #viewer img { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  .tag { position:absolute; top:8px; padding:2px 8px; background:rgba(0,0,0,.55);
         font-size:12px; border-radius:3px; pointer-events:none; z-index:2; }
  /* wipe mode */
  #viewer.wipe { overflow:hidden; border-radius:4px; background:#000;
                 cursor:col-resize; touch-action:none; }
  #viewer.wipe #evt { clip-path:inset(0 0 0 50%); }
  #divider { position:absolute; top:0; left:50%; width:2px; height:100%;
             background:#fff; pointer-events:none; }
  #knob { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
          width:28px; height:28px; border-radius:50%; background:#fff; color:#333;
          display:flex; align-items:center; justify-content:center; font-size:14px; }
  /* side-by-side mode */
  #viewer.sbs { display:flex; gap:8px; }
  .pane { position:relative; flex:1; height:100%; overflow:hidden;
          border-radius:4px; background:#000; }
  .boxes { position:absolute; inset:0; pointer-events:none; }
  .box { position:absolute; box-sizing:border-box; }
  .gt { border:2px solid #21c354; }
  .det-kept { border:2px solid #ffd400; }
  .det-rej { border:2px solid #ff4b4b; }
  .det-label { position:absolute; bottom:100%; left:-2px; padding:0 4px;
               background:rgba(0,0,0,.6); color:#ffd400; font-size:11px; white-space:nowrap; }
</style>
"""

CONTROLS_HTML = """
<div id="controls">
  <button id="prev" title="Previous frame">&lt;</button>
  <button id="next" title="Next frame">&gt;</button>
  <input id="num" type="number" min="0" value="0">
  <button id="play" title="Play / pause">&#9654;</button>
  <input id="progress" type="range" min="0" value="0" step="1">
  <span id="label"></span>
  <input id="fps" type="number" min="1" max="60" value="30"><span class="dim">fps</span>
</div>
"""

PLAYER_JS = """
  const RGB = __RGB__;
  const EVT = __EVT__;
  const N = Math.min(RGB.length, EVT.length);
  // This runs in a srcdoc iframe where location.hostname is empty, so take
  // the host the app is actually served from via the parent page.
  const BASE = (() => {
    try {
      return window.parent.location.protocol + "//" + window.parent.location.hostname + ":__PORT__/";
    } catch (e) {
      const m = document.referrer.match(/^(https?:)\\/\\/([^:/]+)/);
      return m ? m[1] + "//" + m[2] + ":__PORT__/" : "http://localhost:__PORT__/";
    }
  })();

  const rgbImg = document.getElementById("rgb"), evtImg = document.getElementById("evt");
  const progress = document.getElementById("progress"), num = document.getElementById("num");
  const label = document.getElementById("label"), playBtn = document.getElementById("play");
  const fpsInput = document.getElementById("fps"), viewer = document.getElementById("viewer");
  progress.max = N - 1;
  num.max = N - 1;

  let idx = 0, timer = null;

  function preload(from) {
    for (let k = from; k < Math.min(from + 8, N); k++) {
      (new Image()).src = BASE + RGB[k];
      (new Image()).src = BASE + EVT[k];
    }
  }
  function setFrame(i) {
    idx = Math.max(0, Math.min(N - 1, i | 0));
    rgbImg.src = BASE + RGB[idx];
    evtImg.src = BASE + EVT[idx];
    progress.value = idx;
    num.value = idx;
    label.textContent = idx + " / " + (N - 1);
    onFrame(idx);  // mode-specific hook, defined below
    preload(idx + 1);
  }
  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.innerHTML = "&#9654;";
  }
  function start() {
    stop();
    const fps = Math.max(1, Math.min(60, +fpsInput.value || 30));
    timer = setInterval(() => { idx >= N - 1 ? stop() : setFrame(idx + 1); }, 1000 / fps);
    playBtn.innerHTML = "&#10074;&#10074;";
  }
  playBtn.onclick = () => {
    if (timer) { stop(); return; }
    if (idx >= N - 1) setFrame(0);  // replay from the beginning
    start();
  };
  document.getElementById("prev").onclick = () => setFrame(idx - 1);
  document.getElementById("next").onclick = () => setFrame(idx + 1);
  progress.addEventListener("input", () => setFrame(+progress.value));  // seek, also while playing
  num.addEventListener("change", () => setFrame(+num.value));
  fpsInput.addEventListener("change", () => { if (timer) start(); });
"""

WIPE_VIEWER_HTML = """
<div id="viewer" class="wipe">
  <img id="rgb"><img id="evt">
  <div id="divider"><div id="knob">&#8596;</div></div>
  <span class="tag" style="left:8px">RGB</span>
  <span class="tag" style="right:8px">Event</span>
</div>
"""

WIPE_JS = """
  function onFrame(i) {}

  // RGB <-> Event wipe: drag anywhere on the image, also while playing
  const divider = document.getElementById("divider");
  function setWipe(clientX) {
    const r = viewer.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, (clientX - r.left) / r.width * 100));
    evtImg.style.clipPath = "inset(0 0 0 " + pct + "%)";
    divider.style.left = pct + "%";
  }
  let dragging = false;
  viewer.addEventListener("pointerdown", e => { dragging = true; viewer.setPointerCapture(e.pointerId); setWipe(e.clientX); });
  viewer.addEventListener("pointermove", e => { if (dragging) setWipe(e.clientX); });
  viewer.addEventListener("pointerup", () => { dragging = false; });
"""

SBS_VIEWER_HTML = """
<div id="viewer" class="sbs">
  <div class="pane">
    <img id="evt"><div class="boxes" id="evtBoxes"></div>
    <span class="tag" style="left:8px">Event</span>
  </div>
  <div class="pane">
    <img id="rgb"><div class="boxes" id="rgbBoxes"></div>
    <span class="tag" style="left:8px">RGB</span>
  </div>
</div>
"""

SBS_JS = """
  const DETS = __DETS__;  // per frame: [x1, y1, x2, y2, kept, score] in image pixels
  const GTS = __GTS__;    // per frame: [cx, cy, w, h] normalized
  const IMG_W = 1280, IMG_H = 720;
  const rgbBoxes = document.getElementById("rgbBoxes"), evtBoxes = document.getElementById("evtBoxes");

  function boxHtml(cls, l, t, w, h, labelText) {
    return '<div class="box ' + cls + '" style="left:' + l + '%;top:' + t + '%;width:' + w + '%;height:' + h + '%">'
      + (labelText ? '<span class="det-label">' + labelText + '</span>' : '') + '</div>';
  }
  function onFrame(i) {
    let html = "";
    for (const d of DETS[i] || []) {
      const kept = d[4];
      html += boxHtml(
        kept ? "det-kept" : "det-rej",
        d[0] / IMG_W * 100, d[1] / IMG_H * 100,
        (d[2] - d[0]) / IMG_W * 100, (d[3] - d[1]) / IMG_H * 100,
        kept ? "DRONE " + d[5].toFixed(2) : ""
      );
    }
    for (const g of GTS[i] || []) {
      html += boxHtml("gt", (g[0] - g[2] / 2) * 100, (g[1] - g[3] / 2) * 100, g[2] * 100, g[3] * 100, "");
    }
    rgbBoxes.innerHTML = html;
    evtBoxes.innerHTML = html;
  }
"""

def _player_html(viewer_html, mode_js, substitutions):
    html = (
        PLAYER_CSS + CONTROLS_HTML + viewer_html
        + "<script>" + PLAYER_JS + mode_js + "\n  setFrame(0);\n</script>"
    )
    substitutions["__PORT__"] = str(FRAME_SERVER_PORT)
    for token, value in substitutions.items():
        html = html.replace(token, value)
    return html

def _rel_paths(paths):
    return json.dumps([os.path.relpath(p, PROCESSED_DIR) for p in paths])

def render_dataset_player(rgb_paths, event_paths, width=800):
    """Client-side player: play/pause, seekable progress bar, RGB<->event wipe."""
    height = round(width * 720 / 1280)
    html = _player_html(WIPE_VIEWER_HTML, WIPE_JS, {
        "__RGB__": _rel_paths(rgb_paths),
        "__EVT__": _rel_paths(event_paths),
        "__VW__": str(width),
        "__VH__": str(height),
    })
    st.iframe(html, width=width, height=height + 50)

def render_side_by_side_player(rgb_paths, event_paths, detections, gt_boxes, width=1300):
    """Client-side player showing event and RGB frames next to each other, with
    the model detections and ground-truth boxes drawn on both."""
    pane_width = (width - 8) // 2
    height = round(pane_width * 720 / 1280)
    html = _player_html(SBS_VIEWER_HTML, SBS_JS, {
        "__RGB__": _rel_paths(rgb_paths),
        "__EVT__": _rel_paths(event_paths),
        "__DETS__": json.dumps(detections),
        "__GTS__": json.dumps(gt_boxes),
        "__VW__": str(width),
        "__VH__": str(height),
    })
    st.iframe(html, width=width, height=height + 50)

def image_slider(img1_file, img2_file): # not really necessary now that we have image paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
            f1.write(img1_file.getvalue())
            path1 = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
        f2.write(img2_file.getvalue())
        path2 = f2.name
    # path1=save_uploaded_file(img1_file)
    # path2=save_uploaded_file(img2_file)
    image_comparison(
        img1=path1,
        img2=path2,
        label1="RGB Frame",
        label2="Event Frame",
        width=800,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
    )

st.set_page_config(page_title="Applied Machine Intelligence Project - Group 7", layout="wide")

start_frame_server()  # serves processed/ frames to the Preview player

st.title("Hybrid Vision - Group 7")
st.write("Web interface to show project results")
st.write("<- open sidebar to start")

# --- SIDEBAR ---
st.sidebar.title("Hybrid Vision - Group 7")
st.sidebar.header("Step 1 - Data Import")
uploaded_zip = st.sidebar.file_uploader("Upload a sequence zip", type="zip")
if uploaded_zip is not None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_choice = uploaded_zip.name
    upload_dest = os.path.join(DATASET_DIR, zip_choice)
    if not os.path.exists(upload_dest):
        with open(upload_dest, "wb") as f:
            f.write(uploaded_zip.getbuffer())
    st.sidebar.caption(f"Ready: {zip_choice}")
else:
    zip_choice = None

st.sidebar.header("Step 2 - Preprocessing")
# The zip chosen in Step 1 is only extracted and aligned once Process is pressed.
if st.sidebar.button("Process"):
    if zip_choice is None:
        st.sidebar.warning("Upload a zip file in Step 1 first.")
    else:
        with st.spinner(f"Preprocessing {zip_choice} (extract + align RGB/event frames)..."):
            try:
                dataset_name, _ = preprocess_zip(
                    os.path.join(DATASET_DIR, zip_choice),
                    dataset_root=DATASET_DIR,
                    processed_root=PROCESSED_DIR,
                    progress=lambda msg: None,
                )
            except Exception as e:
                st.sidebar.error(f"Preprocessing failed: {e}")
            else:
                st.session_state["dataset"] = dataset_name
selected_dataset = st.session_state.get("dataset")
if selected_dataset is not None:
    st.sidebar.caption(f"Loaded: {selected_dataset}")

st.sidebar.header("Step 3 - Visualize!")
st.sidebar.info("Click a visualization tab at the top.")

# --- PROCESS DATASET ---
# if st.button("Run"):
#     # here run program, ignore for now
# for now: load input dataset
input_data=[]


# --- LOAD PROCESSED DATA ---
@st.cache_data
def load_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f]

@st.cache_data
def blind_test_player_data(jsonl_path):
    """Frame paths and boxes from the blind-test JSONL, compacted for the
    side-by-side player (detections as [x1, y1, x2, y2, kept, score])."""
    rgb, evt, dets, gts = [], [], [], []
    for r in load_jsonl(jsonl_path):
        rgb.append(r["rgb_image"])
        evt.append(r["event_image"])
        dets.append([
            [round(v) for v in d["bbox_xyxy"]] + [1 if d["kept"] else 0, round(d["fusion_score"], 2)]
            for d in r.get("detections", [])
        ])
        gts.append([[round(float(v), 4) for v in b] for b in r.get("gt_boxes_norm", [])])
    return rgb, evt, dets, gts

@st.cache_data
def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        return json.load(f)

BLIND_TEST_JSONL = "outputs/web/fusion_detections_blind_test_v4.jsonl"
BLIND_TEST_MANIFEST = "outputs/web/fusion_manifest_blind_test_v4.json"

# --- TOP MENU ---
tabs = st.tabs([
    "Preview", "Side-by-Side", "As a video", "confusion matrices", "Example"
])

# --- PREVIEW TAB ---
with tabs[0]:
    st.subheader("Preview a dataset")
    if selected_dataset is None:
        st.info("Select a zip in the sidebar (Step 1) and press Process (Step 2) to load it.")
    else:
        rgb_frames = dataset_frame_paths(selected_dataset, "rgb")
        event_frames = dataset_frame_paths(selected_dataset, "event")
        n_pairs = min(len(rgb_frames), len(event_frames))
        if n_pairs == 0:
            st.warning(f"No aligned frames found for {selected_dataset}.")
        else:
            st.write(
                f"**{selected_dataset}** - {n_pairs} aligned frame pairs. "
                "Press ▶ to play, drag the bar to seek, "
                "drag on the image to wipe between RGB and Event."
            )
            render_dataset_player(rgb_frames[:n_pairs], event_frames[:n_pairs])


with tabs[1]:
    st.subheader("Model detections on the blind test, event and RGB side by side")
    sbs_rgb, sbs_evt, sbs_dets, sbs_gts = blind_test_player_data(BLIND_TEST_JSONL)
    st.caption(f"{len(sbs_rgb)} frames (seq40 / seq43 / seq46 / seq49)")
    render_side_by_side_player(sbs_rgb, sbs_evt, sbs_dets, sbs_gts)
    st.write(
        ":green[Green] = ground truth, "
        ":orange[Yellow] = kept detection (with fusion score), "
        ":red[Red] = rejected detection."
    )

MODEL_LABELS = {
    "event_yolo_conf0.25": "Event YOLO (conf 0.25)",
    "event_yolo_conf0.50": "Event YOLO (conf 0.50)",
    "fusion_v4": "Fusion v4",
}

# Classic blue confusion-matrix palette; the cells carry their own background
# and text colors, so the cards read well in both light and dark mode.
CM_DARK_BLUE = "#38678f"
CM_LIGHT_BLUE = "#bcd8ee"

def confusion_card(title, m):
    """HTML card with a styled confusion matrix and P/R/F1 for one model.
    Built from divs + CSS grid (not <table>) so Streamlit's markdown table
    styles don't add borders, and capped in width so wide screens don't
    stretch the cells."""
    axis = "color:#8a8f98;font-size:12px;"
    cell = ("aspect-ratio:1/1;display:flex;flex-direction:column;"
            "justify-content:center;align-items:center;text-align:center;border-radius:8px;")
    dark = cell + f"background:{CM_DARK_BLUE};color:#fff;"
    light = cell + f"background:{CM_LIGHT_BLUE};color:#1f4e6e;"
    na = cell + "background:rgba(138,143,152,.15);color:#8a8f98;"
    num = "font-size:24px;font-weight:700;line-height:1.2;"
    sub = "font-size:11px;letter-spacing:.5px;opacity:.85;"
    return f"""
<div style="max-width:420px;margin:0 auto;border:1px solid rgba(138,143,152,.35);border-radius:10px;padding:16px 16px 12px">
  <div style="text-align:center;font-weight:600;font-size:15px;margin-bottom:8px">{title}</div>
  <div style="text-align:center;{axis}margin-bottom:4px">Predicted</div>
  <div style="display:flex;gap:6px">
    <div style="writing-mode:vertical-rl;transform:rotate(180deg);text-align:center;{axis}">Actual</div>
    <div style="flex:1;display:grid;grid-template-columns:auto 1fr 1fr;gap:4px;align-items:center">
      <div></div>
      <div style="{axis}text-align:center">Drone</div>
      <div style="{axis}text-align:center">Background</div>
      <div style="{axis}text-align:right">Drone</div>
      <div style="{dark}"><div style="{num}">{m['TP']:,}</div><div style="{sub}">TP</div></div>
      <div style="{light}"><div style="{num}">{m['FN']:,}</div><div style="{sub}">FN</div></div>
      <div style="{axis}text-align:right">Background</div>
      <div style="{light}"><div style="{num}">{m['FP']:,}</div><div style="{sub}">FP</div></div>
      <div style="{na}"><div style="{num}">&mdash;</div><div style="{sub}">TN n/a</div></div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-around;margin-top:12px;text-align:center">
    <div><div style="font-size:20px;font-weight:600">{m['precision'] * 100:.1f}%</div>
         <div style="{axis}">Precision</div></div>
    <div><div style="font-size:20px;font-weight:600">{m['recall'] * 100:.1f}%</div>
         <div style="{axis}">Recall</div></div>
    <div><div style="font-size:20px;font-weight:600">{m['f1'] * 100:.1f}%</div>
         <div style="{axis}">F1</div></div>
  </div>
</div>
"""

with tabs[3]:
    st.subheader("Confusion matrices - blind test v4")
    manifest = load_manifest(BLIND_TEST_MANIFEST)
    st.caption(
        f"{manifest['n_frames_total']} frames, {manifest['n_detections_total']} detections "
        f"across {', '.join(manifest['sequences'])} - "
        f"fusion threshold {manifest['fusion_threshold']:.3f}. "
        "TN is undefined for open-set detection, hence n/a."
    )
    split = st.radio(
        "Split", ["all", "train", "val", "test"], horizontal=True, key="cm_split"
    )
    split_metrics = manifest["metrics"][split]
    columns = st.columns(len(split_metrics))
    for col, (model_key, m) in zip(columns, split_metrics.items()):
        with col:
            st.markdown(
                confusion_card(MODEL_LABELS.get(model_key, model_key), m),
                unsafe_allow_html=True,
            )
    with st.expander("Original figure (PNG)"):
        st.image("outputs/confusion_matrices_blind_test_v4.png")

with tabs[4]:
    st.write('decide what data to keep in the repository as an example')
