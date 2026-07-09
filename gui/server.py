"""Web interface for the hybrid drone-detection pipeline.

FastAPI backend serving a single-page frontend (gui/static/):

  GET  /                      the app
  GET  /api/sequences         dataset sequences + pipeline state
  POST /api/upload            FRED sequence zip -> dataset/<stem>/
  POST /api/run/{seq}         run the pipeline (background thread)
  GET  /api/status/{seq}      run state + progress log tail
  GET  /api/frames/{seq}      per-frame detections + GT for the players
  GET  /api/tracks/{seq}      Kalman tracking overlay data (cached JSON)
  GET  /api/metrics/{seq}     fusion manifest + merge stats
  GET  /img/{seq}/{modality}/{stem}   one aligned frame image

Start with:  python run_gui.py   (default http://localhost:8501)
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_vision import PipelineConfig, run_sequence  # noqa: E402
from hybrid_vision.common import read_jsonl, yolo_to_xyxy  # noqa: E402
from tracks_api import build_tracks  # noqa: E402

DATASET_DIR = REPO_ROOT / "dataset"
PROCESSED_DIR = REPO_ROOT / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SPLIT = "test"

app = FastAPI(title="Hybrid Vision — Group 7")

# One pipeline run at a time; progress kept for polling.
_runs: dict[str, dict] = {}
_run_lock = threading.Lock()


def _seq_state(seq: str) -> dict:
    aligned = PROCESSED_DIR / seq / "event_yolo" / "images" / SPLIT
    manifest = OUTPUTS_DIR / seq / "web" / f"fusion_manifest_{seq}.json"
    run = _runs.get(seq, {})
    return {
        "name": seq,
        "aligned": aligned.is_dir() and any(aligned.iterdir()),
        "results": manifest.is_file(),
        "running": run.get("state") == "running",
        "error": run.get("error"),
    }


@app.get("/api/sequences")
def sequences() -> list[dict]:
    seqs = []
    if DATASET_DIR.is_dir():
        for d in sorted(DATASET_DIR.iterdir(),
                        key=lambda p: (len(p.name), p.name)):
            if d.is_dir() and (d / "PADDED_RGB").is_dir():
                seqs.append(_seq_state(d.name))
    return seqs


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Upload a FRED sequence zip (e.g. 40.zip)")
    seq = Path(file.filename).stem
    seq_dir = DATASET_DIR / seq
    if (seq_dir / "PADDED_RGB").is_dir():
        return {"sequence": seq, "note": "already extracted, reusing"}

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / file.filename
    with zip_path.open("wb") as fh:
        while chunk := await file.read(64 * 1024 * 1024):
            fh.write(chunk)

    seq_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if member.is_dir() or "__MACOSX" in member.filename:
                continue
            dest = (seq_dir / member.filename).resolve()
            if not dest.is_relative_to(seq_dir.resolve()):
                raise HTTPException(400, f"Unsafe path in zip: {member.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    # zips may wrap everything in one top-level dir — flatten it
    if not (seq_dir / "PADDED_RGB").is_dir():
        inner = [d for d in seq_dir.iterdir() if d.is_dir()]
        if len(inner) == 1 and (inner[0] / "PADDED_RGB").is_dir():
            for item in inner[0].iterdir():
                shutil.move(str(item), str(seq_dir / item.name))
            inner[0].rmdir()
    if not (seq_dir / "PADDED_RGB").is_dir():
        raise HTTPException(400, "Zip does not contain a FRED sequence "
                                 "(PADDED_RGB/ not found)")
    return {"sequence": seq}


# selectable RGB detectors; "v5" is the final calibrated pipeline default
RGB_MODELS = {
    "v5": {"path": "rgb_yolo11m.pt", "imgsz": 640,
           "label": "v5 (default, imgsz 640)"},
    "v6": {"path": "rgb_yolo11m_v6.pt", "imgsz": 1280,
           "label": "v6 retrain (imgsz 1280)"},
}
DEFAULT_SETTINGS = {"rgb_model": "v5", "rgb_min_conf": 0.60}


def _apply_settings(cfg: PipelineConfig, settings: dict) -> None:
    spec = RGB_MODELS[settings["rgb_model"]]
    cfg.rgb_model = REPO_ROOT / "weights" / spec["path"]
    cfg.rgb_imgsz = spec["imgsz"]
    cfg.rgb_min_conf = settings["rgb_min_conf"]


def _invalidate_stale(seq: str, settings: dict) -> None:
    """Clear cached artifacts that the new settings invalidate.

    The event lane never changes. A different RGB model invalidates the RGB
    proposals; any settings change invalidates merge and everything after.
    """
    out = OUTPUTS_DIR / seq
    settings_file = out / "run_settings.json"
    prev = (json.loads(settings_file.read_text(encoding="utf-8"))
            if settings_file.is_file() else None)
    if prev == settings:
        return
    if prev is None or prev.get("rgb_model") != settings["rgb_model"]:
        (out / "proposals_rgb.jsonl").unlink(missing_ok=True)
    for name in ("proposals_merged.jsonl", "merge_stats.json",
                 "scored_rgb.jsonl", "scored_event.jsonl", "tracks.json"):
        (out / name).unlink(missing_ok=True)
    shutil.rmtree(out / "web", ignore_errors=True)


def _run_pipeline(seq: str, overwrite: bool, settings: dict) -> None:
    run = _runs[seq]

    def progress(msg: str) -> None:
        run["log"].append(str(msg))
        del run["log"][:-400]

    try:
        cfg = PipelineConfig()
        _apply_settings(cfg, settings)
        progress(f"settings: RGB {settings['rgb_model']}, "
                 f"gate {settings['rgb_min_conf']:.2f}")
        _invalidate_stale(seq, settings)
        run_sequence(DATASET_DIR / seq, cfg,
                     processed_root=PROCESSED_DIR, outputs_root=OUTPUTS_DIR,
                     overwrite=overwrite, progress=progress)
        (OUTPUTS_DIR / seq / "tracks.json").unlink(missing_ok=True)
        (OUTPUTS_DIR / seq / "run_settings.json").write_text(
            json.dumps(settings), encoding="utf-8")
        run["state"] = "done"
    except Exception as exc:  # surfaced to the UI, not swallowed
        run["state"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc}"


@app.get("/api/settings")
def settings_options() -> dict:
    return {"rgb_models": {k: v["label"] for k, v in RGB_MODELS.items()},
            "defaults": DEFAULT_SETTINGS}


@app.post("/api/run/{seq}")
def run(seq: str, overwrite: bool = False, rgb_model: str = "v5",
        rgb_min_conf: float = 0.60) -> dict:
    if not (DATASET_DIR / seq / "PADDED_RGB").is_dir():
        raise HTTPException(404, f"No sequence '{seq}' in dataset/")
    if rgb_model not in RGB_MODELS:
        raise HTTPException(400, f"rgb_model must be one of {list(RGB_MODELS)}")
    if not 0.05 <= rgb_min_conf <= 1.0:
        raise HTTPException(400, "rgb_min_conf must be in [0.05, 1.0]")
    settings = {"rgb_model": rgb_model, "rgb_min_conf": round(rgb_min_conf, 2)}
    with _run_lock:
        if any(r.get("state") == "running" for r in _runs.values()):
            raise HTTPException(409, "A pipeline run is already in progress")
        _runs[seq] = {"state": "running", "log": [], "error": None}
        threading.Thread(target=_run_pipeline, args=(seq, overwrite, settings),
                         daemon=True).start()
    return {"sequence": seq, "state": "running", "settings": settings}


@app.get("/api/status/{seq}")
def status(seq: str) -> dict:
    run = _runs.get(seq, {})
    return {**_seq_state(seq),
            "state": run.get("state", "idle"),
            "log": run.get("log", [])[-15:]}


def _manifest_path(seq: str) -> Path:
    return OUTPUTS_DIR / seq / "web" / f"fusion_manifest_{seq}.json"


def _require_results(seq: str) -> None:
    if not _manifest_path(seq).is_file():
        raise HTTPException(404, f"No pipeline results for '{seq}' yet")


@app.get("/api/frames/{seq}")
def frames(seq: str) -> JSONResponse:
    _require_results(seq)
    out = []
    for rec in read_jsonl(OUTPUTS_DIR / seq / "web"
                          / f"fusion_detections_{seq}.jsonl"):
        out.append({
            "stem": rec["stem"],
            "i": rec["frame_index"],
            "t": rec["timestamp_sec"],
            "gt": [yolo_to_xyxy(*b, 1280, 720) for b in rec["gt_boxes_norm"]],
            "dets": [
                {"box": d["bbox_xyxy"], "det": d["detector_score"],
                 "fus": d["fusion_score"], "kept": d["kept"],
                 "src": d.get("source", "event"),
                 "cls": d.get("class", "drone")}
                for d in rec["detections"]
            ],
        })
    return JSONResponse({"frames": out, "fps": 30})


@app.get("/api/tracks/{seq}")
def tracks(seq: str) -> JSONResponse:
    _require_results(seq)
    cache = OUTPUTS_DIR / seq / "tracks.json"
    if not cache.is_file():
        data = build_tracks(OUTPUTS_DIR / seq / "web"
                            / f"fusion_detections_{seq}.jsonl")
        cache.write_text(json.dumps(data), encoding="utf-8")
    return JSONResponse(json.loads(cache.read_text(encoding="utf-8")))


@app.get("/api/metrics/{seq}")
def metrics(seq: str) -> dict:
    _require_results(seq)
    manifest = json.loads(_manifest_path(seq).read_text(encoding="utf-8"))
    stats_path = OUTPUTS_DIR / seq / "merge_stats.json"
    stats = (json.loads(stats_path.read_text(encoding="utf-8"))
             if stats_path.is_file() else None)
    return {"manifest": manifest, "merge_stats": stats}


@app.get("/img/{seq}/{modality}/{stem}")
def image(seq: str, modality: str, stem: str) -> FileResponse:
    if modality not in ("rgb", "event"):
        raise HTTPException(404, "modality must be rgb or event")
    ext = ".jpg" if modality == "rgb" else ".png"
    path = (PROCESSED_DIR / seq / f"{modality}_yolo" / "images" / SPLIT
            / f"{stem}{ext}")
    if not path.is_file():
        raise HTTPException(404, f"frame not found: {path.name}")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent / "static",
                           html=True), name="static")
