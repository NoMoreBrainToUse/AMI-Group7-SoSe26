# Example pipeline result — sequence 31

One FRED sequence (31: courtyard at dusk — dark foreground, bright sky,
far passes and hovering) traced through every pipeline stage. All videos are
1280x720 on the manifest ~30 fps grid, so playback is real time and frames
correspond 1:1 across videos. Rebuild with:

    external/V2V/.venv/bin/python scripts/presentation/build_example_result.py --seq 31

| File | Stage |
|---|---|
| `preprocessed_rgb_seq31.mp4` | Stage 2 — time-aligned padded RGB frames |
| `preprocessed_event_seq31.mp4` | Stage 2 — matched event frames (FRED PNGs) |
| `reconstructed_e2vid_seq31.mp4` | Stage 3 — E2VID (rpg_e2vid, 120 fps output sampled at the manifest timestamps) |
| `reconstructed_e2vidpp_seq31.mp4` | Stage 3 — E2VID++ (V2V/EVBIRD, one frame per manifest sample) |
| `hybrid_e2vid_composite_seq31.mp4` | Hybrid reconstruction — RGB background + event-gated E2VID detail injection (see `../hybrid/README.md`) |
| `yolo11m_on_e2vidpp_seq31.mp4` | Stage 4 — YOLO11m trained on E2VID++ reconstructions, run on the E2VID++ video (cyan = predictions + confidence, green = ground truth) |
| `yolo11m_on_e2vid_seq31.mp4` | Stage 4 — YOLO11m trained on E2VID reconstructions, run on the E2VID video (same annotation scheme) |

Each detector runs on its own reconstruction type (in-domain), matching the
quantitative comparison in the main README results table. Note seq 31 is a
*validation* sequence for both models. A YOLO model trained directly on the
hybrid composites is the natural follow-up.
