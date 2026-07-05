# Hybrid reconstruction — Stage A: RGB-anchored event-gated compositing

Prototype for FRED hybrid video reconstruction. Idea: RGB and events are
spatially complementary in FRED — the background is static (RGB is sharp
there, events are silent) and the drone moves (events/e2vid are informative
there, RGB is often blurred or too dark). So instead of symmetric fusion:

1. **Base layer** = RGB frame (background is RGB by construction — no
   hallucinated e2vid background).
2. **Activity mask** from the ~15 ms of events nearest the RGB timestamp:
   count map -> Gaussian density -> relative threshold -> component filter
   -> dilate. Components must pass an *absolute* floor (mass >= 120 events,
   peak density >= 0.12 ev/px, area <= 8000 px): sensor noise and wind-blown
   vegetation sit at mass < ~100 / peak < ~0.06, while the drone (even
   hovering) shows mass > ~180 / peak > ~0.25 (measured on seq 31). When the
   drone is silent the mask goes empty and the output is pure RGB.
3. **Injection**: nearest e2vid frame (120 fps) is tone-matched to RGB
   luminance on a ring around the mask, then fused inside the feathered
   mask by **detail-max**: per pixel, keep the local-contrast detail of
   whichever modality is stronger, on top of the RGB low-frequency base.
   RGB wins where it is sharp (daylight, close drone); e2vid fills in
   where RGB is flat (dark / far / motion-blurred). Chroma stays RGB.

Everything targets the RGB timestamp (`rgb_time_s`), not the label time —
the drone moves >10 px/ms, so the ~5 ms label offset would ghost the
injected detail next to the RGB drone.

## Data (symlinks)

- `data_seq18/` -> preprocessed FRED seq 18 (manifest, matched RGB/event
  frames, per-frame 33 ms event windows, YOLO labels, raw events).
- `e2vid_seq18/` -> e2vid 120 fps reconstruction. **Frame filenames are
  event indices, not timestamps** (last event of each 1/120 s window).
  `out/event_ts_us.npy` + `out/e2vid_names.npy` / `out/e2vid_frame_t.npy`
  map them to real seconds (built by the first run of the mapping snippet;
  see `composite.py::load_index`).

## Run

    python composite.py --seq 31 --t0 0 --t1 999 \
        --video out/composite_grid_seq31.mp4 --dump-frames 12

(symlink `data_seq<N>` / `e2vid_seq<N>` first; the e2vid time mapping is
built automatically on first run)

Outputs a 2x2 grid video (RGB | event frame / e2vid | composite+label box)
plus a clean 1280x720 composite-only `*_clean.mp4`.

## Known limitations / Stage B ideas

- Residual temporal mismatch (<=4 ms to nearest e2vid frame): measured to
  be a non-issue. A velocity-extrapolation compensation (`--motion-comp`,
  `estimate_shift`) was implemented and A/B tested on seq 18: 90% of frames
  need <3 px of shift, and in the 11 fast frames (5-60 px predicted) the
  shift *reduced* the e2vid detail energy captured by the mask (median
  ratio 0.86; scaled and even reversed variants also failed to improve).
  Reason: the mask slice and the e2vid reconstruction are both anchored to
  nearly the same event-window end time, so they are already mutually
  aligned (median e2vid-detail-to-mask-centroid distance: 5 px); shifting
  one against the other only breaks that. Keep the flag off. The remaining
  smear on fast passes is e2vid's ~8 ms intra-window integration, which
  frame-shifting cannot fix (needs reconstruction at RGB timestamps, or
  learned fusion).
- Tone matching is a global gain/offset per frame; a small U-Net taking
  [RGB, event voxel, e2vid] and predicting a residual on RGB is the
  planned Stage B (train on bright FRED sequences with synthetic
  degradation of the RGB input).
- Label boxes in the manifest lead the visible drone by ~5 ms of motion
  (annotation at event time) — relevant when training a detector on
  composites; consider re-timing labels to `rgb_time_s`.
