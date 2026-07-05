#!/usr/bin/env python3
"""Convert a preprocessed FRED sequence to the Monash/TimoStoff HDF5 format
expected by ET-Net (and event_cnn_minimal-style loaders).

- events/xs, ys (uint16), ts (float64 seconds), ps (uint8 0/1)
- images/image{:09d}: grayscale matched RGB frames with attrs
  'timestamp' (rgb_time_s) and 'event_idx' (first event index >= timestamp),
  so voxel_method='between_frames' reconstructs one frame per RGB timestamp.
"""

import argparse
import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="31")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--downscale", type=int, default=1,
                    help="integer factor to downscale coords/frames (e.g. 2 "
                         "-> 640x360, needed for ET-Net's 8000-token "
                         "positional table and attention memory)")
    args = ap.parse_args()

    pre = (args.repo_root / "data" / "preprocessed_all_val"
           / f"preprocessed_fred_{args.seq}")
    suffix = f"_ds{args.downscale}" if args.downscale > 1 else ""
    out = args.out or (args.repo_root / "artifacts" / "etnet"
                       / f"fred_seq{args.seq}{suffix}.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("loading events...")
    ev = pd.read_csv(pre / "eventRaw" / "events.txt", header=None,
                     names=["x", "y", "p", "t"],
                     dtype={"x": np.uint16, "y": np.uint16,
                            "p": np.uint8, "t": np.int64}).values
    ts = ev[:, 3].astype(np.float64) / 1e6
    print(f"{len(ev)} events, t {ts[0]:.3f}..{ts[-1]:.3f}s")

    rows = [r for r in csv.DictReader(open(pre / "paired" / "manifest_val.csv"))
            if r.get("split") == "val"]
    rows.sort(key=lambda r: float(r["event_time_s"]))

    d_ = args.downscale
    H, W = 720 // d_, 1280 // d_
    with h5py.File(out, "w") as f:
        f.attrs["sensor_resolution"] = np.array([H, W], dtype=np.int64)
        f.attrs["num_events"] = len(ev)
        f.attrs["num_imgs"] = len(rows)
        f.attrs["source"] = "fred"
        g = f.create_group("events")
        g.create_dataset("xs", data=(ev[:, 0] // d_).astype(np.uint16))
        g.create_dataset("ys", data=(ev[:, 1] // d_).astype(np.uint16))
        g.create_dataset("ts", data=ts)
        g.create_dataset("ps", data=ev[:, 2].astype(np.uint8))
        gi = f.create_group("images")
        for i, r in enumerate(rows):
            im = cv2.imread(str(pre / r["rgb_image"]), cv2.IMREAD_GRAYSCALE)
            if d_ > 1:
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            t = float(r["rgb_time_s"])
            d = gi.create_dataset(f"image{i:09d}", data=im,
                                  compression="gzip", compression_opts=1)
            d.attrs["timestamp"] = t
            d.attrs["event_idx"] = int(np.searchsorted(ts, t))
            d.attrs["size"] = im.shape
            if i % 500 == 0:
                print(f"  image {i}/{len(rows)}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
