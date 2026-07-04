#!/usr/bin/env python3
"""
Preprocess a single FRED sequence delivered as a zip file (GUI "Process" step).

The zip is expected to contain one sequence at its root, as shipped in dataset/:
  coordinates.txt / interpolated_coordinates.txt / tracks.txt
  PADDED_RGB/Video_*.jpg
  Event/Frames/Video_*_frame_*.png

The zip is extracted to <dataset-root>/<seq>/ (seq = zip filename stem) and then
run through the same time-alignment preprocessing as preprocess_fred_dataset.py
(process_sequence): labels, padded RGB frames, and event frames are matched by
relative time and materialized under

  <processed-root>/preprocessed_fred_<seq>/
    matched/{rgb,event}/images/<split>/seq<seq>_<time_us>.<ext>
    labels/{rgb,event}/<split>/seq<seq>_<time_us>.txt
    paired/manifest_<split>.csv
    report/preprocessing_report.json

Example:
  python src/preprocessing/preprocess_zip.py dataset/40.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# preprocess_fred_dataset.py lives next to this file; make it importable both
# when this module is imported from the GUI and when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_fred_dataset import (  # noqa: E402
    process_sequence,
    write_manifest,
    write_sequence_report,
)

# Same defaults as the original preprocessing run (see processed/preprocessing_report.json).
ANNOTATION_CANDIDATES = ["interpolated_coordinates.txt", "coordinates.txt"]
RGB_DIR_NAME = "PADDED_RGB"
FRAME_PERIOD = 0.033333
MAX_DELTA_S = 0.04

_JUNK_PARTS = {"__MACOSX", ".DS_Store"}


def _is_junk(name: str) -> bool:
    return any(part in _JUNK_PARTS for part in Path(name).parts)


def extract_zip(zip_path: Path, dataset_root: Path, progress=print) -> Path:
    """Extract zip_path to <dataset_root>/<stem>/ and return that directory.

    Skips extraction when the sequence directory already holds the RGB and
    event frame folders. Rejects unsafe member paths (absolute or with '..').
    """
    seq_dir = dataset_root / zip_path.stem
    if (seq_dir / RGB_DIR_NAME).is_dir() and (seq_dir / "Event" / "Frames").is_dir():
        progress(f"{seq_dir} already extracted, skipping unzip.")
        return seq_dir

    progress(f"Extracting {zip_path.name} to {seq_dir}/ ...")
    seq_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if member.is_dir() or _is_junk(name):
                continue
            dest = (seq_dir / name).resolve()
            if not dest.is_relative_to(seq_dir.resolve()):
                raise ValueError(f"Unsafe path in zip: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return seq_dir


def is_already_processed(output_root: Path) -> bool:
    """True if aligned frames already exist under <output_root>/matched/."""
    images_root = output_root / "matched" / "rgb" / "images"
    if not images_root.is_dir():
        return False
    return any(
        f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for split_dir in images_root.iterdir() if split_dir.is_dir()
        for f in split_dir.iterdir()
    )


def preprocess_zip(
    zip_path: str | Path,
    dataset_root: str | Path = "dataset",
    processed_root: str | Path = "processed",
    split: str = "test",
    overwrite: bool = False,
    progress=print,
) -> tuple[str, dict]:
    """Extract a sequence zip and align its RGB/event frames with the labels.

    Returns (dataset_name, summary) where dataset_name is the directory name
    under processed_root (as listed by the GUI) and summary holds the counters
    from process_sequence.
    """
    zip_path = Path(zip_path)
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    seq = zip_path.stem
    dataset_name = f"preprocessed_fred_{seq}"
    output_root = processed_root / dataset_name

    if not overwrite and is_already_processed(output_root):
        progress(f"{output_root} already contains aligned frames, reusing it.")
        return dataset_name, {"reused": True}

    seq_dir = extract_zip(zip_path, dataset_root, progress=progress)

    progress(f"Aligning RGB and event frames for sequence {seq} ...")
    result, rows = process_sequence(
        seq=seq,
        split=split,
        seq_dir=seq_dir,
        output_root=output_root,
        annotation_candidates=ANNOTATION_CANDIDATES,
        rgb_dir_name=RGB_DIR_NAME,
        frame_period=FRAME_PERIOD,
        max_delta_s=MAX_DELTA_S,
        materialize="hardlink",
        overwrite=overwrite,
        include_removed_in_rgb_timeline=False,
        negative_ratio=0.0,
        negative_exclusion_s=0.066,
        negative_seed=42,
    )
    if result.missing_dirs:
        raise ValueError(
            f"Sequence {seq} is missing required inputs: {', '.join(result.missing_dirs)}"
        )

    write_manifest(output_root, split, rows)
    write_sequence_report(
        output_root,
        {
            "source_zip": str(zip_path),
            "sequence": seq,
            "split": split,
            "output_root": str(output_root),
            "rgb_dir": RGB_DIR_NAME,
            "annotation_files": ANNOTATION_CANDIDATES,
            "frame_period": FRAME_PERIOD,
            "max_delta": MAX_DELTA_S,
            "result": result.__dict__,
        },
    )
    progress(f"Done: {result.samples_written} aligned frame pairs written to {output_root}.")
    return dataset_name, result.__dict__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("zip_path", type=Path, help="Sequence zip file, e.g. dataset/40.zip")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--processed-root", type=Path, default=Path("processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    preprocess_zip(
        args.zip_path,
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
        split=args.split,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
