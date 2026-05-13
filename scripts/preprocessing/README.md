# FRED Preprocessing Guide

This document explains the preprocessing workflow implemented in
`scripts/preprocessing/prepare_fred_yolo.py`: the expected inputs and outputs,
how RGB and Event frames are aligned, how labels are generated, how the paired
manifest is used by a fusion model, and why the current pipeline avoids default
denoising.

## 1. Preprocessing Goal

The project is expected to train and compare three detection settings:

- **RGB YOLO**: drone detection using RGB images only.
- **Event YOLO**: drone detection using Event frames only.
- **Fusion model**: drone detection using synchronized RGB and Event inputs.

The preprocessing stage therefore creates three outputs:

```text
processed/fred10/
  rgb_yolo/       # YOLO dataset for RGB-only detection
  event_yolo/     # YOLO dataset for Event-only detection
  paired/         # synchronized RGB/Event sample manifests for fusion
```

The most important preprocessing tasks are:

```text
timestamp alignment + unified bounding boxes + sequence-level train/val/test split
```

## 2. Recommended 10-Sequence Subset

If the full FRED dataset is too large and only about 10 sequence packages can be
downloaded, use a small subset that still covers training, difficult validation,
and final testing.

Recommended downloads:

```text
train/0.zip
train/1.zip
train/11.zip
train/101.zip
train/102.zip
train/103.zip

train/10.zip
test/21.zip

test/34.zip
test/110.zip
```

Recommended split:

```text
train:
  0, 1, 11, 101, 102, 103

val:
  10, 21

test:
  34, 110
```

Use `train` for model fitting, `val` for tuning and difficult-case inspection,
and `test` for final reporting and the web demo.

Do not randomly split frames from the same sequence into train/val/test. Adjacent
frames in a video are highly similar, so frame-level random splitting would leak
near-duplicate samples into the evaluation set and inflate results.

## 3. Expected Raw Input Structure

The default dataset root is:

```text
dataset/
```

The script supports either a flat layout:

```text
dataset/
  0/
  1/
  10/
  21/
  ...
```

or a split-style layout:

```text
dataset/
  train/
    0/
    1/
    10/
    ...
  test/
    21/
    34/
    110/
    ...
```

Each sequence should contain at least:

```text
PADDED_RGB/
Event/Frames/
interpolated_coordinates.txt or coordinates.txt
```

A typical FRED sequence contains:

```text
sequence/
  RGB/
  PADDED_RGB/
  Event/
    Frames/
    events.raw
  Removed_frames/
  coordinates.txt
  interpolated_coordinates.txt
  tracks.txt
```

## 4. Why `PADDED_RGB` Is Used

FRED provides both `RGB/` and `PADDED_RGB/`.

This preprocessing pipeline defaults to:

```text
PADDED_RGB/ + Event/Frames/ + coordinates.txt or interpolated_coordinates.txt
```

`PADDED_RGB` is preferred because it shares the same coordinate space as the
Event frames. FRED's documentation explains that RGB images are padded so RGB and
Event data can be represented in a common image coordinate system. The
`coordinates.txt` file contains extended boxes in that common coordinate system.

If raw `RGB/` is used instead, padding differences must be handled separately,
usually with `coordinates_rgb.txt` or an explicit coordinate transformation. The
current script does not use this route by default.

## 5. How to Run

Windows PowerShell:

```powershell
python scripts\preprocessing\prepare_fred_yolo.py --dataset-root dataset --output-root processed\fred10 --train-seqs 0,1,11,101,102,103 --val-seqs 10,21 --test-seqs 34,110
```

Linux/macOS:

```bash
python scripts/preprocessing/prepare_fred_yolo.py \
  --dataset-root dataset \
  --output-root processed/fred10 \
  --train-seqs 0,1,11,101,102,103 \
  --val-seqs 10,21 \
  --test-seqs 34,110
```

By default, the annotation fallback order is:

```text
interpolated_coordinates.txt,coordinates.txt
```

To use only the official extended-box annotation file:

```bash
python scripts/preprocessing/prepare_fred_yolo.py \
  --annotation-files coordinates.txt
```

## 6. Output Structure

After preprocessing, the output directory looks like:

```text
processed/fred10/
  rgb_yolo/
    images/train/
    images/val/
    images/test/
    labels/train/
    labels/val/
    labels/test/
    data.yaml

  event_yolo/
    images/train/
    images/val/
    images/test/
    labels/train/
    labels/val/
    labels/test/
    data.yaml

  paired/
    manifest_train.csv
    manifest_val.csv
    manifest_test.csv

  preprocessing_report.json
```

`rgb_yolo` is for RGB-only YOLO training.

`event_yolo` is for Event-only YOLO training.

`paired` is for RGB/Event fusion training or late fusion.

`preprocessing_report.json` records sequence-level statistics, such as the
number of samples written, annotations without a matching RGB frame, annotations
without a matching Event frame, invalid boxes, and missing directories/files.

## 7. What the YOLO Label Files Contain

Files under `rgb_yolo/labels/.../*.txt` and
`event_yolo/labels/.../*.txt` use standard YOLO detection format.

Each line represents one drone bounding box:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.54023438 0.45486111 0.02890625 0.03472222
```

Meaning:

```text
0          class id, currently always drone
x_center   normalized bbox center x
y_center   normalized bbox center y
width      normalized bbox width
height     normalized bbox height
```

If a frame contains multiple drones, the corresponding `.txt` file contains
multiple lines.

All FRED object classes are collapsed to:

```text
0: drone
```

This keeps the first project stage focused on drone localization and detection,
rather than drone model classification.

## 8. What the Paired Manifest Contains

`paired/manifest_train.csv`, `manifest_val.csv`, and `manifest_test.csv` are
synchronization index files for the fusion pipeline.

They are not YOLO label files themselves. Instead, they tell a fusion dataset
loader which synchronized RGB/Event files and which target label file to read.

Each row contains:

```text
sequence
split
label_time_s
rgb_time_s
event_time_s
rgb_delta_s
event_delta_s
rgb_image
event_image
rgb_label
event_label
source_rgb
source_event
annotation_file
num_boxes
```

Important fields:

```text
rgb_image      processed RGB image path
event_image    processed Event image path
rgb_label      corresponding RGB YOLO label
event_label    corresponding Event YOLO label
label_time_s   annotation timestamp
rgb_time_s     matched RGB timestamp
event_time_s   matched Event timestamp
rgb_delta_s    absolute RGB/annotation time difference
event_delta_s  absolute Event/annotation time difference
num_boxes      number of boxes in this sample
```

## 9. Which Labels Each Model Should Read

### RGB YOLO

RGB-only YOLO reads:

```text
processed/fred10/rgb_yolo/images/train/
processed/fred10/rgb_yolo/labels/train/
processed/fred10/rgb_yolo/data.yaml
```

Validation and testing use:

```text
processed/fred10/rgb_yolo/images/val/
processed/fred10/rgb_yolo/labels/val/

processed/fred10/rgb_yolo/images/test/
processed/fred10/rgb_yolo/labels/test/
```

Images and labels are matched by filename stem:

```text
images/train/seq0_001333320.jpg
labels/train/seq0_001333320.txt
```

### Event YOLO

Event-only YOLO reads:

```text
processed/fred10/event_yolo/images/train/
processed/fred10/event_yolo/labels/train/
processed/fred10/event_yolo/data.yaml
```

Validation and testing use:

```text
processed/fred10/event_yolo/images/val/
processed/fred10/event_yolo/labels/val/

processed/fred10/event_yolo/images/test/
processed/fred10/event_yolo/labels/test/
```

Images and labels are also matched by filename stem:

```text
images/train/seq0_001333320.png
labels/train/seq0_001333320.txt
```

### Fusion Model

The fusion model should read the paired manifests:

```text
processed/fred10/paired/manifest_train.csv
processed/fred10/paired/manifest_val.csv
processed/fred10/paired/manifest_test.csv
```

For each row, it should load:

```text
rgb_image
event_image
rgb_label or event_label
```

In the current preprocessing script, `rgb_label` and `event_label` are generated
from the same FRED annotation file. Because `PADDED_RGB` and `Event/Frames` are
normally both `1280x720`, the two label files should usually contain the same
normalized boxes.

For fusion training, use `rgb_label` as the unified target by convention:

```text
input:
  RGB image
  Event image

target:
  YOLO boxes from rgb_label
```

Using one target avoids ambiguous "two-label" supervision in the fusion model.

## 10. How RGB and Event Frames Are Timestamp-Aligned

The script does not assume:

```text
len(RGB) == len(Event)
```

FRED sequences often have different numbers of RGB and Event frames. This can
happen because RGB frames are removed, Event frames cover a slightly longer time
range, or annotations only cover the interval where drones appear.

The script uses annotation-centered three-way alignment:

```text
annotation time -> nearest RGB frame
annotation time -> nearest Event frame
```

Only samples with all three components are kept:

```text
RGB frame + Event frame + label
```

If either image modality cannot be matched, the sample is skipped and the count
is recorded in `preprocessing_report.json`.

### RGB Timestamp

FRED RGB/PADDED_RGB data is approximately 30 FPS. The script estimates relative
RGB frame time by frame index:

```text
rgb_time = (frame_index + 1) * 0.033333
```

This mirrors the timestamp assumption used in FRED's example dataloader.

### Event Timestamp

Event frame time is parsed from the numeric suffix of the filename.

Example:

```text
Video_230_100032333.png
```

becomes:

```text
event_time = 100032333 / 1_000_000 = 100.032333s
```

### Matching Threshold

The default matching threshold is:

```text
--max-delta 0.04
```

This is 40 ms. Since one 30 FPS frame is about 33.3 ms, 40 ms allows small
rounding and timestamp differences.

If RGB/Event alignment looks visibly shifted, reduce the threshold:

```bash
--max-delta 0.02
```

or even:

```bash
--max-delta 0.015
```

This may reduce the number of usable samples.

## 11. What Happens When Event Shows a Drone but RGB Does Not

There are two different cases.

### Case A: No RGB Frame Can Be Matched

If an annotation or Event timestamp has no matching RGB frame within the time
threshold, the sample is discarded:

```text
unmatched_rgb += 1
```

Fusion training requires:

```text
RGB frame + Event frame + label
```

### Case B: An RGB Frame Exists but the Drone Is Not Visually Clear

This sample is kept.

The target exists physically, even if RGB makes it hard to see because of low
light, HDR, motion blur, rain, or a cluttered background. These are exactly the
cases where Event data and RGB/Event fusion should help.

The script writes the same bounding box target to both:

```text
rgb_yolo/labels/.../*.txt
event_yolo/labels/.../*.txt
```

RGB-only may fail on such examples, while Event-only or fusion should be more
robust. This is useful for final analysis and reporting.

## 12. What If RGB and Event Labels Disagree

For fusion training, RGB and Event should not use conflicting supervision
targets.

The current script therefore does not read existing `RGB_YOLO/` or `Event_YOLO/`
labels from the raw FRED folders. Instead, it regenerates both RGB and Event
labels from the same annotation file.

This makes the target consistent across modalities.

If existing RGB/Event YOLO labels appear to disagree, likely causes are:

1. Raw `RGB/` and `Event/Frames/` were used together without handling padding.
2. RGB and Event frames were matched to slightly different timestamps.
3. Existing `RGB_YOLO/` and `Event_YOLO/` folders were generated by separate
   pipelines.
4. The drone is visually weak in RGB but still physically present.

Recommended label usage:

```text
RGB-only:
  use rgb_yolo/labels

Event-only:
  use event_yolo/labels

Fusion:
  use paired/manifest_*.csv and use rgb_label as the unified target
```

If needed later, a quality-check script can compare existing raw `RGB_YOLO` and
`Event_YOLO` labels against the regenerated annotation labels using IoU and write
low-IoU samples to `alignment_warnings.csv`.

## 13. `coordinates.txt` vs `interpolated_coordinates.txt`

`coordinates.txt` is the sparser annotation file. It is closer to the original
provided boxes and does not necessarily contain a bounding box for every frame.

`interpolated_coordinates.txt` is denser. It fills or interpolates boxes between
annotated timestamps and is usually better for training YOLO, because it provides
more labeled frames.

Recommended usage:

```text
YOLO training:
  prefer interpolated_coordinates.txt

Conservative/official-style ablation:
  use coordinates.txt
```

The script defaults to:

```text
interpolated_coordinates.txt,coordinates.txt
```

meaning it uses the interpolated file if present, otherwise falls back to
`coordinates.txt`.

## 14. Should Denoising Be Applied

### RGB

Do not apply a separate RGB denoising step in the first pass.

YOLO is reasonably robust to normal brightness variation and compression noise.
Training-time augmentation is more important:

```text
brightness / contrast
HSV jitter
motion blur
random crop
small object augmentation
```

### Event

Light Event denoising can be tested, but it should not be a default dependency.

Possible Event noise sources include:

- background flicker;
- rain noise;
- isolated event points;
- sensor noise.

Possible denoising operations:

```text
median blur 3x3
morphological opening
remove tiny connected components
contrast normalization
```

However, sparse Event pixels may also be the drone signal. Aggressive denoising
can remove the target.

Therefore, the current script keeps raw Event frames as the baseline. If
denoising is added later, create a separate output variant:

```text
event_yolo_raw/
event_yolo_denoised_median3/
```

This makes report comparisons cleaner:

```text
RGB-only
Event-only raw
Event-only denoised
Fusion raw
Fusion denoised
```

## 15. Why Data Must Be Split by Sequence

Since only a subset of FRED is downloaded, the local split should be explicit and
sequence-based:

```text
train:
  0, 1, 11, 101, 102, 103

val:
  10, 21

test:
  34, 110
```

Do not split all frames randomly into 80/10/10. Frames from the same video are
too similar and would cause evaluation leakage.

## 16. Conservative Handling Rules in the Current Script

The script keeps a sample only when:

```text
annotation exists
RGB frame can be matched
Event frame can be matched
bbox is valid after clamping to image bounds
```

The script skips a sample when:

```text
RGB frame is missing
Event frame is missing
bbox has no valid area after clamping
required directory or annotation file is missing
```

The script does not skip a sample when:

```text
RGB visually struggles to show the drone but annotation exists
Event is clearer than RGB
RGB/Event frame counts differ but timestamps can be matched
```

This conservative strategy keeps the pipeline simple, reproducible, and easy to
explain in the report.

