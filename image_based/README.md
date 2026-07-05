# AMI-Group7-SoSe26
Branch: Reconstruction_Image

## Step 1: Preprocessing

python3 scripts/preprocessing/preprocess_fred_dataset.py --dataset-root data/raw --output-root data/preprocessed --train-seqs 1,11,101,102,1
03 --val-seqs 10 --test-seqs 34,110 --rgb-dir PADDED_RGB --annotation-files interpolated_coordinates.txt,coordinates.txt --mater
ialize hardlink --overwrite

Train command:
python3 event_guided_rgb_enhancement.py train --data-root data/preprocessed --out-dir artifacts/enhancer --epochs 30 --batch-size 8 --image-size 512

Inference command:
python3 event_guided_rgb_enhancement.py infer --data-root data/preprocessed --checkpoint artifacts/enhancer/checkpoints/enhancer_epoch_030.pt --split test --out-dir artifacts/enhancer_infer --save-triplets


###
python3 src/event_guided_rgb_contrast_pilot.py train   -
-data-root data/preprocessed   --out-dir artifacts/enhancer_pilot_contrast_fullres_15min   --epochs 1   --batch-size 1   --image-size 512   --max-train-batches 2000   --max-val-batches 100   --base-channels 24   --max-flow-px 4.0   --residual-scale 0.15   --gate-prior-mix 0.9   --gate-prior-temperature 3.0   --align-mix 0.0   --w-recon 1.8   --w-edge-event 1.0   --w-contrast-event 1.2   --w-bg-tv 0.35   --w-gate-event 0.5   --w-gate-bg 0.7   --w-gate-sparse 0.3   --w-flow-reg 0.02   --w-residual-event 0.3   --event-threshold 0.08   --event-blur 5   --peak-temperature 1.8   --save-every 1