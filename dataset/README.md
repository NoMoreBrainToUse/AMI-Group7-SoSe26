# Dataset directory

Place raw FRED event-camera sequences here. Each sequence is a separate subdirectory:

```
dataset/
  40/   ← seq40  (blind test, split: train)
  43/   ← seq43  (blind test, split: val)
  46/   ← seq46  (blind test, split: test)
  49/   ← seq49  (blind test, split: test)
```

## Automatic download

`run_blind_test_v4.sh` downloads missing sequences from Google Drive automatically
using [gdown](https://github.com/wkentaro/gdown):

```bash
pip install gdown
./run_blind_test_v4.sh          # downloads seq40, 43, 46, 49 if absent
```

Downloaded zip files are extracted here and the zip is deleted.

## Skipping the download

If the sequences are already on disk or mounted as a Docker volume, pass
`--no-download`. The script will verify that all required directories exist
and exit with an error if any are missing.

```bash
./run_blind_test_v4.sh --no-download
```

## Docker mount example

```bash
docker run --rm \
  -v /path/to/fred-data:/app/dataset \
  my-image \
  bash -c "./run_blind_test_v4.sh --no-download"
```

## What is not committed

All actual sequence data (`dataset/40/`, `dataset/43/`, etc.) is excluded from
git via `.gitignore` because the sequences are large (several GB each).
Only this README and the `.gitkeep` placeholder are tracked.
