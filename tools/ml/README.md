# ML Scaffold

This folder holds the data prep and (later) training code for the piece-selection model.

## Quick start

1. Create a virtual environment (Python 3.11 recommended) and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place your labeling output at `tools/ml/data/raw/labels.jsonl` (copy from the tool's output folder).
   The converter also accepts a single JSON array file if you export one later.

3. Convert to the intermediate training format:

```bash
python tools/ml/scripts/convert_labels.py \
  --in tools/ml/data/raw/labels.jsonl \
  --out tools/ml/data/processed/labels_v1.jsonl
```

The intermediate format is documented in `tools/ml/schema.md`.
The converter now includes `session_id`, which is used for session-aware
train/validation splits.

## Training (baseline)

```bash
python tools/ml/scripts/train.py \
  --data tools/ml/data/processed/labels_v1.jsonl \
  --epochs 10 \
  --batch-size 128
```

Options:

- `--method multilabel|soft_targets` - training objective (default: multilabel).
- `--soft-decay 0.7` - decay factor for soft-target weights.
- `--mirror-prob 0.0` - probability of mirroring a sample (labels remapped).
- `--checkpoint-dir tools/ml/checkpoints` - where to save checkpoints.
- `--checkpoint-every 10` - save a checkpoint every N epochs (0 disables).
- `--no-hold` - drop the hold piece feature.

Note: training uses a session-based split when `session_id` is present in the
processed file, so keep that field in your converted data.

## Export

```bash
python tools/ml/scripts/export_model.py \
  --checkpoint tools/ml/checkpoints/epoch_10.pt \
  --out tools/ml/exports/model.json
```

## Layout

- `data/raw/` - raw labels exported by the in-game tool.
- `data/processed/` - cleaned, training-ready JSONL.
- `scripts/` - data prep utilities.
- `src/` - (future) training code.

Training code will be added later.
