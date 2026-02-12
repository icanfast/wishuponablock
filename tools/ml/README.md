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
   This raw format is now accepted directly by the training pipeline.

## Training (baseline)

```bash
python tools/ml/scripts/train.py \
  --data tools/ml/data/raw/labels.jsonl \
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

Note: training uses a session-based split, with virtual session chunking enabled
by default (`--virtual-session-size 100`). For labels exported by the in-game
tool, session/sample IDs are inferred from `source.sessionId` and
`source.sampleIndex` when available.

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
