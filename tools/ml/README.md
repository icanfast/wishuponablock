# ML Scaffold

This folder holds the data prep and (later) training code for the piece-selection model.

## Quick start

1. Create a virtual environment (Python 3.11 recommended) and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare a merged training file (local JSONL + D1 CSV/JSONL):

```bash
python tools/ml/scripts/prepare_data.py \
  --input \
    data/snapshots/labeled/labels.jsonl \
    data/online_labels/labels_v1.jsonl \
    data/online_labels/labels_0_2_3.jsonl \
    data/online_labels/online_labels_0_2_3.csv \
  --out tools/ml/data/raw/labels_merged.jsonl
```

## Training (baseline)

```bash
python tools/ml/scripts/train_pipeline.py \
  --data tools/ml/data/raw/labels_merged.jsonl \
  --method soft_targets \
  --soft-decay 0.7
```

Options:

- `--method multilabel|soft_targets` - training objective (default: soft_targets).
- `--soft-decay 0.7` - decay factor for soft-target weights.
- `--mirror-prob 0.0` - probability of mirroring a sample (labels remapped).
- `train.py` still exists for single-phase runs and accepts multiple `--data` files directly.
- `prepare_data.py` dedupes by default (`--no-dedupe` to disable).
- `--no-hold` drops the hold piece feature.

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
