#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ML_ROOT / "scripts" / "train.py"
DEFAULT_CHECKPOINT_ROOT = ML_ROOT / "checkpoints"
DEFAULT_MANIFEST_PATH = DEFAULT_CHECKPOINT_ROOT / "pipeline_manifest.json"


def run_phase(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    cmd = [sys.executable, str(TRAIN_SCRIPT), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_run_name(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Run name cannot be empty.")
    allowed = {"-", "_", "."}
    normalized = "".join(
        ch if ch.isalnum() or ch in allowed else "_" for ch in cleaned
    )
    if not normalized:
        raise ValueError("Run name must include at least one valid character.")
    return normalized


def checkpoint_epoch(path: Path) -> int:
    match = re.match(r"epoch_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    best = checkpoint_dir / "best.pt"
    if best.exists():
        return best

    epoch_files = sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=checkpoint_epoch,
    )
    if not epoch_files:
        return None
    return epoch_files[-1]


def as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def read_checkpoint_summary(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        import torch
    except Exception:
        return {"checkpoint": str(path)}

    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {"checkpoint": str(path)}

    if not isinstance(payload, dict):
        return {"checkpoint": str(path)}

    summary: dict[str, object] = {"checkpoint": str(path)}
    epoch = as_int(payload.get("epoch"))
    train_loss = as_float(payload.get("train_loss"))
    train_acc = as_float(payload.get("train_acc"))
    val_loss = as_float(payload.get("val_loss"))
    val_acc = as_float(payload.get("val_acc"))
    if epoch is not None:
        summary["epoch"] = epoch
    if train_loss is not None:
        summary["train_loss"] = train_loss
    if train_acc is not None:
        summary["train_acc"] = train_acc
    if val_loss is not None:
        summary["val_loss"] = val_loss
    if val_acc is not None:
        summary["val_acc"] = val_acc
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-phase training pipeline.")
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="One or more paths to labels_v1.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument(
        "--session-chunk-size",
        type=int,
        default=100,
        help=(
            "Split groups by session_id + floor(sampleIndex/chunk_size). "
            "Set 0 to split by session_id only."
        ),
    )
    parser.add_argument(
        "--pool-shape",
        default="2,1",
        help=(
            "Adaptive pooling shape passed to train.py as H,W (or HxW). "
            "Default is 2,1."
        ),
    )

    parser.add_argument("--aug-epochs", type=int, default=500)
    parser.add_argument("--aug-min-epochs", type=int, default=50)
    parser.add_argument("--aug-lr", type=float, default=0.002)
    parser.add_argument("--mirror-prob", type=float, default=0.5)
    parser.add_argument("--aug-patience", type=int, default=30)
    parser.add_argument("--aug-min-delta", type=float, default=0.0)

    parser.add_argument("--polish-epochs", type=int, default=100)
    parser.add_argument("--polish-min-epochs", type=int, default=10)
    parser.add_argument("--polish-lr", type=float, default=0.001)
    parser.add_argument("--polish-patience", type=int, default=15)
    parser.add_argument("--polish-min-delta", type=float, default=0.0)

    parser.add_argument(
        "--method",
        choices=["multilabel", "soft_targets"],
        default="soft_targets",
    )
    parser.add_argument(
        "--soft-decay",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run identifier. When set, outputs are written under "
            "<checkpoint-root>/<run-name>/."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
    )
    args = parser.parse_args()

    run_name = None
    if args.run_name is not None:
        try:
            run_name = normalize_run_name(args.run_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    checkpoint_root = Path(args.checkpoint_root)
    if run_name is not None:
        checkpoint_root = checkpoint_root / run_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    aug_dir = checkpoint_root / "phase_aug"
    polish_dir = checkpoint_root / "phase_polish"
    aug_dir.mkdir(parents=True, exist_ok=True)
    polish_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if args.manifest == str(DEFAULT_MANIFEST_PATH):
        manifest_path = checkpoint_root / "pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--data",
        *args.data,
        "--seed",
        str(args.seed),
        "--val-split",
        str(args.val_split),
        "--batch-size",
        str(args.batch_size),
        "--session-chunk-size",
        str(args.session_chunk_size),
        "--pool-shape",
        args.pool_shape,
        "--method",
        args.method,
        "--soft-decay",
        str(args.soft_decay),
    ]
    if args.no_hold:
        base_args.append("--no-hold")
    if args.device:
        base_args.extend(["--device", args.device])

    run_phase(
        "Phase A (augmented)",
        [
            *base_args,
            "--epochs",
            str(args.aug_epochs),
            "--min-epochs",
            str(args.aug_min_epochs),
            "--lr",
            str(args.aug_lr),
            "--mirror-prob",
            str(args.mirror_prob),
            "--checkpoint-dir",
            str(aug_dir),
            "--checkpoint-every",
            "10",
            "--save-best",
            "--early-stop-patience",
            str(args.aug_patience),
            "--early-stop-min-delta",
            str(args.aug_min_delta),
        ],
    )

    resume_path = aug_dir / "best.pt"
    if not resume_path.exists():
        fallback = aug_dir / f"epoch_{args.aug_epochs}.pt"
        if fallback.exists():
            resume_path = fallback
        else:
            resume_path = None

    aug_best = find_best_checkpoint(aug_dir)

    polish_args = [
        *base_args,
        "--epochs",
        str(args.polish_epochs),
        "--min-epochs",
        str(args.polish_min_epochs),
        "--lr",
        str(args.polish_lr),
        "--mirror-prob",
        "0.0",
        "--checkpoint-dir",
        str(polish_dir),
        "--checkpoint-every",
        "10",
        "--save-best",
        "--early-stop-patience",
        str(args.polish_patience),
        "--early-stop-min-delta",
        str(args.polish_min_delta),
    ]
    if resume_path is not None:
        polish_args.extend(["--resume", str(resume_path)])

    run_phase("Phase B (polish)", polish_args)
    polish_best = find_best_checkpoint(polish_dir)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": args.data,
        "run_name": run_name,
        "seed": args.seed,
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "session_chunk_size": args.session_chunk_size,
        "pool_shape": args.pool_shape,
        "method": args.method,
        "soft_decay": args.soft_decay,
        "no_hold": args.no_hold,
        "phase_aug": {
            "epochs": args.aug_epochs,
            "min_epochs": args.aug_min_epochs,
            "lr": args.aug_lr,
            "mirror_prob": args.mirror_prob,
            "patience": args.aug_patience,
            "min_delta": args.aug_min_delta,
            "checkpoint_dir": str(aug_dir),
            "best": read_checkpoint_summary(aug_best),
        },
        "phase_polish": {
            "epochs": args.polish_epochs,
            "min_epochs": args.polish_min_epochs,
            "lr": args.polish_lr,
            "patience": args.polish_patience,
            "min_delta": args.polish_min_delta,
            "checkpoint_dir": str(polish_dir),
            "resume_from": str(resume_path) if resume_path else None,
            "best": read_checkpoint_summary(polish_best),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest to {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
