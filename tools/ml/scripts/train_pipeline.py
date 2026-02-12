#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ML_ROOT / "scripts" / "train.py"


def run_phase(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    cmd = [sys.executable, str(TRAIN_SCRIPT), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


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
        "--virtual-session-size",
        type=int,
        default=100,
        help="Chunk size for virtual session split (0 disables chunking).",
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
        default=str(ML_ROOT / "checkpoints"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ML_ROOT / "checkpoints" / "pipeline_manifest.json"),
    )
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    aug_dir = checkpoint_root / "phase_aug"
    polish_dir = checkpoint_root / "phase_polish"
    aug_dir.mkdir(parents=True, exist_ok=True)
    polish_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--data",
        *args.data,
        "--seed",
        str(args.seed),
        "--val-split",
        str(args.val_split),
        "--batch-size",
        str(args.batch_size),
        "--virtual-session-size",
        str(args.virtual_session_size),
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

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data": args.data,
        "seed": args.seed,
        "val_split": args.val_split,
        "batch_size": args.batch_size,
        "virtual_session_size": args.virtual_session_size,
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
        },
        "phase_polish": {
            "epochs": args.polish_epochs,
            "min_epochs": args.polish_min_epochs,
            "lr": args.polish_lr,
            "patience": args.polish_patience,
            "min_delta": args.polish_min_delta,
            "checkpoint_dir": str(polish_dir),
            "resume_from": str(resume_path) if resume_path else None,
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest to {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
