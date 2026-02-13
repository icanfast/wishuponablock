#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ML_ROOT / "src"))

from wub_ml.dataset import LabelsDataset, collate_samples
from wub_ml.features import FeatureBuilder
from wub_ml.model import BoardNet
from wub_ml.training.multilabel import MultiLabelTraining
from wub_ml.training.soft_targets import SoftTargetsTraining


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_session(
    group_ids: list[str],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    sessions: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        sessions.setdefault(group_id, []).append(idx)

    session_list = list(sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(session_list)

    total = len(group_ids)
    target_val = int(total * val_split)
    val_indices: list[int] = []
    train_indices: list[int] = []
    val_count = 0

    for session_id in session_list:
        indices = sessions[session_id]
        if val_count < target_val:
            val_indices.extend(indices)
            val_count += len(indices)
        else:
            train_indices.extend(indices)

    if not train_indices:
        train_indices, val_indices = val_indices, train_indices

    return train_indices, val_indices


def build_split_group_id(
    session_id: str,
    sample_index: int | None,
    session_chunk_size: int,
) -> str:
    if session_chunk_size <= 0 or sample_index is None or sample_index < 0:
        return session_id
    chunk_id = sample_index // session_chunk_size
    return f"{session_id}::chunk{chunk_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline WUB model.")
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="One or more paths to labels_v1.jsonl.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--method",
        choices=["multilabel", "soft_targets"],
        default="soft_targets",
        help="Training objective to use.",
    )
    parser.add_argument(
        "--soft-decay",
        type=float,
        default=0.7,
        help="Decay factor for soft-target weights (only for soft_targets).",
    )
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument(
        "--mirror-prob",
        type=float,
        default=0.0,
        help="Probability of mirroring a sample during training.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ML_ROOT / "checkpoints"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 to disable).",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best checkpoint as best.pt (by val loss).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume model weights from.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N epochs without val loss improvement (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Required improvement in val loss to reset patience.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--session-chunk-size",
        "--virtual-session-size",
        dest="session_chunk_size",
        type=int,
        default=100,
        help=(
            "Split groups by session_id + floor(sampleIndex/chunk_size). "
            "Set 0 to split by session_id only."
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    base_dataset = LabelsDataset(
        [Path(p) for p in args.data],
        drop_empty_labels=True,
        mirror_prob=0.0,
        virtual_session_size=0,
    )
    if len(base_dataset) == 0:
        raise SystemExit("No samples found after filtering.")

    split_group_ids = [
        build_split_group_id(
            sample.session_id,
            sample.sample_index,
            args.session_chunk_size,
        )
        for sample in base_dataset.samples
    ]
    train_indices, val_indices = split_by_session(
        split_group_ids,
        args.val_split,
        args.seed,
    )

    train_ds = LabelsDataset(
        [Path(p) for p in args.data],
        drop_empty_labels=True,
        mirror_prob=args.mirror_prob,
        virtual_session_size=0,
    )
    val_ds = LabelsDataset(
        [Path(p) for p in args.data],
        drop_empty_labels=True,
        mirror_prob=0.0,
        virtual_session_size=0,
    )

    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    features = FeatureBuilder(include_hold=not args.no_hold)
    model = BoardNet(
        input_channels=base_dataset.input_channels,
        extra_features=features.feature_dim,
    )
    model.to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
        if state:
            model.load_state_dict(state)
            print(f"Loaded model weights from {args.resume}")
        else:
            print(f"Warning: no model_state in {args.resume}")

    if args.method == "multilabel":
        trainer = MultiLabelTraining(num_classes=7)
    else:
        trainer = SoftTargetsTraining(num_classes=7, decay=args.soft_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = None
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            boards = batch["board"].to(device)
            holds = batch["hold"].to(device)
            labels = batch["labels"]

            extra = features.build(holds)
            logits = model(boards, extra)
            loss = trainer.loss(logits, labels, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_acc += trainer.accuracy(logits.detach(), labels)
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                boards = batch["board"].to(device)
                holds = batch["hold"].to(device)
                labels = batch["labels"]

                extra = features.build(holds)
                logits = model(boards, extra)
                loss = trainer.loss(logits, labels, device)

                val_loss += float(loss.item())
                val_acc += trainer.accuracy(logits, labels)
                val_batches += 1

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss / max(1, train_batches):.4f} "
            f"train_acc={train_acc / max(1, train_batches):.4f} "
            f"val_loss={val_loss / max(1, val_batches):.4f} "
            f"val_acc={val_acc / max(1, val_batches):.4f}"
        )

        current_val = val_loss / max(1, val_batches)
        improved = False
        if best_val_loss is None or current_val < (best_val_loss - args.early_stop_min_delta):
            best_val_loss = current_val
            epochs_since_improve = 0
            improved = True
        else:
            epochs_since_improve += 1

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss / max(1, train_batches),
                    "train_acc": train_acc / max(1, train_batches),
                    "val_loss": val_loss / max(1, val_batches),
                    "val_acc": val_acc / max(1, val_batches),
                    "args": vars(args),
                },
                ckpt_path,
            )

        if args.save_best and improved:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss / max(1, train_batches),
                    "train_acc": train_acc / max(1, train_batches),
                    "val_loss": current_val,
                    "val_acc": val_acc / max(1, val_batches),
                    "args": vars(args),
                },
                best_path,
            )

        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and epochs_since_improve >= args.early_stop_patience
        ):
            print(
                "Early stopping: no val loss improvement for "
                f"{epochs_since_improve} epochs."
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
