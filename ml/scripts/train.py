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
    session_ids: list[str],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    sessions: dict[str, list[int]] = {}
    for idx, session_id in enumerate(session_ids):
        sessions.setdefault(session_id, []).append(idx)

    session_list = list(sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(session_list)

    total = len(session_ids)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a baseline WUB model.")
    parser.add_argument("--data", required=True, help="Path to labels_v1.jsonl.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--method",
        choices=["multilabel", "soft_targets"],
        default="multilabel",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    base_dataset = LabelsDataset(
        Path(args.data),
        drop_empty_labels=True,
        mirror_prob=0.0,
    )
    if len(base_dataset) == 0:
        raise SystemExit("No samples found after filtering.")

    train_indices, val_indices = split_by_session(
        base_dataset.session_ids,
        args.val_split,
        args.seed,
    )

    train_ds = LabelsDataset(
        Path(args.data),
        drop_empty_labels=True,
        mirror_prob=args.mirror_prob,
    )
    val_ds = LabelsDataset(
        Path(args.data),
        drop_empty_labels=True,
        mirror_prob=0.0,
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

    if args.method == "multilabel":
        trainer = MultiLabelTraining(num_classes=7)
    else:
        trainer = SoftTargetsTraining(num_classes=7, decay=args.soft_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
