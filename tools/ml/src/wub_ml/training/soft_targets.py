from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SoftTargetsTraining:
    num_classes: int
    decay: float = 0.7

    def __post_init__(self) -> None:
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def build_targets(
        self,
        labels: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        targets = torch.zeros(
            (len(labels), self.num_classes),
            dtype=torch.float32,
            device=device,
        )
        for i, row in enumerate(labels):
            if not row:
                continue
            weights = torch.tensor(
                [self.decay**rank for rank in range(len(row))],
                dtype=torch.float32,
                device=device,
            )
            weights = weights / weights.sum()
            indices = torch.tensor(row, device=device, dtype=torch.long)
            targets[i].index_add_(0, indices, weights)
        return targets

    def loss(
        self,
        logits: torch.Tensor,
        labels: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        targets = self.build_targets(labels, device)
        log_probs = F.log_softmax(logits, dim=1)
        return self.criterion(log_probs, targets)

    def accuracy(self, logits: torch.Tensor, labels: list[list[int]]) -> float:
        preds = torch.argmax(logits, dim=1).tolist()
        hits = 0
        for pred, row in zip(preds, labels):
            if pred in row:
                hits += 1
        return hits / max(1, len(labels))
