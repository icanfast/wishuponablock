from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultiLabelTraining:
    num_classes: int

    def __post_init__(self) -> None:
        self.criterion = nn.BCEWithLogitsLoss()

    def loss(
        self,
        logits: torch.Tensor,
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
            targets[i, row] = 1.0
        return self.criterion(logits, targets)

    def accuracy(self, logits: torch.Tensor, labels: list[list[int]]) -> float:
        preds = torch.argmax(logits, dim=1).tolist()
        hits = 0
        for pred, row in zip(preds, labels):
            if pred in row:
                hits += 1
        return hits / max(1, len(labels))
