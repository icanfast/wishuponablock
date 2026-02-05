from __future__ import annotations

import torch


class FeatureBuilder:
    """Builds extra features for the MLP head.

    This is the primary hook for injecting handcrafted features.
    """

    def __init__(self, include_hold: bool = True) -> None:
        self.include_hold = include_hold

    @property
    def feature_dim(self) -> int:
        return 8 if self.include_hold else 0

    def build(self, holds: torch.Tensor) -> torch.Tensor | None:
        features: list[torch.Tensor] = []
        if self.include_hold:
            features.append(one_hot_hold(holds))
        if not features:
            return None
        return torch.cat(features, dim=1)


def one_hot_hold(holds: torch.Tensor) -> torch.Tensor:
    """One-hot encode hold: index 0 = none, 1..7 = piece."""
    batch = holds.shape[0]
    out = torch.zeros((batch, 8), dtype=torch.float32, device=holds.device)
    idx = torch.where(holds >= 0, holds + 1, torch.zeros_like(holds))
    out.scatter_(1, idx.unsqueeze(1), 1.0)
    return out
