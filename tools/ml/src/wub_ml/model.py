from __future__ import annotations

import torch
from torch import nn


class BoardNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        mlp_hidden: int = 64,
        extra_features: int = 0,
        num_outputs: int = 7,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        mlp_in = conv_channels[-1] + extra_features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_outputs),
        )

    def encode(self, board: torch.Tensor) -> torch.Tensor:
        x = self.conv(board)
        x = self.pool(x)
        return x.flatten(1)

    def forward(
        self,
        board: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.encode(board)
        if extra_features is not None:
            x = torch.cat([x, extra_features], dim=1)
        return self.mlp(x)
