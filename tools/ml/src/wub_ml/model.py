from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BoardNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        mlp_hidden: int = 64,
        extra_features: int = 0,
        num_outputs: int = 7,
        pool_shape: tuple[int, int] = (1, 1),
        feature_norm: str | None = "layernorm",
        feature_norm_eps: float = 1e-5,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in conv_channels:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        pool_h = max(1, int(pool_shape[0]))
        pool_w = max(1, int(pool_shape[1]))
        self.pool_shape = (pool_h, pool_w)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_shape)

        pooled_features = conv_channels[-1] * self.pool_shape[0] * self.pool_shape[1]
        mlp_in = pooled_features + extra_features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_outputs),
        )
        self.dropout = nn.Dropout(p=max(0.0, min(1.0, float(dropout_p))))

        self.input_channels = int(input_channels)
        self.conv_channels = tuple(int(ch) for ch in conv_channels)
        self.mlp_hidden = int(mlp_hidden)
        self.extra_features = int(extra_features)
        self.num_outputs = int(num_outputs)
        normalized_feature_norm = (feature_norm or "").strip().lower()
        self.feature_norm = (
            normalized_feature_norm if normalized_feature_norm == "layernorm" else None
        )
        self.feature_norm_eps = float(feature_norm_eps)
        self.dropout_p = float(self.dropout.p)

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
        if self.feature_norm == "layernorm":
            x = F.layer_norm(x, (x.shape[1],), eps=self.feature_norm_eps)
        x = self.mlp[0](x)
        x = self.mlp[1](x)
        x = self.dropout(x)
        return self.mlp[2](x)

    def export_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "conv_channels": list(self.conv_channels),
            "mlp_hidden": self.mlp_hidden,
            "extra_features": self.extra_features,
            "num_outputs": self.num_outputs,
            "pool_shape": [self.pool_shape[0], self.pool_shape[1]],
            "feature_norm": self.feature_norm,
            "feature_norm_eps": self.feature_norm_eps,
            "dropout_p": self.dropout_p,
        }
