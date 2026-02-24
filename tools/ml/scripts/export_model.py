#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ML_ROOT = Path(__file__).resolve().parents[1]
FULL_BOARD_CHANNELS = [
    "occupancy",
    "holes",
    "row_fill",
    "coord_x",
    "coord_y",
    "col_height",
    "well_depth",
    "reachable_empty",
    "coarse_occ_v2",
]


def load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt
    return {"model_state": ckpt}


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "data": tensor.detach().cpu().flatten().tolist(),
    }


def normalize_pool_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        h = int(value[0])
        w = int(value[1])
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    return [h, w]


def infer_pool_shape_from_area(area: int) -> list[int]:
    if area <= 1:
        return [1, 1]
    root = int(math.sqrt(area))
    for h in range(root, 0, -1):
        if area % h == 0:
            w = area // h
            return [int(h), int(w)]
    return [1, int(area)]


def infer_model_config(
    checkpoint: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model_cfg = checkpoint.get("model_config")
    if isinstance(model_cfg, dict):
        pool_shape = normalize_pool_shape(model_cfg.get("pool_shape")) or [1, 1]
        try:
            return {
                "input_channels": int(model_cfg["input_channels"]),
                "conv_channels": [int(c) for c in model_cfg["conv_channels"]],
                "mlp_hidden": int(model_cfg["mlp_hidden"]),
                "extra_features": int(model_cfg["extra_features"]),
                "num_outputs": int(model_cfg["num_outputs"]),
                "pool_shape": pool_shape,
                "feature_norm": model_cfg.get("feature_norm"),
                "feature_norm_eps": float(model_cfg.get("feature_norm_eps", 1e-5)),
                "dropout_p": float(model_cfg.get("dropout_p", 0.0)),
            }
        except (KeyError, TypeError, ValueError):
            pass

    conv_weights = []
    for name, tensor in state.items():
        if name.startswith("conv.") and name.endswith(".weight"):
            try:
                layer_idx = int(name.split(".")[1])
            except ValueError:
                continue
            conv_weights.append((layer_idx, tensor))
    conv_weights.sort(key=lambda item: item[0])
    if not conv_weights:
        raise ValueError("No conv weights found in checkpoint.")

    conv_channels = [tensor.shape[0] for _, tensor in conv_weights]
    input_channels = conv_weights[0][1].shape[1]

    mlp0 = state.get("mlp.0.weight")
    mlp2 = state.get("mlp.2.weight")
    if mlp0 is None or mlp2 is None:
        raise ValueError("Missing MLP weights in checkpoint.")

    mlp_hidden = mlp0.shape[0]
    mlp_in = mlp0.shape[1]
    args = checkpoint.get("args")
    include_hold = True
    if isinstance(args, dict):
        include_hold = not bool(args.get("no_hold"))
    extra_features = 8 if include_hold else 0
    pooled_features = mlp_in - extra_features
    if pooled_features <= 0:
        extra_features = max(0, mlp_in - conv_channels[-1])
        pooled_features = mlp_in - extra_features
    conv_out = max(1, conv_channels[-1])
    if pooled_features % conv_out == 0:
        pool_area = max(1, pooled_features // conv_out)
    else:
        pool_area = 1
    pool_shape = infer_pool_shape_from_area(pool_area)
    num_outputs = mlp2.shape[0]

    return {
        "input_channels": int(input_channels),
        "conv_channels": [int(c) for c in conv_channels],
        "mlp_hidden": int(mlp_hidden),
        "extra_features": int(extra_features),
        "num_outputs": int(num_outputs),
        "pool_shape": pool_shape,
        "feature_norm": None,
        "feature_norm_eps": 1e-5,
        "dropout_p": 0.0,
    }


def board_channels_for_input_count(input_channels: int) -> list[str]:
    count = max(0, int(input_channels))
    if count <= len(FULL_BOARD_CHANNELS):
        return FULL_BOARD_CHANNELS[:count]
    return [*FULL_BOARD_CHANNELS, *["occupancy"] * (count - len(FULL_BOARD_CHANNELS))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a model checkpoint to JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument(
        "--out",
        default=str(ML_ROOT / "exports" / "model.json"),
        help="Output path for JSON export.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(ckpt_path)
    state = ckpt["model_state"]
    config = infer_model_config(ckpt, state)

    payload = {
        "schema": "wishuponablock.model.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
        },
        "pieces": ["I", "O", "T", "S", "Z", "J", "L"],
        "board_channels": board_channels_for_input_count(config["input_channels"]),
        "model": config,
        "params": {name: tensor_to_payload(tensor) for name, tensor in state.items()},
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    print(f"Exported model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
