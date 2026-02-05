#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


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


def infer_model_config(state: dict[str, torch.Tensor]) -> dict[str, Any]:
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
    extra_features = mlp_in - conv_channels[-1]
    num_outputs = mlp2.shape[0]

    return {
        "input_channels": int(input_channels),
        "conv_channels": [int(c) for c in conv_channels],
        "mlp_hidden": int(mlp_hidden),
        "extra_features": int(extra_features),
        "num_outputs": int(num_outputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a model checkpoint to JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument(
        "--out",
        default="ml/exports/model.json",
        help="Output path for JSON export.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(ckpt_path)
    state = ckpt["model_state"]
    config = infer_model_config(state)

    payload = {
        "schema": "wishuponablock.model.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch"),
        },
        "pieces": ["I", "O", "T", "S", "Z", "J", "L"],
        "board_channels": ["occupancy", "holes", "row_fill"],
        "model": config,
        "params": {name: tensor_to_payload(tensor) for name, tensor in state.items()},
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    print(f"Exported model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
