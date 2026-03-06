#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from wub_env import WubEnvBridge

RAW_PIECES_ORDER = ("I", "O", "T", "S", "Z", "J", "L")
RAW_CONTEXT_DIM = len(RAW_PIECES_ORDER) + (len(RAW_PIECES_ORDER) + 1) + len(RAW_PIECES_ORDER) + 5
DEFAULT_BOARD_ROWS = 20
DEFAULT_BOARD_COLS = 10
PPO_OBS_ADAPTER_LR_SCALE = 0.1


@dataclass(frozen=True)
class PPOConfig:
    mode_id: str
    model_path: str
    observation_space: str
    placement_execution_mode: str
    queue_policy_id: str
    piece_source_profile: str
    alternate_piece_sources: bool
    max_pieces_per_episode: int
    seed: int
    num_envs: int
    total_timesteps: int
    num_steps: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    policy_clip_coef: float
    value_clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    target_kl: float
    warmup_updates: int
    warmup_ent_coef: float
    warmup_target_kl: float
    warmup_policy_lr_scale: float
    warmup_value_lr_scale: float
    curriculum_topk_start: int
    curriculum_topk_end: int
    curriculum_topk_ramp_updates: int
    curriculum_bias_start: float
    curriculum_bias_end: float
    curriculum_bias_ramp_updates: int
    curriculum_danger_height: int
    distill_coef_start: float
    distill_coef_end: float
    distill_coef_ramp_updates: int
    distill_teacher_tau: float
    validation_episodes_per_env: int
    device: str
    save_every_updates: int
    log_every_updates: int
    out_dir: str
    run_name: str
    server_cmd: str | None
    resume_checkpoint: str | None
    init_artifact: str | None
    deterministic_eval: bool
    bc_dataset: str | None
    bc_epochs: int
    bc_batch_size: int
    bc_learning_rate: float
    bc_value_weight: float
    bc_max_records: int | None
    bc_normalize_returns: bool
    bc_return_clip: float
    freeze_encoder_after_bc: bool
    freeze_conv_after_bc: bool


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()

        def init_layer(
            layer: nn.Linear,
            std: float = math.sqrt(2.0),
            bias_const: float = 0.0,
        ) -> nn.Linear:
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        self.policy_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        self.value_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # PPO-friendly init:
        # - Hidden ReLU layers: orthogonal gain sqrt(2)
        # - Policy head: tiny gain so initial logits are near-uniform
        # - Value head: gain 1.0
        init_layer(self.policy_fc1, std=math.sqrt(2.0))
        init_layer(self.policy_fc2, std=math.sqrt(2.0))
        init_layer(self.value_fc1, std=math.sqrt(2.0))
        init_layer(self.value_fc2, std=math.sqrt(2.0))
        init_layer(self.policy_head, std=0.01)
        init_layer(self.value_head, std=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_hidden = torch.relu(self.policy_fc1(obs))
        policy_hidden = torch.relu(self.policy_fc2(policy_hidden))
        logits = self.policy_head(policy_hidden)

        value_hidden = torch.relu(self.value_fc1(obs))
        value_hidden = torch.relu(self.value_fc2(value_hidden))
        value = self.value_head(value_hidden).squeeze(-1)
        return logits, value


class ObservationAdapter(nn.Module):
    def __init__(self, raw_obs_dim: int, policy_obs_dim: int, policy_observation_space: str) -> None:
        super().__init__()
        self.raw_obs_dim = int(raw_obs_dim)
        self.policy_obs_dim = int(policy_obs_dim)
        self.policy_observation_space = policy_observation_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityObservationAdapter(ObservationAdapter):
    def __init__(self, raw_obs_dim: int, policy_observation_space: str) -> None:
        super().__init__(
            raw_obs_dim=raw_obs_dim,
            policy_obs_dim=raw_obs_dim,
            policy_observation_space=policy_observation_space,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class WubHeadFromRawObservationAdapter(ObservationAdapter):
    def __init__(self, raw_obs_dim: int, model_path: str) -> None:
        model_json = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.model_schema = str(model_json.get("schema", "wishuponablock.model.v1"))
        model_cfg = model_json.get("model", {})
        params = model_json.get("params", {})

        input_channels = int(model_cfg.get("input_channels", 0))
        conv_channels = [int(v) for v in model_cfg.get("conv_channels", [])]
        extra_features = int(model_cfg.get("extra_features", 0))
        pool_shape_raw = model_cfg.get("pool_shape")
        if isinstance(pool_shape_raw, list) and len(pool_shape_raw) == 2:
            pool_h = max(1, int(pool_shape_raw[0]))
            pool_w = max(1, int(pool_shape_raw[1]))
        else:
            pool_h, pool_w = 1, 1

        board_channels = model_json.get("board_channels", ["occupancy", "holes", "row_fill"])
        if not isinstance(board_channels, list) or len(board_channels) == 0:
            board_channels = ["occupancy", "holes", "row_fill"]
        board_channels_norm = [str(x) for x in board_channels]

        board_size = int(raw_obs_dim) - RAW_CONTEXT_DIM
        if board_size <= 0:
            raise ValueError(
                f"raw observation dim is too small for raw_v1 layout: obs_dim={raw_obs_dim}"
            )
        rows, cols = DEFAULT_BOARD_ROWS, DEFAULT_BOARD_COLS
        if rows * cols != board_size:
            raise ValueError(
                "raw observation layout mismatch. "
                f"expected board_size={DEFAULT_BOARD_ROWS * DEFAULT_BOARD_COLS}, got={board_size}"
            )

        model_pieces_raw = model_json.get("pieces")
        if isinstance(model_pieces_raw, list) and len(model_pieces_raw) == len(RAW_PIECES_ORDER):
            model_pieces = [str(x) for x in model_pieces_raw]
        else:
            model_pieces = list(RAW_PIECES_ORDER)
        self.model_pieces = list(model_pieces)
        model_piece_index = {piece: i for i, piece in enumerate(model_pieces)}
        hold_map = [0]
        for piece in RAW_PIECES_ORDER:
            mapped = model_piece_index.get(piece, -1)
            hold_map.append(mapped + 1 if mapped >= 0 else 0)
        hold_index_map = torch.tensor(hold_map, dtype=torch.long)

        coord_x = torch.zeros((rows, cols), dtype=torch.float32)
        coord_y = torch.zeros((rows, cols), dtype=torch.float32)
        denom_x = max(1.0, float(cols - 1))
        denom_y = max(1.0, float(rows - 1))
        for y in range(rows):
            for x in range(cols):
                coord_x[y, x] = float(x) / denom_x
                coord_y[y, x] = float(y) / denom_y

        layers: list[nn.Conv2d] = []
        in_ch = input_channels
        for i, out_ch in enumerate(conv_channels):
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            layer_index = i * 2
            weight_key = f"conv.{layer_index}.weight"
            bias_key = f"conv.{layer_index}.bias"
            weight_payload = params.get(weight_key)
            bias_payload = params.get(bias_key)
            if not isinstance(weight_payload, dict) or not isinstance(bias_payload, dict):
                raise ValueError(f"Missing conv params for layer {i}: {weight_key}/{bias_key}")
            weight_shape = weight_payload.get("shape")
            bias_shape = bias_payload.get("shape")
            if not isinstance(weight_shape, list) or not isinstance(bias_shape, list):
                raise ValueError(f"Invalid conv param shapes for layer {i}.")
            weight_data = np.asarray(weight_payload.get("data", []), dtype=np.float32)
            bias_data = np.asarray(bias_payload.get("data", []), dtype=np.float32)
            weight_tensor = torch.from_numpy(weight_data).view(*[int(v) for v in weight_shape])
            bias_tensor = torch.from_numpy(bias_data).view(*[int(v) for v in bias_shape])
            with torch.no_grad():
                conv.weight.copy_(weight_tensor)
                conv.bias.copy_(bias_tensor)
            layers.append(conv)
            in_ch = out_ch
        extra_features_norm = max(0, extra_features)
        feature_norm = (
            str(model_cfg.get("feature_norm")).lower()
            if model_cfg.get("feature_norm") is not None
            else None
        )
        feature_norm_eps = float(model_cfg.get("feature_norm_eps", 1e-5))

        pooled_dim = in_ch * pool_h * pool_w
        policy_obs_dim = pooled_dim + extra_features_norm
        super().__init__(
            raw_obs_dim=raw_obs_dim,
            policy_obs_dim=policy_obs_dim,
            policy_observation_space="model_head_v1",
        )
        self.board_channels = board_channels_norm
        self.rows = rows
        self.cols = cols
        self.board_size = board_size
        self.active_offset = board_size
        self.hold_offset = self.active_offset + len(RAW_PIECES_ORDER)
        self.next_offset = self.hold_offset + len(RAW_PIECES_ORDER) + 1
        self.register_buffer("hold_index_map", hold_index_map, persistent=False)
        self.register_buffer("coord_x", coord_x, persistent=False)
        self.register_buffer("coord_y", coord_y, persistent=False)
        self.conv_layers = nn.ModuleList(layers)
        self.conv_channels = [int(v) for v in conv_channels]
        self.pool_shape = (pool_h, pool_w)
        self.extra_features = extra_features_norm
        self.feature_norm = feature_norm
        self.feature_norm_eps = feature_norm_eps
        self.input_channels = input_channels
        self.num_outputs = int(model_cfg.get("num_outputs", len(RAW_PIECES_ORDER)))
        self.mlp_hidden = int(model_cfg.get("mlp_hidden", 0))

    def _compute_reachable_empty(self, empty: torch.Tensor) -> torch.Tensor:
        # empty: [B, R, C] bool
        reachable = torch.zeros_like(empty)
        reachable[:, 0, :] = empty[:, 0, :]
        max_iters = self.rows * self.cols
        for _ in range(max_iters):
            up = torch.zeros_like(reachable)
            up[:, 1:, :] = reachable[:, :-1, :]
            down = torch.zeros_like(reachable)
            down[:, :-1, :] = reachable[:, 1:, :]
            left = torch.zeros_like(reachable)
            left[:, :, 1:] = reachable[:, :, :-1]
            right = torch.zeros_like(reachable)
            right[:, :, :-1] = reachable[:, :, 1:]
            expanded = reachable | up | down | left | right
            next_reachable = expanded & empty
            if not torch.any(next_reachable != reachable):
                break
            reachable = next_reachable
        return reachable.to(dtype=torch.float32)

    def _build_model_input_channels(self, occupancy: torch.Tensor) -> torch.Tensor:
        # occupancy: [B, R, C] in {0,1}
        bsz = occupancy.shape[0]
        rows = self.rows
        cols = self.cols
        device = occupancy.device
        dtype = occupancy.dtype

        filled = occupancy > 0.5
        seen = torch.cumsum(filled.to(dtype=torch.int32), dim=1) > 0
        holes = ((~filled) & seen).to(dtype=dtype)
        row_fill = occupancy.mean(dim=2, keepdim=True).expand(-1, -1, cols)

        coord_x = self.coord_x.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
        coord_y = self.coord_y.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)

        has_filled_col = filled.any(dim=1)  # [B, C]
        first_filled = torch.argmax(filled.to(dtype=torch.int64), dim=1)  # [B, C]
        col_height_base = torch.where(
            has_filled_col,
            (rows - first_filled).to(dtype=dtype) / max(1.0, float(rows)),
            torch.zeros_like(first_filled, dtype=dtype),
        )
        col_height = col_height_base.unsqueeze(1).expand(-1, rows, -1)

        well_depth = torch.zeros_like(occupancy)
        depth = torch.zeros((bsz, cols), dtype=torch.int32, device=device)
        for y in range(rows):
            occ_row = filled[:, y, :]
            left_blocked = torch.ones((bsz, cols), dtype=torch.bool, device=device)
            right_blocked = torch.ones((bsz, cols), dtype=torch.bool, device=device)
            left_blocked[:, 1:] = occ_row[:, :-1]
            right_blocked[:, :-1] = occ_row[:, 1:]
            in_well = (~occ_row) & left_blocked & right_blocked
            depth = torch.where(occ_row, torch.zeros_like(depth), torch.where(in_well, depth + 1, torch.zeros_like(depth)))
            well_depth[:, y, :] = depth.to(dtype=dtype) / max(1.0, float(rows))

        empty = ~filled
        reachable_empty = self._compute_reachable_empty(empty)

        coarse_occ_v2 = occupancy.clone()
        for y0 in range(0, rows, 2):
            y1 = min(rows, y0 + 2)
            max_band = occupancy[:, y0:y1, :].amax(dim=1, keepdim=True)
            coarse_occ_v2[:, y0:y1, :] = max_band

        source = {
            "occupancy": occupancy,
            "holes": holes,
            "row_fill": row_fill,
            "coord_x": coord_x,
            "coord_y": coord_y,
            "col_height": col_height,
            "well_depth": well_depth,
            "reachable_empty": reachable_empty,
            "coarse_occ_v2": coarse_occ_v2,
        }
        channels: list[torch.Tensor] = []
        zero = torch.zeros_like(occupancy)
        for name in self.board_channels:
            channels.append(source.get(name, zero))
        if len(channels) < self.input_channels:
            channels.extend([zero] * (self.input_channels - len(channels)))
        elif len(channels) > self.input_channels:
            channels = channels[: self.input_channels]
        return torch.stack(channels, dim=1)

    def _build_extra_features(self, obs: torch.Tensor) -> torch.Tensor:
        bsz = obs.shape[0]
        if self.extra_features <= 0:
            return torch.zeros((bsz, 0), dtype=obs.dtype, device=obs.device)
        hold_block = obs[:, self.hold_offset : self.hold_offset + len(RAW_PIECES_ORDER) + 1]
        hold_idx = torch.argmax(hold_block, dim=1).to(dtype=torch.long)
        mapped = self.hold_index_map.to(device=obs.device)[hold_idx]
        extra = torch.zeros((bsz, self.extra_features), dtype=obs.dtype, device=obs.device)
        valid = (mapped >= 0) & (mapped < self.extra_features)
        if torch.any(valid):
            rows = torch.arange(bsz, device=obs.device)[valid]
            cols = mapped[valid]
            extra[rows, cols] = 1.0
        return extra

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        occupancy = obs[:, : self.board_size].view(-1, self.rows, self.cols)
        occupancy = torch.where(
            occupancy > 0.5,
            torch.ones_like(occupancy),
            torch.zeros_like(occupancy),
        )
        model_input = self._build_model_input_channels(occupancy)
        current = model_input
        for conv in self.conv_layers:
            current = torch.relu(conv(current))
        pooled = F.adaptive_avg_pool2d(current, self.pool_shape)
        flat = pooled.flatten(start_dim=1)
        extra = self._build_extra_features(obs)
        features = torch.cat([flat, extra], dim=1)
        if self.feature_norm == "layernorm":
            features = F.layer_norm(
                features,
                normalized_shape=(features.shape[1],),
                eps=self.feature_norm_eps,
            )
        return features

    def to_exported_encoder_model(self) -> dict[str, Any]:
        params: dict[str, dict[str, Any]] = {}
        for i, conv in enumerate(self.conv_layers):
            layer_index = i * 2
            weight = conv.weight.detach().cpu().numpy().astype(np.float32)
            bias = conv.bias.detach().cpu().numpy().astype(np.float32)
            params[f"conv.{layer_index}.weight"] = {
                "shape": [int(v) for v in weight.shape],
                "data": weight.reshape(-1).tolist(),
            }
            params[f"conv.{layer_index}.bias"] = {
                "shape": [int(v) for v in bias.shape],
                "data": bias.reshape(-1).tolist(),
            }
        return {
            "schema": self.model_schema,
            "model": {
                "input_channels": int(self.input_channels),
                "conv_channels": [int(v) for v in self.conv_channels],
                "mlp_hidden": int(self.mlp_hidden),
                "extra_features": int(self.extra_features),
                "num_outputs": int(self.num_outputs),
                "pool_shape": [int(self.pool_shape[0]), int(self.pool_shape[1])],
                "feature_norm": self.feature_norm,
                "feature_norm_eps": float(self.feature_norm_eps),
            },
            "params": params,
            "pieces": [str(v) for v in self.model_pieces],
            "board_channels": [str(v) for v in self.board_channels],
        }

    def load_from_exported_encoder_model(self, payload: dict[str, Any]) -> None:
        model = payload.get("model", {})
        params = payload.get("params", {})
        if not isinstance(model, dict) or not isinstance(params, dict):
            raise ValueError("Invalid encoderModel payload structure.")
        conv_channels = [int(v) for v in model.get("conv_channels", [])]
        if conv_channels != self.conv_channels:
            raise ValueError(
                "encoderModel conv_channels mismatch. "
                f"artifact={conv_channels} runtime={self.conv_channels}"
            )
        input_channels = int(model.get("input_channels", 0))
        if input_channels != self.input_channels:
            raise ValueError(
                "encoderModel input_channels mismatch. "
                f"artifact={input_channels} runtime={self.input_channels}"
            )
        for i, conv in enumerate(self.conv_layers):
            layer_index = i * 2
            weight_payload = params.get(f"conv.{layer_index}.weight")
            bias_payload = params.get(f"conv.{layer_index}.bias")
            if not isinstance(weight_payload, dict) or not isinstance(bias_payload, dict):
                raise ValueError(f"encoderModel missing conv layer params at index={i}.")
            weight_shape = weight_payload.get("shape")
            bias_shape = bias_payload.get("shape")
            weight_data = np.asarray(weight_payload.get("data", []), dtype=np.float32)
            bias_data = np.asarray(bias_payload.get("data", []), dtype=np.float32)
            if not isinstance(weight_shape, list) or not isinstance(bias_shape, list):
                raise ValueError(f"encoderModel invalid tensor shape for layer={i}.")
            weight_tensor = torch.from_numpy(weight_data).view(*[int(v) for v in weight_shape])
            bias_tensor = torch.from_numpy(bias_data).view(*[int(v) for v in bias_shape])
            if tuple(weight_tensor.shape) != tuple(conv.weight.shape):
                raise ValueError(
                    "encoderModel weight shape mismatch. "
                    f"artifact={tuple(weight_tensor.shape)} runtime={tuple(conv.weight.shape)}"
                )
            if tuple(bias_tensor.shape) != tuple(conv.bias.shape):
                raise ValueError(
                    "encoderModel bias shape mismatch. "
                    f"artifact={tuple(bias_tensor.shape)} runtime={tuple(conv.bias.shape)}"
                )
            with torch.no_grad():
                conv.weight.copy_(weight_tensor)
                conv.bias.copy_(bias_tensor)


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(
        description="Train WUB headless bot with full PPO in local PyTorch."
    )
    parser.add_argument("--mode-id", default="practice")
    parser.add_argument("--model-path", default="public/models/model_v4.json")
    parser.add_argument(
        "--observation-space",
        default="raw_v1",
        choices=["model_head_v1", "raw_v1"],
    )
    parser.add_argument(
        "--placement-execution-mode",
        default="teleport",
        choices=["commands", "teleport"],
        help="How env executes placement actions: deterministic command playback or direct teleport+harddrop.",
    )
    parser.add_argument("--queue-policy-id", default="next_piece_v1")
    parser.add_argument(
        "--piece-source-profile",
        default="active_generator",
        choices=["active_generator", "bag7"],
    )
    parser.add_argument(
        "--alternate-piece-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed source schedule per 5 updates: 4 updates on bag7, then 1 update on active_generator.",
    )
    parser.add_argument("--max-pieces-per-episode", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42030)

    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-steps", type=int, default=256)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--policy-clip-coef",
        "--clip-coef",
        dest="policy_clip_coef",
        type=float,
        default=0.2,
        help="PPO policy ratio clipping epsilon.",
    )
    parser.add_argument(
        "--value-clip-coef",
        type=float,
        default=0.4,
        help="Value function clipping epsilon (used when --clip-vloss is enabled).",
    )
    parser.add_argument(
        "--clip-vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--warmup-updates", type=int, default=50)
    parser.add_argument("--warmup-ent-coef", type=float, default=0.001)
    parser.add_argument("--warmup-target-kl", type=float, default=0.01)
    parser.add_argument(
        "--warmup-policy-lr-scale",
        type=float,
        default=0.1,
        help="Scale for policy/trunk/encoder learning rate during warmup updates.",
    )
    parser.add_argument(
        "--warmup-value-lr-scale",
        type=float,
        default=1.0,
        help="Scale for value-head learning rate during warmup updates.",
    )
    parser.add_argument(
        "--curriculum-topk-start",
        type=int,
        default=12,
        help="Curriculum top-K at update 1 (0 disables top-K pruning).",
    )
    parser.add_argument(
        "--curriculum-topk-end",
        type=int,
        default=48,
        help="Curriculum top-K after ramp completes (0 disables top-K pruning).",
    )
    parser.add_argument(
        "--curriculum-topk-ramp-updates",
        type=int,
        default=400,
        help="Number of updates to linearly ramp top-K from start to end.",
    )
    parser.add_argument(
        "--curriculum-bias-start",
        type=float,
        default=0.25,
        help="Heuristic logit-bias strength at update 1 (0 disables bias).",
    )
    parser.add_argument(
        "--curriculum-bias-end",
        type=float,
        default=0.0,
        help="Heuristic logit-bias strength after ramp completes (0 disables bias).",
    )
    parser.add_argument(
        "--curriculum-bias-ramp-updates",
        type=int,
        default=400,
        help="Number of updates to linearly ramp bias strength from start to end.",
    )
    parser.add_argument(
        "--curriculum-danger-height",
        type=int,
        default=14,
        help="Disable top-K pruning when stack max height is at/above this threshold.",
    )
    parser.add_argument(
        "--distill-coef-start",
        type=float,
        default=0.15,
        help="Distillation loss coefficient at update 1.",
    )
    parser.add_argument(
        "--distill-coef-end",
        type=float,
        default=0.0,
        help="Distillation loss coefficient after ramp completes.",
    )
    parser.add_argument(
        "--distill-coef-ramp-updates",
        type=int,
        default=400,
        help="Number of updates to linearly ramp distillation coefficient.",
    )
    parser.add_argument(
        "--distill-teacher-tau",
        type=float,
        default=0.35,
        help="Teacher temperature for distillation softmax(score / tau). Lower is sharper.",
    )
    parser.add_argument(
        "--validation-episodes-per-env",
        type=int,
        default=2,
        help="Validation episodes per env on each logged update (0 disables validation).",
    )

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--log-every-updates", type=int, default=1)
    parser.add_argument("--out-dir", default="tools/bot_env/output")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--server-cmd",
        default=None,
        help='Override bridge server command (example: "npx --yes tsx tools/bot_env/ts/envServer.ts").',
    )
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--init-artifact", default=None)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument(
        "--bc-dataset",
        default=None,
        help="Path to BC dataset JSON produced by tools/bot_env/ts/buildBcDataset.ts",
    )
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=4096)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-value-weight", type=float, default=0.25)
    parser.add_argument(
        "--bc-normalize-returns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize BC return-to-go targets over valid records before value fitting.",
    )
    parser.add_argument(
        "--bc-return-clip",
        type=float,
        default=10.0,
        help="Clip BC value targets after optional normalization (<=0 disables clipping).",
    )
    parser.add_argument(
        "--freeze-encoder-after-bc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze observation encoder params for PPO after BC completes.",
    )
    parser.add_argument(
        "--freeze-conv-after-bc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze only encoder conv layers after BC (keeps policy/value MLP trainable).",
    )
    parser.add_argument(
        "--bc-max-records",
        type=int,
        default=0,
        help="Optional cap for BC records (0 means no cap).",
    )

    args = parser.parse_args()
    run_name = args.run_name.strip() or f"ppo_baseline_{int(time.time())}"
    return PPOConfig(
        mode_id=args.mode_id.strip().lower(),
        model_path=args.model_path,
        observation_space=(
            "raw_v1" if args.observation_space == "raw_v1" else "model_head_v1"
        ),
        placement_execution_mode=(
            "commands"
            if args.placement_execution_mode == "commands"
            else "teleport"
        ),
        queue_policy_id=args.queue_policy_id.strip().lower(),
        piece_source_profile=(
            "bag7" if args.piece_source_profile == "bag7" else "active_generator"
        ),
        alternate_piece_sources=bool(args.alternate_piece_sources),
        max_pieces_per_episode=max(1, int(args.max_pieces_per_episode)),
        seed=max(1, int(args.seed)),
        num_envs=max(1, int(args.num_envs)),
        total_timesteps=max(1, int(args.total_timesteps)),
        num_steps=max(1, int(args.num_steps)),
        hidden_dim=max(8, int(args.hidden_dim)),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        policy_clip_coef=float(args.policy_clip_coef),
        value_clip_coef=float(args.value_clip_coef),
        clip_vloss=bool(args.clip_vloss),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        update_epochs=max(1, int(args.update_epochs)),
        minibatch_size=max(1, int(args.minibatch_size)),
        target_kl=max(0.0, float(args.target_kl)),
        warmup_updates=max(0, int(args.warmup_updates)),
        warmup_ent_coef=max(0.0, float(args.warmup_ent_coef)),
        warmup_target_kl=max(0.0, float(args.warmup_target_kl)),
        warmup_policy_lr_scale=max(0.0, float(args.warmup_policy_lr_scale)),
        warmup_value_lr_scale=max(0.0, float(args.warmup_value_lr_scale)),
        curriculum_topk_start=max(0, int(args.curriculum_topk_start)),
        curriculum_topk_end=max(0, int(args.curriculum_topk_end)),
        curriculum_topk_ramp_updates=max(0, int(args.curriculum_topk_ramp_updates)),
        curriculum_bias_start=max(0.0, float(args.curriculum_bias_start)),
        curriculum_bias_end=max(0.0, float(args.curriculum_bias_end)),
        curriculum_bias_ramp_updates=max(0, int(args.curriculum_bias_ramp_updates)),
        curriculum_danger_height=max(1, int(args.curriculum_danger_height)),
        distill_coef_start=max(0.0, float(args.distill_coef_start)),
        distill_coef_end=max(0.0, float(args.distill_coef_end)),
        distill_coef_ramp_updates=max(0, int(args.distill_coef_ramp_updates)),
        distill_teacher_tau=max(1e-4, float(args.distill_teacher_tau)),
        validation_episodes_per_env=max(0, int(args.validation_episodes_per_env)),
        device=args.device,
        save_every_updates=max(1, int(args.save_every_updates)),
        log_every_updates=max(1, int(args.log_every_updates)),
        out_dir=args.out_dir,
        run_name=run_name,
        server_cmd=args.server_cmd,
        resume_checkpoint=args.resume_checkpoint,
        init_artifact=args.init_artifact,
        deterministic_eval=bool(args.deterministic_eval),
        bc_dataset=(
            args.bc_dataset.strip() if isinstance(args.bc_dataset, str) and args.bc_dataset.strip() else None
        ),
        bc_epochs=max(0, int(args.bc_epochs)),
        bc_batch_size=max(1, int(args.bc_batch_size)),
        bc_learning_rate=float(args.bc_learning_rate),
        bc_value_weight=max(0.0, float(args.bc_value_weight)),
        bc_max_records=(
            max(1, int(args.bc_max_records)) if int(args.bc_max_records) > 0 else None
        ),
        bc_normalize_returns=bool(args.bc_normalize_returns),
        bc_return_clip=float(args.bc_return_clip),
        freeze_encoder_after_bc=bool(args.freeze_encoder_after_bc),
        freeze_conv_after_bc=bool(args.freeze_conv_after_bc),
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_piece_source_for_update(cfg: PPOConfig, update: int) -> str:
    base = "bag7" if cfg.piece_source_profile == "bag7" else "active_generator"
    if not cfg.alternate_piece_sources:
        return base
    # Fixed alternation schedule:
    # 4 updates on bag7, then 1 update on active_generator.
    # Example: bag7, bag7, bag7, bag7, active_generator, ...
    if ((update - 1) % 5) == 4:
        return "active_generator"
    return "bag7"


def curriculum_schedule_value(
    start: float,
    end: float,
    ramp_updates: int,
    update: int,
) -> float:
    if ramp_updates <= 1:
        return float(end)
    clamped_update = max(1, int(update))
    progress = min(1.0, max(0.0, float(clamped_update - 1) / float(ramp_updates - 1)))
    return float(start + (end - start) * progress)


def curriculum_topk_for_update(cfg: PPOConfig, update: int) -> int:
    value = curriculum_schedule_value(
        float(cfg.curriculum_topk_start),
        float(cfg.curriculum_topk_end),
        cfg.curriculum_topk_ramp_updates,
        update,
    )
    return max(0, int(round(value)))


def curriculum_bias_for_update(cfg: PPOConfig, update: int) -> float:
    return max(
        0.0,
        curriculum_schedule_value(
            cfg.curriculum_bias_start,
            cfg.curriculum_bias_end,
            cfg.curriculum_bias_ramp_updates,
            update,
        ),
    )


def distill_coef_for_update(cfg: PPOConfig, update: int) -> float:
    return max(
        0.0,
        curriculum_schedule_value(
            cfg.distill_coef_start,
            cfg.distill_coef_end,
            cfg.distill_coef_ramp_updates,
            update,
        ),
    )


def ensure_action_masks(
    mask_np: np.ndarray,
    repair_stats: dict[str, int] | None = None,
) -> np.ndarray:
    # Guarantee at least one valid action per env row.
    mask = np.asarray(mask_np, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected action mask shape [N, A], got {mask.shape}")
    valid_counts = mask.sum(axis=1)
    invalid_rows = np.where(valid_counts <= 0)[0]
    if invalid_rows.size > 0:
        if repair_stats is not None:
            repair_stats["rows"] = repair_stats.get("rows", 0) + int(invalid_rows.size)
            repair_stats["batches"] = repair_stats.get("batches", 0) + 1
        # Safe fallback: force a single deterministic action slot (0).
        mask[invalid_rows, :] = 0.0
        mask[invalid_rows, 0] = 1.0
    return mask


def ensure_action_biases(
    mask_np: np.ndarray,
    bias_np: np.ndarray | None,
) -> np.ndarray:
    mask = np.asarray(mask_np, dtype=np.float32)
    if bias_np is None:
        return np.zeros_like(mask, dtype=np.float32)
    bias = np.asarray(bias_np, dtype=np.float32)
    if bias.shape != mask.shape:
        return np.zeros_like(mask, dtype=np.float32)
    finite = np.where(np.isfinite(bias), bias, 0.0).astype(np.float32, copy=False)
    finite = np.maximum(finite, 0.0)
    finite *= (mask > 0).astype(np.float32)
    return finite


def ensure_action_scores(
    mask_np: np.ndarray,
    scores_np: np.ndarray | None,
) -> np.ndarray:
    mask = np.asarray(mask_np, dtype=np.float32)
    if scores_np is None:
        return np.zeros_like(mask, dtype=np.float32)
    scores = np.asarray(scores_np, dtype=np.float32)
    if scores.shape != mask.shape:
        return np.zeros_like(mask, dtype=np.float32)
    finite = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32, copy=False)
    finite *= (mask > 0).astype(np.float32)
    return finite


def build_teacher_probs(
    action_mask: torch.Tensor,
    action_scores: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = action_mask > 0
    valid_f = valid.to(dtype=action_mask.dtype)
    valid_count = torch.sum(valid_f, dim=-1, keepdim=True)
    safe_valid_count = torch.clamp(valid_count, min=1.0)
    masked_scores = torch.where(valid, action_scores, torch.zeros_like(action_scores))
    row_mean = torch.sum(masked_scores, dim=-1, keepdim=True) / safe_valid_count
    row_var = torch.sum(
        ((masked_scores - row_mean) * valid_f) ** 2,
        dim=-1,
        keepdim=True,
    ) / safe_valid_count
    row_is_uniform = (row_var <= 1e-12).to(dtype=action_mask.dtype)
    safe_tau = max(1e-4, float(tau))
    large_neg = torch.full_like(action_scores, -1e9)
    teacher_logits = torch.where(valid, action_scores / safe_tau, large_neg)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    return teacher_probs, row_is_uniform


def masked_categorical(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    action_bias: torch.Tensor | None = None,
    bias_alpha: float = 0.0,
) -> Categorical:
    valid_mask = action_mask > 0
    large_neg = torch.full_like(logits, -1e9)
    masked_logits = torch.where(valid_mask, logits, large_neg)

    alpha = float(max(0.0, min(1.0, bias_alpha)))
    if action_bias is not None and alpha > 0.0:
        bias = torch.clamp(action_bias, min=0.0) * valid_mask.to(dtype=logits.dtype)
        valid_count = torch.sum(valid_mask.to(dtype=logits.dtype), dim=-1, keepdim=True)
        bias_mean = torch.sum(bias, dim=-1, keepdim=True) / torch.clamp(
            valid_count, min=1.0
        )
        centered_bias = bias - bias_mean
        masked_logits = torch.where(
            valid_mask,
            masked_logits + alpha * centered_bias,
            large_neg,
        )
    return Categorical(logits=masked_logits)


def infer_action_dim(mask_batch: list[list[float]]) -> int:
    if not mask_batch:
        raise RuntimeError("Bridge returned empty action mask batch.")
    first = mask_batch[0]
    if not first:
        raise RuntimeError("Bridge returned empty action mask vector.")
    return len(first)


def infer_obs_dim(obs_batch: list[list[float]]) -> int:
    if not obs_batch:
        raise RuntimeError("Bridge returned empty observation batch.")
    first = obs_batch[0]
    if not first:
        raise RuntimeError("Bridge returned empty observation vector.")
    return len(first)


def as_tensor(np_array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np_array).to(device)


def trainable_parameters(
    model: PolicyValueNet, obs_adapter: ObservationAdapter
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    params.extend([p for p in obs_adapter.parameters() if p.requires_grad])
    return params


def scale_obs_adapter_gradients(obs_adapter: ObservationAdapter, scale: float) -> None:
    if not math.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-12:
        return
    for param in obs_adapter.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        param.grad.mul_(scale)


def build_ppo_optimizer(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    learning_rate: float,
) -> torch.optim.Optimizer:
    # Split optimizer groups so we can slow policy/trunk updates during PPO warmup
    # while keeping value-head updates at full speed.
    policy_params: list[nn.Parameter] = []
    policy_params.extend(
        [p for p in model.policy_fc1.parameters() if p.requires_grad]
    )
    policy_params.extend(
        [p for p in model.policy_fc2.parameters() if p.requires_grad]
    )
    policy_params.extend(
        [p for p in model.policy_head.parameters() if p.requires_grad]
    )
    policy_params.extend([p for p in obs_adapter.parameters() if p.requires_grad])

    value_params: list[nn.Parameter] = []
    value_params.extend(
        [p for p in model.value_fc1.parameters() if p.requires_grad]
    )
    value_params.extend(
        [p for p in model.value_fc2.parameters() if p.requires_grad]
    )
    value_params.extend([p for p in model.value_head.parameters() if p.requires_grad])

    param_groups: list[dict[str, Any]] = []
    if policy_params:
        param_groups.append(
            {
                "params": policy_params,
                "lr": learning_rate,
                "group_name": "policy",
            }
        )
    if value_params:
        param_groups.append(
            {
                "params": value_params,
                "lr": learning_rate,
                "group_name": "value",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found for PPO optimizer.")
    return torch.optim.Adam(param_groups, lr=learning_rate, eps=1e-5)


def apply_warmup_lr_schedule(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    warmup_active: bool,
    warmup_policy_lr_scale: float,
    warmup_value_lr_scale: float,
) -> tuple[float, float]:
    policy_lr = base_lr * (warmup_policy_lr_scale if warmup_active else 1.0)
    value_lr = base_lr * (warmup_value_lr_scale if warmup_active else 1.0)
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", "policy"))
        if group_name == "value":
            group["lr"] = value_lr
        else:
            group["lr"] = policy_lr
    return float(policy_lr), float(value_lr)


def export_bot_policy_artifact(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    cfg: PPOConfig,
    obs_dim: int,
    action_dim: int,
    policy_observation_space: str,
    pipeline_id: str = "bot_ppo_offline_v1",
) -> dict[str, Any]:
    with torch.no_grad():
        # Policy tower (kept on legacy keys for client/runtime backward compatibility).
        w1 = model.policy_fc1.weight.detach().cpu().numpy().astype(np.float32)  # [H, I]
        b1 = model.policy_fc1.bias.detach().cpu().numpy().astype(np.float32)  # [H]
        w2 = model.policy_fc2.weight.detach().cpu().numpy().astype(np.float32)  # [H, H]
        b2 = model.policy_fc2.bias.detach().cpu().numpy().astype(np.float32)  # [H]
        wp = (
            model.policy_head.weight.detach().cpu().numpy().astype(np.float32)
        )  # [A, H]
        bp = model.policy_head.bias.detach().cpu().numpy().astype(np.float32)  # [A]

        # Value tower (new in split actor/critic architecture).
        wv1 = model.value_fc1.weight.detach().cpu().numpy().astype(np.float32)  # [H, I]
        bv1 = model.value_fc1.bias.detach().cpu().numpy().astype(np.float32)  # [H]
        wv2 = model.value_fc2.weight.detach().cpu().numpy().astype(np.float32)  # [H, H]
        bv2 = model.value_fc2.bias.detach().cpu().numpy().astype(np.float32)  # [H]
        wv = model.value_head.weight.detach().cpu().numpy().astype(np.float32)  # [1, H]
        bv = model.value_head.bias.detach().cpu().numpy().astype(np.float32)  # [1]

    now_ms = int(time.time() * 1000)
    artifact = {
        "id": f"bot_policy_baseline_{now_ms}",
        "modeId": cfg.mode_id,
        "archId": "full",
        "queuePolicyId": cfg.queue_policy_id,
        "pipelineId": pipeline_id,
        "pieceSourceProfile": cfg.piece_source_profile,
        "observationSpace": policy_observation_space,
        "createdAtMs": now_ms,
        "inputDim": int(obs_dim),
        "hiddenDim": int(cfg.hidden_dim),
        "actionDim": int(action_dim),
        "actionSpaceKind": "placement_full_v1",
        "placementActionDim": int(action_dim),
        "weights": {
            "w1": w1.T.reshape(-1).tolist(),  # [I, H]
            "b1": b1.reshape(-1).tolist(),
            "w2": w2.T.reshape(-1).tolist(),  # [H, H]
            "b2": b2.reshape(-1).tolist(),
            "wp": wp.T.reshape(-1).tolist(),  # [H, A]
            "bp": bp.reshape(-1).tolist(),
            "wv1": wv1.T.reshape(-1).tolist(),  # [I, H]
            "bv1": bv1.reshape(-1).tolist(),
            "wv2": wv2.T.reshape(-1).tolist(),  # [H, H]
            "bv2": bv2.reshape(-1).tolist(),
            "wv": wv.reshape(-1).tolist(),  # [H]
            "bv": bv.reshape(-1).tolist(),  # [1]
        },
    }
    if isinstance(obs_adapter, WubHeadFromRawObservationAdapter):
        artifact["encoderModel"] = obs_adapter.to_exported_encoder_model()
    return artifact


def load_from_artifact(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    artifact_path: Path,
) -> str:
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    observation_space_raw = raw.get("observationSpace")
    observation_space = (
        "raw_v1" if observation_space_raw == "raw_v1" else "model_head_v1"
    )
    weights = raw.get("weights", {})
    input_dim = int(raw.get("inputDim", 0))
    hidden_dim = int(raw.get("hiddenDim", 0))
    action_dim = int(raw.get("actionDim", 0))

    if input_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
        raise ValueError("Invalid artifact dims.")

    if model.policy_fc1.in_features != input_dim:
        raise ValueError(
            f"Artifact inputDim mismatch. artifact={input_dim} model={model.policy_fc1.in_features}"
        )
    if model.policy_fc1.out_features != hidden_dim:
        raise ValueError(
            f"Artifact hiddenDim mismatch. artifact={hidden_dim} model={model.policy_fc1.out_features}"
        )
    if model.policy_head.out_features != action_dim:
        raise ValueError(
            f"Artifact actionDim mismatch. artifact={action_dim} model={model.policy_head.out_features}"
        )

    w1 = np.asarray(weights.get("w1", []), dtype=np.float32).reshape(input_dim, hidden_dim)
    b1 = np.asarray(weights.get("b1", []), dtype=np.float32).reshape(hidden_dim)
    w2_payload = weights.get("w2")
    b2_payload = weights.get("b2")
    has_second_layer = isinstance(w2_payload, list) and isinstance(b2_payload, list)
    if has_second_layer:
        w2 = np.asarray(w2_payload, dtype=np.float32).reshape(hidden_dim, hidden_dim)
        b2 = np.asarray(b2_payload, dtype=np.float32).reshape(hidden_dim)
    else:
        # Backward compatibility for one-hidden-layer artifacts.
        w2 = np.eye(hidden_dim, dtype=np.float32)
        b2 = np.zeros((hidden_dim,), dtype=np.float32)
    wp = np.asarray(weights.get("wp", []), dtype=np.float32).reshape(hidden_dim, action_dim)
    bp = np.asarray(weights.get("bp", []), dtype=np.float32).reshape(action_dim)

    wv1_payload = weights.get("wv1")
    bv1_payload = weights.get("bv1")
    wv2_payload = weights.get("wv2")
    bv2_payload = weights.get("bv2")
    has_split_value_tower = (
        isinstance(wv1_payload, list)
        and isinstance(bv1_payload, list)
        and isinstance(wv2_payload, list)
        and isinstance(bv2_payload, list)
    )
    if has_split_value_tower:
        wv1 = np.asarray(wv1_payload, dtype=np.float32).reshape(input_dim, hidden_dim)
        bv1 = np.asarray(bv1_payload, dtype=np.float32).reshape(hidden_dim)
        wv2 = np.asarray(wv2_payload, dtype=np.float32).reshape(hidden_dim, hidden_dim)
        bv2 = np.asarray(bv2_payload, dtype=np.float32).reshape(hidden_dim)
    else:
        # Backward compatibility for artifacts before split actor/critic towers.
        # Start value tower from policy tower weights and keep old value head.
        wv1 = w1.copy()
        bv1 = b1.copy()
        wv2 = w2.copy()
        bv2 = b2.copy()

    wv = np.asarray(weights.get("wv", []), dtype=np.float32).reshape(hidden_dim)
    bv = np.asarray(weights.get("bv", []), dtype=np.float32).reshape(1)

    with torch.no_grad():
        model.policy_fc1.weight.copy_(torch.from_numpy(w1.T))
        model.policy_fc1.bias.copy_(torch.from_numpy(b1))
        model.policy_fc2.weight.copy_(torch.from_numpy(w2.T))
        model.policy_fc2.bias.copy_(torch.from_numpy(b2))
        model.policy_head.weight.copy_(torch.from_numpy(wp.T))
        model.policy_head.bias.copy_(torch.from_numpy(bp))
        model.value_fc1.weight.copy_(torch.from_numpy(wv1.T))
        model.value_fc1.bias.copy_(torch.from_numpy(bv1))
        model.value_fc2.weight.copy_(torch.from_numpy(wv2.T))
        model.value_fc2.bias.copy_(torch.from_numpy(bv2))
        model.value_head.weight.copy_(torch.from_numpy(wv.reshape(1, hidden_dim)))
        model.value_head.bias.copy_(torch.from_numpy(bv))
    encoder_payload = raw.get("encoderModel")
    if isinstance(obs_adapter, WubHeadFromRawObservationAdapter) and isinstance(
        encoder_payload, dict
    ):
        obs_adapter.load_from_exported_encoder_model(encoder_payload)
    return observation_space


def save_checkpoint(
    checkpoint_path: Path,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    optimizer: torch.optim.Optimizer,
    cfg: PPOConfig,
    obs_dim: int,
    action_dim: int,
    global_step: int,
    update: int,
    stats: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "obs_adapter_state_dict": obs_adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "update": int(update),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "config": cfg.__dict__,
        "stats": stats,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    optimizer: torch.optim.Optimizer,
    map_device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
    model_state = checkpoint["model_state_dict"]
    if isinstance(model_state, dict):
        # Backward compatibility for checkpoints from shared-torso models.
        if "policy_fc1.weight" not in model_state and "fc1.weight" in model_state:
            fc1_w = model_state.get("fc1.weight")
            fc1_b = model_state.get("fc1.bias")
            fc2_w = model_state.get("fc2.weight")
            fc2_b = model_state.get("fc2.bias")
            if fc1_w is not None and fc1_b is not None:
                model_state["policy_fc1.weight"] = fc1_w
                model_state["policy_fc1.bias"] = fc1_b
                model_state["value_fc1.weight"] = fc1_w
                model_state["value_fc1.bias"] = fc1_b
            if fc2_w is not None and fc2_b is not None:
                model_state["policy_fc2.weight"] = fc2_w
                model_state["policy_fc2.bias"] = fc2_b
                model_state["value_fc2.weight"] = fc2_w
                model_state["value_fc2.bias"] = fc2_b

        # Backward compatibility for checkpoints saved before second hidden layer existed.
        if "policy_fc2.weight" not in model_state or "policy_fc2.bias" not in model_state:
            with torch.no_grad():
                eye = torch.eye(
                    model.policy_fc2.out_features,
                    model.policy_fc2.in_features,
                    dtype=model.policy_fc2.weight.dtype,
                    device=model.policy_fc2.weight.device,
                )
                model.policy_fc2.weight.copy_(eye)
                model.policy_fc2.bias.zero_()
        if "value_fc2.weight" not in model_state or "value_fc2.bias" not in model_state:
            with torch.no_grad():
                eye = torch.eye(
                    model.value_fc2.out_features,
                    model.value_fc2.in_features,
                    dtype=model.value_fc2.weight.dtype,
                    device=model.value_fc2.weight.device,
                )
                model.value_fc2.weight.copy_(eye)
                model.value_fc2.bias.zero_()
    model.load_state_dict(model_state, strict=False)
    adapter_state = checkpoint.get("obs_adapter_state_dict")
    if isinstance(adapter_state, dict):
        obs_adapter.load_state_dict(adapter_state, strict=False)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer_state, dict):
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as error:
            print(
                "[ppo] warning: optimizer state was not loaded from checkpoint "
                f"(param-group mismatch). Using fresh optimizer state. detail={error}"
            )
    global_step = int(checkpoint.get("global_step", 0))
    update = int(checkpoint.get("update", 0))
    return global_step, update


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_json_any(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def pop_latest_trajectory(
    env: WubEnvBridge,
    max_drain: int = 128,
) -> tuple[dict[str, Any] | None, int]:
    latest: dict[str, Any] | None = None
    drained = 0
    for _ in range(max(1, int(max_drain))):
        result = env.pop_trajectory()
        candidate = result.get("trajectory")
        if not isinstance(candidate, dict):
            break
        latest = candidate
        drained += 1
    return latest, drained


def capture_single_policy_rollout(
    env: WubEnvBridge,
    env_id: int,
    seed: int,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    device: torch.device,
    max_steps: int,
    deterministic: bool = True,
    bias_alpha: float = 0.0,
) -> dict[str, Any]:
    reset = env.reset_many(env_ids=[env_id], seeds=[seed])
    obs = np.asarray(reset["obs"], dtype=np.float32)
    mask = ensure_action_masks(np.asarray(reset["action_masks"], dtype=np.float32))
    action_bias = ensure_action_biases(
        mask,
        np.asarray(reset.get("action_biases", []), dtype=np.float32)
        if "action_biases" in reset
        else None,
    )
    if obs.shape[0] <= 0 or mask.shape[0] <= 0:
        return {
            "episode_return": 0.0,
            "episode_length": 0,
            "done": False,
            "trajectory": None,
            "drained": 0,
        }

    episode_return = 0.0
    episode_length = 0
    done = False

    for _ in range(max(1, int(max_steps))):
        raw_obs_t = as_tensor(obs, device)
        mask_t = as_tensor(mask, device)
        action_bias_t = as_tensor(action_bias, device)
        with torch.no_grad():
            features_t = obs_adapter(raw_obs_t)
            logits_t, _values_t = model(features_t)
            dist_t = masked_categorical(
                logits_t,
                mask_t,
                action_bias=action_bias_t,
                bias_alpha=bias_alpha,
            )
            if deterministic:
                action_t = torch.argmax(dist_t.probs, dim=-1)
            else:
                action_t = dist_t.sample()
        action = int(action_t.detach().cpu().numpy()[0])
        step = env.step_many(env_ids=[env_id], actions=[action])
        rewards = np.asarray(step.get("rewards", [0.0]), dtype=np.float32)
        dones = np.asarray(step.get("dones", [False]), dtype=np.float32)
        episode_return += float(rewards[0]) if rewards.size > 0 else 0.0
        episode_length += 1
        done = bool(dones[0] > 0.5) if dones.size > 0 else False
        if done:
            break
        obs = np.asarray(step["obs"], dtype=np.float32)
        mask = ensure_action_masks(
            np.asarray(step["action_masks"], dtype=np.float32)
        )
        action_bias = ensure_action_biases(
            mask,
            np.asarray(step.get("action_biases", []), dtype=np.float32)
            if "action_biases" in step
            else None,
        )

    trajectory, drained = pop_latest_trajectory(env, max_drain=64)
    return {
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "done": bool(done),
        "trajectory": trajectory,
        "drained": int(drained),
    }


def run_validation_eval(
    *,
    cfg: PPOConfig,
    repo_root: Path,
    server_cmd: list[str] | None,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    device: torch.device,
    update: int,
    piece_source_profile: str,
    reward_component_aliases: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    episodes_per_env = max(0, int(cfg.validation_episodes_per_env))
    if episodes_per_env <= 0:
        return {"enabled": False}

    was_model_training = model.training
    was_adapter_training = obs_adapter.training
    model.eval()
    obs_adapter.eval()
    try:
        with WubEnvBridge(server_cmd=server_cmd, cwd=repo_root) as val_env:
            init_result = val_env.init(
                mode_id=cfg.mode_id,
                num_envs=cfg.num_envs,
                model_path=cfg.model_path,
                observation_space=cfg.observation_space,
                placement_execution_mode=cfg.placement_execution_mode,
                piece_source_profile=piece_source_profile,
                queue_policy_id=cfg.queue_policy_id,
                max_pieces_per_episode=cfg.max_pieces_per_episode,
                seed=cfg.seed + update * 1777 + 31,
            )
            env_ids: list[int] = [int(v) for v in init_result.get("env_ids", [])]
            if not env_ids:
                return {"enabled": False, "error": "validation init returned no env ids"}

            val_env.set_piece_source(piece_source_profile)
            val_env.set_curriculum(
                top_k=0,
                bias_strength=0.0,
                danger_height=cfg.curriculum_danger_height,
            )

            returns: list[float] = []
            lengths: list[int] = []
            term_values: dict[str, list[float]] = {
                key: [] for key in reward_component_aliases
            }
            max_steps = max(8, int(cfg.max_pieces_per_episode) * 2)

            for episode_round in range(episodes_per_env):
                seeds = [
                    cfg.seed
                    + update * 1_000_003
                    + episode_round * 10_007
                    + env_idx * 101
                    for env_idx in range(len(env_ids))
                ]
                reset_result = val_env.reset_many(env_ids=env_ids, seeds=seeds)
                obs_np = np.asarray(reset_result["obs"], dtype=np.float32)
                mask_np = ensure_action_masks(
                    np.asarray(reset_result["action_masks"], dtype=np.float32)
                )
                action_bias_np = ensure_action_biases(
                    mask_np,
                    np.asarray(reset_result.get("action_biases", []), dtype=np.float32)
                    if "action_biases" in reset_result
                    else None,
                )

                env_count = obs_np.shape[0]
                ep_return = np.zeros(env_count, dtype=np.float64)
                ep_length = np.zeros(env_count, dtype=np.int64)
                ep_terms = {
                    key: np.zeros(env_count, dtype=np.float64)
                    for key in reward_component_aliases
                }
                done_mask = np.zeros(env_count, dtype=np.bool_)

                for _ in range(max_steps):
                    raw_obs_t = as_tensor(obs_np, device)
                    mask_t = as_tensor(mask_np, device)
                    action_bias_t = as_tensor(action_bias_np, device)
                    with torch.no_grad():
                        features_t = obs_adapter(raw_obs_t)
                        logits_t, _values_t = model(features_t)
                        dist_t = masked_categorical(
                            logits_t,
                            mask_t,
                            action_bias=action_bias_t,
                            bias_alpha=0.0,  # Validation is actor-only.
                        )
                        actions_t = torch.argmax(dist_t.probs, dim=-1)
                    actions_np = (
                        actions_t.detach().cpu().numpy().astype(np.int64).tolist()
                    )

                    step_result = val_env.step_many(env_ids=env_ids, actions=actions_np)
                    rewards_np = np.asarray(step_result["rewards"], dtype=np.float32)
                    dones_np = np.asarray(step_result["dones"], dtype=np.float32)
                    infos_raw = step_result.get("infos", [])

                    active_mask = ~done_mask
                    ep_return[active_mask] += rewards_np[active_mask].astype(np.float64)
                    ep_length[active_mask] += 1

                    if isinstance(infos_raw, list):
                        max_info = min(len(infos_raw), env_count)
                        for env_idx in range(max_info):
                            if not active_mask[env_idx]:
                                continue
                            info = infos_raw[env_idx]
                            reward_final = _info_num(
                                info,
                                reward_component_aliases["reward_final"],
                                default=float(rewards_np[env_idx]),
                            )
                            ep_terms["reward_final"][env_idx] += reward_final
                            for key in reward_component_aliases.keys():
                                if key == "reward_final":
                                    continue
                                ep_terms[key][env_idx] += _info_num(
                                    info,
                                    reward_component_aliases[key],
                                    default=0.0,
                                )
                    else:
                        ep_terms["reward_final"][active_mask] += rewards_np[
                            active_mask
                        ].astype(np.float64)

                    done_mask |= dones_np > 0.5
                    if bool(np.all(done_mask)):
                        break

                    obs_np = np.asarray(step_result["obs"], dtype=np.float32)
                    mask_np = ensure_action_masks(
                        np.asarray(step_result["action_masks"], dtype=np.float32)
                    )
                    action_bias_np = ensure_action_biases(
                        mask_np,
                        np.asarray(step_result.get("action_biases", []), dtype=np.float32)
                        if "action_biases" in step_result
                        else None,
                    )

                returns.extend(ep_return.tolist())
                lengths.extend(ep_length.tolist())
                for key, arr in ep_terms.items():
                    term_values[key].extend(arr.tolist())

            return {
                "enabled": True,
                "episodes": int(len(returns)),
                "piece_source_profile": piece_source_profile,
                "mean_return": _safe_recent_mean(returns, window=len(returns)),
                "mean_length": _safe_recent_mean(lengths, window=len(lengths)),
                "terms": {
                    key: _safe_recent_mean(values, window=len(values))
                    for key, values in term_values.items()
                },
            }
    finally:
        if was_model_training:
            model.train()
        if was_adapter_training:
            obs_adapter.train()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _profile_num(profile: Any, key: str) -> float:
    if not isinstance(profile, dict):
        return 0.0
    value = profile.get(key)
    if not _is_finite_number(value):
        return 0.0
    return float(value)


def _info_num(info: Any, keys: tuple[str, ...], default: float = 0.0) -> float:
    if not isinstance(info, dict):
        return default
    for key in keys:
        value = info.get(key)
        if _is_finite_number(value):
            return float(value)
    return default


def _safe_recent_mean(values: list[float], window: int = 100) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values[-max(1, int(window)) :]))


def _fmt_float(value: Any, precision: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{precision}f}"
    return "nan"


def load_bc_dataset(
    dataset_path: Path,
    obs_dim: int,
    action_dim: int,
    max_records: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = raw.get("records")
    if not isinstance(records, list):
        raise ValueError("BC dataset missing records array.")

    obs_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    action_rows: list[int] = []
    return_rows: list[float] = []
    return_mask_rows: list[float] = []

    skipped = {
        "invalid_record": 0,
        "invalid_obs": 0,
        "invalid_mask": 0,
        "invalid_action": 0,
    }

    for rec in records:
        if max_records is not None and len(obs_rows) >= max_records:
            break
        if not isinstance(rec, dict):
            skipped["invalid_record"] += 1
            continue

        obs = rec.get("obs")
        action_mask = rec.get("actionMask")
        action_index = rec.get("actionIndex")
        return_to_go = rec.get("returnToGo")

        if (
            not isinstance(obs, list)
            or len(obs) != obs_dim
            or any(not _is_finite_number(v) for v in obs)
        ):
            skipped["invalid_obs"] += 1
            continue
        if (
            not isinstance(action_mask, list)
            or len(action_mask) != action_dim
            or any(not _is_finite_number(v) for v in action_mask)
        ):
            skipped["invalid_mask"] += 1
            continue
        if not isinstance(action_index, int):
            skipped["invalid_action"] += 1
            continue
        if action_index < 0 or action_index >= action_dim:
            skipped["invalid_action"] += 1
            continue

        mask = [1.0 if float(v) > 0 else 0.0 for v in action_mask]
        if sum(mask) <= 0 or mask[action_index] <= 0:
            skipped["invalid_action"] += 1
            continue

        obs_rows.append([float(v) for v in obs])
        mask_rows.append(mask)
        action_rows.append(action_index)
        if _is_finite_number(return_to_go):
            return_rows.append(float(return_to_go))
            return_mask_rows.append(1.0)
        else:
            return_rows.append(0.0)
            return_mask_rows.append(0.0)

    if len(obs_rows) == 0:
        raise ValueError("BC dataset produced zero compatible records.")

    payload = {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "masks": np.asarray(mask_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.int64),
        "returns": np.asarray(return_rows, dtype=np.float32),
        "returns_mask": np.asarray(return_mask_rows, dtype=np.float32),
    }
    stats = {
        "total_records": len(records),
        "used_records": int(payload["obs"].shape[0]),
        "with_returns": int(np.sum(payload["returns_mask"])),
        **skipped,
    }
    return payload, stats


def run_bc_pretrain(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    cfg: PPOConfig,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    if not cfg.bc_dataset or cfg.bc_epochs <= 0:
        return {"enabled": False}

    dataset_path = Path(cfg.bc_dataset).resolve()
    batch, dataset_stats = load_bc_dataset(
        dataset_path=dataset_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_records=cfg.bc_max_records,
    )

    obs_t = as_tensor(batch["obs"], device)
    masks_t = as_tensor(batch["masks"], device)
    actions_t = torch.from_numpy(batch["actions"]).to(device)
    returns_t = as_tensor(batch["returns"], device)
    returns_mask_t = as_tensor(batch["returns_mask"], device)
    returns_target_t = returns_t.clone()

    valid_return_count = int(torch.sum(returns_mask_t).detach().cpu().item())
    return_norm_mean = 0.0
    return_norm_std = 1.0
    return_clip_used = (
        float(cfg.bc_return_clip)
        if math.isfinite(cfg.bc_return_clip) and cfg.bc_return_clip > 0
        else None
    )
    if valid_return_count > 0:
        valid_mask_bool = returns_mask_t > 0.5
        valid_returns = returns_t[valid_mask_bool]
        if cfg.bc_normalize_returns:
            return_norm_mean = float(valid_returns.mean().detach().cpu().item())
            return_norm_std = float(
                valid_returns.std(unbiased=False).detach().cpu().item()
            )
            if not math.isfinite(return_norm_std) or return_norm_std < 1e-6:
                return_norm_std = 1.0
            returns_target_t[valid_mask_bool] = (
                valid_returns - return_norm_mean
            ) / return_norm_std
        if return_clip_used is not None:
            returns_target_t[valid_mask_bool] = torch.clamp(
                returns_target_t[valid_mask_bool],
                -return_clip_used,
                return_clip_used,
            )

    sample_count = int(obs_t.shape[0])
    batch_size = min(cfg.bc_batch_size, sample_count)
    optimizer = torch.optim.Adam(
        trainable_parameters(model, obs_adapter), lr=cfg.bc_learning_rate, eps=1e-5
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_adapter_state: dict[str, torch.Tensor] | None = None
    epoch_logs: list[dict[str, float]] = []
    print(
        "[bc] "
        f"returns preprocess: normalize={'y' if cfg.bc_normalize_returns else 'n'} "
        f"clip={return_clip_used if return_clip_used is not None else 'off'} "
        f"valid_targets={valid_return_count} "
        f"mean={return_norm_mean:.4f} std={return_norm_std:.4f}"
    )

    for epoch in range(1, cfg.bc_epochs + 1):
        perm = torch.randperm(sample_count, device=device)
        actor_loss_sum = 0.0
        value_loss_sum = 0.0
        total_loss_sum = 0.0
        batch_count = 0

        for start in range(0, sample_count, batch_size):
            idx = perm[start : start + batch_size]
            obs_features = obs_adapter(obs_t[idx])
            logits, values = model(obs_features)
            dist = masked_categorical(logits, masks_t[idx])
            actor_loss = -dist.log_prob(actions_t[idx]).mean()

            value_loss = torch.zeros((), device=device)
            if cfg.bc_value_weight > 0:
                mb_return_mask = returns_mask_t[idx]
                valid_returns = torch.sum(mb_return_mask)
                if float(valid_returns.detach().cpu().item()) > 0:
                    sq_err = (values - returns_target_t[idx]) ** 2
                    value_loss = 0.5 * torch.sum(sq_err * mb_return_mask) / valid_returns

            total_loss = actor_loss + cfg.bc_value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                trainable_parameters(model, obs_adapter), cfg.max_grad_norm
            )
            optimizer.step()

            actor_loss_sum += float(actor_loss.detach().cpu().item())
            value_loss_sum += float(value_loss.detach().cpu().item())
            total_loss_sum += float(total_loss.detach().cpu().item())
            batch_count += 1

        denom = max(1, batch_count)
        mean_actor_loss = actor_loss_sum / denom
        mean_value_loss = value_loss_sum / denom
        mean_total_loss = total_loss_sum / denom

        epoch_log = {
            "epoch": float(epoch),
            "actor_loss": mean_actor_loss,
            "value_loss": mean_value_loss,
            "total_loss": mean_total_loss,
        }
        epoch_logs.append(epoch_log)
        print(
            "[bc] "
            f"epoch={epoch}/{cfg.bc_epochs} "
            f"actor={mean_actor_loss:.4f} "
            f"value={mean_value_loss:.4f} "
            f"total={mean_total_loss:.4f}"
        )

        if math.isfinite(mean_total_loss) and mean_total_loss < best_loss:
            best_loss = mean_total_loss
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            best_adapter_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in obs_adapter.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_adapter_state is not None:
        obs_adapter.load_state_dict(best_adapter_state, strict=False)

    return {
        "enabled": True,
        "dataset_path": str(dataset_path),
        "epochs": cfg.bc_epochs,
        "batch_size": batch_size,
        "learning_rate": cfg.bc_learning_rate,
        "value_weight": cfg.bc_value_weight,
        "normalize_returns": cfg.bc_normalize_returns,
        "return_norm_mean": return_norm_mean,
        "return_norm_std": return_norm_std,
        "return_clip_used": return_clip_used,
        "returns_with_targets": valid_return_count,
        "dataset_stats": dataset_stats,
        "best_total_loss": best_loss,
        "epoch_logs": epoch_logs,
    }


def train(cfg: PPOConfig) -> None:
    set_global_seed(cfg.seed)
    device = choose_device(cfg.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(cfg.out_dir).resolve() / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    server_cmd = (
        shlex.split(cfg.server_cmd.strip())
        if cfg.server_cmd and cfg.server_cmd.strip()
        else None
    )

    with WubEnvBridge(server_cmd=server_cmd, cwd=repo_root) as env:
        init_result = env.init(
            mode_id=cfg.mode_id,
            num_envs=cfg.num_envs,
            model_path=cfg.model_path,
            observation_space=cfg.observation_space,
            placement_execution_mode=cfg.placement_execution_mode,
            piece_source_profile=cfg.piece_source_profile,
            queue_policy_id=cfg.queue_policy_id,
            max_pieces_per_episode=cfg.max_pieces_per_episode,
            seed=cfg.seed,
        )
        env_ids: list[int] = [int(v) for v in init_result.get("env_ids", [])]
        if not env_ids:
            raise RuntimeError("Bridge init returned no env ids.")
        if len(env_ids) != cfg.num_envs:
            raise RuntimeError(
                f"Bridge init env count mismatch. expected={cfg.num_envs} got={len(env_ids)}"
            )

        reset_seeds = [cfg.seed + i * 101 for i in range(cfg.num_envs)]
        reset_result = env.reset_many(env_ids=env_ids, seeds=reset_seeds)
        obs_np = np.asarray(reset_result["obs"], dtype=np.float32)
        mask_repair_total = {"rows": 0, "batches": 0}
        mask_np = ensure_action_masks(
            np.asarray(reset_result["action_masks"], dtype=np.float32),
            repair_stats=mask_repair_total,
        )
        action_bias_np = ensure_action_biases(
            mask_np,
            np.asarray(reset_result.get("action_biases", []), dtype=np.float32)
            if "action_biases" in reset_result
            else None,
        )
        action_score_np = ensure_action_scores(
            mask_np,
            np.asarray(reset_result.get("action_scores", []), dtype=np.float32)
            if "action_scores" in reset_result
            else None,
        )
        current_piece_source = (
            "bag7"
            if cfg.piece_source_profile == "bag7"
            else "active_generator"
        )
        raw_obs_dim = infer_obs_dim(reset_result["obs"])
        action_dim = infer_action_dim(reset_result["action_masks"])

        if cfg.observation_space == "raw_v1":
            obs_adapter: ObservationAdapter = WubHeadFromRawObservationAdapter(
                raw_obs_dim=raw_obs_dim,
                model_path=cfg.model_path,
            ).to(device)
        else:
            obs_adapter = IdentityObservationAdapter(
                raw_obs_dim=raw_obs_dim,
                policy_observation_space=cfg.observation_space,
            ).to(device)
        obs_adapter.train()
        obs_dim = int(obs_adapter.policy_obs_dim)
        policy_observation_space = obs_adapter.policy_observation_space

        model = PolicyValueNet(obs_dim, cfg.hidden_dim, action_dim).to(device)
        optimizer: torch.optim.Optimizer

        global_step = 0
        start_update = 0
        if cfg.resume_checkpoint:
            optimizer = build_ppo_optimizer(model, obs_adapter, cfg.learning_rate)
            checkpoint_path = Path(cfg.resume_checkpoint).resolve()
            global_step, start_update = load_checkpoint(
                checkpoint_path, model, obs_adapter, optimizer, device
            )
            print(
                f"[ppo] resumed checkpoint: {checkpoint_path} "
                f"(global_step={global_step}, update={start_update})"
            )
        elif cfg.init_artifact:
            artifact_path = Path(cfg.init_artifact).resolve()
            artifact_observation_space = load_from_artifact(
                model, obs_adapter, artifact_path
            )
            if artifact_observation_space != policy_observation_space:
                raise ValueError(
                    "Init artifact observationSpace mismatch. "
                    f"artifact={artifact_observation_space} policy={policy_observation_space}"
                )
            print(f"[ppo] initialized from bot artifact: {artifact_path}")

        encoder_frozen_for_ppo = False
        encoder_freeze_mode_applied = "none"

        if not cfg.resume_checkpoint:
            bc_stats = run_bc_pretrain(
                model=model,
                obs_adapter=obs_adapter,
                cfg=cfg,
                device=device,
                obs_dim=raw_obs_dim,
                action_dim=action_dim,
            )
            if bc_stats.get("enabled"):
                write_json(out_dir / "bc_stats.json", bc_stats)
                post_bc_artifact = export_bot_policy_artifact(
                    model,
                    obs_adapter,
                    cfg,
                    obs_dim,
                    action_dim,
                    policy_observation_space,
                    pipeline_id="bot_ppo_offline_post_bc_v1",
                )
                post_bc_artifact_path = artifacts_dir / "bot_policy_post_bc.json"
                write_json(post_bc_artifact_path, post_bc_artifact)

                snapshot_seed = cfg.seed + 900_001
                rollout = capture_single_policy_rollout(
                    env=env,
                    env_id=env_ids[0],
                    seed=snapshot_seed,
                    model=model,
                    obs_adapter=obs_adapter,
                    device=device,
                    max_steps=max(1, cfg.max_pieces_per_episode * 4),
                    deterministic=True,
                )
                post_bc_trajectory_path: Path | None = None
                trajectory_payload = rollout.get("trajectory")
                if isinstance(trajectory_payload, dict):
                    post_bc_trajectory_path = trajectories_dir / "trajectory_post_bc.json"
                    write_json_any(post_bc_trajectory_path, trajectory_payload)

                post_bc_stats = {
                    "enabled": True,
                    "seed": int(snapshot_seed),
                    "policy_path": str(post_bc_artifact_path),
                    "trajectory_path": (
                        str(post_bc_trajectory_path)
                        if post_bc_trajectory_path is not None
                        else None
                    ),
                    "trajectory_drained": int(rollout.get("drained", 0)),
                    "episode_return": float(rollout.get("episode_return", 0.0)),
                    "episode_length": int(rollout.get("episode_length", 0)),
                    "episode_done": bool(rollout.get("done", False)),
                    "trajectory_session_id": (
                        trajectory_payload.get("sessionId")
                        if isinstance(trajectory_payload, dict)
                        else None
                    ),
                    "trajectory_samples": (
                        len(trajectory_payload.get("samples", []))
                        if isinstance(trajectory_payload, dict)
                        and isinstance(trajectory_payload.get("samples"), list)
                        else 0
                    ),
                }
                write_json(out_dir / "post_bc_stats.json", post_bc_stats)
                print(
                    "[ppo] "
                    f"post_bc snapshot saved "
                    f"(policy={post_bc_artifact_path.name}, "
                    f"trajectory={'yes' if post_bc_trajectory_path is not None else 'no'}, "
                    f"ret={post_bc_stats['episode_return']:.3f}, "
                    f"len={post_bc_stats['episode_length']})"
                )
            if cfg.freeze_encoder_after_bc or cfg.freeze_conv_after_bc:
                if bc_stats.get("enabled"):
                    if cfg.freeze_encoder_after_bc:
                        encoder_param_total = 0
                        encoder_param_trainable = 0
                        for param in obs_adapter.parameters():
                            param_count = int(param.numel())
                            encoder_param_total += param_count
                            if param.requires_grad:
                                encoder_param_trainable += param_count
                            param.requires_grad = False
                        encoder_frozen_for_ppo = encoder_param_total > 0
                        if encoder_frozen_for_ppo:
                            encoder_freeze_mode_applied = "all"
                            print(
                                "[ppo] encoder frozen after BC "
                                f"(params={encoder_param_total}, trainable_before={encoder_param_trainable})"
                            )
                        else:
                            print(
                                "[ppo] encoder freeze requested after BC, "
                                "but no trainable encoder params were found."
                            )
                    elif cfg.freeze_conv_after_bc:
                        if isinstance(obs_adapter, WubHeadFromRawObservationAdapter):
                            conv_param_total = 0
                            conv_param_trainable = 0
                            for conv in obs_adapter.conv_layers:
                                for param in conv.parameters():
                                    param_count = int(param.numel())
                                    conv_param_total += param_count
                                    if param.requires_grad:
                                        conv_param_trainable += param_count
                                    param.requires_grad = False
                            encoder_frozen_for_ppo = conv_param_total > 0
                            if encoder_frozen_for_ppo:
                                encoder_freeze_mode_applied = "conv"
                                print(
                                    "[ppo] encoder conv frozen after BC "
                                    f"(params={conv_param_total}, trainable_before={conv_param_trainable})"
                                )
                            else:
                                print(
                                    "[ppo] conv freeze requested after BC, "
                                    "but no trainable conv params were found."
                                )
                        else:
                            print(
                                "[ppo] conv freeze requested after BC, "
                                "but observation adapter has no conv layers."
                            )
                else:
                    freeze_label = (
                        "encoder"
                        if cfg.freeze_encoder_after_bc
                        else "encoder conv"
                    )
                    print(
                        f"[ppo] {freeze_label} freeze requested after BC, "
                        "but BC was skipped/disabled."
                    )
            optimizer = build_ppo_optimizer(model, obs_adapter, cfg.learning_rate)

            # Reinitialize env batch after post-BC snapshot capture so PPO
            # always starts from a clean synchronized state.
            reset_seeds = [cfg.seed + i * 101 for i in range(cfg.num_envs)]
            reset_result = env.reset_many(env_ids=env_ids, seeds=reset_seeds)
            obs_np = np.asarray(reset_result["obs"], dtype=np.float32)
            mask_np = ensure_action_masks(
                np.asarray(reset_result["action_masks"], dtype=np.float32),
                repair_stats=mask_repair_total,
            )
            action_bias_np = ensure_action_biases(
                mask_np,
                np.asarray(reset_result.get("action_biases", []), dtype=np.float32)
                if "action_biases" in reset_result
                else None,
            )
            action_score_np = ensure_action_scores(
                mask_np,
                np.asarray(reset_result.get("action_scores", []), dtype=np.float32)
                if "action_scores" in reset_result
                else None,
            )

        batch_size = cfg.num_envs * cfg.num_steps
        if cfg.minibatch_size > batch_size:
            raise ValueError(
                f"minibatch_size ({cfg.minibatch_size}) exceeds batch_size ({batch_size})."
            )
        num_updates = max(1, cfg.total_timesteps // batch_size)

        print(
            "[ppo] starting training "
            f"(device={device.type}, env_obs_space={cfg.observation_space}, "
            f"placement_exec={cfg.placement_execution_mode}, "
            f"policy_obs_space={policy_observation_space}, "
            f"raw_obs_dim={raw_obs_dim}, policy_obs_dim={obs_dim}, action_dim={action_dim}, "
            f"batch_size={batch_size}, updates={num_updates}, "
            f"piece_source_base={cfg.piece_source_profile}, "
            f"alternate_sources={'y' if cfg.alternate_piece_sources else 'n'}, "
            f"policy_clip_coef={cfg.policy_clip_coef:.4f}, "
            f"value_clip_coef={cfg.value_clip_coef:.4f}, "
            f"warmup_policy_lr_scale={cfg.warmup_policy_lr_scale:.3f}, "
            f"warmup_value_lr_scale={cfg.warmup_value_lr_scale:.3f}, "
            f"curriculum_topk={cfg.curriculum_topk_start}->{cfg.curriculum_topk_end}/"
            f"{cfg.curriculum_topk_ramp_updates}, "
            f"curriculum_bias={cfg.curriculum_bias_start:.3f}->{cfg.curriculum_bias_end:.3f}/"
            f"{cfg.curriculum_bias_ramp_updates}, "
            f"curriculum_danger_height={cfg.curriculum_danger_height}, "
            f"distill_coef={cfg.distill_coef_start:.4f}->{cfg.distill_coef_end:.4f}/"
            f"{cfg.distill_coef_ramp_updates}, "
            f"distill_teacher_tau={cfg.distill_teacher_tau:.4f}, "
            f"obs_adapter_lr_scale={PPO_OBS_ADAPTER_LR_SCALE:.3f}, "
            f"encoder_frozen_after_bc={'y' if encoder_frozen_for_ppo else 'n'}, "
            f"encoder_freeze_mode={encoder_freeze_mode_applied})"
        )
        write_json(
            out_dir / "config.json",
            {
                **cfg.__dict__,
                "device_resolved": device.type,
                "raw_obs_dim": raw_obs_dim,
                "policy_obs_dim": obs_dim,
                "policy_observation_space": policy_observation_space,
                "observation_adapter": obs_adapter.__class__.__name__,
                "encoder_frozen_after_bc_applied": encoder_frozen_for_ppo,
                "encoder_freeze_mode_applied": encoder_freeze_mode_applied,
                "action_dim": action_dim,
                "batch_size": batch_size,
                "num_updates": num_updates,
            },
        )

        # Episode trackers for logging.
        ep_return = np.zeros(cfg.num_envs, dtype=np.float64)
        ep_length = np.zeros(cfg.num_envs, dtype=np.int64)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        reward_component_aliases: dict[str, tuple[str, ...]] = {
            "reward_final": ("rewardFinal",),
            "reward_base": ("rewardBase",),
            "top_out_penalty": ("topOutPenalty",),
            "term_lines": ("rewardTermLines",),
            "term_score": ("rewardTermScore",),
            "term_time": ("rewardTermTime",),
            "term_height": ("rewardTermHeight",),
            "term_holes": ("rewardTermHoles",),
            "term_bumpiness": ("rewardTermBumpiness",),
            "term_board_score": ("rewardTermBoardScore",),
            "term_board_quality": ("rewardTermBoardQuality",),
        }
        ep_reward_component_sums = {
            key: np.zeros(cfg.num_envs, dtype=np.float64)
            for key in reward_component_aliases
        }
        completed_reward_component_sums: dict[str, list[float]] = {
            key: [] for key in reward_component_aliases
        }

        training_start = time.time()
        best_score = float("-inf")
        best_update = 0
        best_stats: dict[str, Any] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_adapter_state_dict: dict[str, torch.Tensor] | None = None
        did_interrupt = False
        try:
            for update in range(start_update + 1, num_updates + 1):
                warmup_active = cfg.warmup_updates > 0 and update <= cfg.warmup_updates
                ent_coef_now = (
                    cfg.warmup_ent_coef if warmup_active else cfg.ent_coef
                )
                target_kl_now = (
                    cfg.warmup_target_kl if warmup_active else cfg.target_kl
                )
                policy_lr_now, value_lr_now = apply_warmup_lr_schedule(
                    optimizer=optimizer,
                    base_lr=cfg.learning_rate,
                    warmup_active=warmup_active,
                    warmup_policy_lr_scale=cfg.warmup_policy_lr_scale,
                    warmup_value_lr_scale=cfg.warmup_value_lr_scale,
                )
                update_start_wall = time.time()
                update_start_perf = time.perf_counter()
                profile_policy_forward_s = 0.0
                profile_env_step_s = 0.0
                profile_env_reset_s = 0.0
                profile_env_step_batch_s = 0.0
                profile_env_step_core_s = 0.0
                profile_env_step_choices_current_s = 0.0
                profile_env_step_choices_next_s = 0.0
                profile_env_step_runner_s = 0.0
                profile_env_step_reward_s = 0.0
                profile_env_step_obs_s = 0.0
                profile_env_reset_batch_s = 0.0
                profile_env_reset_core_s = 0.0
                profile_env_reset_obs_s = 0.0
                profile_env_reset_choices_s = 0.0
                profile_gae_s = 0.0
                profile_opt_s = 0.0
                profile_io_s = 0.0
                mask_repair_update = {"rows": 0, "batches": 0}
                curriculum_topk_now = curriculum_topk_for_update(cfg, update)
                curriculum_bias_now = curriculum_bias_for_update(cfg, update)
                distill_coef_now = distill_coef_for_update(cfg, update)
                env.set_curriculum(
                    top_k=curriculum_topk_now,
                    bias_strength=curriculum_bias_now,
                    danger_height=cfg.curriculum_danger_height,
                )

                desired_piece_source = resolve_piece_source_for_update(cfg, update)
                if desired_piece_source != current_piece_source:
                    switch_result = env.set_piece_source(desired_piece_source)
                    current_piece_source = str(
                        switch_result.get("piece_source_profile", desired_piece_source)
                    )
                    source_reset_seeds = [
                        cfg.seed + update * 100_003 + i * 101 for i in range(cfg.num_envs)
                    ]
                    env_reset_start = time.perf_counter()
                    source_reset = env.reset_many(env_ids=env_ids, seeds=source_reset_seeds)
                    profile_env_reset_s += time.perf_counter() - env_reset_start
                    reset_profile = source_reset.get("profile", {})
                    profile_env_reset_batch_s += _profile_num(
                        reset_profile, "batch_total_s"
                    )
                    profile_env_reset_core_s += _profile_num(
                        reset_profile, "reset_env_total_s"
                    )
                    profile_env_reset_obs_s += _profile_num(
                        reset_profile, "reset_obs_s"
                    )
                    profile_env_reset_choices_s += _profile_num(
                        reset_profile, "reset_choices_s"
                    )
                    obs_np = np.asarray(source_reset["obs"], dtype=np.float32)
                    mask_np = ensure_action_masks(
                        np.asarray(source_reset["action_masks"], dtype=np.float32),
                        repair_stats=mask_repair_update,
                    )
                    action_bias_np = ensure_action_biases(
                        mask_np,
                        np.asarray(source_reset.get("action_biases", []), dtype=np.float32)
                        if "action_biases" in source_reset
                        else None,
                    )
                    action_score_np = ensure_action_scores(
                        mask_np,
                        np.asarray(source_reset.get("action_scores", []), dtype=np.float32)
                        if "action_scores" in source_reset
                        else None,
                    )
                    ep_return.fill(0.0)
                    ep_length.fill(0)
                    for component_sum in ep_reward_component_sums.values():
                        component_sum.fill(0.0)

                rollout_start_perf = time.perf_counter()
                raw_obs_buf = torch.zeros(
                    (cfg.num_steps, cfg.num_envs, raw_obs_dim), device=device
                )
                mask_buf = torch.zeros((cfg.num_steps, cfg.num_envs, action_dim), device=device)
                action_bias_buf = torch.zeros(
                    (cfg.num_steps, cfg.num_envs, action_dim), device=device
                )
                action_score_buf = torch.zeros(
                    (cfg.num_steps, cfg.num_envs, action_dim), device=device
                )
                action_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device, dtype=torch.long)
                logprob_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                reward_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                done_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                value_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

                for step in range(cfg.num_steps):
                    raw_obs_t = as_tensor(obs_np, device)
                    mask_t = as_tensor(mask_np, device)
                    action_bias_t = as_tensor(action_bias_np, device)
                    action_score_t = as_tensor(action_score_np, device)
                    raw_obs_buf[step] = raw_obs_t
                    mask_buf[step] = mask_t
                    action_bias_buf[step] = action_bias_t
                    action_score_buf[step] = action_score_t

                    policy_forward_start = time.perf_counter()
                    with torch.no_grad():
                        obs_t = obs_adapter(raw_obs_t)
                        logits, values = model(obs_t)
                        dist = masked_categorical(
                            logits,
                            mask_t,
                            action_bias=action_bias_t,
                            bias_alpha=curriculum_bias_now,
                        )
                        if cfg.deterministic_eval:
                            actions_t = torch.argmax(dist.probs, dim=-1)
                        else:
                            actions_t = dist.sample()
                        logprob_t = dist.log_prob(actions_t)
                    profile_policy_forward_s += time.perf_counter() - policy_forward_start

                    action_buf[step] = actions_t
                    logprob_buf[step] = logprob_t
                    value_buf[step] = values

                    actions_np = actions_t.detach().cpu().numpy().astype(np.int64).tolist()
                    env_step_start = time.perf_counter()
                    step_result = env.step_many(env_ids=env_ids, actions=actions_np)
                    profile_env_step_s += time.perf_counter() - env_step_start
                    step_profile = step_result.get("profile", {})
                    profile_env_step_batch_s += _profile_num(step_profile, "batch_total_s")
                    profile_env_step_core_s += _profile_num(step_profile, "step_env_total_s")
                    profile_env_step_choices_current_s += _profile_num(
                        step_profile, "step_choices_current_s"
                    )
                    profile_env_step_choices_next_s += _profile_num(
                        step_profile, "step_choices_next_s"
                    )
                    profile_env_step_runner_s += _profile_num(step_profile, "step_runner_s")
                    profile_env_step_reward_s += _profile_num(step_profile, "step_reward_s")
                    profile_env_step_obs_s += _profile_num(step_profile, "step_obs_s")
                    next_obs_np = np.asarray(step_result["obs"], dtype=np.float32)
                    next_mask_np = ensure_action_masks(
                        np.asarray(step_result["action_masks"], dtype=np.float32),
                        repair_stats=mask_repair_update,
                    )
                    next_action_bias_np = ensure_action_biases(
                        next_mask_np,
                        np.asarray(step_result.get("action_biases", []), dtype=np.float32)
                        if "action_biases" in step_result
                        else None,
                    )
                    next_action_score_np = ensure_action_scores(
                        next_mask_np,
                        np.asarray(step_result.get("action_scores", []), dtype=np.float32)
                        if "action_scores" in step_result
                        else None,
                    )
                    rewards_np = np.asarray(step_result["rewards"], dtype=np.float32)
                    infos_raw = step_result.get("infos", [])
                    dones_np = np.asarray(step_result["dones"], dtype=np.float32)

                    reward_buf[step] = as_tensor(rewards_np, device)
                    done_buf[step] = as_tensor(dones_np, device)

                    ep_return += rewards_np.astype(np.float64)
                    ep_length += 1
                    if isinstance(infos_raw, list):
                        max_info = min(len(infos_raw), cfg.num_envs)
                        for env_idx in range(max_info):
                            info = infos_raw[env_idx]
                            reward_final = _info_num(
                                info,
                                reward_component_aliases["reward_final"],
                                default=float(rewards_np[env_idx]),
                            )
                            ep_reward_component_sums["reward_final"][
                                env_idx
                            ] += reward_final
                            for key in (
                                "reward_base",
                                "top_out_penalty",
                                "term_lines",
                                "term_score",
                                "term_time",
                                "term_height",
                                "term_holes",
                                "term_bumpiness",
                                "term_board_score",
                                "term_board_quality",
                            ):
                                ep_reward_component_sums[key][env_idx] += _info_num(
                                    info,
                                    reward_component_aliases[key],
                                    default=0.0,
                                )
                        if max_info < cfg.num_envs:
                            ep_reward_component_sums["reward_final"][
                                max_info:cfg.num_envs
                            ] += rewards_np[max_info:cfg.num_envs].astype(np.float64)
                    else:
                        ep_reward_component_sums["reward_final"] += rewards_np.astype(
                            np.float64
                        )

                    done_indices = np.where(dones_np > 0.5)[0]
                    if done_indices.size > 0:
                        completed_returns.extend(ep_return[done_indices].tolist())
                        completed_lengths.extend(ep_length[done_indices].tolist())
                        for key, sums in ep_reward_component_sums.items():
                            completed_reward_component_sums[key].extend(
                                sums[done_indices].tolist()
                            )
                        ep_return[done_indices] = 0.0
                        ep_length[done_indices] = 0
                        for sums in ep_reward_component_sums.values():
                            sums[done_indices] = 0.0

                        done_env_ids = [env_ids[int(i)] for i in done_indices.tolist()]
                        done_seeds = [
                            cfg.seed + global_step + int(i) * 991 + update * 131
                            for i in done_indices.tolist()
                        ]
                        env_reset_start = time.perf_counter()
                        reset_done = env.reset_many(env_ids=done_env_ids, seeds=done_seeds)
                        profile_env_reset_s += time.perf_counter() - env_reset_start
                        reset_profile = reset_done.get("profile", {})
                        profile_env_reset_batch_s += _profile_num(
                            reset_profile, "batch_total_s"
                        )
                        profile_env_reset_core_s += _profile_num(
                            reset_profile, "reset_env_total_s"
                        )
                        profile_env_reset_obs_s += _profile_num(
                            reset_profile, "reset_obs_s"
                        )
                        profile_env_reset_choices_s += _profile_num(
                            reset_profile, "reset_choices_s"
                        )
                        reset_obs = np.asarray(reset_done["obs"], dtype=np.float32)
                        reset_masks = ensure_action_masks(
                            np.asarray(reset_done["action_masks"], dtype=np.float32),
                            repair_stats=mask_repair_update,
                        )
                        reset_action_biases = ensure_action_biases(
                            reset_masks,
                            np.asarray(reset_done.get("action_biases", []), dtype=np.float32)
                            if "action_biases" in reset_done
                            else None,
                        )
                        reset_action_scores = ensure_action_scores(
                            reset_masks,
                            np.asarray(reset_done.get("action_scores", []), dtype=np.float32)
                            if "action_scores" in reset_done
                            else None,
                        )
                        for local_pos, env_idx in enumerate(done_indices.tolist()):
                            next_obs_np[env_idx] = reset_obs[local_pos]
                            next_mask_np[env_idx] = reset_masks[local_pos]
                            next_action_bias_np[env_idx] = reset_action_biases[local_pos]
                            next_action_score_np[env_idx] = reset_action_scores[local_pos]

                    obs_np = next_obs_np
                    mask_np = next_mask_np
                    action_bias_np = next_action_bias_np
                    action_score_np = next_action_score_np
                    global_step += cfg.num_envs

                profile_rollout_s = time.perf_counter() - rollout_start_perf

                gae_start = time.perf_counter()
                with torch.no_grad():
                    next_raw_obs_t = as_tensor(obs_np, device)
                    next_obs_t = obs_adapter(next_raw_obs_t)
                    _, next_value = model(next_obs_t)

                advantages = torch.zeros_like(reward_buf, device=device)
                lastgaelam = torch.zeros(cfg.num_envs, device=device)
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        next_non_terminal = 1.0 - done_buf[t]
                        next_values = next_value
                    else:
                        # done_buf[t] marks whether transition t ended the episode.
                        # Use done_t for both TD bootstrap and GAE recursion masks.
                        next_non_terminal = 1.0 - done_buf[t]
                        next_values = value_buf[t + 1]
                    delta = reward_buf[t] + cfg.gamma * next_values * next_non_terminal - value_buf[t]
                    lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + value_buf

                b_raw_obs = raw_obs_buf.reshape((-1, raw_obs_dim))
                b_masks = mask_buf.reshape((-1, action_dim))
                b_action_bias = action_bias_buf.reshape((-1, action_dim))
                b_action_scores = action_score_buf.reshape((-1, action_dim))
                b_actions = action_buf.reshape(-1)
                b_logprobs = logprob_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = value_buf.reshape(-1)

                adv_mean = b_advantages.mean()
                adv_std = b_advantages.std(unbiased=False) + 1e-8
                b_advantages = (b_advantages - adv_mean) / adv_std
                profile_gae_s = time.perf_counter() - gae_start

                batch_inds = np.arange(batch_size)
                clipfracs: list[float] = []
                approx_kl_value = 0.0
                policy_loss_value = 0.0
                value_loss_value = 0.0
                entropy_value = 0.0
                distill_loss_value = 0.0
                total_loss_value = 0.0
                policy_term_value = 0.0
                value_term_value = 0.0
                entropy_term_value = 0.0
                distill_term_value = 0.0
                policy_loss_sum = 0.0
                value_loss_sum = 0.0
                entropy_sum = 0.0
                distill_loss_sum = 0.0
                total_loss_sum = 0.0
                policy_term_sum = 0.0
                value_term_sum = 0.0
                entropy_term_sum = 0.0
                distill_term_sum = 0.0
                teacher_entropy_value = 0.0
                teacher_max_prob_value = 0.0
                teacher_uniform_row_frac_value = 0.0
                teacher_bias_row_frac_value = 0.0
                teacher_prob_sum_value = 0.0
                teacher_valid_actions_value = 0.0
                teacher_entropy_sum = 0.0
                teacher_max_prob_sum = 0.0
                teacher_uniform_row_frac_sum = 0.0
                teacher_bias_row_frac_sum = 0.0
                teacher_prob_sum_sum = 0.0
                teacher_valid_actions_sum = 0.0
                updates_done = 0
                early_stopped = False
                early_stop_epoch: int | None = None
                approx_kl_values: list[float] = []

                optimize_start = time.perf_counter()
                for _epoch in range(cfg.update_epochs):
                    epoch_approx_kl_values: list[float] = []
                    np.random.shuffle(batch_inds)
                    for start in range(0, batch_size, cfg.minibatch_size):
                        end = start + cfg.minibatch_size
                        mb_inds_np = batch_inds[start:end]
                        mb_inds = torch.as_tensor(
                            mb_inds_np, device=device, dtype=torch.long
                        )

                        mb_features = obs_adapter(b_raw_obs[mb_inds])
                        logits, new_values = model(mb_features)
                        mb_mask = b_masks[mb_inds]
                        mb_action_bias = b_action_bias[mb_inds]
                        mb_action_scores = b_action_scores[mb_inds]
                        dist = masked_categorical(
                            logits,
                            mb_mask,
                            action_bias=mb_action_bias,
                            bias_alpha=curriculum_bias_now,
                        )
                        new_logprob = dist.log_prob(b_actions[mb_inds])
                        entropy = dist.entropy().mean()

                        logratio = new_logprob - b_logprobs[mb_inds]
                        ratio = torch.exp(logratio)

                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - logratio).mean()
                            clipfrac = (
                                (ratio - 1.0).abs() > cfg.policy_clip_coef
                            ).float().mean()
                            approx_kl_value = float(approx_kl.detach().cpu().item())
                            approx_kl_values.append(approx_kl_value)
                            epoch_approx_kl_values.append(approx_kl_value)
                            clipfracs.append(float(clipfrac.detach().cpu().item()))

                        mb_adv = b_advantages[mb_inds]
                        pg_loss_1 = -mb_adv * ratio
                        pg_loss_2 = -mb_adv * torch.clamp(
                            ratio,
                            1.0 - cfg.policy_clip_coef,
                            1.0 + cfg.policy_clip_coef,
                        )
                        policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                        value_pred = new_values
                        if cfg.clip_vloss:
                            value_pred_clipped = b_values[mb_inds] + (
                                value_pred - b_values[mb_inds]
                            ).clamp(-cfg.value_clip_coef, cfg.value_clip_coef)
                            value_losses = (value_pred - b_returns[mb_inds]) ** 2
                            value_losses_clipped = (
                                value_pred_clipped - b_returns[mb_inds]
                            ) ** 2
                            value_loss = (
                                0.5
                                * torch.max(value_losses, value_losses_clipped).mean()
                            )
                        else:
                            value_loss = (
                                0.5 * ((value_pred - b_returns[mb_inds]) ** 2).mean()
                            )

                        # Distillation: teacher from heuristic action_bias, student from
                        # raw masked policy logits (no heuristic prior injection).
                        student_valid = mb_mask > 0
                        student_large_neg = torch.full_like(logits, -1e9)
                        student_logits = torch.where(
                            student_valid, logits, student_large_neg
                        )
                        student_log_probs = torch.log_softmax(
                            student_logits, dim=-1
                        )
                        teacher_probs, teacher_uniform_rows = build_teacher_probs(
                            mb_mask,
                            mb_action_scores,
                            cfg.distill_teacher_tau,
                        )
                        distill_loss = -torch.sum(
                            teacher_probs * student_log_probs, dim=-1
                        ).mean()

                        policy_term = policy_loss
                        value_term = cfg.vf_coef * value_loss
                        entropy_term = -ent_coef_now * entropy
                        distill_term = distill_coef_now * distill_loss
                        loss = (
                            policy_term
                            + value_term
                            + entropy_term
                            + distill_term
                        )

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        scale_obs_adapter_gradients(
                            obs_adapter, PPO_OBS_ADAPTER_LR_SCALE
                        )
                        nn.utils.clip_grad_norm_(
                            trainable_parameters(model, obs_adapter), cfg.max_grad_norm
                        )
                        optimizer.step()

                        policy_loss_value = float(policy_loss.detach().cpu().item())
                        value_loss_value = float(value_loss.detach().cpu().item())
                        entropy_value = float(entropy.detach().cpu().item())
                        distill_loss_value = float(
                            distill_loss.detach().cpu().item()
                        )
                        policy_term_value = float(policy_term.detach().cpu().item())
                        value_term_value = float(value_term.detach().cpu().item())
                        entropy_term_value = float(entropy_term.detach().cpu().item())
                        distill_term_value = float(distill_term.detach().cpu().item())
                        total_loss_value = float(loss.detach().cpu().item())
                        policy_loss_sum += policy_loss_value
                        value_loss_sum += value_loss_value
                        entropy_sum += entropy_value
                        distill_loss_sum += distill_loss_value
                        policy_term_sum += policy_term_value
                        value_term_sum += value_term_value
                        entropy_term_sum += entropy_term_value
                        distill_term_sum += distill_term_value
                        total_loss_sum += total_loss_value
                        valid_f = (mb_mask > 0).to(dtype=teacher_probs.dtype)
                        teacher_uniform_row_frac_value = float(
                            teacher_uniform_rows.mean().detach().cpu().item()
                        )
                        teacher_bias_row_frac_value = 1.0 - teacher_uniform_row_frac_value
                        teacher_entropy_value = float(
                            (
                                -torch.sum(
                                    teacher_probs * torch.log(torch.clamp(teacher_probs, min=1e-8)),
                                    dim=-1,
                                ).mean()
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        teacher_max_prob_value = float(
                            teacher_probs.max(dim=-1).values.mean().detach().cpu().item()
                        )
                        teacher_prob_sum_value = float(
                            teacher_probs.sum(dim=-1).mean().detach().cpu().item()
                        )
                        teacher_valid_actions_value = float(
                            valid_f.sum(dim=-1).mean().detach().cpu().item()
                        )
                        teacher_entropy_sum += teacher_entropy_value
                        teacher_max_prob_sum += teacher_max_prob_value
                        teacher_uniform_row_frac_sum += teacher_uniform_row_frac_value
                        teacher_bias_row_frac_sum += teacher_bias_row_frac_value
                        teacher_prob_sum_sum += teacher_prob_sum_value
                        teacher_valid_actions_sum += teacher_valid_actions_value
                        updates_done += 1

                    epoch_approx_kl_mean = (
                        float(np.mean(epoch_approx_kl_values))
                        if epoch_approx_kl_values
                        else 0.0
                    )
                    if target_kl_now > 0 and epoch_approx_kl_mean > target_kl_now:
                        early_stopped = True
                        early_stop_epoch = _epoch + 1
                        print(
                            "[ppo] "
                            f"target kl early stop epoch {early_stop_epoch}/{cfg.update_epochs} "
                            f"(update {update}/{num_updates}, "
                            f"mean_kl={epoch_approx_kl_mean:.5f}, target_kl={target_kl_now:.5f})"
                        )
                        break
                profile_opt_s = time.perf_counter() - optimize_start
                updates_done_denom = float(max(1, updates_done))
                policy_loss_value = policy_loss_sum / updates_done_denom
                value_loss_value = value_loss_sum / updates_done_denom
                entropy_value = entropy_sum / updates_done_denom
                distill_loss_value = distill_loss_sum / updates_done_denom
                policy_term_value = policy_term_sum / updates_done_denom
                value_term_value = value_term_sum / updates_done_denom
                entropy_term_value = entropy_term_sum / updates_done_denom
                distill_term_value = distill_term_sum / updates_done_denom
                total_loss_value = total_loss_sum / updates_done_denom
                teacher_entropy_value = teacher_entropy_sum / updates_done_denom
                teacher_max_prob_value = teacher_max_prob_sum / updates_done_denom
                teacher_uniform_row_frac_value = (
                    teacher_uniform_row_frac_sum / updates_done_denom
                )
                teacher_bias_row_frac_value = (
                    teacher_bias_row_frac_sum / updates_done_denom
                )
                teacher_prob_sum_value = teacher_prob_sum_sum / updates_done_denom
                teacher_valid_actions_value = teacher_valid_actions_sum / updates_done_denom

                y_pred = b_values.detach().cpu().numpy()
                y_true = b_returns.detach().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    float("nan")
                    if var_y <= 1e-12
                    else 1.0 - float(np.var(y_true - y_pred) / var_y)
                )

                update_seconds = max(1e-6, time.time() - update_start_wall)
                total_seconds = max(1e-6, time.time() - training_start)
                sps = int(global_step / total_seconds)

                approx_kl_mean = (
                    float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
                )
                stats = {
                    "update": update,
                    "global_step": global_step,
                    "piece_source_profile": current_piece_source,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                    "entropy": entropy_value,
                    "distill_loss": distill_loss_value,
                    "loss_total": total_loss_value,
                    "loss_policy_term": policy_term_value,
                    "loss_value_term": value_term_value,
                    "loss_entropy_term": entropy_term_value,
                    "loss_distill_term": distill_term_value,
                    "approx_kl": approx_kl_mean,
                    "clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "explained_variance": explained_var,
                    "updates_done": updates_done,
                    "early_stopped_kl": early_stopped,
                    "warmup_active": warmup_active,
                    "ent_coef_used": ent_coef_now,
                    "target_kl_used": target_kl_now,
                    "policy_clip_coef_used": cfg.policy_clip_coef,
                    "value_clip_coef_used": cfg.value_clip_coef,
                    "policy_lr_used": policy_lr_now,
                    "value_lr_used": value_lr_now,
                    "distill_coef_used": distill_coef_now,
                    "distill_teacher_tau_used": cfg.distill_teacher_tau,
                    "curriculum_topk_used": curriculum_topk_now,
                    "curriculum_bias_used": curriculum_bias_now,
                    "curriculum_danger_height_used": cfg.curriculum_danger_height,
                    "teacher_entropy": teacher_entropy_value,
                    "teacher_max_prob": teacher_max_prob_value,
                    "teacher_uniform_row_frac": teacher_uniform_row_frac_value,
                    "teacher_bias_row_frac": teacher_bias_row_frac_value,
                    "teacher_prob_sum": teacher_prob_sum_value,
                    "teacher_valid_actions": teacher_valid_actions_value,
                    "sps": sps,
                    "update_seconds": update_seconds,
                    "mean_episode_return_recent": (
                        float(np.mean(completed_returns[-100:]))
                        if completed_returns
                        else float("nan")
                    ),
                    "mean_episode_length_recent": (
                        float(np.mean(completed_lengths[-100:]))
                        if completed_lengths
                        else float("nan")
                    ),
                    "ret100_terms": {
                        key: _safe_recent_mean(values, window=100)
                        for key, values in completed_reward_component_sums.items()
                    },
                }
                should_log = (
                    update % cfg.log_every_updates == 0
                    or update == 1
                    or update == num_updates
                )
                validation: dict[str, Any] | None = None
                if should_log and cfg.validation_episodes_per_env > 0:
                    try:
                        validation = run_validation_eval(
                            cfg=cfg,
                            repo_root=repo_root,
                            server_cmd=server_cmd,
                            model=model,
                            obs_adapter=obs_adapter,
                            device=device,
                            update=update,
                            piece_source_profile=current_piece_source,
                            reward_component_aliases=reward_component_aliases,
                        )
                    except Exception as error:
                        validation = {"enabled": False, "error": str(error)}
                        print(
                            "[ppo] "
                            f"validation failed at update={update}: {error}"
                        )
                if isinstance(validation, dict):
                    stats["validation"] = validation
                    if bool(validation.get("enabled", False)):
                        stats["validation_mean_return"] = validation.get(
                            "mean_return", float("nan")
                        )
                        stats["validation_mean_length"] = validation.get(
                            "mean_length", float("nan")
                        )
                        stats["validation_terms"] = validation.get("terms", {})
                stats["profile_rollout_s"] = profile_rollout_s
                stats["profile_env_step_s"] = profile_env_step_s
                stats["profile_env_reset_s"] = profile_env_reset_s
                stats["profile_env_step_batch_s"] = profile_env_step_batch_s
                stats["profile_env_step_core_s"] = profile_env_step_core_s
                stats["profile_env_step_choices_current_s"] = (
                    profile_env_step_choices_current_s
                )
                stats["profile_env_step_choices_next_s"] = profile_env_step_choices_next_s
                stats["profile_env_step_runner_s"] = profile_env_step_runner_s
                stats["profile_env_step_reward_s"] = profile_env_step_reward_s
                stats["profile_env_step_obs_s"] = profile_env_step_obs_s
                stats["profile_env_reset_batch_s"] = profile_env_reset_batch_s
                stats["profile_env_reset_core_s"] = profile_env_reset_core_s
                stats["profile_env_reset_obs_s"] = profile_env_reset_obs_s
                stats["profile_env_reset_choices_s"] = profile_env_reset_choices_s
                stats["profile_policy_forward_s"] = profile_policy_forward_s
                stats["profile_gae_s"] = profile_gae_s
                stats["profile_opt_s"] = profile_opt_s
                stats["profile_io_s"] = profile_io_s
                stats["mask_repair_rows_update"] = mask_repair_update["rows"]
                stats["mask_repair_batches_update"] = mask_repair_update["batches"]
                mask_repair_total["rows"] += mask_repair_update["rows"]
                mask_repair_total["batches"] += mask_repair_update["batches"]
                stats["mask_repair_rows_total"] = mask_repair_total["rows"]
                stats["mask_repair_batches_total"] = mask_repair_total["batches"]
                stats["profile_update_s"] = max(
                    1e-6, time.perf_counter() - update_start_perf
                )

                score_raw = stats["mean_episode_return_recent"]
                score = (
                    float(score_raw)
                    if isinstance(score_raw, (float, int))
                    and math.isfinite(float(score_raw))
                    else float("-inf")
                )
                if best_state_dict is None or score > best_score:
                    io_start = time.perf_counter()
                    best_score = score
                    best_update = update
                    best_stats = dict(stats)
                    best_state_dict = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in model.state_dict().items()
                    }
                    best_adapter_state_dict = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in obs_adapter.state_dict().items()
                    }
                    best_ckpt_path = checkpoints_dir / "ppo_best.pt"
                    save_checkpoint(
                        checkpoint_path=best_ckpt_path,
                        model=model,
                        obs_adapter=obs_adapter,
                        optimizer=optimizer,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        global_step=global_step,
                        update=update,
                        stats=stats,
                    )
                    best_artifact = export_bot_policy_artifact(
                        model,
                        obs_adapter,
                        cfg,
                        obs_dim,
                        action_dim,
                        policy_observation_space,
                    )
                    write_json(out_dir / "bot_policy_best.json", best_artifact)
                    write_json(out_dir / "best_stats.json", best_stats)
                    profile_io_s += time.perf_counter() - io_start
                    print("[ppo] " f"new_best update={update} ret100={best_score:.3f}")

                if should_log:
                    profile_env_total_s = profile_env_step_s + profile_env_reset_s
                    profile_env_batch_s = profile_env_step_batch_s + profile_env_reset_batch_s
                    profile_env_core_s = profile_env_step_core_s + profile_env_reset_core_s
                    profile_env_choices_s = (
                        profile_env_step_choices_current_s
                        + profile_env_step_choices_next_s
                        + profile_env_reset_choices_s
                    )
                    profile_env_obs_s = profile_env_step_obs_s + profile_env_reset_obs_s
                    profile_env_ipc_s = max(0.0, profile_env_total_s - profile_env_batch_s)
                    profile_accounted_s = (
                        profile_rollout_s
                        + profile_gae_s
                        + profile_opt_s
                        + profile_io_s
                    )
                    profile_overhead_s = max(0.0, update_seconds - profile_accounted_s)
                    print(
                        "[ppo] "
                        f"update={update}/{num_updates} "
                        f"step={global_step} "
                        f"ploss={stats['policy_loss']:.4f} "
                        f"vloss={stats['value_loss']:.4f} "
                        f"dloss={stats['distill_loss']:.4f} "
                        f"loss={stats['loss_total']:.4f} "
                        f"loss_terms("
                        f"p={stats['loss_policy_term']:.4f},"
                        f"v={stats['loss_value_term']:.4f},"
                        f"ent={stats['loss_entropy_term']:.4f},"
                        f"dist={stats['loss_distill_term']:.4f}"
                        f") "
                        f"ent={stats['entropy']:.4f} "
                        f"kl={stats['approx_kl']:.5f} "
                        f"clip={stats['clip_fraction']:.3f} "
                        f"ent_coef={ent_coef_now:.5f} "
                        f"distill_coef={distill_coef_now:.5f} "
                        f"tau={cfg.distill_teacher_tau:.3f} "
                        f"pclip={cfg.policy_clip_coef:.4f} "
                        f"vclip={cfg.value_clip_coef:.4f} "
                        f"p_lr={policy_lr_now:.6g} "
                        f"v_lr={value_lr_now:.6g} "
                        f"target_kl={target_kl_now:.5f} "
                        f"topk={curriculum_topk_now} "
                        f"bias={curriculum_bias_now:.3f} "
                        f"src={'ml' if current_piece_source == 'active_generator' else 'bag7'} "
                        f"warmup={'y' if warmup_active else 'n'} "
                        f"teacher("
                        f"ent={stats['teacher_entropy']:.3f},"
                        f"maxp={stats['teacher_max_prob']:.3f},"
                        f"uniform={stats['teacher_uniform_row_frac']:.3f},"
                        f"bias_rows={stats['teacher_bias_row_frac']:.3f},"
                        f"psum={stats['teacher_prob_sum']:.3f},"
                        f"valid={stats['teacher_valid_actions']:.1f}"
                        f") "
                        f"ev={stats['explained_variance']:.3f} "
                        f"ret100={stats['mean_episode_return_recent']:.3f} "
                        f"ret100_terms("
                        f"lines={_fmt_float(stats['ret100_terms'].get('term_lines'))},"
                        f"score={_fmt_float(stats['ret100_terms'].get('term_score'))},"
                        f"time={_fmt_float(stats['ret100_terms'].get('term_time'))},"
                        f"height={_fmt_float(stats['ret100_terms'].get('term_height'))},"
                        f"holes={_fmt_float(stats['ret100_terms'].get('term_holes'))},"
                        f"bump={_fmt_float(stats['ret100_terms'].get('term_bumpiness'))},"
                        f"board={_fmt_float(stats['ret100_terms'].get('term_board_score'))},"
                        f"q={_fmt_float(stats['ret100_terms'].get('term_board_quality'))},"
                        f"topout={_fmt_float(stats['ret100_terms'].get('top_out_penalty'))},"
                        f"base={_fmt_float(stats['ret100_terms'].get('reward_base'))},"
                        f"final={_fmt_float(stats['ret100_terms'].get('reward_final'))}"
                        f") "
                        f"val={_fmt_float(stats.get('validation_mean_return'))} "
                        f"val_terms("
                        f"lines={_fmt_float((stats.get('validation_terms') or {}).get('term_lines'))},"
                        f"score={_fmt_float((stats.get('validation_terms') or {}).get('term_score'))},"
                        f"time={_fmt_float((stats.get('validation_terms') or {}).get('term_time'))},"
                        f"height={_fmt_float((stats.get('validation_terms') or {}).get('term_height'))},"
                        f"holes={_fmt_float((stats.get('validation_terms') or {}).get('term_holes'))},"
                        f"bump={_fmt_float((stats.get('validation_terms') or {}).get('term_bumpiness'))},"
                        f"board={_fmt_float((stats.get('validation_terms') or {}).get('term_board_score'))},"
                        f"q={_fmt_float((stats.get('validation_terms') or {}).get('term_board_quality'))},"
                        f"topout={_fmt_float((stats.get('validation_terms') or {}).get('top_out_penalty'))},"
                        f"base={_fmt_float((stats.get('validation_terms') or {}).get('reward_base'))},"
                        f"final={_fmt_float((stats.get('validation_terms') or {}).get('reward_final'))}"
                        f") "
                        f"sps={stats['sps']} "
                        f"t_upd={update_seconds:.2f}s "
                        f"t_roll={profile_rollout_s:.2f}s "
                        f"t_env={profile_env_total_s:.2f}s "
                        f"t_env_batch={profile_env_batch_s:.2f}s "
                        f"t_env_core={profile_env_core_s:.2f}s "
                        f"t_env_ipc={profile_env_ipc_s:.2f}s "
                        f"t_env_runner={profile_env_step_runner_s:.2f}s "
                        f"t_env_choices={profile_env_choices_s:.2f}s "
                        f"t_env_obs={profile_env_obs_s:.2f}s "
                        f"t_env_reward={profile_env_step_reward_s:.2f}s "
                        f"t_fwd={profile_policy_forward_s:.2f}s "
                        f"t_gae={profile_gae_s:.2f}s "
                        f"t_opt={profile_opt_s:.2f}s "
                        f"t_io={profile_io_s:.2f}s "
                        f"t_ovh={profile_overhead_s:.2f}s "
                        f"mask_fix_rows={mask_repair_update['rows']} "
                        f"mask_fix_rows_total={mask_repair_total['rows']}"
                    )

                if update % cfg.save_every_updates == 0 or update == num_updates:
                    io_start = time.perf_counter()
                    ckpt_path = checkpoints_dir / f"ppo_update_{update:06d}.pt"
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        obs_adapter=obs_adapter,
                        optimizer=optimizer,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        global_step=global_step,
                        update=update,
                        stats=stats,
                    )
                    artifact = export_bot_policy_artifact(
                        model,
                        obs_adapter,
                        cfg,
                        obs_dim,
                        action_dim,
                        policy_observation_space,
                    )
                    artifact_path = artifacts_dir / f"bot_policy_update_{update:06d}.json"
                    write_json(artifact_path, artifact)
                    write_json(out_dir / "last_stats.json", stats)
                    latest_trajectory, drained = pop_latest_trajectory(env)
                    if latest_trajectory is not None:
                        trajectory_path = (
                            trajectories_dir / f"trajectory_update_{update:06d}.json"
                        )
                        write_json_any(trajectory_path, latest_trajectory)
                        sample_count = latest_trajectory.get("samples")
                        sample_total = (
                            len(sample_count) if isinstance(sample_count, list) else -1
                        )
                        session_id = latest_trajectory.get("sessionId")
                        print(
                            "[ppo] "
                            f"saved trajectory update={update} "
                            f"session={session_id} "
                            f"samples={sample_total} "
                            f"drained={drained} "
                            f"path={trajectory_path}"
                        )
                    else:
                        print(
                            "[ppo] "
                            f"no completed trajectory available at update={update}."
                        )
                    profile_io_s += time.perf_counter() - io_start

        except KeyboardInterrupt:
            did_interrupt = True
            print("[ppo] keyboard interrupt received; finishing with best available policy...")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        if best_adapter_state_dict is not None:
            obs_adapter.load_state_dict(best_adapter_state_dict, strict=False)
        if best_stats is not None:
            write_json(out_dir / "best_stats.json", best_stats)

        artifact_final = export_bot_policy_artifact(
            model,
            obs_adapter,
            cfg,
            obs_dim,
            action_dim,
            policy_observation_space,
        )
        write_json(out_dir / "bot_policy_final.json", artifact_final)
        if did_interrupt:
            if best_update > 0 and math.isfinite(best_score):
                print(
                    f"[ppo] training interrupted. output={out_dir} "
                    f"(final=best update={best_update}, ret100={best_score:.3f})"
                )
            else:
                print(f"[ppo] training interrupted. output={out_dir} (final=latest)")
        elif best_update > 0 and math.isfinite(best_score):
            print(
                f"[ppo] training complete. output={out_dir} "
                f"(final=best update={best_update}, ret100={best_score:.3f})"
            )
        else:
            print(f"[ppo] training complete. output={out_dir} (final=latest)")


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
