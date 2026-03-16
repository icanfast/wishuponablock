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
RAW_ACTIVE_DIM = len(RAW_PIECES_ORDER)
RAW_HOLD_DIM = len(RAW_PIECES_ORDER) + 1
RAW_NEXT_DIM = len(RAW_PIECES_ORDER)
RAW_SCALAR_CONTEXT_DIM = 5
RAW_CONTEXT_DIM = RAW_ACTIVE_DIM + RAW_HOLD_DIM + RAW_NEXT_DIM + RAW_SCALAR_CONTEXT_DIM
DEFAULT_BOARD_ROWS = 20
DEFAULT_BOARD_COLS = 10
RAW_BOARD_SIZE = DEFAULT_BOARD_ROWS * DEFAULT_BOARD_COLS
RAW_LEGACY_OBS_DIM = RAW_BOARD_SIZE + RAW_CONTEXT_DIM
RAW_VISIBLE_QUEUE_SLOTS = 5
RAW_QUEUE_INPUT_DIM = RAW_VISIBLE_QUEUE_SLOTS * len(RAW_PIECES_ORDER)
RAW_EXTENDED_OBS_DIM = RAW_LEGACY_OBS_DIM + RAW_QUEUE_INPUT_DIM
DEFAULT_OBS_ADAPTER_LR_SCALE = 0.1
DEFAULT_QUEUE_ENCODER_HIDDEN_DIM = 32
DEFAULT_QUEUE_ENCODER_LR_SCALE = 5.0
VALID_GENERATOR_SOURCES = ("bag7", "active_generator", "random")
VALID_ACTION_SPACE_KINDS = ("placement_full_v1", "placement_hold_step_v2")


@dataclass(frozen=True)
class PPOConfig:
    mode_id: str
    model_path: str
    observation_space: str
    phase_context_enabled: bool
    placement_execution_mode: str
    action_space_kind: str
    queue_policy_id: str
    generator_schedule: tuple[str, ...]
    reward_functions: tuple[str, ...]
    max_pieces_per_episode_train: int
    max_pieces_per_episode_val: int
    reward_blend_span: int
    reward_blend_unit: str
    seed: int
    num_envs: int
    total_timesteps: int
    num_steps: int
    hidden_dim: int
    learning_rate: float
    obs_adapter_lr_scale: float
    queue_encoder_hidden_dim: int
    queue_encoder_lr_scale: float
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
    distill_teacher_alpha: float
    distill_teacher_tau: float
    distill_teacher_top_m: int
    hold_margin_probe: bool
    hold_margin_eps: float
    hold_margin_good_threshold: float
    hold_margin_penalty_base: float
    hold_margin_penalty_threshold: float
    hold_swap_probe: bool
    hold_swap_distill_coef_start: float
    hold_swap_distill_coef_end: float
    hold_swap_distill_coef_ramp_updates: int
    hold_swap_teacher_tau: float
    validation_episodes_per_env: int
    validate_every_updates: int
    device: str
    save_every_updates: int
    log_every_updates: int
    out_dir: str
    run_name: str
    server_cmd: str | None
    resume_checkpoint: str | None
    resume_mode: str
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
    bc_val_fraction: float
    bc_min_epochs: int
    bc_early_stop_patience: int
    bc_eval_every_epochs: int
    bc_rollout_eval_every_epochs: int
    bc_kl_probe_size: int
    bc_policy_lr_scale: float
    bc_policy_head_lr_scale: float
    bc_value_lr_scale: float
    bc_adapter_lr_scale: float
    bc_queue_encoder_lr_scale: float
    bc_hold_output_lr_scale: float
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


def split_raw_v1_board_and_queue_tensor(
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if obs.ndim != 2:
        raise ValueError(f"Expected rank-2 observation batch, got shape={tuple(obs.shape)}")
    if obs.shape[1] < RAW_LEGACY_OBS_DIM:
        raise ValueError(
            "raw_v1 observation is too short. "
            f"expected_at_least={RAW_LEGACY_OBS_DIM} got={int(obs.shape[1])}"
        )
    legacy = obs[:, :RAW_LEGACY_OBS_DIM]
    queue = torch.zeros(
        (obs.shape[0], RAW_QUEUE_INPUT_DIM),
        dtype=obs.dtype,
        device=obs.device,
    )
    available = min(RAW_QUEUE_INPUT_DIM, max(0, int(obs.shape[1]) - RAW_LEGACY_OBS_DIM))
    if available > 0:
        queue[:, :available] = obs[
            :,
            RAW_LEGACY_OBS_DIM : RAW_LEGACY_OBS_DIM + available,
        ]
    return legacy, queue


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


class RawV1BoardQueueObservationAdapter(ObservationAdapter):
    def __init__(
        self,
        raw_obs_dim: int,
        model_path: str,
        queue_hidden_dim: int,
    ) -> None:
        nn.Module.__init__(self)
        self.board_adapter = WubHeadFromRawObservationAdapter(
            raw_obs_dim=RAW_LEGACY_OBS_DIM,
            model_path=model_path,
        )
        self.queue_input_dim = RAW_QUEUE_INPUT_DIM
        self.queue_hidden_dim = max(1, int(queue_hidden_dim))
        queue_policy_obs_dim = self.board_adapter.policy_obs_dim + self.queue_hidden_dim
        self.raw_obs_dim = int(raw_obs_dim)
        self.policy_obs_dim = int(queue_policy_obs_dim)
        self.policy_observation_space = "raw_v1"
        self.queue_fc1 = nn.Linear(self.queue_input_dim, self.queue_hidden_dim)
        nn.init.orthogonal_(self.queue_fc1.weight, math.sqrt(2.0))
        nn.init.constant_(self.queue_fc1.bias, 0.0)
        self.board_obs_dim = int(self.board_adapter.policy_obs_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        legacy_obs, queue_obs = split_raw_v1_board_and_queue_tensor(obs)
        board_features = self.board_adapter(legacy_obs)
        queue_features = torch.relu(self.queue_fc1(queue_obs))
        return torch.cat([board_features, queue_features], dim=1)


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
        "--phase-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether raw/model-head observations include phase scalars "
            "(progress, time, level, score). Default off for generalist policies."
        ),
    )
    parser.add_argument(
        "--placement-execution-mode",
        default="teleport",
        choices=["commands", "teleport"],
        help="How env executes placement actions: deterministic command playback or direct teleport+harddrop.",
    )
    parser.add_argument(
        "--action-space-kind",
        default="placement_full_v1",
        choices=list(VALID_ACTION_SPACE_KINDS),
        help=(
            "Policy output/action-space variant. "
            "placement_full_v1 = flat no-hold + hold-placement actions. "
            "placement_hold_step_v2 = no-hold placements + one explicit HOLD action."
        ),
    )
    parser.add_argument("--queue-policy-id", default="next_piece_v1")
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help=(
            "Piece-source specification for proportional per-env mixing each update. "
            "Duplicates act as weights across env slots (cycled by env index). "
            "Accepts space/comma-separated list of: bag7, active_generator, random. "
            "Examples: --generators bag7 random (50/50 with even num-envs) | "
            "--generators bag7 bag7 bag7 bag7 active_generator (~80/20)"
        ),
    )
    parser.add_argument(
        "--piece-source-profile",
        default="active_generator",
        choices=["active_generator", "bag7", "random"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alternate-piece-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-pieces-per-episode-train", type=int, default=1024)
    parser.add_argument("--max-pieces-per-episode-val", type=int, default=512)
    parser.add_argument(
        "--max-pieces-per-episode",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reward-functions",
        nargs="+",
        default=None,
        help=(
            "Reward schedule definition. Accepts one or two items (space/comma-separated). "
            "One item ('v1' / 'v2' / 'v3') uses that reward only. "
            "Two items (for example: 'v1 v3') blend first->second using --reward-blend-updates."
        ),
    )
    parser.add_argument(
        "--reward-blend-updates",
        type=int,
        default=10,
        help=(
            "Number of PPO updates used to linearly blend first->second reward "
            "when --reward-functions has two items (e.g. v1 v2). "
            "Ignored in single-reward mode."
        ),
    )
    parser.add_argument(
        "--reward-blend-timesteps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42030)

    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-steps", type=int, default=256)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--obs-adapter-lr-scale",
        type=float,
        default=DEFAULT_OBS_ADAPTER_LR_SCALE,
        help=(
            "Gradient scale for observation adapter (conv stack in raw_v1). "
            "Effective adapter LR ~= learning_rate * obs_adapter_lr_scale."
        ),
    )
    parser.add_argument(
        "--queue-encoder-hidden-dim",
        type=int,
        default=DEFAULT_QUEUE_ENCODER_HIDDEN_DIM,
        help="Hidden/output width of the visible-queue encoder branch.",
    )
    parser.add_argument(
        "--queue-encoder-lr-scale",
        type=float,
        default=DEFAULT_QUEUE_ENCODER_LR_SCALE,
        help=(
            "Learning-rate scale for the new visible-queue encoder branch during PPO."
        ),
    )
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
        help="Teacher temperature for distillation softmax(teacher_logits / tau).",
    )
    parser.add_argument(
        "--distill-teacher-alpha",
        type=float,
        default=0.25,
        help=(
            "Heuristic prior strength for teacher construction. "
            "Independent from rollout curriculum bias."
        ),
    )
    parser.add_argument(
        "--distill-teacher-top-m",
        type=int,
        default=0,
        help=(
            "Optional sparse teacher support size over legal actions. "
            "0 disables top-M sparsification."
        ),
    )
    parser.add_argument(
        "--hold-margin-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Probe hold justification using immediate reward(no hold tax)+gamma*V(next).",
    )
    parser.add_argument(
        "--hold-margin-eps",
        type=float,
        default=0.1,
        help="Threshold for counting hold margins as near-zero / unnecessary.",
    )
    parser.add_argument(
        "--hold-margin-good-threshold",
        type=float,
        default=1.0,
        help="Threshold for counting hold margins as clearly justified.",
    )
    parser.add_argument(
        "--hold-margin-penalty-base",
        type=float,
        default=0.0,
        help="Optional margin-based penalty base for chosen hold actions (0 disables shaping).",
    )
    parser.add_argument(
        "--hold-margin-penalty-threshold",
        type=float,
        default=1.0,
        help="Margin threshold at/above which hold penalty becomes zero.",
    )
    parser.add_argument(
        "--hold-swap-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Probe explicit HOLD usefulness via swap utility: "
            "best hold-placement score minus best no-hold placement score."
        ),
    )
    parser.add_argument(
        "--hold-swap-distill-coef-start",
        type=float,
        default=0.0,
        help="Hold-swap auxiliary loss coefficient at update 1.",
    )
    parser.add_argument(
        "--hold-swap-distill-coef-end",
        type=float,
        default=0.0,
        help="Hold-swap auxiliary loss coefficient after ramp completes.",
    )
    parser.add_argument(
        "--hold-swap-distill-coef-ramp-updates",
        type=int,
        default=1,
        help="Number of updates to linearly ramp the hold-swap auxiliary loss.",
    )
    parser.add_argument(
        "--hold-swap-teacher-tau",
        type=float,
        default=0.25,
        help="Temperature for sigmoid(margin_swap / tau) hold teacher.",
    )
    parser.add_argument(
        "--validation-episodes-per-env",
        type=int,
        default=2,
        help="Validation episodes per env per validation run (0 disables validation).",
    )
    parser.add_argument(
        "--validate-every-updates",
        type=int,
        default=10,
        help=(
            "Run validation once every N updates (independent of --log-every-updates). "
            "Use 1 to validate every update."
        ),
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
    parser.add_argument(
        "--resume-mode",
        choices=["fresh", "continue"],
        default="fresh",
        help=(
            "When using --resume-checkpoint: "
            "'fresh' warm-starts from checkpoint weights but resets training clock "
            "(update/step schedules restart from 1); "
            "'continue' preserves checkpoint progress and optimizer state."
        ),
    )
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
        "--bc-val-fraction",
        type=float,
        default=0.1,
        help="Session-level validation split fraction for BC (0 disables val split).",
    )
    parser.add_argument(
        "--bc-min-epochs",
        type=int,
        default=3,
        help="Minimum BC epochs before early stopping can trigger.",
    )
    parser.add_argument(
        "--bc-early-stop-patience",
        type=int,
        default=5,
        help="BC early-stop patience on validation total loss (0 disables early stop).",
    )
    parser.add_argument(
        "--bc-eval-every-epochs",
        type=int,
        default=1,
        help="Run full BC train/val metrics every N epochs.",
    )
    parser.add_argument(
        "--bc-rollout-eval-every-epochs",
        type=int,
        default=1,
        help="Run deterministic BC rollout eval once every N epochs (0 disables).",
    )
    parser.add_argument(
        "--bc-kl-probe-size",
        type=int,
        default=512,
        help="Fixed train/val probe size for KL(epoch_t-1 || epoch_t) logging.",
    )
    parser.add_argument(
        "--bc-policy-lr-scale",
        type=float,
        default=1.0,
        help="BC LR scale for inherited policy torso.",
    )
    parser.add_argument(
        "--bc-policy-head-lr-scale",
        type=float,
        default=1.0,
        help="BC LR scale for policy head weights/bias.",
    )
    parser.add_argument(
        "--bc-value-lr-scale",
        type=float,
        default=1.0,
        help="BC LR scale for value tower.",
    )
    parser.add_argument(
        "--bc-adapter-lr-scale",
        type=float,
        default=0.25,
        help="BC LR scale for the observation encoder / board adapter.",
    )
    parser.add_argument(
        "--bc-queue-encoder-lr-scale",
        type=float,
        default=1.0,
        help="BC LR scale for the visible-queue encoder branch.",
    )
    parser.add_argument(
        "--bc-hold-output-lr-scale",
        type=float,
        default=4.0,
        help=(
            "Extra BC gradient scale for the explicit HOLD logit in "
            "placement_hold_step_v2."
        ),
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

    def _normalize_generator_source(value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {
            "ml": "active_generator",
            "active": "active_generator",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in VALID_GENERATOR_SOURCES:
            raise ValueError(
                f"Unsupported generator source '{value}'. "
                f"Allowed: {', '.join(VALID_GENERATOR_SOURCES)}"
            )
        return normalized

    def _parse_generator_schedule() -> tuple[str, ...]:
        raw_values = args.generators
        expanded: list[str] = []
        if isinstance(raw_values, list):
            for token in raw_values:
                for part in str(token).split(","):
                    stripped = part.strip()
                    if stripped:
                        expanded.append(stripped)
        if expanded:
            return tuple(_normalize_generator_source(v) for v in expanded)
        # Backward compatibility fallback for old flags.
        base = _normalize_generator_source(args.piece_source_profile)
        if bool(args.alternate_piece_sources):
            return ("bag7", "bag7", "bag7", "bag7", "active_generator")
        return (base,)

    def _normalize_reward_function(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in ("v1", "v2", "v3"):
            raise ValueError(
                f"Unsupported reward function '{value}'. Allowed: v1, v2, v3."
            )
        return normalized

    def _parse_reward_functions() -> tuple[str, ...]:
        raw_values = args.reward_functions
        expanded: list[str] = []
        if isinstance(raw_values, list):
            for token in raw_values:
                for part in str(token).split(","):
                    stripped = part.strip()
                    if stripped:
                        expanded.append(stripped)
        if not expanded:
            expanded = ["v1", "v2"]
        parsed = tuple(_normalize_reward_function(v) for v in expanded)
        if len(parsed) == 1:
            return parsed
        if len(parsed) == 2:
            if parsed[0] == parsed[1]:
                raise ValueError(
                    "When two reward functions are provided, they must be different."
                )
            return parsed
        raise ValueError(
            "Reward schedule must contain one function (v1/v2/v3) or two distinct functions."
        )

    try:
        generator_schedule = _parse_generator_schedule()
        reward_functions = _parse_reward_functions()
    except ValueError as error:
        parser.error(str(error))

    reward_blend_unit = "updates"
    reward_blend_span = max(1, int(args.reward_blend_updates))
    if args.reward_blend_timesteps is not None:
        reward_blend_unit = "timesteps"
        reward_blend_span = max(1, int(args.reward_blend_timesteps))

    max_pieces_train = max(1, int(args.max_pieces_per_episode_train))
    max_pieces_val = max(1, int(args.max_pieces_per_episode_val))
    if args.max_pieces_per_episode is not None:
        legacy_cap = max(1, int(args.max_pieces_per_episode))
        max_pieces_train = legacy_cap
        max_pieces_val = legacy_cap

    run_name = args.run_name.strip() or f"ppo_baseline_{int(time.time())}"
    return PPOConfig(
        mode_id=args.mode_id.strip().lower(),
        model_path=args.model_path,
        observation_space=(
            "raw_v1" if args.observation_space == "raw_v1" else "model_head_v1"
        ),
        phase_context_enabled=bool(args.phase_context),
        placement_execution_mode=(
            "commands"
            if args.placement_execution_mode == "commands"
            else "teleport"
        ),
        action_space_kind=(
            "placement_hold_step_v2"
            if args.action_space_kind == "placement_hold_step_v2"
            else "placement_full_v1"
        ),
        queue_policy_id=args.queue_policy_id.strip().lower(),
        generator_schedule=generator_schedule,
        reward_functions=reward_functions,
        max_pieces_per_episode_train=max_pieces_train,
        max_pieces_per_episode_val=max_pieces_val,
        reward_blend_span=reward_blend_span,
        reward_blend_unit=reward_blend_unit,
        seed=max(1, int(args.seed)),
        num_envs=max(1, int(args.num_envs)),
        total_timesteps=max(1, int(args.total_timesteps)),
        num_steps=max(1, int(args.num_steps)),
        hidden_dim=max(8, int(args.hidden_dim)),
        learning_rate=float(args.learning_rate),
        obs_adapter_lr_scale=max(0.0, float(args.obs_adapter_lr_scale)),
        queue_encoder_hidden_dim=max(1, int(args.queue_encoder_hidden_dim)),
        queue_encoder_lr_scale=max(0.0, float(args.queue_encoder_lr_scale)),
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
        distill_teacher_alpha=max(0.0, float(args.distill_teacher_alpha)),
        distill_teacher_tau=max(1e-4, float(args.distill_teacher_tau)),
        distill_teacher_top_m=max(0, int(args.distill_teacher_top_m)),
        hold_margin_probe=bool(args.hold_margin_probe),
        hold_margin_eps=max(0.0, float(args.hold_margin_eps)),
        hold_margin_good_threshold=max(0.0, float(args.hold_margin_good_threshold)),
        hold_margin_penalty_base=max(0.0, float(args.hold_margin_penalty_base)),
        hold_margin_penalty_threshold=max(
            1e-6, float(args.hold_margin_penalty_threshold)
        ),
        hold_swap_probe=bool(args.hold_swap_probe),
        hold_swap_distill_coef_start=max(
            0.0, float(args.hold_swap_distill_coef_start)
        ),
        hold_swap_distill_coef_end=max(0.0, float(args.hold_swap_distill_coef_end)),
        hold_swap_distill_coef_ramp_updates=max(
            0, int(args.hold_swap_distill_coef_ramp_updates)
        ),
        hold_swap_teacher_tau=max(1e-4, float(args.hold_swap_teacher_tau)),
        validation_episodes_per_env=max(0, int(args.validation_episodes_per_env)),
        validate_every_updates=max(1, int(args.validate_every_updates)),
        device=args.device,
        save_every_updates=max(1, int(args.save_every_updates)),
        log_every_updates=max(1, int(args.log_every_updates)),
        out_dir=args.out_dir,
        run_name=run_name,
        server_cmd=args.server_cmd,
        resume_checkpoint=args.resume_checkpoint,
        resume_mode=str(args.resume_mode),
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
        bc_val_fraction=min(0.95, max(0.0, float(args.bc_val_fraction))),
        bc_min_epochs=max(1, int(args.bc_min_epochs)),
        bc_early_stop_patience=max(0, int(args.bc_early_stop_patience)),
        bc_eval_every_epochs=max(1, int(args.bc_eval_every_epochs)),
        bc_rollout_eval_every_epochs=max(0, int(args.bc_rollout_eval_every_epochs)),
        bc_kl_probe_size=max(0, int(args.bc_kl_probe_size)),
        bc_policy_lr_scale=max(0.0, float(args.bc_policy_lr_scale)),
        bc_policy_head_lr_scale=max(0.0, float(args.bc_policy_head_lr_scale)),
        bc_value_lr_scale=max(0.0, float(args.bc_value_lr_scale)),
        bc_adapter_lr_scale=max(0.0, float(args.bc_adapter_lr_scale)),
        bc_queue_encoder_lr_scale=max(0.0, float(args.bc_queue_encoder_lr_scale)),
        bc_hold_output_lr_scale=max(0.0, float(args.bc_hold_output_lr_scale)),
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


def build_env_piece_source_assignment(
    cfg: PPOConfig, env_ids: list[int]
) -> tuple[list[str], dict[str, int]]:
    schedule = cfg.generator_schedule if cfg.generator_schedule else ("bag7",)
    normalized_schedule = tuple(str(v).strip().lower() for v in schedule if str(v).strip())
    if not normalized_schedule:
        normalized_schedule = ("bag7",)
    piece_sources: list[str] = []
    counts: dict[str, int] = {}
    for i in range(len(env_ids)):
        source = normalized_schedule[i % len(normalized_schedule)]
        piece_sources.append(source)
        counts[source] = counts.get(source, 0) + 1
    return piece_sources, counts


def unique_generator_sources(cfg: PPOConfig) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in cfg.generator_schedule if cfg.generator_schedule else ("bag7",):
        source = str(raw).strip().lower()
        if not source or source in seen:
            continue
        seen.add(source)
        ordered.append(source)
    if not ordered:
        ordered.append("bag7")
    return ordered


def reward_blend_total_for_env(cfg: PPOConfig) -> int:
    if cfg.reward_blend_unit == "updates":
        return max(1, int(cfg.reward_blend_span) - 1)
    return max(1, int(cfg.reward_blend_span))


def fixed_reward_blend_step_for_cfg(cfg: PPOConfig) -> int | None:
    reward_functions = tuple(str(v).strip().lower() for v in cfg.reward_functions)
    if len(reward_functions) != 1:
        return None
    if reward_functions[0] == "v1":
        return 0
    if reward_functions[0] in ("v2", "v3"):
        # Force fully-target reward from the first training step.
        return 1_000_000_000
    return None


def effective_reward_blend_unit(cfg: PPOConfig) -> str:
    # Single-reward mode is fixed, so we keep blend progression disabled.
    return "updates" if fixed_reward_blend_step_for_cfg(cfg) is not None else cfg.reward_blend_unit


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


def hold_swap_distill_coef_for_update(cfg: PPOConfig, update: int) -> float:
    return max(
        0.0,
        curriculum_schedule_value(
            cfg.hold_swap_distill_coef_start,
            cfg.hold_swap_distill_coef_end,
            cfg.hold_swap_distill_coef_ramp_updates,
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
    student_logits: torch.Tensor,
    action_scores: torch.Tensor,
    tau: float,
    teacher_alpha: float,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Distill toward the policy that would result from adding a heuristic prior
    # to the current actor logits, with actor logits detached for a fixed target.
    valid = action_mask > 0
    valid_f = valid.to(dtype=student_logits.dtype)
    valid_count = torch.sum(valid_f, dim=-1, keepdim=True)
    safe_valid_count = torch.clamp(valid_count, min=1.0)
    masked_scores = torch.where(valid, action_scores, torch.zeros_like(action_scores))
    row_mean = torch.sum(masked_scores, dim=-1, keepdim=True) / safe_valid_count
    centered_scores = (masked_scores - row_mean) * valid_f
    row_var = torch.sum(centered_scores**2, dim=-1, keepdim=True) / safe_valid_count
    row_has_signal = row_var > 1e-12
    row_prior_inactive = (~row_has_signal).to(dtype=student_logits.dtype)
    safe_tau = max(1e-4, float(tau))
    large_neg = torch.full_like(student_logits, -1e9)
    teacher_logits = torch.where(valid, student_logits.detach(), large_neg)
    # Distillation prior strength is independent from rollout policy steering.
    prior_alpha = float(max(0.0, min(1.0, teacher_alpha)))
    if prior_alpha <= 0.0:
        row_prior_inactive = torch.ones_like(row_prior_inactive)
    if prior_alpha > 0.0:
        teacher_logits = torch.where(
            valid, teacher_logits + prior_alpha * centered_scores, large_neg
        )

    row_topm_applied = torch.zeros_like(row_prior_inactive)
    if top_m and top_m > 0:
        action_dim = int(action_mask.shape[-1])
        keep_k = max(1, min(int(top_m), action_dim))
        if keep_k < action_dim:
            heuristic_for_topk = torch.where(valid, centered_scores, large_neg)
            topk_idx = torch.topk(heuristic_for_topk, k=keep_k, dim=-1).indices
            keep_mask = torch.zeros_like(valid, dtype=torch.bool)
            keep_mask.scatter_(1, topk_idx, True)
            keep_mask = keep_mask & valid
            teacher_logits = torch.where(keep_mask, teacher_logits, large_neg)
            row_topm_applied = (valid_count > float(keep_k)).to(
                dtype=student_logits.dtype
            )

    teacher_logits = teacher_logits / safe_tau
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    return teacher_probs, row_prior_inactive, row_topm_applied


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


def is_hold_probe_supported(action_space_kind: str) -> bool:
    return str(action_space_kind).strip().lower() == "placement_full_v1"


def action_index_uses_hold(
    action_index: int,
    action_dim: int,
    action_space_kind: str = "placement_full_v1",
) -> bool:
    if action_dim <= 0:
        return False
    if str(action_space_kind).strip().lower() == "placement_hold_step_v2":
        return int(action_index) == max(0, int(action_dim) - 1)
    hold_stride = max(1, action_dim // 2)
    return int(action_index) >= hold_stride


def compute_hold_margin_penalty(
    margin: float | None,
    base_penalty: float,
    margin_threshold: float,
) -> float:
    if margin is None or not math.isfinite(margin):
        return 0.0
    if base_penalty <= 0.0:
        return 0.0
    threshold = max(1e-6, float(margin_threshold))
    scaled = 1.0 - (float(margin) / threshold)
    return float(base_penalty) * max(0.0, min(1.0, scaled))


def _hold_probe_summary(
    values: list[float],
    eps: float,
    good_threshold: float,
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "frac_negative": float("nan"),
            "frac_below_eps": float("nan"),
            "frac_above_good": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "frac_negative": float(np.mean(arr < 0.0)),
        "frac_below_eps": float(np.mean(arr < float(eps))),
        "frac_above_good": float(np.mean(arr > float(good_threshold))),
    }


def _hold_swap_margin_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "frac_positive": float("nan"),
            "frac_negative": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "frac_positive": float(np.mean(arr > 0.0)),
        "frac_negative": float(np.mean(arr < 0.0)),
    }


def hold_swap_teacher_prob_from_margin(margin: float, tau: float) -> float:
    safe_tau = max(1e-4, float(tau))
    scaled = float(margin) / safe_tau
    if scaled >= 0.0:
        z = math.exp(-scaled)
        return float(1.0 / (1.0 + z))
    z = math.exp(scaled)
    return float(z / (1.0 + z))


def evaluate_hold_margin_for_actions(
    *,
    env: WubEnvBridge,
    env_ids: list[int],
    actions: list[int],
    env_piece_sources: list[str],
    obs_adapter: ObservationAdapter,
    model: PolicyValueNet,
    device: torch.device,
    gamma: float,
    action_dim: int,
    action_space_kind: str,
    eps: float,
    good_threshold: float,
) -> tuple[dict[int, dict[str, float | None]], dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    probe_env_ids: list[int] = []
    probe_env_sources: list[str] = []
    chosen_actions_by_env_id: dict[int, int] = {}

    for env_id, action, source in zip(env_ids, actions, env_piece_sources):
        source_stats = by_source.setdefault(
            source,
            {
                "hold_actions": 0,
                "non_hold_actions": 0,
                "unavailable": 0,
                "margins": [],
                "best_hold_scores": [],
                "best_no_hold_scores": [],
                "best_hold_immediate": [],
                "best_no_hold_immediate": [],
                "best_hold_value": [],
                "best_no_hold_value": [],
                "chosen_hold_scores": [],
                "chosen_hold_immediate": [],
                "chosen_hold_value": [],
            },
        )
        if action_index_uses_hold(int(action), action_dim, action_space_kind):
            source_stats["hold_actions"] += 1
            probe_env_ids.append(int(env_id))
            probe_env_sources.append(source)
            chosen_actions_by_env_id[int(env_id)] = int(action)
        else:
            source_stats["non_hold_actions"] += 1

    if not probe_env_ids:
        return {}, {
            source: {
                "hold_actions": int(stats["hold_actions"]),
                "non_hold_actions": int(stats["non_hold_actions"]),
                "unavailable": int(stats["unavailable"]),
                "margin": _hold_probe_summary([], eps, good_threshold),
                "mean_best_hold_score": float("nan"),
                "mean_best_no_hold_score": float("nan"),
                "mean_best_hold_immediate": float("nan"),
                "mean_best_no_hold_immediate": float("nan"),
                "mean_best_hold_value": float("nan"),
                "mean_best_no_hold_value": float("nan"),
                "mean_chosen_hold_score": float("nan"),
                "mean_chosen_hold_immediate": float("nan"),
                "mean_chosen_hold_value": float("nan"),
            }
            for source, stats in by_source.items()
        }

    probe_result = env.evaluate_hold_candidates_many(env_ids=probe_env_ids)
    candidate_batches = probe_result.get("candidates", [])
    flat_obs: list[list[float]] = []
    flat_mapping: list[tuple[int, int]] = []
    for batch_idx, candidates in enumerate(candidate_batches):
        if not isinstance(candidates, list):
            continue
        for cand_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            if bool(candidate.get("done", False)):
                continue
            obs = candidate.get("obs")
            if not isinstance(obs, list) or not obs:
                continue
            flat_obs.append(obs)
            flat_mapping.append((batch_idx, cand_idx))

    continuation_values: dict[tuple[int, int], float] = {}
    if flat_obs:
        raw_obs = torch.as_tensor(np.asarray(flat_obs, dtype=np.float32), device=device)
        with torch.no_grad():
            probe_features = obs_adapter(raw_obs)
            _probe_logits, probe_values = model(probe_features)
        probe_values_np = probe_values.detach().cpu().numpy().astype(np.float64)
        for mapping, value in zip(flat_mapping, probe_values_np.tolist()):
            continuation_values[mapping] = float(value)

    per_env: dict[int, dict[str, float | None]] = {}
    for batch_idx, env_id in enumerate(probe_env_ids):
        source = probe_env_sources[batch_idx]
        source_stats = by_source[source]
        candidates_raw = (
            candidate_batches[batch_idx]
            if batch_idx < len(candidate_batches) and isinstance(candidate_batches[batch_idx], list)
            else []
        )
        scored_hold: list[dict[str, float]] = []
        scored_no_hold: list[dict[str, float]] = []
        chosen_action = chosen_actions_by_env_id[int(env_id)]
        chosen_score: float | None = None
        chosen_immediate: float | None = None
        chosen_value: float | None = None
        for cand_idx, candidate in enumerate(candidates_raw):
            if not isinstance(candidate, dict):
                continue
            action_index = int(candidate.get("action_index", -1))
            immediate = float(candidate.get("immediate_reward_no_hold_tax", 0.0))
            done = bool(candidate.get("done", False))
            value_term = 0.0 if done else float(continuation_values.get((batch_idx, cand_idx), 0.0))
            score = immediate + (0.0 if done else float(gamma) * value_term)
            scored = {
                "score": score,
                "immediate": immediate,
                "value": value_term,
            }
            if bool(candidate.get("hold_used", False)):
                scored_hold.append(scored)
            else:
                scored_no_hold.append(scored)
            if action_index == chosen_action:
                chosen_score = score
                chosen_immediate = immediate
                chosen_value = value_term

        if not scored_hold or not scored_no_hold:
            source_stats["unavailable"] += 1
            per_env[int(env_id)] = {
                "margin": None,
                "penalty": None,
                "chosen_score": chosen_score,
                "chosen_immediate": chosen_immediate,
                "chosen_value": chosen_value,
                "best_hold_score": None,
                "best_no_hold_score": None,
                "best_hold_immediate": None,
                "best_no_hold_immediate": None,
                "best_hold_value": None,
                "best_no_hold_value": None,
            }
            continue

        best_hold = max(scored_hold, key=lambda item: item["score"])
        best_no_hold = max(scored_no_hold, key=lambda item: item["score"])
        margin = float(best_hold["score"] - best_no_hold["score"])
        source_stats["margins"].append(margin)
        source_stats["best_hold_scores"].append(float(best_hold["score"]))
        source_stats["best_no_hold_scores"].append(float(best_no_hold["score"]))
        source_stats["best_hold_immediate"].append(float(best_hold["immediate"]))
        source_stats["best_no_hold_immediate"].append(float(best_no_hold["immediate"]))
        source_stats["best_hold_value"].append(float(best_hold["value"]))
        source_stats["best_no_hold_value"].append(float(best_no_hold["value"]))
        if chosen_score is not None:
            source_stats["chosen_hold_scores"].append(float(chosen_score))
        if chosen_immediate is not None:
            source_stats["chosen_hold_immediate"].append(float(chosen_immediate))
        if chosen_value is not None:
            source_stats["chosen_hold_value"].append(float(chosen_value))
        per_env[int(env_id)] = {
            "margin": margin,
            "penalty": None,
            "chosen_score": chosen_score,
            "chosen_immediate": chosen_immediate,
            "chosen_value": chosen_value,
            "best_hold_score": float(best_hold["score"]),
            "best_no_hold_score": float(best_no_hold["score"]),
            "best_hold_immediate": float(best_hold["immediate"]),
            "best_no_hold_immediate": float(best_no_hold["immediate"]),
            "best_hold_value": float(best_hold["value"]),
            "best_no_hold_value": float(best_no_hold["value"]),
        }

    summary_by_source: dict[str, Any] = {}
    for source, stats in by_source.items():
        summary_by_source[source] = {
            "hold_actions": int(stats["hold_actions"]),
            "non_hold_actions": int(stats["non_hold_actions"]),
            "unavailable": int(stats["unavailable"]),
            "margin": _hold_probe_summary(stats["margins"], eps, good_threshold),
            "mean_best_hold_score": _safe_recent_mean(
                stats["best_hold_scores"], window=len(stats["best_hold_scores"])
            ),
            "mean_best_no_hold_score": _safe_recent_mean(
                stats["best_no_hold_scores"], window=len(stats["best_no_hold_scores"])
            ),
            "mean_best_hold_immediate": _safe_recent_mean(
                stats["best_hold_immediate"], window=len(stats["best_hold_immediate"])
            ),
            "mean_best_no_hold_immediate": _safe_recent_mean(
                stats["best_no_hold_immediate"], window=len(stats["best_no_hold_immediate"])
            ),
            "mean_best_hold_value": _safe_recent_mean(
                stats["best_hold_value"], window=len(stats["best_hold_value"])
            ),
            "mean_best_no_hold_value": _safe_recent_mean(
                stats["best_no_hold_value"], window=len(stats["best_no_hold_value"])
            ),
            "mean_chosen_hold_score": _safe_recent_mean(
                stats["chosen_hold_scores"], window=len(stats["chosen_hold_scores"])
            ),
            "mean_chosen_hold_immediate": _safe_recent_mean(
                stats["chosen_hold_immediate"], window=len(stats["chosen_hold_immediate"])
            ),
            "mean_chosen_hold_value": _safe_recent_mean(
                stats["chosen_hold_value"], window=len(stats["chosen_hold_value"])
            ),
        }
    return per_env, summary_by_source


def new_hold_probe_accumulator() -> dict[str, Any]:
    return {
        "hold_actions": 0,
        "non_hold_actions": 0,
        "unavailable": 0,
        "margins": [],
        "best_hold_scores": [],
        "best_no_hold_scores": [],
        "chosen_hold_scores": [],
        "chosen_hold_immediate": [],
        "chosen_hold_value": [],
        "best_hold_immediate": [],
        "best_no_hold_immediate": [],
        "best_hold_value": [],
        "best_no_hold_value": [],
        "penalties": [],
    }


def evaluate_hold_swap_teacher_for_envs(
    *,
    env: WubEnvBridge,
    env_ids: list[int],
    env_piece_sources: list[str],
    obs_adapter: ObservationAdapter,
    model: PolicyValueNet,
    device: torch.device,
    gamma: float,
    teacher_tau: float,
) -> tuple[dict[int, dict[str, float | None]], dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for source in env_piece_sources:
        by_source.setdefault(
            source,
            {
                "available": 0,
                "unavailable": 0,
                "margins": [],
                "teacher_hold_probs": [],
                "best_hold_scores": [],
                "best_no_hold_scores": [],
            },
        )

    if not env_ids:
        return {}, {}

    probe_result = env.evaluate_hold_candidates_many(env_ids=env_ids)
    candidate_batches = probe_result.get("candidates", [])
    flat_obs: list[list[float]] = []
    flat_mapping: list[tuple[int, int]] = []
    for batch_idx, candidates in enumerate(candidate_batches):
        if not isinstance(candidates, list):
            continue
        for cand_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            if bool(candidate.get("done", False)):
                continue
            obs = candidate.get("obs")
            if not isinstance(obs, list) or not obs:
                continue
            flat_obs.append(obs)
            flat_mapping.append((batch_idx, cand_idx))

    continuation_values: dict[tuple[int, int], float] = {}
    if flat_obs:
        raw_obs = torch.as_tensor(np.asarray(flat_obs, dtype=np.float32), device=device)
        with torch.no_grad():
            probe_features = obs_adapter(raw_obs)
            _probe_logits, probe_values = model(probe_features)
        probe_values_np = probe_values.detach().cpu().numpy().astype(np.float64)
        for mapping, value in zip(flat_mapping, probe_values_np.tolist()):
            continuation_values[mapping] = float(value)

    per_env: dict[int, dict[str, float | None]] = {}
    for batch_idx, (env_id, source) in enumerate(zip(env_ids, env_piece_sources)):
        source_stats = by_source.setdefault(
            source,
            {
                "available": 0,
                "unavailable": 0,
                "margins": [],
                "teacher_hold_probs": [],
                "best_hold_scores": [],
                "best_no_hold_scores": [],
            },
        )
        candidates_raw = (
            candidate_batches[batch_idx]
            if batch_idx < len(candidate_batches)
            and isinstance(candidate_batches[batch_idx], list)
            else []
        )
        scored_hold: list[float] = []
        scored_no_hold: list[float] = []
        for cand_idx, candidate in enumerate(candidates_raw):
            if not isinstance(candidate, dict):
                continue
            immediate = float(candidate.get("immediate_reward_no_hold_tax", 0.0))
            done = bool(candidate.get("done", False))
            value_term = (
                0.0
                if done
                else float(continuation_values.get((batch_idx, cand_idx), 0.0))
            )
            score = immediate + (0.0 if done else float(gamma) * value_term)
            if bool(candidate.get("hold_used", False)):
                scored_hold.append(score)
            else:
                scored_no_hold.append(score)
        if not scored_hold or not scored_no_hold:
            source_stats["unavailable"] += 1
            per_env[int(env_id)] = {
                "margin": None,
                "teacher_hold_prob": None,
                "best_hold_score": None,
                "best_no_hold_score": None,
            }
            continue
        best_hold_score = float(max(scored_hold))
        best_no_hold_score = float(max(scored_no_hold))
        margin = best_hold_score - best_no_hold_score
        teacher_hold_prob = hold_swap_teacher_prob_from_margin(
            margin, teacher_tau
        )
        source_stats["available"] += 1
        source_stats["margins"].append(margin)
        source_stats["teacher_hold_probs"].append(teacher_hold_prob)
        source_stats["best_hold_scores"].append(best_hold_score)
        source_stats["best_no_hold_scores"].append(best_no_hold_score)
        per_env[int(env_id)] = {
            "margin": margin,
            "teacher_hold_prob": teacher_hold_prob,
            "best_hold_score": best_hold_score,
            "best_no_hold_score": best_no_hold_score,
        }

    summary_by_source: dict[str, Any] = {}
    for source, stats in by_source.items():
        summary_by_source[source] = {
            "available": int(stats["available"]),
            "unavailable": int(stats["unavailable"]),
            "margin": _hold_swap_margin_summary(stats["margins"]),
            "mean_teacher_hold_prob": _safe_recent_mean(
                stats["teacher_hold_probs"],
                window=len(stats["teacher_hold_probs"]),
            ),
            "mean_best_hold_score": _safe_recent_mean(
                stats["best_hold_scores"],
                window=len(stats["best_hold_scores"]),
            ),
            "mean_best_no_hold_score": _safe_recent_mean(
                stats["best_no_hold_scores"],
                window=len(stats["best_no_hold_scores"]),
            ),
        }
    return per_env, summary_by_source


def new_hold_swap_teacher_accumulator() -> dict[str, Any]:
    return {
        "available": 0,
        "unavailable": 0,
        "margins": [],
        "teacher_hold_probs": [],
        "best_hold_scores": [],
        "best_no_hold_scores": [],
    }


def accumulate_hold_swap_teacher_metrics(
    accumulator: dict[str, dict[str, Any]],
    *,
    env_ids: list[int],
    env_piece_sources: list[str],
    per_env: dict[int, dict[str, float | None]],
) -> None:
    for env_id, source in zip(env_ids, env_piece_sources):
        source_acc = accumulator.setdefault(source, new_hold_swap_teacher_accumulator())
        env_result = per_env.get(int(env_id))
        if not env_result or env_result.get("margin") is None:
            source_acc["unavailable"] += 1
            continue
        source_acc["available"] += 1
        for key, acc_key in (
            ("margin", "margins"),
            ("teacher_hold_prob", "teacher_hold_probs"),
            ("best_hold_score", "best_hold_scores"),
            ("best_no_hold_score", "best_no_hold_scores"),
        ):
            value = env_result.get(key)
            if value is None or not math.isfinite(float(value)):
                continue
            source_acc[acc_key].append(float(value))


def summarize_hold_swap_teacher_accumulator(
    accumulator: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source, stats in accumulator.items():
        summary[source] = {
            "available": int(stats["available"]),
            "unavailable": int(stats["unavailable"]),
            "margin": _hold_swap_margin_summary(stats["margins"]),
            "mean_teacher_hold_prob": _safe_recent_mean(
                stats["teacher_hold_probs"],
                window=len(stats["teacher_hold_probs"]),
            ),
            "mean_best_hold_score": _safe_recent_mean(
                stats["best_hold_scores"],
                window=len(stats["best_hold_scores"]),
            ),
            "mean_best_no_hold_score": _safe_recent_mean(
                stats["best_no_hold_scores"],
                window=len(stats["best_no_hold_scores"]),
            ),
        }
    return summary


def accumulate_hold_action_counts(
    accumulator: dict[str, dict[str, Any]],
    *,
    env_piece_sources: list[str],
    actions: list[int],
    action_dim: int,
    action_space_kind: str,
) -> None:
    for source, action in zip(env_piece_sources, actions):
        source_acc = accumulator.setdefault(source, new_hold_probe_accumulator())
        if action_index_uses_hold(int(action), action_dim, action_space_kind):
            source_acc["hold_actions"] += 1
        else:
            source_acc["non_hold_actions"] += 1


def accumulate_hold_probe_metrics(
    accumulator: dict[str, dict[str, Any]],
    *,
    env_ids: list[int],
    env_piece_sources: list[str],
    actions: list[int],
    action_dim: int,
    action_space_kind: str,
    per_env: dict[int, dict[str, float | None]],
    penalty_base: float,
    penalty_threshold: float,
) -> dict[int, float]:
    penalties_by_env_id: dict[int, float] = {}
    for env_id, source, action in zip(env_ids, env_piece_sources, actions):
        source_acc = accumulator.setdefault(source, new_hold_probe_accumulator())
        if action_index_uses_hold(int(action), action_dim, action_space_kind):
            source_acc["hold_actions"] += 1
            env_result = per_env.get(int(env_id))
            if not env_result or env_result.get("margin") is None:
                source_acc["unavailable"] += 1
                continue
            margin = float(env_result["margin"])
            penalty = compute_hold_margin_penalty(
                margin,
                base_penalty=penalty_base,
                margin_threshold=penalty_threshold,
            )
            penalties_by_env_id[int(env_id)] = penalty
            source_acc["margins"].append(margin)
            source_acc["penalties"].append(penalty)
            for key in (
                "best_hold_score",
                "best_no_hold_score",
                "best_hold_immediate",
                "best_no_hold_immediate",
                "best_hold_value",
                "best_no_hold_value",
                "chosen_score",
                "chosen_immediate",
                "chosen_value",
            ):
                value = env_result.get(key)
                if value is None or not math.isfinite(float(value)):
                    continue
                if key == "best_hold_score":
                    source_acc["best_hold_scores"].append(float(value))
                elif key == "best_no_hold_score":
                    source_acc["best_no_hold_scores"].append(float(value))
                elif key == "best_hold_immediate":
                    source_acc["best_hold_immediate"].append(float(value))
                elif key == "best_no_hold_immediate":
                    source_acc["best_no_hold_immediate"].append(float(value))
                elif key == "best_hold_value":
                    source_acc["best_hold_value"].append(float(value))
                elif key == "best_no_hold_value":
                    source_acc["best_no_hold_value"].append(float(value))
                elif key == "chosen_score":
                    source_acc["chosen_hold_scores"].append(float(value))
                elif key == "chosen_immediate":
                    source_acc["chosen_hold_immediate"].append(float(value))
                elif key == "chosen_value":
                    source_acc["chosen_hold_value"].append(float(value))
        else:
            source_acc["non_hold_actions"] += 1
    return penalties_by_env_id


def summarize_hold_probe_accumulator(
    accumulator: dict[str, dict[str, Any]],
    *,
    eps: float,
    good_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source, stats in accumulator.items():
        hold_actions = int(stats["hold_actions"])
        non_hold_actions = int(stats["non_hold_actions"])
        total_actions = hold_actions + non_hold_actions
        if non_hold_actions > 0:
            hold_rate = float(hold_actions / non_hold_actions)
        elif hold_actions > 0:
            hold_rate = float("inf")
        else:
            hold_rate = float("nan")
        summary[source] = {
            "hold_actions": hold_actions,
            "non_hold_actions": non_hold_actions,
            "total_actions": total_actions,
            "hold_rate": hold_rate,
            "unavailable": int(stats["unavailable"]),
            "margin": _hold_probe_summary(stats["margins"], eps, good_threshold),
            "mean_best_hold_score": _safe_recent_mean(
                stats["best_hold_scores"], window=len(stats["best_hold_scores"])
            ),
            "mean_best_no_hold_score": _safe_recent_mean(
                stats["best_no_hold_scores"], window=len(stats["best_no_hold_scores"])
            ),
            "mean_best_hold_immediate": _safe_recent_mean(
                stats["best_hold_immediate"], window=len(stats["best_hold_immediate"])
            ),
            "mean_best_no_hold_immediate": _safe_recent_mean(
                stats["best_no_hold_immediate"], window=len(stats["best_no_hold_immediate"])
            ),
            "mean_best_hold_value": _safe_recent_mean(
                stats["best_hold_value"], window=len(stats["best_hold_value"])
            ),
            "mean_best_no_hold_value": _safe_recent_mean(
                stats["best_no_hold_value"], window=len(stats["best_no_hold_value"])
            ),
            "mean_chosen_hold_score": _safe_recent_mean(
                stats["chosen_hold_scores"], window=len(stats["chosen_hold_scores"])
            ),
            "mean_chosen_hold_immediate": _safe_recent_mean(
                stats["chosen_hold_immediate"], window=len(stats["chosen_hold_immediate"])
            ),
            "mean_chosen_hold_value": _safe_recent_mean(
                stats["chosen_hold_value"], window=len(stats["chosen_hold_value"])
            ),
            "mean_penalty": _safe_recent_mean(
                stats["penalties"], window=len(stats["penalties"])
            ),
        }
    return summary


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
    queue_encoder_lr_scale: float,
) -> torch.optim.Optimizer:
    # Split optimizer groups so we can slow policy/trunk updates during PPO warmup
    # while keeping value-head updates at full speed.
    policy_params: list[nn.Parameter] = []
    policy_params.extend([p for p in model.policy_fc1.parameters() if p.requires_grad])
    policy_params.extend([p for p in model.policy_fc2.parameters() if p.requires_grad])
    policy_params.extend([p for p in model.policy_head.parameters() if p.requires_grad])

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
    if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
        board_adapter_params = [
            p for p in obs_adapter.board_adapter.parameters() if p.requires_grad
        ]
        queue_params = [p for p in obs_adapter.queue_fc1.parameters() if p.requires_grad]
        if board_adapter_params:
            param_groups.append(
                {
                    "params": board_adapter_params,
                    "lr": learning_rate,
                    "group_name": "adapter_board",
                }
            )
        if queue_params:
            param_groups.append(
                {
                    "params": queue_params,
                    "lr": learning_rate * max(0.0, float(queue_encoder_lr_scale)),
                    "group_name": "queue",
                }
            )
    else:
        adapter_params = [p for p in obs_adapter.parameters() if p.requires_grad]
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": learning_rate,
                    "group_name": "adapter",
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
    obs_adapter_lr_scale: float,
    queue_encoder_lr_scale: float,
) -> tuple[float, float, float, float]:
    policy_lr = base_lr * (warmup_policy_lr_scale if warmup_active else 1.0)
    value_lr = base_lr * (warmup_value_lr_scale if warmup_active else 1.0)
    board_adapter_lr = policy_lr * max(0.0, float(obs_adapter_lr_scale))
    queue_lr = base_lr * max(0.0, float(queue_encoder_lr_scale))
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", "policy"))
        if group_name == "value":
            group["lr"] = value_lr
        elif group_name == "queue":
            group["lr"] = queue_lr
        elif group_name in ("adapter_board", "adapter"):
            group["lr"] = board_adapter_lr
        else:
            group["lr"] = policy_lr
    return (
        float(policy_lr),
        float(value_lr),
        float(board_adapter_lr),
        float(queue_lr),
    )


def summarize_adapter_state(
    obs_adapter: ObservationAdapter,
    optimizer: torch.optim.Optimizer,
    policy_lr: float,
    obs_adapter_lr_scale: float,
    queue_lr: float = 0.0,
) -> dict[str, Any]:
    adapter_params = list(obs_adapter.parameters())
    adapter_total = int(sum(int(p.numel()) for p in adapter_params))
    adapter_trainable = int(
        sum(int(p.numel()) for p in adapter_params if p.requires_grad)
    )
    trainable_ids = {id(p) for p in adapter_params if p.requires_grad}
    optimizer_adapter_params = 0
    optimizer_groups: set[str] = set()
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", "policy"))
        for param in group.get("params", []):
            if id(param) in trainable_ids:
                optimizer_adapter_params += int(param.numel())
                optimizer_groups.add(group_name)

    conv_total = 0
    conv_trainable = 0
    queue_total = 0
    queue_trainable = 0
    board_adapter = (
        obs_adapter.board_adapter
        if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter)
        else obs_adapter
    )
    if isinstance(board_adapter, WubHeadFromRawObservationAdapter):
        for conv in board_adapter.conv_layers:
            for param in conv.parameters():
                count = int(param.numel())
                conv_total += count
                if param.requires_grad:
                    conv_trainable += count
    if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
        for param in obs_adapter.queue_fc1.parameters():
            count = int(param.numel())
            queue_total += count
            if param.requires_grad:
                queue_trainable += count

    adapter_frozen = adapter_trainable <= 0
    in_optimizer = optimizer_adapter_params > 0
    effective_lr = (
        float(policy_lr) * float(obs_adapter_lr_scale)
        if (adapter_trainable > 0 and in_optimizer)
        else 0.0
    )
    if adapter_total <= 0:
        status = "absent"
    elif adapter_frozen:
        status = "frozen"
    elif optimizer_adapter_params == adapter_trainable:
        status = "trainable"
    elif in_optimizer:
        status = "trainable_partial_optimizer"
    else:
        status = "trainable_not_in_optimizer"

    conv_status = "n/a"
    if isinstance(board_adapter, WubHeadFromRawObservationAdapter):
        if conv_total <= 0:
            conv_status = "none"
        elif conv_trainable <= 0:
            conv_status = "frozen"
        elif conv_trainable >= conv_total:
            conv_status = "trainable"
        else:
            conv_status = "partial"
    if queue_total <= 0:
        queue_status = "absent"
    elif queue_trainable <= 0:
        queue_status = "frozen"
    elif queue_trainable >= queue_total:
        queue_status = "trainable"
    else:
        queue_status = "partial"

    checks: list[str] = []
    if adapter_trainable > 0 and optimizer_adapter_params <= 0:
        checks.append("adapter_trainable_but_missing_from_optimizer")
    if adapter_trainable > 0 and optimizer_adapter_params != adapter_trainable:
        checks.append(
            f"optimizer_adapter_param_count_mismatch(trainable={adapter_trainable},optimizer={optimizer_adapter_params})"
        )
    if adapter_frozen and optimizer_adapter_params > 0:
        checks.append("adapter_frozen_but_present_in_optimizer")
    if conv_status == "partial":
        checks.append("adapter_conv_partially_frozen")
    return {
        "status": status,
        "adapter_total": adapter_total,
        "adapter_trainable": adapter_trainable,
        "adapter_in_optimizer": in_optimizer,
        "optimizer_adapter_params": optimizer_adapter_params,
        "optimizer_groups": sorted(optimizer_groups),
        "obs_adapter_lr_scale": float(obs_adapter_lr_scale),
        "policy_lr": float(policy_lr),
        "effective_lr": float(effective_lr),
        "conv_total": conv_total,
        "conv_trainable": conv_trainable,
        "has_conv": isinstance(board_adapter, WubHeadFromRawObservationAdapter),
        "conv_status": conv_status,
        "queue_total": queue_total,
        "queue_trainable": queue_trainable,
        "queue_status": queue_status,
        "queue_lr": float(queue_lr),
        "checks_ok": len(checks) == 0,
        "checks": checks,
    }


def format_adapter_state_log(summary: dict[str, Any]) -> str:
    groups = summary.get("optimizer_groups") or []
    if isinstance(groups, list) and groups:
        groups_str = ",".join(str(v) for v in groups)
    else:
        groups_str = "-"
    checks_ok = bool(summary.get("checks_ok", False))
    checks = summary.get("checks") or []
    if checks_ok:
        checks_str = "ok"
    elif isinstance(checks, list) and checks:
        checks_str = ";".join(str(v) for v in checks)
    else:
        checks_str = "unknown"
    return (
        "adapter("
        f"status={summary.get('status')},"
        f"conv={summary.get('conv_status')},"
        f"queue={summary.get('queue_status')},"
        f"trainable={summary.get('adapter_trainable')}/{summary.get('adapter_total')},"
        f"in_opt={'y' if summary.get('adapter_in_optimizer') else 'n'},"
        f"opt_params={summary.get('optimizer_adapter_params')},"
        f"groups={groups_str},"
        f"lr_scale={float(summary.get('obs_adapter_lr_scale', 0.0)):.3f},"
        f"eff_lr={float(summary.get('effective_lr', 0.0)):.6g},"
        f"queue_lr={float(summary.get('queue_lr', 0.0)):.6g},"
        f"checks={checks_str}"
        ")"
    )


def adapt_widened_input_weight(
    target_weight: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor | None:
    if target_weight.ndim != 2 or loaded_weight.ndim != 2:
        return None
    if int(target_weight.shape[0]) != int(loaded_weight.shape[0]):
        return None
    target_in = int(target_weight.shape[1])
    loaded_in = int(loaded_weight.shape[1])
    if loaded_in > target_in:
        return None
    expanded = target_weight.detach().clone()
    expanded.zero_()
    expanded[:, :loaded_in] = loaded_weight.to(
        device=expanded.device,
        dtype=expanded.dtype,
    )
    return expanded


def prepare_model_state_for_load(
    model: PolicyValueNet,
    model_state: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, torch.Tensor]:
    current_state = model.state_dict()
    prepared: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        target_tensor = current_state.get(key)
        if target_tensor is None:
            continue
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value_tensor = value.detach().to(dtype=target_tensor.dtype)
        if tuple(value_tensor.shape) == tuple(target_tensor.shape):
            prepared[key] = value_tensor
            continue
        if key in ("policy_fc1.weight", "value_fc1.weight"):
            expanded = adapt_widened_input_weight(target_tensor, value_tensor)
            if expanded is not None:
                prepared[key] = expanded
                print(
                    "[ppo] "
                    f"adapted widened {key} from {source_label} "
                    f"(loaded_shape={tuple(value_tensor.shape)}, target_shape={tuple(target_tensor.shape)})"
                )
                continue
        if key == "policy_head.weight":
            adapted = adapt_hold_step_policy_head_weight(target_tensor, value_tensor)
            if adapted is not None:
                prepared[key] = adapted
                print(
                    "[ppo] "
                    f"adapted hold-step {key} from {source_label} "
                    f"(loaded_shape={tuple(value_tensor.shape)}, target_shape={tuple(target_tensor.shape)})"
                )
                continue
        if key == "policy_head.bias":
            adapted = adapt_hold_step_policy_head_bias(target_tensor, value_tensor)
            if adapted is not None:
                prepared[key] = adapted
                print(
                    "[ppo] "
                    f"adapted hold-step {key} from {source_label} "
                    f"(loaded_shape={tuple(value_tensor.shape)}, target_shape={tuple(target_tensor.shape)})"
                )
                continue
        print(
            "[ppo] warning: skipped incompatible tensor from "
            f"{source_label} for key={key} "
            f"(loaded_shape={tuple(value_tensor.shape)}, target_shape={tuple(target_tensor.shape)})"
        )
    return prepared


def remap_adapter_state_for_queue_transfer(
    obs_adapter: ObservationAdapter,
    adapter_state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
        return adapter_state
    has_board_prefix = any(str(key).startswith("board_adapter.") for key in adapter_state)
    has_queue_prefix = any(str(key).startswith("queue_fc1.") for key in adapter_state)
    if has_board_prefix or has_queue_prefix:
        return adapter_state
    return {
        f"board_adapter.{key}": value
        for key, value in adapter_state.items()
    }


def adapt_hold_step_policy_head_weight(
    target_weight: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor | None:
    if target_weight.ndim != 2 or loaded_weight.ndim != 2:
        return None
    loaded_out, loaded_in = loaded_weight.shape
    target_out, target_in = target_weight.shape
    if loaded_in != target_in or loaded_out <= 0 or loaded_out % 2 != 0:
        return None
    loaded_no_hold = loaded_out // 2
    if target_out != loaded_no_hold + 1:
        return None
    adapted = target_weight.detach().clone()
    adapted.zero_()
    adapted[:loaded_no_hold, :] = loaded_weight[:loaded_no_hold, :].to(
        device=adapted.device,
        dtype=adapted.dtype,
    )
    return adapted


def adapt_hold_step_policy_head_bias(
    target_bias: torch.Tensor,
    loaded_bias: torch.Tensor,
) -> torch.Tensor | None:
    if target_bias.ndim != 1 or loaded_bias.ndim != 1:
        return None
    loaded_out = int(loaded_bias.shape[0])
    target_out = int(target_bias.shape[0])
    if loaded_out <= 0 or loaded_out % 2 != 0:
        return None
    loaded_no_hold = loaded_out // 2
    if target_out != loaded_no_hold + 1:
        return None
    adapted = target_bias.detach().clone()
    adapted.zero_()
    adapted[:loaded_no_hold] = loaded_bias[:loaded_no_hold].to(
        device=adapted.device,
        dtype=adapted.dtype,
    )
    adapted[-1] = -0.5
    return adapted


def adapt_artifact_policy_head_for_hold_step_transfer(
    wp: np.ndarray,
    bp: np.ndarray,
    target_action_dim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if wp.ndim != 2 or bp.ndim != 1:
        return None
    loaded_action_dim = int(wp.shape[1])
    if loaded_action_dim <= 0 or loaded_action_dim % 2 != 0:
        return None
    loaded_no_hold = loaded_action_dim // 2
    if int(target_action_dim) != loaded_no_hold + 1:
        return None
    adapted_wp = np.zeros((wp.shape[0], int(target_action_dim)), dtype=np.float32)
    adapted_bp = np.zeros((int(target_action_dim),), dtype=np.float32)
    adapted_wp[:, :loaded_no_hold] = wp[:, :loaded_no_hold]
    adapted_bp[:loaded_no_hold] = bp[:loaded_no_hold]
    adapted_bp[-1] = -0.5
    return adapted_wp, adapted_bp


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
        "archId": (
            "hold_step"
            if cfg.action_space_kind == "placement_hold_step_v2"
            else "full"
        ),
        "queuePolicyId": cfg.queue_policy_id,
        "pipelineId": pipeline_id,
        "pieceSourceProfile": cfg.generator_schedule[0],
        "pieceSourceSchedule": list(cfg.generator_schedule),
        "observationSpace": policy_observation_space,
        "phaseContextEnabled": bool(cfg.phase_context_enabled),
        "createdAtMs": now_ms,
        "inputDim": int(obs_dim),
        "hiddenDim": int(cfg.hidden_dim),
        "actionDim": int(action_dim),
        "actionSpaceKind": str(cfg.action_space_kind),
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
    if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
        artifact["encoderModel"] = obs_adapter.board_adapter.to_exported_encoder_model()
        queue_w = (
            obs_adapter.queue_fc1.weight.detach().cpu().numpy().astype(np.float32)
        )
        queue_b = (
            obs_adapter.queue_fc1.bias.detach().cpu().numpy().astype(np.float32)
        )
        artifact["queueEncoder"] = {
            "inputDim": int(obs_adapter.queue_input_dim),
            "hiddenDim": int(obs_adapter.queue_hidden_dim),
            "w1": queue_w.T.reshape(-1).tolist(),  # [I, H]
            "b1": queue_b.reshape(-1).tolist(),
        }
    elif isinstance(obs_adapter, WubHeadFromRawObservationAdapter):
        artifact["encoderModel"] = obs_adapter.to_exported_encoder_model()
    return artifact


def load_from_artifact(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    artifact_path: Path,
) -> tuple[str, bool, str]:
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    observation_space_raw = raw.get("observationSpace")
    observation_space = (
        "raw_v1" if observation_space_raw == "raw_v1" else "model_head_v1"
    )
    phase_context_enabled = bool(raw.get("phaseContextEnabled", True))
    action_space_kind_raw = raw.get("actionSpaceKind")
    action_space_kind = (
        "placement_hold_step_v2"
        if action_space_kind_raw == "placement_hold_step_v2"
        else "placement_full_v1"
    )
    weights = raw.get("weights", {})
    input_dim = int(raw.get("inputDim", 0))
    hidden_dim = int(raw.get("hiddenDim", 0))
    action_dim = int(raw.get("actionDim", 0))
    queue_encoder_payload = raw.get("queueEncoder")

    if input_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
        raise ValueError("Invalid artifact dims.")

    target_input_dim = int(model.policy_fc1.in_features)
    if input_dim > target_input_dim:
        raise ValueError(
            f"Artifact inputDim mismatch. artifact={input_dim} model={target_input_dim}"
        )
    if model.policy_fc1.out_features != hidden_dim:
        raise ValueError(
            f"Artifact hiddenDim mismatch. artifact={hidden_dim} model={model.policy_fc1.out_features}"
        )
    target_action_dim = int(model.policy_head.out_features)
    if target_action_dim != action_dim:
        if not (
            action_space_kind == "placement_full_v1"
            and adapt_artifact_policy_head_for_hold_step_transfer(
                np.zeros((hidden_dim, action_dim), dtype=np.float32),
                np.zeros((action_dim,), dtype=np.float32),
                target_action_dim,
            )
            is not None
        ):
            raise ValueError(
                f"Artifact actionDim mismatch. artifact={action_dim} model={target_action_dim}"
            )
    if model.value_head.out_features != 1:
        raise ValueError(
            "Invalid model value head shape."
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
    if target_action_dim != action_dim:
        adapted_policy_head = adapt_artifact_policy_head_for_hold_step_transfer(
            wp,
            bp,
            target_action_dim,
        )
        if adapted_policy_head is None:
            raise ValueError(
                f"Artifact actionDim mismatch. artifact={action_dim} model={target_action_dim}"
            )
        wp, bp = adapted_policy_head
        print(
            "[ppo] adapted hold-step policy head from artifact "
            f"{artifact_path.name} (artifact_action_dim={action_dim}, target_action_dim={target_action_dim})"
        )
        action_dim = target_action_dim

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

    expanded_w1 = np.zeros((target_input_dim, hidden_dim), dtype=np.float32)
    expanded_w1[:input_dim, :] = w1
    expanded_wv1 = np.zeros((target_input_dim, hidden_dim), dtype=np.float32)
    expanded_wv1[:input_dim, :] = wv1

    with torch.no_grad():
        model.policy_fc1.weight.copy_(torch.from_numpy(expanded_w1.T))
        model.policy_fc1.bias.copy_(torch.from_numpy(b1))
        model.policy_fc2.weight.copy_(torch.from_numpy(w2.T))
        model.policy_fc2.bias.copy_(torch.from_numpy(b2))
        model.policy_head.weight.copy_(torch.from_numpy(wp.T))
        model.policy_head.bias.copy_(torch.from_numpy(bp))
        model.value_fc1.weight.copy_(torch.from_numpy(expanded_wv1.T))
        model.value_fc1.bias.copy_(torch.from_numpy(bv1))
        model.value_fc2.weight.copy_(torch.from_numpy(wv2.T))
        model.value_fc2.bias.copy_(torch.from_numpy(bv2))
        model.value_head.weight.copy_(torch.from_numpy(wv.reshape(1, hidden_dim)))
        model.value_head.bias.copy_(torch.from_numpy(bv))
    encoder_payload = raw.get("encoderModel")
    if isinstance(encoder_payload, dict):
        if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
            obs_adapter.board_adapter.load_from_exported_encoder_model(encoder_payload)
        elif isinstance(obs_adapter, WubHeadFromRawObservationAdapter):
            obs_adapter.load_from_exported_encoder_model(encoder_payload)
    if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter) and isinstance(
        queue_encoder_payload, dict
    ):
        queue_input_dim = int(queue_encoder_payload.get("inputDim", 0))
        queue_hidden_dim = int(queue_encoder_payload.get("hiddenDim", 0))
        if queue_input_dim != obs_adapter.queue_input_dim:
            raise ValueError(
                "Artifact queueEncoder inputDim mismatch. "
                f"artifact={queue_input_dim} runtime={obs_adapter.queue_input_dim}"
            )
        if queue_hidden_dim != obs_adapter.queue_hidden_dim:
            raise ValueError(
                "Artifact queueEncoder hiddenDim mismatch. "
                f"artifact={queue_hidden_dim} runtime={obs_adapter.queue_hidden_dim}"
            )
        queue_w = np.asarray(
            queue_encoder_payload.get("w1", []),
            dtype=np.float32,
        ).reshape(queue_input_dim, queue_hidden_dim)
        queue_b = np.asarray(
            queue_encoder_payload.get("b1", []),
            dtype=np.float32,
        ).reshape(queue_hidden_dim)
        with torch.no_grad():
            obs_adapter.queue_fc1.weight.copy_(torch.from_numpy(queue_w.T))
            obs_adapter.queue_fc1.bias.copy_(torch.from_numpy(queue_b))
    return observation_space, phase_context_enabled, action_space_kind


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
    load_optimizer_state: bool = True,
    expected_phase_context_enabled: bool | None = None,
    expected_action_space_kind: str | None = None,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
    model_state = checkpoint["model_state_dict"]
    checkpoint_config = checkpoint.get("config")
    checkpoint_policy_head_shape_mismatch = False
    if isinstance(model_state, dict):
        loaded_policy_head_weight = model_state.get("policy_head.weight")
        if torch.is_tensor(loaded_policy_head_weight):
            checkpoint_policy_head_shape_mismatch = tuple(
                loaded_policy_head_weight.shape
            ) != tuple(model.policy_head.weight.shape)
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
    prepared_model_state = prepare_model_state_for_load(
        model,
        model_state,
        source_label=f"checkpoint:{checkpoint_path.name}",
    )
    model.load_state_dict(prepared_model_state, strict=False)
    adapter_state = checkpoint.get("obs_adapter_state_dict")
    if isinstance(adapter_state, dict):
        prepared_adapter_state = remap_adapter_state_for_queue_transfer(
            obs_adapter,
            adapter_state,
        )
        obs_adapter.load_state_dict(prepared_adapter_state, strict=False)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    allow_optimizer_state = load_optimizer_state
    if allow_optimizer_state and checkpoint_policy_head_shape_mismatch:
        allow_optimizer_state = False
        print(
            "[ppo] warning: optimizer state was not loaded from checkpoint "
            "(policy head shape mismatch requires fresh optimizer state)."
        )
    if (
        allow_optimizer_state
        and expected_action_space_kind is not None
        and isinstance(checkpoint_config, dict)
        and "action_space_kind" in checkpoint_config
    ):
        checkpoint_action_space_kind = str(
            checkpoint_config.get("action_space_kind")
        ).strip()
        if checkpoint_action_space_kind != str(expected_action_space_kind).strip():
            allow_optimizer_state = False
            print(
                "[ppo] warning: optimizer state was not loaded from checkpoint "
                "(action-space mismatch requires fresh optimizer state)."
            )
    if load_optimizer_state and isinstance(optimizer_state, dict):
        try:
            if allow_optimizer_state:
                optimizer.load_state_dict(optimizer_state)
        except ValueError as error:
            print(
                "[ppo] warning: optimizer state was not loaded from checkpoint "
                f"(param-group mismatch). Using fresh optimizer state. detail={error}"
            )
    if (
        expected_phase_context_enabled is not None
        and isinstance(checkpoint_config, dict)
        and "phase_context_enabled" in checkpoint_config
    ):
        checkpoint_phase_context_enabled = bool(
            checkpoint_config.get("phase_context_enabled")
        )
        if checkpoint_phase_context_enabled != expected_phase_context_enabled:
            print(
                "[ppo] warning: checkpoint phase-context mismatch "
                f"(checkpoint={'on' if checkpoint_phase_context_enabled else 'off'}, "
                f"run={'on' if expected_phase_context_enabled else 'off'})."
            )
    if (
        expected_action_space_kind is not None
        and isinstance(checkpoint_config, dict)
        and "action_space_kind" in checkpoint_config
    ):
        checkpoint_action_space_kind = str(
            checkpoint_config.get("action_space_kind")
        ).strip()
        if checkpoint_action_space_kind != str(expected_action_space_kind).strip():
            print(
                "[ppo] warning: checkpoint action-space mismatch "
                f"(checkpoint={checkpoint_action_space_kind}, "
                f"run={expected_action_space_kind})."
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
    reward_blend_step: int,
    reward_blend_unit: str,
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
        reward_fn_from = (
            cfg.reward_functions[0]
            if len(cfg.reward_functions) >= 1
            else "v1"
        )
        reward_fn_to = (
            cfg.reward_functions[1]
            if len(cfg.reward_functions) >= 2
            else reward_fn_from
        )
        with WubEnvBridge(server_cmd=server_cmd, cwd=repo_root) as val_env:
            init_result = val_env.init(
                mode_id=cfg.mode_id,
                num_envs=cfg.num_envs,
                model_path=cfg.model_path,
                observation_space=cfg.observation_space,
                phase_context_enabled=cfg.phase_context_enabled,
                placement_execution_mode=cfg.placement_execution_mode,
                action_space_kind=cfg.action_space_kind,
                piece_source_profile=piece_source_profile,
                queue_policy_id=cfg.queue_policy_id,
                max_pieces_per_episode=cfg.max_pieces_per_episode_val,
                reward_function_from=reward_fn_from,
                reward_function_to=reward_fn_to,
                reward_blend_timesteps=reward_blend_total_for_env(cfg),
                reward_blend_unit=reward_blend_unit,
                reward_blend_start_step=max(0, int(reward_blend_step)),
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
            diag_holes_created_values: list[float] = []
            diag_max_height_values: list[float] = []
            diag_kick_assisted_locks_values: list[float] = []
            diag_hold_uses_values: list[float] = []
            hold_probe_accumulator: dict[str, dict[str, Any]] = {}
            hold_swap_accumulator: dict[str, dict[str, Any]] = {}
            blend_step_term_values: dict[str, list[float]] = {
                key: [] for key in BLEND_STEP_TERM_KEYS
            }
            episode_term_keys = [
                key
                for key in reward_component_aliases.keys()
                if key not in BLEND_STEP_TERM_KEYS
            ]
            max_steps = max(8, int(cfg.max_pieces_per_episode_val) * 2)

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
                    for key in episode_term_keys
                }
                ep_holes_created = np.zeros(env_count, dtype=np.float64)
                ep_max_height = np.zeros(env_count, dtype=np.float64)
                ep_kick_assisted_locks = np.zeros(env_count, dtype=np.float64)
                ep_hold_uses = np.zeros(env_count, dtype=np.float64)
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
                    hold_swap_coef_now = hold_swap_distill_coef_for_update(cfg, update)
                    if cfg.hold_margin_probe or cfg.hold_margin_penalty_base > 0.0:
                        if not is_hold_probe_supported(cfg.action_space_kind):
                            accumulate_hold_action_counts(
                                hold_probe_accumulator,
                                env_piece_sources=[piece_source_profile] * env_count,
                                actions=actions_np,
                                action_dim=int(mask_np.shape[1]),
                                action_space_kind=cfg.action_space_kind,
                            )
                    if is_hold_probe_supported(cfg.action_space_kind) and (
                        cfg.hold_margin_probe or cfg.hold_margin_penalty_base > 0.0
                    ):
                        per_env_hold_probe, _hold_probe_summary_unused = (
                            evaluate_hold_margin_for_actions(
                                env=val_env,
                                env_ids=env_ids,
                                actions=actions_np,
                                env_piece_sources=[piece_source_profile] * env_count,
                                obs_adapter=obs_adapter,
                                model=model,
                                device=device,
                                gamma=cfg.gamma,
                                action_dim=int(mask_np.shape[1]),
                                action_space_kind=cfg.action_space_kind,
                                eps=cfg.hold_margin_eps,
                                good_threshold=cfg.hold_margin_good_threshold,
                            )
                        )
                        accumulate_hold_probe_metrics(
                            hold_probe_accumulator,
                            env_ids=env_ids,
                            env_piece_sources=[piece_source_profile] * env_count,
                            actions=actions_np,
                            action_dim=int(mask_np.shape[1]),
                            action_space_kind=cfg.action_space_kind,
                            per_env=per_env_hold_probe,
                            penalty_base=0.0,
                            penalty_threshold=cfg.hold_margin_penalty_threshold,
                        )
                    if (
                        str(cfg.action_space_kind).strip().lower()
                        == "placement_hold_step_v2"
                        and (
                            cfg.hold_swap_probe
                            or hold_swap_coef_now > 0.0
                        )
                    ):
                        hold_idx = int(mask_np.shape[1]) - 1
                        hold_env_pairs = [
                            (env_id, piece_source_profile)
                            for env_idx, env_id in enumerate(env_ids)
                            if hold_idx >= 0 and mask_np[env_idx, hold_idx] > 0.5
                        ]
                        if hold_env_pairs:
                            hold_env_ids = [env_id for env_id, _source in hold_env_pairs]
                            hold_sources = [
                                source for _env_id, source in hold_env_pairs
                            ]
                            per_env_hold_swap, _unused_summary = (
                                evaluate_hold_swap_teacher_for_envs(
                                    env=val_env,
                                    env_ids=hold_env_ids,
                                    env_piece_sources=hold_sources,
                                    obs_adapter=obs_adapter,
                                    model=model,
                                    device=device,
                                    gamma=cfg.gamma,
                                    teacher_tau=cfg.hold_swap_teacher_tau,
                                )
                            )
                            accumulate_hold_swap_teacher_metrics(
                                hold_swap_accumulator,
                                env_ids=hold_env_ids,
                                env_piece_sources=hold_sources,
                                per_env=per_env_hold_swap,
                            )

                    step_result = val_env.step_many(env_ids=env_ids, actions=actions_np)
                    rewards_np = np.asarray(step_result["rewards"], dtype=np.float32)
                    dones_np = np.asarray(step_result["dones"], dtype=np.float32)
                    infos_raw = step_result.get("infos", [])
                    next_action_masks_np = np.asarray(
                        step_result["action_masks"], dtype=np.float32
                    )

                    active_mask = ~done_mask
                    ep_return[active_mask] += rewards_np[active_mask].astype(np.float64)
                    ep_length[active_mask] += 1

                    if isinstance(infos_raw, list):
                        max_info = min(len(infos_raw), env_count)
                        for blend_key in BLEND_STEP_TERM_KEYS:
                            alias_keys = reward_component_aliases.get(blend_key)
                            if not alias_keys:
                                continue
                            values: list[float] = []
                            for env_idx in range(max_info):
                                if not active_mask[env_idx]:
                                    continue
                                info = infos_raw[env_idx]
                                value = _info_num(
                                    info,
                                    alias_keys,
                                    default=float("nan"),
                                )
                                if math.isfinite(value):
                                    values.append(value)
                            if values:
                                blend_step_term_values[blend_key].append(
                                    float(np.mean(values))
                                )
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
                            holes_delta = _info_num(
                                info,
                                ("holesDelta",),
                                default=0.0,
                            )
                            if holes_delta > 0:
                                ep_holes_created[env_idx] += holes_delta
                            stack_height_after = _info_num(
                                info,
                                ("stackHeightAfter",),
                                default=float("nan"),
                            )
                            if math.isfinite(stack_height_after):
                                ep_max_height[env_idx] = max(
                                    ep_max_height[env_idx], stack_height_after
                                )
                            lock_observed = (
                                _info_num(info, ("lockObserved",), default=0.0) > 0.5
                            )
                            if lock_observed:
                                if (
                                    _info_num(
                                        info,
                                        ("placementSrsKickCount",),
                                        default=0.0,
                                    )
                                    > 0.0
                                ):
                                    ep_kick_assisted_locks[env_idx] += 1.0
                                if (
                                    _info_num(
                                        info,
                                        ("placementHoldUsed",),
                                        default=0.0,
                                    )
                                    > 0.5
                                ):
                                    ep_hold_uses[env_idx] += 1.0
                            elif (
                                _info_num(
                                    info,
                                    ("placementHoldUsed",),
                                    default=0.0,
                                )
                                > 0.5
                            ):
                                ep_hold_uses[env_idx] += 1.0
                            for key in episode_term_keys:
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
                        next_action_masks_np
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
                diag_holes_created_values.extend(ep_holes_created.tolist())
                diag_max_height_values.extend(ep_max_height.tolist())
                diag_kick_assisted_locks_values.extend(
                    ep_kick_assisted_locks.tolist()
                )
                diag_hold_uses_values.extend(ep_hold_uses.tolist())

            terms = {
                key: _safe_recent_mean(values, window=len(values))
                for key, values in term_values.items()
            }
            for key in BLEND_STEP_TERM_KEYS:
                terms[key] = _safe_recent_mean(
                    blend_step_term_values.get(key, []),
                    window=100,
                )

            return {
                "enabled": True,
                "episodes": int(len(returns)),
                "piece_source_profile": piece_source_profile,
                "mean_return": _safe_recent_mean(returns, window=len(returns)),
                "mean_length": _safe_recent_mean(lengths, window=len(lengths)),
                "terms": terms,
                "diagnostics": {
                    "holes_created_total": _safe_recent_mean(
                        diag_holes_created_values,
                        window=len(diag_holes_created_values),
                    ),
                    "max_height_reached": _safe_recent_mean(
                        diag_max_height_values,
                        window=len(diag_max_height_values),
                    ),
                    "kick_assisted_locks": _safe_recent_mean(
                        diag_kick_assisted_locks_values,
                        window=len(diag_kick_assisted_locks_values),
                    ),
                    "hold_uses": _safe_recent_mean(
                        diag_hold_uses_values,
                        window=len(diag_hold_uses_values),
                    ),
                    "hold_probe": summarize_hold_probe_accumulator(
                        hold_probe_accumulator,
                        eps=cfg.hold_margin_eps,
                        good_threshold=cfg.hold_margin_good_threshold,
                    ).get(piece_source_profile),
                    "hold_swap": summarize_hold_swap_teacher_accumulator(
                        hold_swap_accumulator
                    ).get(piece_source_profile),
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


def _format_hold_probe_log_lines(prefix: str, probe: Any) -> list[str]:
    if not isinstance(probe, dict):
        return []
    margin = probe.get("margin")
    margin_dict = margin if isinstance(margin, dict) else {}
    lines = [
        (
            f"{prefix}: "
            f"hold={int(probe.get('hold_actions', 0))} "
            f"non_hold={int(probe.get('non_hold_actions', 0))} "
            f"rate={_fmt_float(probe.get('hold_rate'))} "
            f"unavail={int(probe.get('unavailable', 0))} "
            f"penalty={_fmt_float(probe.get('mean_penalty'))}"
        )
    ]
    margin_count = int(margin_dict.get("count", 0)) if isinstance(margin_dict, dict) else 0
    if margin_count <= 0:
        return lines
    lines.extend(
        [
            (
                "      margin: "
                f"mean={_fmt_float(margin_dict.get('mean'))} "
                f"med={_fmt_float(margin_dict.get('median'))} "
                f"p25={_fmt_float(margin_dict.get('p25'))} "
                f"p75={_fmt_float(margin_dict.get('p75'))} "
                f"neg={_fmt_float(margin_dict.get('frac_negative'))} "
                f"lt_eps={_fmt_float(margin_dict.get('frac_below_eps'))} "
                f"gt_good={_fmt_float(margin_dict.get('frac_above_good'))}"
            ),
            (
                "      scores: "
                f"chosen={_fmt_float(probe.get('mean_chosen_hold_score'))} "
                f"best_hold={_fmt_float(probe.get('mean_best_hold_score'))} "
                f"best_no_hold={_fmt_float(probe.get('mean_best_no_hold_score'))} "
                f"| imm(chosen={_fmt_float(probe.get('mean_chosen_hold_immediate'))},"
                f"hold={_fmt_float(probe.get('mean_best_hold_immediate'))},"
                f"no_hold={_fmt_float(probe.get('mean_best_no_hold_immediate'))}) "
                f"| v(chosen={_fmt_float(probe.get('mean_chosen_hold_value'))},"
                f"hold={_fmt_float(probe.get('mean_best_hold_value'))},"
                f"no_hold={_fmt_float(probe.get('mean_best_no_hold_value'))})"
            ),
        ]
    )
    return lines


def _format_hold_swap_log_lines(prefix: str, summary: Any) -> list[str]:
    if not isinstance(summary, dict):
        return []
    margin = summary.get("margin")
    margin_dict = margin if isinstance(margin, dict) else {}
    return [
        (
            f"{prefix}: "
            f"avail={int(summary.get('available', 0))} "
            f"unavail={int(summary.get('unavailable', 0))} "
            f"p_hold={_fmt_float(summary.get('mean_teacher_hold_prob'))} "
            f"best_hold={_fmt_float(summary.get('mean_best_hold_score'))} "
            f"best_no_hold={_fmt_float(summary.get('mean_best_no_hold_score'))}"
        ),
        (
            "      margin: "
            f"mean={_fmt_float(margin_dict.get('mean'))} "
            f"med={_fmt_float(margin_dict.get('median'))} "
            f"p25={_fmt_float(margin_dict.get('p25'))} "
            f"p75={_fmt_float(margin_dict.get('p75'))} "
            f"pos={_fmt_float(margin_dict.get('frac_positive'))} "
            f"neg={_fmt_float(margin_dict.get('frac_negative'))}"
        ),
    ]


BLEND_STEP_TERM_KEYS: tuple[str, ...] = (
    "blend_t",
    "blend_legacy_weight",
    "blend_target_weight",
    "blend_transition_step",
    "blend_transition_total_steps",
)


def _reward_hierarchy_from_terms(terms: dict[str, Any] | None) -> dict[str, Any]:
    data = terms if isinstance(terms, dict) else {}
    return {
        "overall": {
            "final": data.get("reward_final"),
            "base": data.get("reward_base"),
            "hold_margin_penalty": data.get("term_hold_margin_penalty"),
            "top_out_term": (
                data.get("top_out_term")
                if data.get("top_out_term") is not None
                else data.get("top_out_penalty")
            ),
        },
        "blend": {
            "t": data.get("blend_t"),
            "legacy_weight": data.get("blend_legacy_weight"),
            "target_weight": data.get("blend_target_weight"),
            "transition_step": data.get("blend_transition_step"),
            "transition_total_steps": data.get("blend_transition_total_steps"),
        },
        "v1": {
            "base": data.get("v1_base"),
            "base_raw": data.get("v1_base_raw"),
            "terms": {
                "lines": data.get("v1_term_lines"),
                "score": data.get("v1_term_score"),
                "time": data.get("v1_term_time"),
                "height": data.get("v1_term_height"),
                "holes": data.get("v1_term_holes"),
                "bumpiness": data.get("v1_term_bumpiness"),
                "board_score": data.get("v1_term_board_score"),
                "board_quality": data.get("v1_term_board_quality"),
                "board_quality_absolute": data.get("v1_term_board_quality_abs"),
                "full_clear": data.get("v1_term_full_clear"),
                "top_out": data.get("v1_term_top_out"),
            },
        },
        "v2": {
            "base": data.get("v2_base"),
            "base_raw": data.get("v2_base_raw"),
            "terms": {
                "lines": data.get("v2_term_lines"),
                "score": data.get("v2_term_score"),
                "time": data.get("v2_term_time"),
                "height": data.get("v2_term_height"),
                "holes": data.get("v2_term_holes"),
                "bumpiness": data.get("v2_term_bumpiness"),
                "board_score": data.get("v2_term_board_score"),
                "board_quality": data.get("v2_term_board_quality"),
                "board_quality_absolute": data.get("v2_term_board_quality_abs"),
                "full_clear": data.get("v2_term_full_clear"),
                "top_out": data.get("v2_term_top_out"),
            },
        },
        "blended_terms": {
            "lines": data.get("term_lines"),
            "score": data.get("term_score"),
            "time": data.get("term_time"),
            "height": data.get("term_height"),
            "holes": data.get("term_holes"),
            "bumpiness": data.get("term_bumpiness"),
            "board_score": data.get("term_board_score"),
            "board_quality": data.get("term_board_quality"),
            "board_quality_absolute": data.get("term_board_quality_abs"),
            "full_clear": data.get("term_full_clear"),
            "hold_margin_penalty": data.get("term_hold_margin_penalty"),
            "top_out": data.get("term_top_out"),
        },
    }


def load_bc_dataset(
    dataset_path: Path,
    obs_dim: int,
    action_dim: int,
    max_records: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = raw.get("records")
    if not isinstance(records, list):
        raise ValueError("BC dataset missing records array.")

    obs_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    action_rows: list[int] = []
    return_rows: list[float] = []
    return_mask_rows: list[float] = []
    session_id_rows: list[str] = []

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

        if not isinstance(obs, list) or any(not _is_finite_number(v) for v in obs):
            skipped["invalid_obs"] += 1
            continue
        obs_values = [float(v) for v in obs]
        if len(obs_values) < obs_dim:
            if len(obs_values) < RAW_LEGACY_OBS_DIM:
                skipped["invalid_obs"] += 1
                continue
            obs_values = obs_values + [0.0] * (obs_dim - len(obs_values))
        elif len(obs_values) > obs_dim:
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

        obs_rows.append(obs_values)
        mask_rows.append(mask)
        action_rows.append(action_index)
        source = rec.get("source")
        session_id: str | None = None
        if isinstance(source, dict):
            candidate = source.get("sessionId")
            if isinstance(candidate, str) and candidate.strip():
                session_id = candidate.strip()
        if session_id is None:
            candidate = rec.get("sessionId")
            if isinstance(candidate, str) and candidate.strip():
                session_id = candidate.strip()
        session_id_rows.append(session_id or f"record_{len(session_id_rows)}")
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
        "session_ids": np.asarray(session_id_rows, dtype=object),
    }
    stats = {
        "total_records": len(records),
        "used_records": int(payload["obs"].shape[0]),
        "with_returns": int(np.sum(payload["returns_mask"])),
        "sessions": int(len(set(session_id_rows))),
        **skipped,
    }
    return payload, stats


def split_bc_indices_by_session(
    session_ids: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    session_list = [str(v) for v in session_ids.tolist()]
    unique_sessions = list(dict.fromkeys(session_list))
    session_count = len(unique_sessions)
    if session_count <= 1 or val_fraction <= 0.0:
        all_indices = np.arange(len(session_list), dtype=np.int64)
        return all_indices, np.zeros((0,), dtype=np.int64), {
            "session_count": session_count,
            "train_sessions": session_count,
            "val_sessions": 0,
            "train_records": int(len(all_indices)),
            "val_records": 0,
            "val_fraction_effective": 0.0,
        }

    shuffled_sessions = list(unique_sessions)
    random.Random(seed).shuffle(shuffled_sessions)
    val_session_count = int(round(session_count * float(val_fraction)))
    val_session_count = max(1, min(session_count - 1, val_session_count))
    val_sessions = set(shuffled_sessions[:val_session_count])
    train_indices = np.asarray(
        [idx for idx, sid in enumerate(session_list) if sid not in val_sessions],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [idx for idx, sid in enumerate(session_list) if sid in val_sessions],
        dtype=np.int64,
    )
    if train_indices.size == 0 or val_indices.size == 0:
        all_indices = np.arange(len(session_list), dtype=np.int64)
        return all_indices, np.zeros((0,), dtype=np.int64), {
            "session_count": session_count,
            "train_sessions": session_count,
            "val_sessions": 0,
            "train_records": int(len(all_indices)),
            "val_records": 0,
            "val_fraction_effective": 0.0,
        }
    return train_indices, val_indices, {
        "session_count": session_count,
        "train_sessions": int(session_count - val_session_count),
        "val_sessions": int(val_session_count),
        "train_records": int(train_indices.size),
        "val_records": int(val_indices.size),
        "val_fraction_effective": float(val_indices.size / max(1, len(session_list))),
    }


def build_bc_optimizer(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    *,
    learning_rate: float,
    policy_lr_scale: float,
    policy_head_lr_scale: float,
    value_lr_scale: float,
    adapter_lr_scale: float,
    queue_lr_scale: float,
) -> torch.optim.Optimizer:
    param_groups: list[dict[str, Any]] = []

    policy_torso_params: list[nn.Parameter] = []
    policy_torso_params.extend(
        [p for p in model.policy_fc1.parameters() if p.requires_grad]
    )
    policy_torso_params.extend(
        [p for p in model.policy_fc2.parameters() if p.requires_grad]
    )
    if policy_torso_params:
        param_groups.append(
            {
                "params": policy_torso_params,
                "lr": learning_rate * max(0.0, float(policy_lr_scale)),
                "group_name": "bc_policy",
            }
        )

    policy_head_params = [p for p in model.policy_head.parameters() if p.requires_grad]
    if policy_head_params:
        param_groups.append(
            {
                "params": policy_head_params,
                "lr": learning_rate * max(0.0, float(policy_head_lr_scale)),
                "group_name": "bc_policy_head",
            }
        )

    value_params: list[nn.Parameter] = []
    value_params.extend([p for p in model.value_fc1.parameters() if p.requires_grad])
    value_params.extend([p for p in model.value_fc2.parameters() if p.requires_grad])
    value_params.extend([p for p in model.value_head.parameters() if p.requires_grad])
    if value_params:
        param_groups.append(
            {
                "params": value_params,
                "lr": learning_rate * max(0.0, float(value_lr_scale)),
                "group_name": "bc_value",
            }
        )

    if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter):
        board_params = [
            p for p in obs_adapter.board_adapter.parameters() if p.requires_grad
        ]
        if board_params:
            param_groups.append(
                {
                    "params": board_params,
                    "lr": learning_rate * max(0.0, float(adapter_lr_scale)),
                    "group_name": "bc_adapter_board",
                }
            )
        queue_params = [p for p in obs_adapter.queue_fc1.parameters() if p.requires_grad]
        if queue_params:
            param_groups.append(
                {
                    "params": queue_params,
                    "lr": learning_rate * max(0.0, float(queue_lr_scale)),
                    "group_name": "bc_queue",
                }
            )
    else:
        adapter_params = [p for p in obs_adapter.parameters() if p.requires_grad]
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": learning_rate * max(0.0, float(adapter_lr_scale)),
                    "group_name": "bc_adapter",
                }
            )

    if not param_groups:
        raise ValueError("No trainable parameters found for BC optimizer.")
    return torch.optim.Adam(param_groups, lr=learning_rate, eps=1e-5)


def scale_bc_hold_output_gradients(
    model: PolicyValueNet,
    *,
    action_space_kind: str,
    hold_output_lr_scale: float,
) -> None:
    if str(action_space_kind).strip().lower() != "placement_hold_step_v2":
        return
    scale = float(hold_output_lr_scale)
    if not math.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-12:
        return
    weight_grad = model.policy_head.weight.grad
    if weight_grad is not None and weight_grad.ndim == 2 and weight_grad.shape[0] > 0:
        weight_grad[-1].mul_(scale)
    bias_grad = model.policy_head.bias.grad
    if bias_grad is not None and bias_grad.ndim == 1 and bias_grad.shape[0] > 0:
        bias_grad[-1].mul_(scale)


def summarize_optimizer_group_lrs(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in optimizer.param_groups:
        out[str(group.get("group_name", f"group_{len(out)}"))] = float(
            group.get("lr", float("nan"))
        )
    return out


def _bc_probe_indices(
    indices: np.ndarray,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    if size <= 0 or indices.size <= 0:
        return np.zeros((0,), dtype=np.int64)
    if indices.size <= size:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.choice(indices, size=size, replace=False), dtype=np.int64)


def _bc_mean_kl(prev_probs: np.ndarray | None, curr_probs: np.ndarray | None) -> float:
    if prev_probs is None or curr_probs is None:
        return float("nan")
    if prev_probs.shape != curr_probs.shape or prev_probs.size == 0:
        return float("nan")
    eps = 1e-12
    safe_prev = np.clip(prev_probs, eps, 1.0)
    safe_curr = np.clip(curr_probs, eps, 1.0)
    kl = np.sum(
        np.where(
            prev_probs > 0.0,
            prev_probs * (np.log(safe_prev) - np.log(safe_curr)),
            0.0,
        ),
        axis=1,
    )
    return float(np.mean(kl, dtype=np.float64)) if kl.size > 0 else float("nan")


def collect_bc_policy_probe_probs(
    *,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    device: torch.device,
    obs_t: torch.Tensor,
    masks_t: torch.Tensor,
    indices_np: np.ndarray,
    batch_size: int,
) -> np.ndarray | None:
    if indices_np.size <= 0:
        return None
    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(indices_np.size), max(1, int(batch_size))):
            batch_indices_np = indices_np[start : start + max(1, int(batch_size))]
            batch_indices = torch.as_tensor(
                batch_indices_np,
                device=device,
                dtype=torch.long,
            )
            features = obs_adapter(obs_t[batch_indices])
            logits, _values = model(features)
            dist = masked_categorical(logits, masks_t[batch_indices])
            probs_chunks.append(
                dist.probs.detach().cpu().numpy().astype(np.float32, copy=False)
            )
    if not probs_chunks:
        return None
    return np.concatenate(probs_chunks, axis=0)


def evaluate_bc_split(
    *,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    device: torch.device,
    obs_t: torch.Tensor,
    masks_t: torch.Tensor,
    actions_t: torch.Tensor,
    returns_target_t: torch.Tensor,
    returns_mask_t: torch.Tensor,
    indices_np: np.ndarray,
    batch_size: int,
    action_space_kind: str,
    action_dim: int,
    value_weight: float,
) -> dict[str, float]:
    if indices_np.size <= 0:
        return {
            "records": 0.0,
            "actor_loss": float("nan"),
            "value_loss": float("nan"),
            "total_loss": float("nan"),
            "accuracy": float("nan"),
            "top5_accuracy": float("nan"),
            "top10_accuracy": float("nan"),
            "chosen_logprob": float("nan"),
            "entropy": float("nan"),
            "hold_label_rate": float("nan"),
            "hold_pred_rate": float("nan"),
            "hold_precision": float("nan"),
            "hold_recall": float("nan"),
            "hold_f1": float("nan"),
        }

    actor_loss_sum = 0.0
    value_loss_weighted_sum = 0.0
    chosen_logprob_sum = 0.0
    entropy_sum = 0.0
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_count = 0
    valid_return_total = 0.0
    top5_k = min(5, max(1, int(action_dim)))
    top10_k = min(10, max(1, int(action_dim)))
    hold_index = int(action_dim - 1)
    hold_label_count = 0
    hold_pred_count = 0
    hold_tp = 0
    hold_fp = 0
    hold_fn = 0
    hold_metrics_enabled = (
        str(action_space_kind).strip().lower() == "placement_hold_step_v2"
    )

    with torch.no_grad():
        for start in range(0, int(indices_np.size), max(1, int(batch_size))):
            batch_indices_np = indices_np[start : start + max(1, int(batch_size))]
            batch_indices = torch.as_tensor(
                batch_indices_np,
                device=device,
                dtype=torch.long,
            )
            features = obs_adapter(obs_t[batch_indices])
            logits, values = model(features)
            dist = masked_categorical(logits, masks_t[batch_indices])
            log_probs = dist.log_prob(actions_t[batch_indices])
            probs = dist.probs
            target_actions = actions_t[batch_indices]
            top1 = torch.argmax(probs, dim=-1)
            top5 = torch.topk(probs, k=top5_k, dim=-1).indices
            top10 = torch.topk(probs, k=top10_k, dim=-1).indices

            batch_count = int(target_actions.shape[0])
            total_count += batch_count
            actor_loss_sum += float(
                torch.sum(-log_probs).detach().cpu().item()
            )
            chosen_logprob_sum += float(torch.sum(log_probs).detach().cpu().item())
            entropy_sum += float(torch.sum(dist.entropy()).detach().cpu().item())
            correct_top1 += int(torch.sum(top1 == target_actions).detach().cpu().item())
            correct_top5 += int(
                torch.sum(torch.any(top5 == target_actions.unsqueeze(1), dim=1))
                .detach()
                .cpu()
                .item()
            )
            correct_top10 += int(
                torch.sum(torch.any(top10 == target_actions.unsqueeze(1), dim=1))
                .detach()
                .cpu()
                .item()
            )

            mb_return_mask = returns_mask_t[batch_indices]
            valid_returns = float(torch.sum(mb_return_mask).detach().cpu().item())
            if valid_returns > 0:
                sq_err = (values - returns_target_t[batch_indices]) ** 2
                weighted_value_loss = 0.5 * torch.sum(sq_err * mb_return_mask)
                value_loss_weighted_sum += float(
                    weighted_value_loss.detach().cpu().item()
                )
                valid_return_total += valid_returns

            if hold_metrics_enabled:
                label_hold = target_actions == hold_index
                pred_hold = top1 == hold_index
                hold_label_count += int(torch.sum(label_hold).detach().cpu().item())
                hold_pred_count += int(torch.sum(pred_hold).detach().cpu().item())
                hold_tp += int(torch.sum(label_hold & pred_hold).detach().cpu().item())
                hold_fp += int(
                    torch.sum((~label_hold) & pred_hold).detach().cpu().item()
                )
                hold_fn += int(
                    torch.sum(label_hold & (~pred_hold)).detach().cpu().item()
                )

    actor_mean = actor_loss_sum / max(1, total_count)
    value_mean = (
        value_loss_weighted_sum / max(1.0, valid_return_total)
        if valid_return_total > 0
        else 0.0
    )
    total_mean = actor_mean + float(value_weight) * value_mean
    hold_precision = (
        float(hold_tp / max(1, hold_tp + hold_fp))
        if (hold_tp + hold_fp) > 0
        else float("nan")
    )
    hold_recall = (
        float(hold_tp / max(1, hold_tp + hold_fn))
        if (hold_tp + hold_fn) > 0
        else float("nan")
    )
    hold_f1 = (
        float(2.0 * hold_precision * hold_recall / (hold_precision + hold_recall))
        if math.isfinite(hold_precision)
        and math.isfinite(hold_recall)
        and (hold_precision + hold_recall) > 0.0
        else float("nan")
    )
    return {
        "records": float(total_count),
        "actor_loss": float(actor_mean),
        "value_loss": float(value_mean),
        "total_loss": float(total_mean),
        "accuracy": float(correct_top1 / max(1, total_count)),
        "top5_accuracy": float(correct_top5 / max(1, total_count)),
        "top10_accuracy": float(correct_top10 / max(1, total_count)),
        "chosen_logprob": float(chosen_logprob_sum / max(1, total_count)),
        "entropy": float(entropy_sum / max(1, total_count)),
        "hold_label_rate": (
            float(hold_label_count / max(1, total_count))
            if hold_metrics_enabled
            else float("nan")
        ),
        "hold_pred_rate": (
            float(hold_pred_count / max(1, total_count))
            if hold_metrics_enabled
            else float("nan")
        ),
        "hold_precision": hold_precision,
        "hold_recall": hold_recall,
        "hold_f1": hold_f1,
    }


def run_bc_rollout_evals(
    *,
    env: WubEnvBridge,
    env_id: int,
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    device: torch.device,
    cfg: PPOConfig,
    epoch: int,
    sources: list[str],
) -> dict[str, dict[str, float | bool]]:
    if cfg.bc_rollout_eval_every_epochs <= 0:
        return {}
    if epoch % max(1, int(cfg.bc_rollout_eval_every_epochs)) != 0:
        return {}
    env.set_curriculum(
        top_k=0,
        bias_strength=0.0,
        danger_height=cfg.curriculum_danger_height,
    )
    out: dict[str, dict[str, float | bool]] = {}
    max_steps = max(8, int(cfg.max_pieces_per_episode_val) * 4)
    for source_index, source in enumerate(sources):
        env.set_piece_source(source)
        rollout = capture_single_policy_rollout(
            env=env,
            env_id=env_id,
            seed=cfg.seed + 7_000_001 + epoch * 10_007 + source_index * 101,
            model=model,
            obs_adapter=obs_adapter,
            device=device,
            max_steps=max_steps,
            deterministic=True,
        )
        out[source] = {
            "episode_return": float(rollout.get("episode_return", 0.0)),
            "episode_length": float(rollout.get("episode_length", 0)),
            "done": bool(rollout.get("done", False)),
        }
    return out


def format_bc_metrics_line(
    prefix: str,
    metrics: dict[str, float],
    *,
    hold_metrics: bool,
) -> str:
    out = (
        f"{prefix}: "
        f"actor={_fmt_float(metrics.get('actor_loss'), 4)} "
        f"value={_fmt_float(metrics.get('value_loss'), 4)} "
        f"total={_fmt_float(metrics.get('total_loss'), 4)} "
        f"acc={_fmt_float(metrics.get('accuracy'))} "
        f"top5={_fmt_float(metrics.get('top5_accuracy'))} "
        f"top10={_fmt_float(metrics.get('top10_accuracy'))} "
        f"logp={_fmt_float(metrics.get('chosen_logprob'), 4)} "
        f"ent={_fmt_float(metrics.get('entropy'), 4)} "
        f"kl_prev={_fmt_float(metrics.get('kl_prev'), 4)}"
    )
    if hold_metrics:
        out += (
            " "
            f"hold(lbl={_fmt_float(metrics.get('hold_label_rate'))},"
            f"pred={_fmt_float(metrics.get('hold_pred_rate'))},"
            f"p={_fmt_float(metrics.get('hold_precision'))},"
            f"r={_fmt_float(metrics.get('hold_recall'))},"
            f"f1={_fmt_float(metrics.get('hold_f1'))})"
        )
    return out


def run_bc_pretrain(
    model: PolicyValueNet,
    obs_adapter: ObservationAdapter,
    cfg: PPOConfig,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
    env: WubEnvBridge,
    env_ids: list[int],
    rollout_sources: list[str],
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
    session_ids_np = np.asarray(batch["session_ids"], dtype=object)
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
    train_indices_np, val_indices_np, split_stats = split_bc_indices_by_session(
        session_ids_np,
        val_fraction=cfg.bc_val_fraction,
        seed=cfg.seed,
    )
    train_indices_t = torch.as_tensor(train_indices_np, device=device, dtype=torch.long)
    val_enabled = val_indices_np.size > 0
    optimizer = build_bc_optimizer(
        model,
        obs_adapter,
        learning_rate=cfg.bc_learning_rate,
        policy_lr_scale=cfg.bc_policy_lr_scale,
        policy_head_lr_scale=cfg.bc_policy_head_lr_scale,
        value_lr_scale=cfg.bc_value_lr_scale,
        adapter_lr_scale=cfg.bc_adapter_lr_scale,
        queue_lr_scale=cfg.bc_queue_encoder_lr_scale,
    )
    optimizer_group_lrs = summarize_optimizer_group_lrs(optimizer)

    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_adapter_state: dict[str, torch.Tensor] | None = None
    epoch_logs: list[dict[str, Any]] = []
    hold_metrics_enabled = (
        str(cfg.action_space_kind).strip().lower() == "placement_hold_step_v2"
    )
    patience_used = 0
    stopped_early = False
    stop_reason: str | None = None

    train_probe_indices_np = _bc_probe_indices(
        train_indices_np,
        size=cfg.bc_kl_probe_size,
        seed=cfg.seed + 17,
    )
    val_probe_indices_np = _bc_probe_indices(
        val_indices_np,
        size=cfg.bc_kl_probe_size,
        seed=cfg.seed + 31,
    )
    prev_train_probe_probs = collect_bc_policy_probe_probs(
        model=model,
        obs_adapter=obs_adapter,
        device=device,
        obs_t=obs_t,
        masks_t=masks_t,
        indices_np=train_probe_indices_np,
        batch_size=batch_size,
    )
    prev_val_probe_probs = collect_bc_policy_probe_probs(
        model=model,
        obs_adapter=obs_adapter,
        device=device,
        obs_t=obs_t,
        masks_t=masks_t,
        indices_np=val_probe_indices_np,
        batch_size=batch_size,
    )
    print(
        "[bc] "
        f"returns preprocess: normalize={'y' if cfg.bc_normalize_returns else 'n'} "
        f"clip={return_clip_used if return_clip_used is not None else 'off'} "
        f"valid_targets={valid_return_count} "
        f"mean={return_norm_mean:.4f} std={return_norm_std:.4f}"
    )
    print(
        "[bc] "
        f"split: sessions(train={split_stats['train_sessions']}, val={split_stats['val_sessions']}) "
        f"records(train={split_stats['train_records']}, val={split_stats['val_records']}) "
        f"val_fraction={split_stats['val_fraction_effective']:.3f}"
    )
    print(
        "[bc] optimizer: "
        + ", ".join(
            f"{key}={value:.6g}" for key, value in optimizer_group_lrs.items()
        )
        + (
            f", hold_output_scale={cfg.bc_hold_output_lr_scale:.3f}"
            if hold_metrics_enabled
            else ""
        )
    )

    for epoch in range(1, cfg.bc_epochs + 1):
        perm = torch.randperm(int(train_indices_t.shape[0]), device=device)

        for start in range(0, int(train_indices_t.shape[0]), batch_size):
            idx = train_indices_t[perm[start : start + batch_size]]
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
            scale_bc_hold_output_gradients(
                model,
                action_space_kind=cfg.action_space_kind,
                hold_output_lr_scale=cfg.bc_hold_output_lr_scale,
            )
            nn.utils.clip_grad_norm_(
                trainable_parameters(model, obs_adapter), cfg.max_grad_norm
            )
            optimizer.step()

        was_model_training = model.training
        was_adapter_training = obs_adapter.training
        model.eval()
        obs_adapter.eval()
        try:
            train_metrics = evaluate_bc_split(
                model=model,
                obs_adapter=obs_adapter,
                device=device,
                obs_t=obs_t,
                masks_t=masks_t,
                actions_t=actions_t,
                returns_target_t=returns_target_t,
                returns_mask_t=returns_mask_t,
                indices_np=train_indices_np,
                batch_size=batch_size,
                action_space_kind=cfg.action_space_kind,
                action_dim=action_dim,
                value_weight=cfg.bc_value_weight,
            )
            val_metrics = evaluate_bc_split(
                model=model,
                obs_adapter=obs_adapter,
                device=device,
                obs_t=obs_t,
                masks_t=masks_t,
                actions_t=actions_t,
                returns_target_t=returns_target_t,
                returns_mask_t=returns_mask_t,
                indices_np=val_indices_np,
                batch_size=batch_size,
                action_space_kind=cfg.action_space_kind,
                action_dim=action_dim,
                value_weight=cfg.bc_value_weight,
            )
            current_train_probe_probs = collect_bc_policy_probe_probs(
                model=model,
                obs_adapter=obs_adapter,
                device=device,
                obs_t=obs_t,
                masks_t=masks_t,
                indices_np=train_probe_indices_np,
                batch_size=batch_size,
            )
            current_val_probe_probs = collect_bc_policy_probe_probs(
                model=model,
                obs_adapter=obs_adapter,
                device=device,
                obs_t=obs_t,
                masks_t=masks_t,
                indices_np=val_probe_indices_np,
                batch_size=batch_size,
            )
            train_metrics["kl_prev"] = _bc_mean_kl(
                prev_train_probe_probs,
                current_train_probe_probs,
            )
            val_metrics["kl_prev"] = _bc_mean_kl(
                prev_val_probe_probs,
                current_val_probe_probs,
            )
            prev_train_probe_probs = current_train_probe_probs
            prev_val_probe_probs = current_val_probe_probs
            rollout_eval = run_bc_rollout_evals(
                env=env,
                env_id=env_ids[0],
                model=model,
                obs_adapter=obs_adapter,
                device=device,
                cfg=cfg,
                epoch=epoch,
                sources=rollout_sources,
            )
        finally:
            if was_model_training:
                model.train()
            if was_adapter_training:
                obs_adapter.train()

        selection_metric = (
            float(val_metrics.get("total_loss", float("nan")))
            if val_enabled
            else float(train_metrics.get("total_loss", float("nan")))
        )
        improved = math.isfinite(selection_metric) and (
            not math.isfinite(best_metric) or selection_metric < (best_metric - 1e-6)
        )
        if improved:
            best_metric = float(selection_metric)
            best_epoch = int(epoch)
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            best_adapter_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in obs_adapter.state_dict().items()
            }
            patience_used = 0
        elif val_enabled and cfg.bc_early_stop_patience > 0 and epoch >= cfg.bc_min_epochs:
            patience_used += 1
            if patience_used >= cfg.bc_early_stop_patience:
                stopped_early = True
                stop_reason = (
                    f"no val improvement for {cfg.bc_early_stop_patience} epochs"
                )

        epoch_log = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
            "rollout": rollout_eval,
            "selection_metric": selection_metric,
            "improved": bool(improved),
            "patience_used": int(patience_used),
        }
        epoch_logs.append(epoch_log)

        log_eval_due = (
            epoch == 1
            or epoch == cfg.bc_epochs
            or epoch % max(1, cfg.bc_eval_every_epochs) == 0
            or stopped_early
        )
        if log_eval_due:
            print(f"[bc] epoch={epoch}/{cfg.bc_epochs}")
            print(
                "  "
                + format_bc_metrics_line(
                    "train",
                    train_metrics,
                    hold_metrics=hold_metrics_enabled,
                )
            )
            if val_enabled:
                print(
                    "  "
                    + format_bc_metrics_line(
                        "val",
                        val_metrics,
                        hold_metrics=hold_metrics_enabled,
                    )
                )
            if rollout_eval:
                print(
                    "  rollout: "
                    + ", ".join(
                        (
                            f"{source}(ret={_fmt_float(result.get('episode_return'))},"
                            f"len={_fmt_float(result.get('episode_length'), 0)},"
                            f"done={'y' if bool(result.get('done')) else 'n'})"
                        )
                        for source, result in rollout_eval.items()
                    )
                )
            if val_enabled and cfg.bc_early_stop_patience > 0:
                print(
                    "  "
                    f"early_stop: best_epoch={best_epoch} "
                    f"best_val={_fmt_float(best_metric, 4)} "
                    f"patience={patience_used}/{cfg.bc_early_stop_patience}"
                )

        if stopped_early:
            print(f"[bc] early stop at epoch {epoch}: {stop_reason}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_adapter_state is not None:
        obs_adapter.load_state_dict(best_adapter_state, strict=False)

    return {
        "enabled": True,
        "dataset_path": str(dataset_path),
        "epochs": cfg.bc_epochs,
        "epochs_ran": len(epoch_logs),
        "batch_size": batch_size,
        "learning_rate": cfg.bc_learning_rate,
        "value_weight": cfg.bc_value_weight,
        "val_fraction": cfg.bc_val_fraction,
        "kl_probe_size": cfg.bc_kl_probe_size,
        "normalize_returns": cfg.bc_normalize_returns,
        "return_norm_mean": return_norm_mean,
        "return_norm_std": return_norm_std,
        "return_clip_used": return_clip_used,
        "returns_with_targets": valid_return_count,
        "dataset_stats": dataset_stats,
        "split": split_stats,
        "optimizer_group_lrs": optimizer_group_lrs,
        "optimizer_group_scales": {
            "policy": cfg.bc_policy_lr_scale,
            "policy_head": cfg.bc_policy_head_lr_scale,
            "value": cfg.bc_value_lr_scale,
            "adapter": cfg.bc_adapter_lr_scale,
            "queue": cfg.bc_queue_encoder_lr_scale,
            "hold_output": cfg.bc_hold_output_lr_scale,
        },
        "best_total_loss": best_metric,
        "best_epoch": best_epoch,
        "early_stopped": stopped_early,
        "stop_reason": stop_reason,
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
        fixed_blend_step = fixed_reward_blend_step_for_cfg(cfg)
        blend_unit_effective = effective_reward_blend_unit(cfg)
        blend_start_step = (
            int(fixed_blend_step) if fixed_blend_step is not None else 0
        )
        reward_fn_from = (
            cfg.reward_functions[0]
            if len(cfg.reward_functions) >= 1
            else "v1"
        )
        reward_fn_to = (
            cfg.reward_functions[1]
            if len(cfg.reward_functions) >= 2
            else reward_fn_from
        )
        init_result = env.init(
            mode_id=cfg.mode_id,
            num_envs=cfg.num_envs,
            model_path=cfg.model_path,
            observation_space=cfg.observation_space,
            phase_context_enabled=cfg.phase_context_enabled,
            placement_execution_mode=cfg.placement_execution_mode,
            action_space_kind=cfg.action_space_kind,
            piece_source_profile=cfg.generator_schedule[0],
            queue_policy_id=cfg.queue_policy_id,
            max_pieces_per_episode=cfg.max_pieces_per_episode_train,
            reward_function_from=reward_fn_from,
            reward_function_to=reward_fn_to,
            reward_blend_timesteps=reward_blend_total_for_env(cfg),
            reward_blend_unit=blend_unit_effective,
            reward_blend_start_step=blend_start_step,
            seed=cfg.seed,
        )
        env_ids: list[int] = [int(v) for v in init_result.get("env_ids", [])]
        if not env_ids:
            raise RuntimeError("Bridge init returned no env ids.")
        if len(env_ids) != cfg.num_envs:
            raise RuntimeError(
                f"Bridge init env count mismatch. expected={cfg.num_envs} got={len(env_ids)}"
            )

        env_piece_sources, env_piece_source_counts = build_env_piece_source_assignment(
            cfg, env_ids
        )
        env.set_piece_sources(env_ids=env_ids, piece_source_profiles=env_piece_sources)
        validation_sources = unique_generator_sources(cfg)
        source_mix_label = ",".join(
            f"{key}:{env_piece_source_counts[key]}"
            for key in sorted(env_piece_source_counts.keys())
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
        raw_obs_dim = infer_obs_dim(reset_result["obs"])
        action_dim = infer_action_dim(reset_result["action_masks"])

        if cfg.observation_space == "raw_v1":
            obs_adapter: ObservationAdapter = RawV1BoardQueueObservationAdapter(
                raw_obs_dim=raw_obs_dim,
                model_path=cfg.model_path,
                queue_hidden_dim=cfg.queue_encoder_hidden_dim,
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
        optimizer: torch.optim.Optimizer | None = None

        global_step = 0
        start_update = 0
        if cfg.resume_checkpoint:
            optimizer = build_ppo_optimizer(
                model,
                obs_adapter,
                cfg.learning_rate,
                cfg.queue_encoder_lr_scale,
            )
            checkpoint_path = Path(cfg.resume_checkpoint).resolve()
            loaded_global_step, loaded_update = load_checkpoint(
                checkpoint_path,
                model,
                obs_adapter,
                optimizer,
                device,
                load_optimizer_state=(cfg.resume_mode == "continue"),
                expected_phase_context_enabled=cfg.phase_context_enabled,
                expected_action_space_kind=cfg.action_space_kind,
            )
            if cfg.resume_mode == "continue":
                global_step = loaded_global_step
                start_update = loaded_update
                blend_resume_step = (
                    int(fixed_blend_step)
                    if fixed_blend_step is not None
                    else (
                        start_update
                        if blend_unit_effective == "updates"
                        else global_step
                    )
                )
                blend_result = env.set_reward_blend_step(blend_resume_step)
                blend_step = int(
                    blend_result.get("transition_step", blend_resume_step)
                )
                print(
                    f"[ppo] resumed checkpoint (continue): {checkpoint_path} "
                    f"(global_step={global_step}, update={start_update}, "
                    f"blend_transition_step={blend_step})"
                )
            else:
                global_step = 0
                start_update = 0
                print(
                    f"[ppo] warm-started from checkpoint (fresh): {checkpoint_path} "
                    f"(loaded_global_step={loaded_global_step}, loaded_update={loaded_update}, "
                    "training clock reset to step=0/update=0)"
                )
        elif cfg.init_artifact:
            artifact_path = Path(cfg.init_artifact).resolve()
            (
                artifact_observation_space,
                artifact_phase_context_enabled,
                artifact_action_space_kind,
            ) = load_from_artifact(
                model, obs_adapter, artifact_path
            )
            artifact_obs_compatible = artifact_observation_space == policy_observation_space or (
                isinstance(obs_adapter, RawV1BoardQueueObservationAdapter)
                and artifact_observation_space == "model_head_v1"
                and policy_observation_space == "raw_v1"
            )
            if not artifact_obs_compatible:
                raise ValueError(
                    "Init artifact observationSpace mismatch. "
                    f"artifact={artifact_observation_space} policy={policy_observation_space}"
                )
            if artifact_phase_context_enabled != cfg.phase_context_enabled:
                print(
                    "[ppo] init artifact phase-context mismatch: "
                    f"artifact={'on' if artifact_phase_context_enabled else 'off'} "
                    f"run={'on' if cfg.phase_context_enabled else 'off'}"
                )
            if artifact_action_space_kind != cfg.action_space_kind:
                print(
                    "[ppo] init artifact action-space mismatch: "
                    f"artifact={artifact_action_space_kind} run={cfg.action_space_kind}"
                )
            print(f"[ppo] initialized from bot artifact: {artifact_path}")

        encoder_frozen_for_ppo = False
        encoder_freeze_mode_applied = "none"

        bc_requested = bool(cfg.bc_dataset) and cfg.bc_epochs > 0
        bc_allowed = (not cfg.resume_checkpoint) or cfg.resume_mode == "fresh"
        bc_should_run = bc_requested and bc_allowed
        if bc_requested and not bc_allowed:
            print(
                "[ppo] warning: BC was requested but skipped because "
                "--resume-checkpoint is being used with --resume-mode continue."
            )

        bc_stats: dict[str, Any] = {"enabled": False}
        if bc_should_run:
            bc_stats = run_bc_pretrain(
                model=model,
                obs_adapter=obs_adapter,
                cfg=cfg,
                device=device,
                obs_dim=raw_obs_dim,
                action_dim=action_dim,
                env=env,
                env_ids=env_ids,
                rollout_sources=validation_sources,
            )
            env.set_piece_sources(
                env_ids=env_ids,
                piece_source_profiles=env_piece_sources,
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
                    max_steps=max(1, cfg.max_pieces_per_episode_val * 4),
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
                    board_adapter = (
                        obs_adapter.board_adapter
                        if isinstance(obs_adapter, RawV1BoardQueueObservationAdapter)
                        else obs_adapter
                    )
                    if isinstance(board_adapter, WubHeadFromRawObservationAdapter):
                        conv_param_total = 0
                        conv_param_trainable = 0
                        for conv in board_adapter.conv_layers:
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

        if bc_should_run:
            env.set_piece_sources(
                env_ids=env_ids,
                piece_source_profiles=env_piece_sources,
            )

        if optimizer is None or bc_should_run:
            optimizer = build_ppo_optimizer(
                model,
                obs_adapter,
                cfg.learning_rate,
                cfg.queue_encoder_lr_scale,
            )

        if bc_should_run:
            # Reinitialize env batch after BC eval/snapshot capture so PPO
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
        first_update_index = start_update + 1
        first_update_warmup_active = (
            cfg.warmup_updates > 0 and first_update_index <= cfg.warmup_updates
        )
        first_update_policy_lr = cfg.learning_rate * (
            cfg.warmup_policy_lr_scale if first_update_warmup_active else 1.0
        )
        first_update_queue_lr = cfg.learning_rate * cfg.queue_encoder_lr_scale
        startup_adapter_state = summarize_adapter_state(
            obs_adapter=obs_adapter,
            optimizer=optimizer,
            policy_lr=first_update_policy_lr,
            obs_adapter_lr_scale=cfg.obs_adapter_lr_scale,
            queue_lr=first_update_queue_lr,
        )
        print("[ppo] " + format_adapter_state_log(startup_adapter_state))

        print(
            "[ppo] starting training "
            f"(device={device.type}, env_obs_space={cfg.observation_space}, "
            f"phase_context={'on' if cfg.phase_context_enabled else 'off'}, "
            f"placement_exec={cfg.placement_execution_mode}, "
            f"action_space={cfg.action_space_kind}, "
            f"policy_obs_space={policy_observation_space}, "
            f"raw_obs_dim={raw_obs_dim}, policy_obs_dim={obs_dim}, action_dim={action_dim}, "
            f"queue_hidden_dim={cfg.queue_encoder_hidden_dim}, "
            f"queue_lr_scale={cfg.queue_encoder_lr_scale:.3f}, "
            f"batch_size={batch_size}, updates={num_updates}, "
            f"max_pieces_train={cfg.max_pieces_per_episode_train}, "
            f"max_pieces_val={cfg.max_pieces_per_episode_val}, "
            f"generators_spec={list(cfg.generator_schedule)}, "
            f"generators_mix={source_mix_label}, "
            f"reward_functions={list(cfg.reward_functions)}, "
            f"reward_from={reward_fn_from}, reward_to={reward_fn_to}, "
            f"validation_sources={validation_sources}, "
            f"validate_every_updates={cfg.validate_every_updates}, "
            f"reward_blend={cfg.reward_blend_span}({blend_unit_effective},env_total={reward_blend_total_for_env(cfg)},fixed_step={fixed_blend_step}), "
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
            f"distill_teacher_alpha={cfg.distill_teacher_alpha:.3f}, "
            f"distill_teacher_tau={cfg.distill_teacher_tau:.4f}, "
            f"distill_teacher_top_m={cfg.distill_teacher_top_m}, "
            f"hold_margin_probe={'y' if cfg.hold_margin_probe else 'n'}, "
            f"hold_margin_penalty_base={cfg.hold_margin_penalty_base:.4f}, "
            f"hold_margin_penalty_threshold={cfg.hold_margin_penalty_threshold:.4f}, "
            f"hold_swap_probe={'y' if cfg.hold_swap_probe else 'n'}, "
            f"hold_swap_distill_coef={cfg.hold_swap_distill_coef_start:.4f}->{cfg.hold_swap_distill_coef_end:.4f}/"
            f"{cfg.hold_swap_distill_coef_ramp_updates}, "
            f"hold_swap_teacher_tau={cfg.hold_swap_teacher_tau:.4f}, "
            f"obs_adapter_lr_scale={cfg.obs_adapter_lr_scale:.3f}, "
            f"encoder_frozen_after_bc={'y' if encoder_frozen_for_ppo else 'n'}, "
            f"encoder_freeze_mode={encoder_freeze_mode_applied})"
        )
        if not is_hold_probe_supported(cfg.action_space_kind) and (
            cfg.hold_margin_probe or cfg.hold_margin_penalty_base > 0.0
        ):
            print(
                "[ppo] note: hold-margin probe/shaping is ignored for "
                f"action_space={cfg.action_space_kind}. "
                "The old flat-action hold probe does not apply to explicit HOLD steps."
            )
        if str(cfg.action_space_kind).strip().lower() != "placement_hold_step_v2" and (
            cfg.hold_swap_probe or cfg.hold_swap_distill_coef_start > 0.0
        ):
            print(
                "[ppo] note: hold-swap probe/distillation is ignored for "
                f"action_space={cfg.action_space_kind}. "
                "Swap-utility teaching only applies to explicit HOLD action spaces."
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
                "action_space_kind": cfg.action_space_kind,
                "action_dim": action_dim,
                "batch_size": batch_size,
                "num_updates": num_updates,
                "generator_mix_by_env": env_piece_source_counts,
                "validation_sources": validation_sources,
                "phase_context_enabled": cfg.phase_context_enabled,
                "reward_function_from": reward_fn_from,
                "reward_function_to": reward_fn_to,
                "reward_blend_env_total": reward_blend_total_for_env(cfg),
                "reward_blend_unit_effective": blend_unit_effective,
                "reward_blend_fixed_step": fixed_blend_step,
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
            "top_out_term": ("topOutTerm", "topOutPenalty"),
            "term_hold_margin_penalty": (),
            "blend_t": ("rewardBlendT",),
            "blend_legacy_weight": ("rewardBlendLegacyWeight",),
            "blend_target_weight": ("rewardBlendTargetWeight",),
            "blend_transition_step": ("rewardBlendTransitionStep",),
            "blend_transition_total_steps": ("rewardBlendTransitionTotalSteps",),
            "v1_base_raw": ("rewardLegacyBase",),
            "v2_base_raw": ("rewardTargetBase",),
            "v1_base": ("rewardLegacyContribution",),
            "v2_base": ("rewardTargetContribution",),
            "term_lines": ("rewardTermLines",),
            "term_score": ("rewardTermScore",),
            "term_time": ("rewardTermTime",),
            "term_height": ("rewardTermHeight",),
            "term_holes": ("rewardTermHoles",),
            "term_bumpiness": ("rewardTermBumpiness",),
            "term_board_score": ("rewardTermBoardScore",),
            "term_board_quality": ("rewardTermBoardQuality",),
            "term_board_quality_abs": ("rewardTermBoardQualityAbsolute",),
            "term_full_clear": ("rewardTermFullClear",),
            "term_top_out": ("rewardTermTopOut",),
            "v1_term_lines": ("rewardLegacyContributionTermLines",),
            "v1_term_score": ("rewardLegacyContributionTermScore",),
            "v1_term_time": ("rewardLegacyContributionTermTime",),
            "v1_term_height": ("rewardLegacyContributionTermHeight",),
            "v1_term_holes": ("rewardLegacyContributionTermHoles",),
            "v1_term_bumpiness": ("rewardLegacyContributionTermBumpiness",),
            "v1_term_board_score": ("rewardLegacyContributionTermBoardScore",),
            "v1_term_board_quality": ("rewardLegacyContributionTermBoardQuality",),
            "v1_term_board_quality_abs": (
                "rewardLegacyContributionTermBoardQualityAbsolute",
            ),
            "v1_term_full_clear": ("rewardLegacyContributionTermFullClear",),
            "v1_term_top_out": ("rewardLegacyContributionTermTopOut",),
            "v2_term_lines": ("rewardTargetContributionTermLines",),
            "v2_term_score": ("rewardTargetContributionTermScore",),
            "v2_term_time": ("rewardTargetContributionTermTime",),
            "v2_term_height": ("rewardTargetContributionTermHeight",),
            "v2_term_holes": ("rewardTargetContributionTermHoles",),
            "v2_term_bumpiness": ("rewardTargetContributionTermBumpiness",),
            "v2_term_board_score": ("rewardTargetContributionTermBoardScore",),
            "v2_term_board_quality": ("rewardTargetContributionTermBoardQuality",),
            "v2_term_board_quality_abs": (
                "rewardTargetContributionTermBoardQualityAbsolute",
            ),
            "v2_term_full_clear": ("rewardTargetContributionTermFullClear",),
            "v2_term_top_out": ("rewardTargetContributionTermTopOut",),
        }
        episode_reward_term_keys = [
            key
            for key in reward_component_aliases.keys()
            if key not in BLEND_STEP_TERM_KEYS
        ]
        ep_reward_component_sums = {
            key: np.zeros(cfg.num_envs, dtype=np.float64)
            for key in episode_reward_term_keys
        }
        completed_reward_component_sums: dict[str, list[float]] = {
            key: [] for key in reward_component_aliases
        }
        step_reward_component_values: dict[str, list[float]] = {
            key: [] for key in BLEND_STEP_TERM_KEYS
        }

        training_start = time.time()
        best_score = float("-inf")
        best_update = 0
        best_stats: dict[str, Any] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_adapter_state_dict: dict[str, torch.Tensor] | None = None
        did_interrupt = False
        last_log_wall: float | None = None
        try:
            for update in range(start_update + 1, num_updates + 1):
                warmup_active = cfg.warmup_updates > 0 and update <= cfg.warmup_updates
                ent_coef_now = (
                    cfg.warmup_ent_coef if warmup_active else cfg.ent_coef
                )
                target_kl_now = (
                    cfg.warmup_target_kl if warmup_active else cfg.target_kl
                )
                (
                    policy_lr_now,
                    value_lr_now,
                    _adapter_lr_now,
                    queue_lr_now,
                ) = apply_warmup_lr_schedule(
                    optimizer=optimizer,
                    base_lr=cfg.learning_rate,
                    warmup_active=warmup_active,
                    warmup_policy_lr_scale=cfg.warmup_policy_lr_scale,
                    warmup_value_lr_scale=cfg.warmup_value_lr_scale,
                    obs_adapter_lr_scale=cfg.obs_adapter_lr_scale,
                    queue_encoder_lr_scale=cfg.queue_encoder_lr_scale,
                )
                adapter_state_now = summarize_adapter_state(
                    obs_adapter=obs_adapter,
                    optimizer=optimizer,
                    policy_lr=policy_lr_now,
                    obs_adapter_lr_scale=cfg.obs_adapter_lr_scale,
                    queue_lr=queue_lr_now,
                )
                update_start_wall = time.time()
                update_start_perf = time.perf_counter()
                validation_wall_s = 0.0
                save_wall_s = 0.0
                log_emit_wall_s = 0.0
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
                hold_probe_update_accumulator: dict[str, dict[str, Any]] = {}
                hold_swap_update_accumulator: dict[str, dict[str, Any]] = {}
                curriculum_topk_now = curriculum_topk_for_update(cfg, update)
                curriculum_bias_now = curriculum_bias_for_update(cfg, update)
                distill_coef_now = distill_coef_for_update(cfg, update)
                hold_swap_distill_coef_now = hold_swap_distill_coef_for_update(
                    cfg, update
                )
                env.set_curriculum(
                    top_k=curriculum_topk_now,
                    bias_strength=curriculum_bias_now,
                    danger_height=cfg.curriculum_danger_height,
                )
                if fixed_blend_step is not None:
                    blend_step_now = int(fixed_blend_step)
                else:
                    blend_step_now = (
                        max(0, int(update - 1))
                        if blend_unit_effective == "updates"
                        else max(0, int(global_step))
                    )
                    if blend_unit_effective == "updates":
                        env.set_reward_blend_step(blend_step_now)

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
                hold_swap_teacher_prob_buf = torch.zeros(
                    (cfg.num_steps, cfg.num_envs), device=device
                )
                hold_swap_teacher_valid_buf = torch.zeros(
                    (cfg.num_steps, cfg.num_envs), device=device
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
                    hold_penalties_by_env_id: dict[int, float] = {}
                    if cfg.hold_margin_probe or cfg.hold_margin_penalty_base > 0.0:
                        if not is_hold_probe_supported(cfg.action_space_kind):
                            accumulate_hold_action_counts(
                                hold_probe_update_accumulator,
                                env_piece_sources=env_piece_sources,
                                actions=actions_np,
                                action_dim=action_dim,
                                action_space_kind=cfg.action_space_kind,
                            )
                    if is_hold_probe_supported(cfg.action_space_kind) and (
                        cfg.hold_margin_probe or cfg.hold_margin_penalty_base > 0.0
                    ):
                        per_env_hold_probe, _hold_probe_summary_unused = (
                            evaluate_hold_margin_for_actions(
                                env=env,
                                env_ids=env_ids,
                                actions=actions_np,
                                env_piece_sources=env_piece_sources,
                                obs_adapter=obs_adapter,
                                model=model,
                                device=device,
                                gamma=cfg.gamma,
                                action_dim=action_dim,
                                action_space_kind=cfg.action_space_kind,
                                eps=cfg.hold_margin_eps,
                                good_threshold=cfg.hold_margin_good_threshold,
                            )
                        )
                        hold_penalties_by_env_id = accumulate_hold_probe_metrics(
                            hold_probe_update_accumulator,
                            env_ids=env_ids,
                            env_piece_sources=env_piece_sources,
                            actions=actions_np,
                            action_dim=action_dim,
                            action_space_kind=cfg.action_space_kind,
                            per_env=per_env_hold_probe,
                            penalty_base=cfg.hold_margin_penalty_base,
                            penalty_threshold=cfg.hold_margin_penalty_threshold,
                        )
                    if (
                        str(cfg.action_space_kind).strip().lower()
                        == "placement_hold_step_v2"
                        and (
                            cfg.hold_swap_probe
                            or hold_swap_distill_coef_now > 0.0
                        )
                    ):
                        hold_idx = action_dim - 1
                        hold_env_pairs = [
                            (env_idx, env_id, env_piece_sources[env_idx])
                            for env_idx, env_id in enumerate(env_ids)
                            if hold_idx >= 0 and mask_np[env_idx, hold_idx] > 0.5
                        ]
                        if hold_env_pairs:
                            hold_env_ids = [
                                env_id for _env_idx, env_id, _source in hold_env_pairs
                            ]
                            hold_sources = [
                                source
                                for _env_idx, _env_id, source in hold_env_pairs
                            ]
                            per_env_hold_swap, _unused_summary = (
                                evaluate_hold_swap_teacher_for_envs(
                                    env=env,
                                    env_ids=hold_env_ids,
                                    env_piece_sources=hold_sources,
                                    obs_adapter=obs_adapter,
                                    model=model,
                                    device=device,
                                    gamma=cfg.gamma,
                                    teacher_tau=cfg.hold_swap_teacher_tau,
                                )
                            )
                            accumulate_hold_swap_teacher_metrics(
                                hold_swap_update_accumulator,
                                env_ids=hold_env_ids,
                                env_piece_sources=hold_sources,
                                per_env=per_env_hold_swap,
                            )
                            for env_idx, env_id, _source in hold_env_pairs:
                                env_result = per_env_hold_swap.get(int(env_id))
                                if not env_result:
                                    continue
                                teacher_hold_prob = env_result.get("teacher_hold_prob")
                                if teacher_hold_prob is None or not math.isfinite(
                                    float(teacher_hold_prob)
                                ):
                                    continue
                                hold_swap_teacher_prob_buf[step, env_idx] = float(
                                    teacher_hold_prob
                                )
                                hold_swap_teacher_valid_buf[step, env_idx] = 1.0
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
                    if hold_penalties_by_env_id:
                        for env_idx, env_id in enumerate(env_ids):
                            penalty = hold_penalties_by_env_id.get(int(env_id))
                            if penalty and penalty > 0.0:
                                rewards_np[env_idx] -= np.float32(penalty)

                    reward_buf[step] = as_tensor(rewards_np, device)
                    done_buf[step] = as_tensor(dones_np, device)

                    ep_return += rewards_np.astype(np.float64)
                    ep_length += 1
                    if isinstance(infos_raw, list):
                        max_info = min(len(infos_raw), cfg.num_envs)
                        for blend_key in BLEND_STEP_TERM_KEYS:
                            alias_keys = reward_component_aliases.get(blend_key)
                            if not alias_keys:
                                continue
                            values: list[float] = []
                            for env_idx in range(max_info):
                                info = infos_raw[env_idx]
                                value = _info_num(
                                    info,
                                    alias_keys,
                                    default=float("nan"),
                                )
                                if math.isfinite(value):
                                    values.append(value)
                            if values:
                                step_reward_component_values[blend_key].append(
                                    float(np.mean(values))
                                )
                        for env_idx in range(max_info):
                            info = infos_raw[env_idx]
                            hold_penalty = float(
                                hold_penalties_by_env_id.get(int(env_ids[env_idx]), 0.0)
                            )
                            reward_final = _info_num(
                                info,
                                reward_component_aliases["reward_final"],
                                default=float(rewards_np[env_idx]),
                            )
                            ep_reward_component_sums["reward_final"][
                                env_idx
                            ] += reward_final - hold_penalty
                            ep_reward_component_sums["term_hold_margin_penalty"][
                                env_idx
                            ] += -hold_penalty
                            for key in episode_reward_term_keys:
                                if key in ("reward_final", "term_hold_margin_penalty"):
                                    continue
                                ep_reward_component_sums[key][env_idx] += _info_num(
                                    info,
                                    reward_component_aliases[key],
                                    default=0.0,
                                )
                        if max_info < cfg.num_envs:
                            ep_reward_component_sums["reward_final"][
                                max_info:cfg.num_envs
                            ] += rewards_np[max_info:cfg.num_envs].astype(np.float64)
                            if hold_penalties_by_env_id:
                                for env_idx in range(max_info, cfg.num_envs):
                                    hold_penalty = float(
                                        hold_penalties_by_env_id.get(
                                            int(env_ids[env_idx]), 0.0
                                        )
                                    )
                                    ep_reward_component_sums[
                                        "term_hold_margin_penalty"
                                    ][env_idx] += -hold_penalty
                    else:
                        ep_reward_component_sums["reward_final"] += rewards_np.astype(
                            np.float64
                        )
                        if hold_penalties_by_env_id:
                            for env_idx, env_id in enumerate(env_ids):
                                hold_penalty = float(
                                    hold_penalties_by_env_id.get(int(env_id), 0.0)
                                )
                                ep_reward_component_sums["term_hold_margin_penalty"][
                                    env_idx
                                ] += -hold_penalty

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
                b_hold_swap_teacher_prob = hold_swap_teacher_prob_buf.reshape(-1)
                b_hold_swap_teacher_valid = hold_swap_teacher_valid_buf.reshape(-1)
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
                hold_swap_distill_loss_value = 0.0
                total_loss_value = 0.0
                policy_term_value = 0.0
                value_term_value = 0.0
                entropy_term_value = 0.0
                distill_term_value = 0.0
                hold_swap_distill_term_value = 0.0
                policy_loss_sum = 0.0
                value_loss_sum = 0.0
                entropy_sum = 0.0
                distill_loss_sum = 0.0
                hold_swap_distill_loss_sum = 0.0
                total_loss_sum = 0.0
                policy_term_sum = 0.0
                value_term_sum = 0.0
                entropy_term_sum = 0.0
                distill_term_sum = 0.0
                hold_swap_distill_term_sum = 0.0
                teacher_entropy_value = 0.0
                teacher_max_prob_value = 0.0
                teacher_uniform_row_frac_value = 0.0
                teacher_bias_row_frac_value = 0.0
                teacher_topm_row_frac_value = 0.0
                teacher_prob_sum_value = 0.0
                teacher_valid_actions_value = 0.0
                teacher_entropy_sum = 0.0
                teacher_max_prob_sum = 0.0
                teacher_uniform_row_frac_sum = 0.0
                teacher_bias_row_frac_sum = 0.0
                teacher_topm_row_frac_sum = 0.0
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

                        # Distillation: teacher from stop-grad steered policy
                        # (student logits + heuristic prior in logit space).
                        student_valid = mb_mask > 0
                        student_large_neg = torch.full_like(logits, -1e9)
                        student_logits = torch.where(
                            student_valid, logits, student_large_neg
                        )
                        student_log_probs = torch.log_softmax(
                            student_logits, dim=-1
                        )
                        teacher_probs, teacher_prior_inactive_rows, teacher_topm_rows = (
                            build_teacher_probs(
                                mb_mask,
                                student_logits,
                                mb_action_scores,
                                cfg.distill_teacher_tau,
                                cfg.distill_teacher_alpha,
                                cfg.distill_teacher_top_m,
                            )
                        )
                        distill_loss = -torch.sum(
                            teacher_probs * student_log_probs, dim=-1
                        ).mean()
                        hold_swap_distill_loss = torch.zeros(
                            (), device=device, dtype=logits.dtype
                        )
                        if (
                            hold_swap_distill_coef_now > 0.0
                            and str(cfg.action_space_kind).strip().lower()
                            == "placement_hold_step_v2"
                            and action_dim > 1
                        ):
                            mb_hold_swap_valid = (
                                b_hold_swap_teacher_valid[mb_inds] > 0.5
                            )
                            if bool(torch.any(mb_hold_swap_valid).item()):
                                hold_idx = action_dim - 1
                                non_hold_logsumexp = torch.logsumexp(
                                    student_logits[:, :hold_idx], dim=-1
                                )
                                hold_gate_logit = (
                                    student_logits[:, hold_idx] - non_hold_logsumexp
                                )
                                teacher_hold_prob = torch.clamp(
                                    b_hold_swap_teacher_prob[mb_inds],
                                    min=1e-4,
                                    max=1.0 - 1e-4,
                                )
                                hold_swap_distill_loss = (
                                    F.binary_cross_entropy_with_logits(
                                        hold_gate_logit[mb_hold_swap_valid],
                                        teacher_hold_prob[mb_hold_swap_valid],
                                        reduction="mean",
                                    )
                                )

                        policy_term = policy_loss
                        value_term = cfg.vf_coef * value_loss
                        entropy_term = -ent_coef_now * entropy
                        distill_term = distill_coef_now * distill_loss
                        hold_swap_distill_term = (
                            hold_swap_distill_coef_now * hold_swap_distill_loss
                        )
                        loss = (
                            policy_term
                            + value_term
                            + entropy_term
                            + distill_term
                            + hold_swap_distill_term
                        )

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
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
                        hold_swap_distill_loss_value = float(
                            hold_swap_distill_loss.detach().cpu().item()
                        )
                        policy_term_value = float(policy_term.detach().cpu().item())
                        value_term_value = float(value_term.detach().cpu().item())
                        entropy_term_value = float(entropy_term.detach().cpu().item())
                        distill_term_value = float(distill_term.detach().cpu().item())
                        hold_swap_distill_term_value = float(
                            hold_swap_distill_term.detach().cpu().item()
                        )
                        total_loss_value = float(loss.detach().cpu().item())
                        policy_loss_sum += policy_loss_value
                        value_loss_sum += value_loss_value
                        entropy_sum += entropy_value
                        distill_loss_sum += distill_loss_value
                        hold_swap_distill_loss_sum += hold_swap_distill_loss_value
                        policy_term_sum += policy_term_value
                        value_term_sum += value_term_value
                        entropy_term_sum += entropy_term_value
                        distill_term_sum += distill_term_value
                        hold_swap_distill_term_sum += hold_swap_distill_term_value
                        total_loss_sum += total_loss_value
                        valid_f = (mb_mask > 0).to(dtype=teacher_probs.dtype)
                        teacher_uniform_row_frac_value = float(
                            teacher_prior_inactive_rows.mean().detach().cpu().item()
                        )
                        teacher_bias_row_frac_value = 1.0 - teacher_uniform_row_frac_value
                        teacher_topm_row_frac_value = float(
                            teacher_topm_rows.mean().detach().cpu().item()
                        )
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
                        teacher_topm_row_frac_sum += teacher_topm_row_frac_value
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
                hold_swap_distill_loss_value = (
                    hold_swap_distill_loss_sum / updates_done_denom
                )
                policy_term_value = policy_term_sum / updates_done_denom
                value_term_value = value_term_sum / updates_done_denom
                entropy_term_value = entropy_term_sum / updates_done_denom
                distill_term_value = distill_term_sum / updates_done_denom
                hold_swap_distill_term_value = (
                    hold_swap_distill_term_sum / updates_done_denom
                )
                total_loss_value = total_loss_sum / updates_done_denom
                teacher_entropy_value = teacher_entropy_sum / updates_done_denom
                teacher_max_prob_value = teacher_max_prob_sum / updates_done_denom
                teacher_uniform_row_frac_value = (
                    teacher_uniform_row_frac_sum / updates_done_denom
                )
                teacher_bias_row_frac_value = (
                    teacher_bias_row_frac_sum / updates_done_denom
                )
                teacher_topm_row_frac_value = (
                    teacher_topm_row_frac_sum / updates_done_denom
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

                update_core_seconds = max(1e-6, time.time() - update_start_wall)
                total_seconds = max(1e-6, time.time() - training_start)
                sps = int(global_step / total_seconds)

                approx_kl_mean = (
                    float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
                )
                ret100_terms = {
                    key: _safe_recent_mean(values, window=100)
                    for key, values in completed_reward_component_sums.items()
                }
                for key in BLEND_STEP_TERM_KEYS:
                    ret100_terms[key] = _safe_recent_mean(
                        step_reward_component_values.get(key, []),
                        window=100,
                    )
                hold_probe_summary = summarize_hold_probe_accumulator(
                    hold_probe_update_accumulator,
                    eps=cfg.hold_margin_eps,
                    good_threshold=cfg.hold_margin_good_threshold,
                )
                hold_swap_summary = summarize_hold_swap_teacher_accumulator(
                    hold_swap_update_accumulator
                )
                stats = {
                    "update": update,
                    "global_step": global_step,
                    "piece_source_profile": "mixed",
                    "piece_source_mix": dict(env_piece_source_counts),
                    "reward_functions": list(cfg.reward_functions),
                    "reward_blend_unit": cfg.reward_blend_unit,
                    "reward_blend_unit_effective": blend_unit_effective,
                    "reward_blend_span": cfg.reward_blend_span,
                    "reward_blend_step_used": blend_step_now,
                    "reward_blend_fixed_step": fixed_blend_step,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                    "entropy": entropy_value,
                    "distill_loss": distill_loss_value,
                    "hold_swap_distill_loss": hold_swap_distill_loss_value,
                    "loss_total": total_loss_value,
                    "loss_policy_term": policy_term_value,
                    "loss_value_term": value_term_value,
                    "loss_entropy_term": entropy_term_value,
                    "loss_distill_term": distill_term_value,
                    "loss_hold_swap_distill_term": hold_swap_distill_term_value,
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
                    "adapter_status": adapter_state_now.get("status"),
                    "adapter_conv_status": adapter_state_now.get("conv_status"),
                    "adapter_trainable_params": int(
                        adapter_state_now.get("adapter_trainable", 0)
                    ),
                    "adapter_total_params": int(
                        adapter_state_now.get("adapter_total", 0)
                    ),
                    "adapter_in_optimizer": bool(
                        adapter_state_now.get("adapter_in_optimizer", False)
                    ),
                    "adapter_optimizer_params": int(
                        adapter_state_now.get("optimizer_adapter_params", 0)
                    ),
                    "adapter_optimizer_groups": list(
                        adapter_state_now.get("optimizer_groups", [])
                    ),
                    "adapter_lr_scale_used": float(
                        adapter_state_now.get("obs_adapter_lr_scale", 0.0)
                    ),
                    "adapter_effective_lr_used": float(
                        adapter_state_now.get("effective_lr", 0.0)
                    ),
                    "adapter_checks_ok": bool(
                        adapter_state_now.get("checks_ok", False)
                    ),
                    "adapter_checks": list(adapter_state_now.get("checks", [])),
                    "distill_coef_used": distill_coef_now,
                    "distill_teacher_alpha_used": cfg.distill_teacher_alpha,
                    "distill_teacher_tau_used": cfg.distill_teacher_tau,
                    "distill_teacher_top_m_used": cfg.distill_teacher_top_m,
                    "hold_swap_distill_coef_used": hold_swap_distill_coef_now,
                    "hold_swap_teacher_tau_used": cfg.hold_swap_teacher_tau,
                    "curriculum_topk_used": curriculum_topk_now,
                    "curriculum_bias_used": curriculum_bias_now,
                    "curriculum_danger_height_used": cfg.curriculum_danger_height,
                    "teacher_entropy": teacher_entropy_value,
                    "teacher_max_prob": teacher_max_prob_value,
                    "teacher_uniform_row_frac": teacher_uniform_row_frac_value,
                    "teacher_bias_row_frac": teacher_bias_row_frac_value,
                    "teacher_topm_row_frac": teacher_topm_row_frac_value,
                    "teacher_prob_sum": teacher_prob_sum_value,
                    "teacher_valid_actions": teacher_valid_actions_value,
                    "sps": sps,
                    "update_seconds": update_core_seconds,
                    "update_core_seconds": update_core_seconds,
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
                    "ret100_terms": ret100_terms,
                    "hold_probe_by_source": hold_probe_summary,
                    "hold_swap_by_source": hold_swap_summary,
                }
                stats["ret100_hierarchy"] = _reward_hierarchy_from_terms(
                    stats.get("ret100_terms")
                )
                should_log = (
                    update % cfg.log_every_updates == 0
                    or update == 1
                    or update == num_updates
                )
                should_validate = (
                    cfg.validation_episodes_per_env > 0
                    and (
                        update % cfg.validate_every_updates == 0
                        or update == num_updates
                    )
                )
                validation: dict[str, Any] | None = None
                validation_by_source: dict[str, Any] = {}
                if should_validate:
                    validation_wall_start = time.time()
                    for validation_source in validation_sources:
                        try:
                            validation_item = run_validation_eval(
                                cfg=cfg,
                                repo_root=repo_root,
                                server_cmd=server_cmd,
                                model=model,
                            obs_adapter=obs_adapter,
                            device=device,
                            update=update,
                            reward_blend_step=blend_step_now,
                            reward_blend_unit=blend_unit_effective,
                            piece_source_profile=validation_source,
                            reward_component_aliases=reward_component_aliases,
                        )
                        except Exception as error:
                            validation_item = {"enabled": False, "error": str(error)}
                            print(
                                "[ppo] "
                                f"validation failed at update={update} "
                                f"source={validation_source}: {error}"
                            )
                        validation_by_source[validation_source] = validation_item
                    if validation_sources:
                        validation = validation_by_source.get(validation_sources[0])
                    validation_wall_s += max(0.0, time.time() - validation_wall_start)
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
                        stats["validation_hierarchy"] = _reward_hierarchy_from_terms(
                            stats.get("validation_terms")
                        )
                if validation_by_source:
                    stats["validation_by_source"] = validation_by_source
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
                    profile_overhead_s = max(
                        0.0, update_core_seconds - profile_accounted_s
                    )
                    ret_terms = stats["ret100_terms"]
                    validation_by_source = stats.get("validation_by_source") or {}
                    source_label = source_mix_label
                    reward_blend_active = reward_fn_from != reward_fn_to
                    adapter_checks_label = (
                        "ok" if stats["adapter_checks_ok"] else "warn"
                    )
                    if reward_blend_active:
                        ret100_lines = [
                            (
                                "  ret100: "
                                f"final={_fmt_float(ret_terms.get('reward_final'))} "
                                f"base={_fmt_float(ret_terms.get('reward_base'))} "
                                f"{reward_fn_from}={_fmt_float(ret_terms.get('v1_base'))} "
                                f"{reward_fn_to}={_fmt_float(ret_terms.get('v2_base'))} "
                                f"{reward_fn_from}_raw={_fmt_float(ret_terms.get('v1_base_raw'))} "
                                f"{reward_fn_to}_raw={_fmt_float(ret_terms.get('v2_base_raw'))} "
                                f"hold_pen={_fmt_float(ret_terms.get('term_hold_margin_penalty'))} "
                                f"w1={_fmt_float(ret_terms.get('blend_legacy_weight'))} "
                                f"w2={_fmt_float(ret_terms.get('blend_target_weight'))} "
                                f"t={_fmt_float(ret_terms.get('blend_t'))} "
                                f"topout={_fmt_float(ret_terms.get('top_out_term'))}"
                            ),
                            (
                                f"    {reward_fn_from}_terms: "
                                f"lines={_fmt_float(ret_terms.get('v1_term_lines'))} "
                                f"score={_fmt_float(ret_terms.get('v1_term_score'))} "
                                f"time={_fmt_float(ret_terms.get('v1_term_time'))} "
                                f"height={_fmt_float(ret_terms.get('v1_term_height'))} "
                                f"holes={_fmt_float(ret_terms.get('v1_term_holes'))} "
                                f"bump={_fmt_float(ret_terms.get('v1_term_bumpiness'))} "
                                f"board={_fmt_float(ret_terms.get('v1_term_board_score'))} "
                                f"q={_fmt_float(ret_terms.get('v1_term_board_quality'))} "
                                f"q_abs={_fmt_float(ret_terms.get('v1_term_board_quality_abs'))} "
                                f"full_clear={_fmt_float(ret_terms.get('v1_term_full_clear'))} "
                                f"topout={_fmt_float(ret_terms.get('v1_term_top_out'))}"
                            ),
                            (
                                f"    {reward_fn_to}_terms: "
                                f"lines={_fmt_float(ret_terms.get('v2_term_lines'))} "
                                f"score={_fmt_float(ret_terms.get('v2_term_score'))} "
                                f"time={_fmt_float(ret_terms.get('v2_term_time'))} "
                                f"height={_fmt_float(ret_terms.get('v2_term_height'))} "
                                f"holes={_fmt_float(ret_terms.get('v2_term_holes'))} "
                                f"bump={_fmt_float(ret_terms.get('v2_term_bumpiness'))} "
                                f"board={_fmt_float(ret_terms.get('v2_term_board_score'))} "
                                f"q={_fmt_float(ret_terms.get('v2_term_board_quality'))} "
                                f"q_abs={_fmt_float(ret_terms.get('v2_term_board_quality_abs'))} "
                                f"full_clear={_fmt_float(ret_terms.get('v2_term_full_clear'))} "
                                f"topout={_fmt_float(ret_terms.get('v2_term_top_out'))}"
                            ),
                            (
                                "    blend_terms: "
                                f"lines={_fmt_float(ret_terms.get('term_lines'))} "
                                f"score={_fmt_float(ret_terms.get('term_score'))} "
                                f"time={_fmt_float(ret_terms.get('term_time'))} "
                                f"height={_fmt_float(ret_terms.get('term_height'))} "
                                f"holes={_fmt_float(ret_terms.get('term_holes'))} "
                                f"bump={_fmt_float(ret_terms.get('term_bumpiness'))} "
                                f"board={_fmt_float(ret_terms.get('term_board_score'))} "
                                f"q={_fmt_float(ret_terms.get('term_board_quality'))} "
                                f"q_abs={_fmt_float(ret_terms.get('term_board_quality_abs'))} "
                                f"full_clear={_fmt_float(ret_terms.get('term_full_clear'))} "
                                f"hold_pen={_fmt_float(ret_terms.get('term_hold_margin_penalty'))} "
                                f"topout={_fmt_float(ret_terms.get('term_top_out'))}"
                            ),
                        ]
                    else:
                        active_reward_fn = reward_fn_from
                        ret100_lines = [
                            (
                                "  ret100: "
                                f"final={_fmt_float(ret_terms.get('reward_final'))} "
                                f"base={_fmt_float(ret_terms.get('reward_base'))} "
                                f"{active_reward_fn}={_fmt_float(ret_terms.get('reward_base'))} "
                                f"hold_pen={_fmt_float(ret_terms.get('term_hold_margin_penalty'))} "
                                f"topout={_fmt_float(ret_terms.get('top_out_term'))}"
                            ),
                            (
                                f"    {active_reward_fn}_terms: "
                                f"lines={_fmt_float(ret_terms.get('term_lines'))} "
                                f"score={_fmt_float(ret_terms.get('term_score'))} "
                                f"time={_fmt_float(ret_terms.get('term_time'))} "
                                f"height={_fmt_float(ret_terms.get('term_height'))} "
                                f"holes={_fmt_float(ret_terms.get('term_holes'))} "
                                f"bump={_fmt_float(ret_terms.get('term_bumpiness'))} "
                                f"board={_fmt_float(ret_terms.get('term_board_score'))} "
                                f"q={_fmt_float(ret_terms.get('term_board_quality'))} "
                                f"q_abs={_fmt_float(ret_terms.get('term_board_quality_abs'))} "
                                f"full_clear={_fmt_float(ret_terms.get('term_full_clear'))} "
                                f"hold_pen={_fmt_float(ret_terms.get('term_hold_margin_penalty'))} "
                                f"topout={_fmt_float(ret_terms.get('term_top_out'))}"
                            ),
                        ]
                    log_lines = [
                        (
                            f"[ppo] update={update}/{num_updates} "
                            f"step={global_step} src={source_label} "
                            f"warmup={'y' if warmup_active else 'n'} "
                            f"sps={stats['sps']}"
                        ),
                        (
                            "  losses: "
                            f"ploss={stats['policy_loss']:.4f} "
                            f"vloss={stats['value_loss']:.4f} "
                            f"dloss={stats['distill_loss']:.4f} "
                            f"hsloss={stats['hold_swap_distill_loss']:.4f} "
                            f"total={stats['loss_total']:.4f} "
                            f"| terms(p={stats['loss_policy_term']:.4f},"
                            f"v={stats['loss_value_term']:.4f},"
                            f"ent={stats['loss_entropy_term']:.4f},"
                            f"dist={stats['loss_distill_term']:.4f},"
                            f"hold={stats['loss_hold_swap_distill_term']:.4f}) "
                            f"| ent={stats['entropy']:.4f} "
                            f"kl={stats['approx_kl']:.5f} "
                            f"clip={stats['clip_fraction']:.3f} "
                            f"ev={stats['explained_variance']:.3f}"
                        ),
                        (
                            "  schedule: "
                            f"ent_coef={ent_coef_now:.5f} "
                            f"distill_coef={distill_coef_now:.5f} "
                            f"hold_swap_coef={hold_swap_distill_coef_now:.5f} "
                            f"teacher_alpha={cfg.distill_teacher_alpha:.3f} "
                            f"tau={cfg.distill_teacher_tau:.3f} "
                            f"teacher_topm={cfg.distill_teacher_top_m} "
                            f"hold_tau={cfg.hold_swap_teacher_tau:.3f} "
                            f"pclip={cfg.policy_clip_coef:.4f} "
                            f"vclip={cfg.value_clip_coef:.4f} "
                            f"p_lr={policy_lr_now:.6g} "
                            f"v_lr={value_lr_now:.6g} "
                            f"target_kl={target_kl_now:.5f} "
                            f"validate={'y' if should_validate else 'n'}"
                        ),
                        (
                            "  adapter: "
                            f"status={stats['adapter_status']} "
                            f"conv={stats['adapter_conv_status']} "
                            f"lr_scale={stats['adapter_lr_scale_used']:.3f} "
                            f"eff_lr={stats['adapter_effective_lr_used']:.6g} "
                            f"checks={adapter_checks_label}"
                        ),
                        (
                            "  curriculum: "
                            f"topk={curriculum_topk_now} "
                            f"bias={curriculum_bias_now:.3f} "
                            f"teacher(ent={stats['teacher_entropy']:.3f},"
                            f"maxp={stats['teacher_max_prob']:.3f},"
                            f"uniform={stats['teacher_uniform_row_frac']:.3f},"
                            f"bias_rows={stats['teacher_bias_row_frac']:.3f},"
                            f"topm_rows={stats['teacher_topm_row_frac']:.3f},"
                            f"psum={stats['teacher_prob_sum']:.3f},"
                            f"valid={stats['teacher_valid_actions']:.1f})"
                        ),
                    ]
                    log_lines.extend(ret100_lines)
                    log_lines.extend(
                        [
                            (
                                "  timing: "
                                f"t_upd_core={update_core_seconds:.2f}s "
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
                                f"t_ovh={profile_overhead_s:.2f}s"
                            ),
                            (
                                "  mask_fix: "
                                f"rows={mask_repair_update['rows']} "
                                f"rows_total={mask_repair_total['rows']}"
                            ),
                        ]
                    )
                    for source in validation_sources:
                        probe_item = hold_probe_summary.get(source)
                        if probe_item is None:
                            continue
                        log_lines.extend(
                            _format_hold_probe_log_lines(
                                f"  hold_probe[{source}]",
                                probe_item,
                            )
                        )
                    for source in validation_sources:
                        hold_swap_item = hold_swap_summary.get(source)
                        if hold_swap_item is None:
                            continue
                        log_lines.extend(
                            _format_hold_swap_log_lines(
                                f"  hold_swap[{source}]",
                                hold_swap_item,
                            )
                        )
                    if not validation_by_source:
                        log_lines.append("  validation: skipped")
                    else:
                        for source in validation_sources:
                            source_item = (
                                validation_by_source.get(source)
                                if isinstance(validation_by_source, dict)
                                else None
                            )
                            source_dict = (
                                source_item if isinstance(source_item, dict) else {}
                            )
                            source_terms = source_dict.get("terms")
                            source_terms_dict = (
                                source_terms if isinstance(source_terms, dict) else {}
                            )
                            source_diagnostics = source_dict.get("diagnostics")
                            source_diag_dict = (
                                source_diagnostics
                                if isinstance(source_diagnostics, dict)
                                else {}
                            )
                            if bool(source_dict.get("enabled", False)):
                                if reward_blend_active:
                                    log_lines.append(
                                        "  validation["
                                        + source
                                        + "]: "
                                        + f"ret={_fmt_float(source_dict.get('mean_return'))} "
                                        + f"final={_fmt_float(source_terms_dict.get('reward_final'))} "
                                        + f"base={_fmt_float(source_terms_dict.get('reward_base'))} "
                                        + f"{reward_fn_from}={_fmt_float(source_terms_dict.get('v1_base'))} "
                                        + f"{reward_fn_to}={_fmt_float(source_terms_dict.get('v2_base'))} "
                                        + f"{reward_fn_from}_raw={_fmt_float(source_terms_dict.get('v1_base_raw'))} "
                                        + f"{reward_fn_to}_raw={_fmt_float(source_terms_dict.get('v2_base_raw'))} "
                                        + f"hold_pen={_fmt_float(source_terms_dict.get('term_hold_margin_penalty'))} "
                                        + f"w1={_fmt_float(source_terms_dict.get('blend_legacy_weight'))} "
                                        + f"w2={_fmt_float(source_terms_dict.get('blend_target_weight'))} "
                                        + f"t={_fmt_float(source_terms_dict.get('blend_t'))} "
                                        + f"topout={_fmt_float(source_terms_dict.get('top_out_term'))}"
                                    )
                                    log_lines.append(
                                        f"    val_{reward_fn_from}_terms["
                                        + source
                                        + "]: "
                                        + f"lines={_fmt_float(source_terms_dict.get('v1_term_lines'))} "
                                        + f"score={_fmt_float(source_terms_dict.get('v1_term_score'))} "
                                        + f"time={_fmt_float(source_terms_dict.get('v1_term_time'))} "
                                        + f"height={_fmt_float(source_terms_dict.get('v1_term_height'))} "
                                        + f"holes={_fmt_float(source_terms_dict.get('v1_term_holes'))} "
                                        + f"bump={_fmt_float(source_terms_dict.get('v1_term_bumpiness'))} "
                                        + f"board={_fmt_float(source_terms_dict.get('v1_term_board_score'))} "
                                        + f"q={_fmt_float(source_terms_dict.get('v1_term_board_quality'))} "
                                        + f"q_abs={_fmt_float(source_terms_dict.get('v1_term_board_quality_abs'))} "
                                        + f"full_clear={_fmt_float(source_terms_dict.get('v1_term_full_clear'))} "
                                        + f"topout={_fmt_float(source_terms_dict.get('v1_term_top_out'))}"
                                    )
                                    log_lines.append(
                                        f"    val_{reward_fn_to}_terms["
                                        + source
                                        + "]: "
                                        + f"lines={_fmt_float(source_terms_dict.get('v2_term_lines'))} "
                                        + f"score={_fmt_float(source_terms_dict.get('v2_term_score'))} "
                                        + f"time={_fmt_float(source_terms_dict.get('v2_term_time'))} "
                                        + f"height={_fmt_float(source_terms_dict.get('v2_term_height'))} "
                                        + f"holes={_fmt_float(source_terms_dict.get('v2_term_holes'))} "
                                        + f"bump={_fmt_float(source_terms_dict.get('v2_term_bumpiness'))} "
                                        + f"board={_fmt_float(source_terms_dict.get('v2_term_board_score'))} "
                                        + f"q={_fmt_float(source_terms_dict.get('v2_term_board_quality'))} "
                                        + f"q_abs={_fmt_float(source_terms_dict.get('v2_term_board_quality_abs'))} "
                                        + f"full_clear={_fmt_float(source_terms_dict.get('v2_term_full_clear'))} "
                                        + f"topout={_fmt_float(source_terms_dict.get('v2_term_top_out'))}"
                                    )
                                else:
                                    active_reward_fn = reward_fn_from
                                    log_lines.append(
                                        "  validation["
                                        + source
                                        + "]: "
                                        + f"ret={_fmt_float(source_dict.get('mean_return'))} "
                                        + f"final={_fmt_float(source_terms_dict.get('reward_final'))} "
                                        + f"base={_fmt_float(source_terms_dict.get('reward_base'))} "
                                        + f"{active_reward_fn}={_fmt_float(source_terms_dict.get('reward_base'))} "
                                        + f"hold_pen={_fmt_float(source_terms_dict.get('term_hold_margin_penalty'))} "
                                        + f"topout={_fmt_float(source_terms_dict.get('top_out_term'))}"
                                    )
                                    log_lines.append(
                                        f"    val_{active_reward_fn}_terms["
                                        + source
                                        + "]: "
                                        + f"lines={_fmt_float(source_terms_dict.get('term_lines'))} "
                                        + f"score={_fmt_float(source_terms_dict.get('term_score'))} "
                                        + f"time={_fmt_float(source_terms_dict.get('term_time'))} "
                                        + f"height={_fmt_float(source_terms_dict.get('term_height'))} "
                                        + f"holes={_fmt_float(source_terms_dict.get('term_holes'))} "
                                        + f"bump={_fmt_float(source_terms_dict.get('term_bumpiness'))} "
                                        + f"board={_fmt_float(source_terms_dict.get('term_board_score'))} "
                                        + f"q={_fmt_float(source_terms_dict.get('term_board_quality'))} "
                                        + f"q_abs={_fmt_float(source_terms_dict.get('term_board_quality_abs'))} "
                                        + f"full_clear={_fmt_float(source_terms_dict.get('term_full_clear'))} "
                                        + f"hold_pen={_fmt_float(source_terms_dict.get('term_hold_margin_penalty'))} "
                                        + f"topout={_fmt_float(source_terms_dict.get('term_top_out'))}"
                                    )
                                log_lines.append(
                                    "    val_diag["
                                    + source
                                    + "]: "
                                    + f"holes_created={_fmt_float(source_diag_dict.get('holes_created_total'))} "
                                    + f"max_height={_fmt_float(source_diag_dict.get('max_height_reached'))} "
                                    + f"kick_locks={_fmt_float(source_diag_dict.get('kick_assisted_locks'))} "
                                    + f"hold_uses={_fmt_float(source_diag_dict.get('hold_uses'))}"
                                )
                                log_lines.extend(
                                    _format_hold_probe_log_lines(
                                        f"    val_hold_probe[{source}]",
                                        source_diag_dict.get("hold_probe"),
                                    )
                                )
                                log_lines.extend(
                                    _format_hold_swap_log_lines(
                                        f"    val_hold_swap[{source}]",
                                        source_diag_dict.get("hold_swap"),
                                    )
                                )
                            else:
                                log_lines.append(
                                    "  validation["
                                    + source
                                    + "]: "
                                    + (
                                        f"error={source_dict.get('error')}"
                                        if source_dict.get("error")
                                        else "disabled"
                                    )
                                )
                    log_emit_start = time.time()
                    print("\n".join(log_lines))
                    if not stats["adapter_checks_ok"]:
                        print(
                            "[ppo] adapter integrity warning "
                            + format_adapter_state_log(adapter_state_now)
                        )
                    log_emit_wall_s += max(0.0, time.time() - log_emit_start)

                if update % cfg.save_every_updates == 0 or update == num_updates:
                    save_wall_start = time.time()
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
                    save_wall_s += max(0.0, time.time() - save_wall_start)

                update_total_seconds = max(1e-6, time.time() - update_start_wall)
                update_outside_core_seconds = max(
                    0.0, update_total_seconds - update_core_seconds
                )
                stats["update_seconds"] = update_total_seconds
                stats["update_validation_seconds"] = validation_wall_s
                stats["update_save_seconds"] = save_wall_s
                stats["update_log_emit_seconds"] = log_emit_wall_s
                stats["update_outside_core_seconds"] = update_outside_core_seconds

                if should_log:
                    now_wall = time.time()
                    log_gap_seconds = (
                        max(0.0, now_wall - last_log_wall)
                        if last_log_wall is not None
                        else float("nan")
                    )
                    last_log_wall = now_wall
                    print(
                        "[ppo] timing_ext "
                        f"update={update}/{num_updates} "
                        f"t_core={update_core_seconds:.2f}s "
                        f"t_val={validation_wall_s:.2f}s "
                        f"t_save={save_wall_s:.2f}s "
                        f"t_log={log_emit_wall_s:.2f}s "
                        f"t_outside={update_outside_core_seconds:.2f}s "
                        f"t_total={update_total_seconds:.2f}s "
                        f"t_since_last_log={_fmt_float(log_gap_seconds)}s"
                    )

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
