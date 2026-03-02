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
from torch.distributions import Categorical

from wub_env import WubEnvBridge


@dataclass(frozen=True)
class PPOConfig:
    mode_id: str
    model_path: str
    queue_policy_id: str
    piece_source_profile: str
    max_pieces_per_episode: int
    seed: int
    num_envs: int
    total_timesteps: int
    num_steps: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_vloss: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    target_kl: float
    device: str
    save_every_updates: int
    log_every_updates: int
    out_dir: str
    run_name: str
    server_cmd: str | None
    resume_checkpoint: str | None
    init_artifact: str | None
    deterministic_eval: bool


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.policy_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.policy_head.bias)
        nn.init.kaiming_uniform_(self.value_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.fc1(obs))
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(
        description="Train WUB headless bot with full PPO in local PyTorch."
    )
    parser.add_argument("--mode-id", default="charcuterie")
    parser.add_argument("--model-path", default="public/models/model_v4.json")
    parser.add_argument("--queue-policy-id", default="next_piece_v1")
    parser.add_argument("--piece-source-profile", default="bag7")
    parser.add_argument("--max-pieces-per-episode", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42030)

    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-steps", type=int, default=256)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--target-kl", type=float, default=0.02)

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

    args = parser.parse_args()
    run_name = args.run_name.strip() or f"ppo_{args.mode_id}_{int(time.time())}"
    return PPOConfig(
        mode_id=args.mode_id.strip().lower(),
        model_path=args.model_path,
        queue_policy_id=args.queue_policy_id.strip().lower(),
        piece_source_profile=(
            "active_generator"
            if args.piece_source_profile == "active_generator"
            else "bag7"
        ),
        max_pieces_per_episode=max(1, int(args.max_pieces_per_episode)),
        seed=max(1, int(args.seed)),
        num_envs=max(1, int(args.num_envs)),
        total_timesteps=max(1, int(args.total_timesteps)),
        num_steps=max(1, int(args.num_steps)),
        hidden_dim=max(8, int(args.hidden_dim)),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_coef=float(args.clip_coef),
        clip_vloss=bool(args.clip_vloss),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        update_epochs=max(1, int(args.update_epochs)),
        minibatch_size=max(1, int(args.minibatch_size)),
        target_kl=max(0.0, float(args.target_kl)),
        device=args.device,
        save_every_updates=max(1, int(args.save_every_updates)),
        log_every_updates=max(1, int(args.log_every_updates)),
        out_dir=args.out_dir,
        run_name=run_name,
        server_cmd=args.server_cmd,
        resume_checkpoint=args.resume_checkpoint,
        init_artifact=args.init_artifact,
        deterministic_eval=bool(args.deterministic_eval),
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


def ensure_action_masks(mask_np: np.ndarray) -> np.ndarray:
    # Guarantee at least one valid action per env row.
    mask = np.asarray(mask_np, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected action mask shape [N, A], got {mask.shape}")
    valid_counts = mask.sum(axis=1)
    invalid_rows = np.where(valid_counts <= 0)[0]
    if invalid_rows.size > 0:
        mask[invalid_rows, :] = 1.0
    return mask


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    huge = torch.tensor(1e9, device=logits.device, dtype=logits.dtype)
    masked_logits = logits - (1.0 - action_mask) * huge
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


def export_bot_policy_artifact(
    model: PolicyValueNet,
    cfg: PPOConfig,
    obs_dim: int,
    action_dim: int,
    pipeline_id: str = "bot_ppo_offline_v1",
) -> dict[str, Any]:
    with torch.no_grad():
        w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # [H, I]
        b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)  # [H]
        wp = (
            model.policy_head.weight.detach().cpu().numpy().astype(np.float32)
        )  # [A, H]
        bp = model.policy_head.bias.detach().cpu().numpy().astype(np.float32)  # [A]
        wv = model.value_head.weight.detach().cpu().numpy().astype(np.float32)  # [1, H]
        bv = model.value_head.bias.detach().cpu().numpy().astype(np.float32)  # [1]

    now_ms = int(time.time() * 1000)
    artifact = {
        "id": f"bot_policy_{cfg.mode_id}_{now_ms}",
        "modeId": cfg.mode_id,
        "archId": "full",
        "queuePolicyId": cfg.queue_policy_id,
        "pipelineId": pipeline_id,
        "pieceSourceProfile": cfg.piece_source_profile,
        "createdAtMs": now_ms,
        "inputDim": int(obs_dim),
        "hiddenDim": int(cfg.hidden_dim),
        "actionDim": int(action_dim),
        "actionSpaceKind": "placement_v1",
        "placementActionDim": int(action_dim),
        "weights": {
            "w1": w1.T.reshape(-1).tolist(),  # [I, H]
            "b1": b1.reshape(-1).tolist(),
            "wp": wp.T.reshape(-1).tolist(),  # [H, A]
            "bp": bp.reshape(-1).tolist(),
            "wv": wv.reshape(-1).tolist(),  # [H]
            "bv": bv.reshape(-1).tolist(),  # [1]
        },
    }
    return artifact


def load_from_artifact(model: PolicyValueNet, artifact_path: Path) -> None:
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    weights = raw.get("weights", {})
    input_dim = int(raw.get("inputDim", 0))
    hidden_dim = int(raw.get("hiddenDim", 0))
    action_dim = int(raw.get("actionDim", 0))

    if input_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
        raise ValueError("Invalid artifact dims.")

    if model.fc1.in_features != input_dim:
        raise ValueError(
            f"Artifact inputDim mismatch. artifact={input_dim} model={model.fc1.in_features}"
        )
    if model.fc1.out_features != hidden_dim:
        raise ValueError(
            f"Artifact hiddenDim mismatch. artifact={hidden_dim} model={model.fc1.out_features}"
        )
    if model.policy_head.out_features != action_dim:
        raise ValueError(
            f"Artifact actionDim mismatch. artifact={action_dim} model={model.policy_head.out_features}"
        )

    w1 = np.asarray(weights.get("w1", []), dtype=np.float32).reshape(input_dim, hidden_dim)
    b1 = np.asarray(weights.get("b1", []), dtype=np.float32).reshape(hidden_dim)
    wp = np.asarray(weights.get("wp", []), dtype=np.float32).reshape(hidden_dim, action_dim)
    bp = np.asarray(weights.get("bp", []), dtype=np.float32).reshape(action_dim)
    wv = np.asarray(weights.get("wv", []), dtype=np.float32).reshape(hidden_dim)
    bv = np.asarray(weights.get("bv", []), dtype=np.float32).reshape(1)

    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(w1.T))
        model.fc1.bias.copy_(torch.from_numpy(b1))
        model.policy_head.weight.copy_(torch.from_numpy(wp.T))
        model.policy_head.bias.copy_(torch.from_numpy(bp))
        model.value_head.weight.copy_(torch.from_numpy(wv.reshape(1, hidden_dim)))
        model.value_head.bias.copy_(torch.from_numpy(bv))


def save_checkpoint(
    checkpoint_path: Path,
    model: PolicyValueNet,
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
    optimizer: torch.optim.Optimizer,
    map_device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    global_step = int(checkpoint.get("global_step", 0))
    update = int(checkpoint.get("update", 0))
    return global_step, update


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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
        mask_np = ensure_action_masks(np.asarray(reset_result["action_masks"], dtype=np.float32))
        obs_dim = infer_obs_dim(reset_result["obs"])
        action_dim = infer_action_dim(reset_result["action_masks"])

        model = PolicyValueNet(obs_dim, cfg.hidden_dim, action_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

        global_step = 0
        start_update = 0
        if cfg.resume_checkpoint:
            checkpoint_path = Path(cfg.resume_checkpoint).resolve()
            global_step, start_update = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(
                f"[ppo] resumed checkpoint: {checkpoint_path} "
                f"(global_step={global_step}, update={start_update})"
            )
        elif cfg.init_artifact:
            artifact_path = Path(cfg.init_artifact).resolve()
            load_from_artifact(model, artifact_path)
            print(f"[ppo] initialized from bot artifact: {artifact_path}")

        batch_size = cfg.num_envs * cfg.num_steps
        if cfg.minibatch_size > batch_size:
            raise ValueError(
                f"minibatch_size ({cfg.minibatch_size}) exceeds batch_size ({batch_size})."
            )
        num_updates = max(1, cfg.total_timesteps // batch_size)

        print(
            "[ppo] starting training "
            f"(device={device.type}, obs_dim={obs_dim}, action_dim={action_dim}, "
            f"batch_size={batch_size}, updates={num_updates})"
        )
        write_json(
            out_dir / "config.json",
            {
                **cfg.__dict__,
                "device_resolved": device.type,
                "obs_dim": obs_dim,
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

        training_start = time.time()
        best_score = float("-inf")
        best_update = 0
        best_stats: dict[str, Any] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        did_interrupt = False
        try:
            for update in range(start_update + 1, num_updates + 1):
                update_start = time.time()

                obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
                mask_buf = torch.zeros((cfg.num_steps, cfg.num_envs, action_dim), device=device)
                action_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device, dtype=torch.long)
                logprob_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                reward_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                done_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
                value_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

                for step in range(cfg.num_steps):
                    obs_t = as_tensor(obs_np, device)
                    mask_t = as_tensor(mask_np, device)
                    obs_buf[step] = obs_t
                    mask_buf[step] = mask_t

                    with torch.no_grad():
                        logits, values = model(obs_t)
                        dist = masked_categorical(logits, mask_t)
                        if cfg.deterministic_eval:
                            actions_t = torch.argmax(dist.probs, dim=-1)
                        else:
                            actions_t = dist.sample()
                        logprob_t = dist.log_prob(actions_t)

                    action_buf[step] = actions_t
                    logprob_buf[step] = logprob_t
                    value_buf[step] = values

                    actions_np = actions_t.detach().cpu().numpy().astype(np.int64).tolist()
                    step_result = env.step_many(env_ids=env_ids, actions=actions_np)
                    next_obs_np = np.asarray(step_result["obs"], dtype=np.float32)
                    next_mask_np = ensure_action_masks(
                        np.asarray(step_result["action_masks"], dtype=np.float32)
                    )
                    rewards_np = np.asarray(step_result["rewards"], dtype=np.float32)
                    dones_np = np.asarray(step_result["dones"], dtype=np.float32)

                    reward_buf[step] = as_tensor(rewards_np, device)
                    done_buf[step] = as_tensor(dones_np, device)

                    ep_return += rewards_np.astype(np.float64)
                    ep_length += 1

                    done_indices = np.where(dones_np > 0.5)[0]
                    if done_indices.size > 0:
                        completed_returns.extend(ep_return[done_indices].tolist())
                        completed_lengths.extend(ep_length[done_indices].tolist())
                        ep_return[done_indices] = 0.0
                        ep_length[done_indices] = 0

                        done_env_ids = [env_ids[int(i)] for i in done_indices.tolist()]
                        done_seeds = [
                            cfg.seed + global_step + int(i) * 991 + update * 131
                            for i in done_indices.tolist()
                        ]
                        reset_done = env.reset_many(env_ids=done_env_ids, seeds=done_seeds)
                        reset_obs = np.asarray(reset_done["obs"], dtype=np.float32)
                        reset_masks = ensure_action_masks(
                            np.asarray(reset_done["action_masks"], dtype=np.float32)
                        )
                        for local_pos, env_idx in enumerate(done_indices.tolist()):
                            next_obs_np[env_idx] = reset_obs[local_pos]
                            next_mask_np[env_idx] = reset_masks[local_pos]

                    obs_np = next_obs_np
                    mask_np = next_mask_np
                    global_step += cfg.num_envs

                with torch.no_grad():
                    next_obs_t = as_tensor(obs_np, device)
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

                b_obs = obs_buf.reshape((-1, obs_dim))
                b_masks = mask_buf.reshape((-1, action_dim))
                b_actions = action_buf.reshape(-1)
                b_logprobs = logprob_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = value_buf.reshape(-1)

                adv_mean = b_advantages.mean()
                adv_std = b_advantages.std(unbiased=False) + 1e-8
                b_advantages = (b_advantages - adv_mean) / adv_std

                batch_inds = np.arange(batch_size)
                clipfracs: list[float] = []
                approx_kl_value = 0.0
                policy_loss_value = 0.0
                value_loss_value = 0.0
                entropy_value = 0.0
                updates_done = 0
                early_stopped = False

                for _epoch in range(cfg.update_epochs):
                    np.random.shuffle(batch_inds)
                    for start in range(0, batch_size, cfg.minibatch_size):
                        end = start + cfg.minibatch_size
                        mb_inds_np = batch_inds[start:end]
                        mb_inds = torch.as_tensor(
                            mb_inds_np, device=device, dtype=torch.long
                        )

                        logits, new_values = model(b_obs[mb_inds])
                        dist = masked_categorical(logits, b_masks[mb_inds])
                        new_logprob = dist.log_prob(b_actions[mb_inds])
                        entropy = dist.entropy().mean()

                        logratio = new_logprob - b_logprobs[mb_inds]
                        ratio = torch.exp(logratio)

                        with torch.no_grad():
                            approx_kl = (b_logprobs[mb_inds] - new_logprob).mean()
                            clipfrac = (
                                (ratio - 1.0).abs() > cfg.clip_coef
                            ).float().mean()
                            approx_kl_value = float(approx_kl.detach().cpu().item())
                            clipfracs.append(float(clipfrac.detach().cpu().item()))

                        mb_adv = b_advantages[mb_inds]
                        pg_loss_1 = -mb_adv * ratio
                        pg_loss_2 = -mb_adv * torch.clamp(
                            ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                        )
                        policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                        value_pred = new_values
                        if cfg.clip_vloss:
                            value_pred_clipped = b_values[mb_inds] + (
                                value_pred - b_values[mb_inds]
                            ).clamp(-cfg.clip_coef, cfg.clip_coef)
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

                        loss = (
                            policy_loss
                            + cfg.vf_coef * value_loss
                            - cfg.ent_coef * entropy
                        )

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        optimizer.step()

                        policy_loss_value = float(policy_loss.detach().cpu().item())
                        value_loss_value = float(value_loss.detach().cpu().item())
                        entropy_value = float(entropy.detach().cpu().item())
                        updates_done += 1

                    if cfg.target_kl > 0 and approx_kl_value > cfg.target_kl:
                        early_stopped = True
                        break

                y_pred = b_values.detach().cpu().numpy()
                y_true = b_returns.detach().cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    float("nan")
                    if var_y <= 1e-12
                    else 1.0 - float(np.var(y_true - y_pred) / var_y)
                )

                update_seconds = max(1e-6, time.time() - update_start)
                total_seconds = max(1e-6, time.time() - training_start)
                sps = int(global_step / total_seconds)

                stats = {
                    "update": update,
                    "global_step": global_step,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                    "entropy": entropy_value,
                    "approx_kl": approx_kl_value,
                    "clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "explained_variance": explained_var,
                    "updates_done": updates_done,
                    "early_stopped_kl": early_stopped,
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
                }

                score_raw = stats["mean_episode_return_recent"]
                score = (
                    float(score_raw)
                    if isinstance(score_raw, (float, int))
                    and math.isfinite(float(score_raw))
                    else float("-inf")
                )
                if best_state_dict is None or score > best_score:
                    best_score = score
                    best_update = update
                    best_stats = dict(stats)
                    best_state_dict = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in model.state_dict().items()
                    }
                    best_ckpt_path = checkpoints_dir / "ppo_best.pt"
                    save_checkpoint(
                        checkpoint_path=best_ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        global_step=global_step,
                        update=update,
                        stats=stats,
                    )
                    best_artifact = export_bot_policy_artifact(
                        model, cfg, obs_dim, action_dim
                    )
                    write_json(out_dir / "bot_policy_best.json", best_artifact)
                    write_json(out_dir / "best_stats.json", best_stats)
                    print("[ppo] " f"new_best update={update} ret100={best_score:.3f}")

                if update % cfg.log_every_updates == 0 or update == 1 or update == num_updates:
                    print(
                        "[ppo] "
                        f"update={update}/{num_updates} "
                        f"step={global_step} "
                        f"ploss={stats['policy_loss']:.4f} "
                        f"vloss={stats['value_loss']:.4f} "
                        f"ent={stats['entropy']:.4f} "
                        f"kl={stats['approx_kl']:.5f} "
                        f"clip={stats['clip_fraction']:.3f} "
                        f"ev={stats['explained_variance']:.3f} "
                        f"ret100={stats['mean_episode_return_recent']:.3f} "
                        f"sps={stats['sps']}"
                    )

                if update % cfg.save_every_updates == 0 or update == num_updates:
                    ckpt_path = checkpoints_dir / f"ppo_update_{update:06d}.pt"
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        global_step=global_step,
                        update=update,
                        stats=stats,
                    )
                    artifact = export_bot_policy_artifact(model, cfg, obs_dim, action_dim)
                    artifact_path = artifacts_dir / f"bot_policy_update_{update:06d}.json"
                    write_json(artifact_path, artifact)
                    write_json(out_dir / "last_stats.json", stats)
        except KeyboardInterrupt:
            did_interrupt = True
            print("[ppo] keyboard interrupt received; finishing with best available policy...")

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            if best_stats is not None:
                write_json(out_dir / "best_stats.json", best_stats)

        artifact_final = export_bot_policy_artifact(model, cfg, obs_dim, action_dim)
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
