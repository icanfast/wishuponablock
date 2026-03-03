#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from wub_env import WubEnvBridge


def first_valid_actions(action_masks: list[list[float]]) -> list[int]:
    out: list[int] = []
    for mask in action_masks:
        idx = 0
        for i, value in enumerate(mask):
            if value > 0:
                idx = i
                break
        out.append(idx)
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with WubEnvBridge(cwd=repo_root) as env:
        init_result = env.init(
            mode_id="practice",
            num_envs=2,
            model_path="public/models/model_v4.json",
            piece_source_profile="bag7",
            max_pieces_per_episode=50,
            seed=42_030,
        )
        env_ids = init_result.get("env_ids", [0, 1])
        print("init env_ids:", env_ids)

        reset_result = env.reset_many(env_ids=env_ids, seeds=[1001, 1002])
        print(
            "reset:",
            "batch",
            len(reset_result["obs"]),
            "obs_dim",
            len(reset_result["obs"][0]) if reset_result["obs"] else 0,
            "action_dim",
            len(reset_result["action_masks"][0]) if reset_result["action_masks"] else 0,
        )

        for t in range(10):
            actions = first_valid_actions(reset_result["action_masks"])
            step_result = env.step_many(env_ids=env_ids, actions=actions)
            rewards = step_result["rewards"]
            dones = step_result["dones"]
            print(f"step {t}: rewards={rewards} dones={dones}")
            reset_result = step_result


if __name__ == "__main__":
    main()
