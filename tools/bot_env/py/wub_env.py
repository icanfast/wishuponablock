#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _resolve_default_server_cmd() -> list[str]:
    if os.name == "nt":
        candidates = ("npx.cmd", "npx.exe", "npx")
    else:
        candidates = ("npx",)
    for candidate in candidates:
        if shutil.which(candidate):
            return [candidate, "--yes", "tsx", "tools/bot_env/ts/envServer.ts"]
    # Keep a predictable fallback so caller gets a clear process launch error.
    return [candidates[0], "--yes", "tsx", "tools/bot_env/ts/envServer.ts"]


class _EnvServerClient:
    def __init__(
        self,
        server_cmd: list[str],
        cwd: Path,
    ) -> None:
        self.server_cmd = list(server_cmd)
        self.cwd = Path(cwd)
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self.server_cmd,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._request("close", {})
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def init(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("init", payload)

    def reset_many(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("reset_many", payload)

    def set_piece_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("set_piece_source", payload)

    def set_piece_sources(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("set_piece_sources", payload)

    def set_curriculum(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("set_curriculum", payload)

    def set_reward_blend_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("set_reward_blend_step", payload)

    def step_many(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("step_many", payload)

    def evaluate_hold_candidates_many(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("evaluate_hold_candidates_many", payload)

    def probe_actions_many(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("probe_actions_many", payload)

    def pop_trajectory(self) -> dict[str, Any]:
        return self._request("pop_trajectory", {})

    def _request(self, cmd: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            self.start()
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        request_id = self._next_id
        self._next_id += 1
        req = {"id": request_id, "cmd": cmd, "payload": payload}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr = ""
            if self._proc.stderr is not None:
                try:
                    stderr = self._proc.stderr.read()
                except Exception:
                    stderr = ""
            raise RuntimeError(f"Bridge server exited unexpectedly. stderr={stderr}")
        resp = json.loads(line)
        if int(resp.get("id", -1)) != request_id:
            raise RuntimeError(
                f"Mismatched response id. expected={request_id} got={resp.get('id')}"
            )
        if not bool(resp.get("ok", False)):
            raise RuntimeError(str(resp.get("error", "Unknown bridge error.")))
        return dict(resp.get("result", {}))


class WubEnvBridge:
    def __init__(
        self,
        server_cmd: list[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        shard_count: int = 1,
    ) -> None:
        self.cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[3]
        self.server_cmd = (
            server_cmd if server_cmd is not None else _resolve_default_server_cmd()
        )
        self.shard_count = max(1, int(shard_count))
        self._clients: list[_EnvServerClient] = []
        self._active_shard_count = 0
        self._executor: ThreadPoolExecutor | None = None
        self._env_to_shard: dict[int, tuple[int, int]] = {}
        self._next_pop_shard = 0

    def start(self) -> None:
        return

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for client in self._clients:
            client.close()
        self._clients = []
        self._active_shard_count = 0
        self._env_to_shard.clear()
        self._next_pop_shard = 0

    def __enter__(self) -> "WubEnvBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_active_clients(self, count: int) -> None:
        target = max(1, int(count))
        if len(self._clients) > target:
            for client in self._clients[target:]:
                client.close()
            self._clients = self._clients[:target]
        while len(self._clients) < target:
            self._clients.append(_EnvServerClient(self.server_cmd, self.cwd))
        for client in self._clients:
            client.start()
        self._active_shard_count = target
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=target)

    def _active_clients(self) -> list[_EnvServerClient]:
        return self._clients[: self._active_shard_count]

    def _split_env_counts(self, num_envs: int) -> list[int]:
        active_shards = max(1, min(self.shard_count, max(1, int(num_envs))))
        base = int(num_envs) // active_shards
        rem = int(num_envs) % active_shards
        return [base + (1 if shard_idx < rem else 0) for shard_idx in range(active_shards)]

    def _run_shard_calls(self, calls: list[tuple[int, Any]]) -> dict[int, Any]:
        if not calls:
            return {}
        if len(calls) <= 1 or self._executor is None:
            return {shard_idx: fn() for shard_idx, fn in calls}
        future_map = {
            self._executor.submit(fn): shard_idx for shard_idx, fn in calls
        }
        results: dict[int, Any] = {}
        for future, shard_idx in future_map.items():
            results[shard_idx] = future.result()
        return results

    def _require_env_mapping(self, env_id: int) -> tuple[int, int]:
        key = int(env_id)
        mapping = self._env_to_shard.get(key)
        if mapping is None:
            raise KeyError(f"Unknown env id: {key}")
        return mapping

    def _partition_env_ids(
        self,
        env_ids: list[int],
        *,
        seeds: list[int] | None = None,
        initial_boards: list[list[list[int]] | None] | None = None,
        actions: list[int] | None = None,
        piece_source_profiles: list[str] | None = None,
    ) -> dict[int, dict[str, Any]]:
        partitions: dict[int, dict[str, Any]] = {}
        for pos, env_id in enumerate(env_ids):
            shard_idx, local_env_id = self._require_env_mapping(int(env_id))
            part = partitions.setdefault(
                shard_idx,
                {
                    "positions": [],
                    "env_ids": [],
                    "seeds": [],
                    "initial_boards": [],
                    "actions": [],
                    "piece_source_profiles": [],
                },
            )
            part["positions"].append(pos)
            part["env_ids"].append(local_env_id)
            if seeds is not None:
                part["seeds"].append(int(seeds[pos]))
            if initial_boards is not None:
                part["initial_boards"].append(initial_boards[pos])
            if actions is not None:
                part["actions"].append(int(actions[pos]))
            if piece_source_profiles is not None:
                part["piece_source_profiles"].append(
                    str(piece_source_profiles[pos]).strip().lower()
                )
        return partitions

    def _merge_step_like_results(
        self,
        env_ids: list[int],
        partitions: dict[int, dict[str, Any]],
        shard_results: dict[int, dict[str, Any]],
        *,
        profile_keys: tuple[str, ...],
        profile_wall_s: float,
    ) -> dict[str, Any]:
        obs: list[Any] = [None] * len(env_ids)
        action_masks: list[Any] = [None] * len(env_ids)
        action_biases: list[Any] = [None] * len(env_ids)
        action_scores: list[Any] = [None] * len(env_ids)
        rewards: list[Any] = [0] * len(env_ids)
        dones: list[Any] = [False] * len(env_ids)
        infos: list[Any] = [None] * len(env_ids)
        profile: dict[str, Any] = {
            "batch_total_s": float(profile_wall_s),
            "env_count": len(env_ids),
        }
        for key in profile_keys:
            profile[key] = 0.0
        for shard_idx, part in partitions.items():
            result = shard_results[shard_idx]
            shard_obs = list(result.get("obs", []))
            shard_masks = list(result.get("action_masks", []))
            shard_biases = list(result.get("action_biases", []))
            shard_scores = list(result.get("action_scores", []))
            shard_rewards = list(result.get("rewards", []))
            shard_dones = list(result.get("dones", []))
            shard_infos = list(result.get("infos", []))
            for local_pos, global_pos in enumerate(part["positions"]):
                obs[global_pos] = shard_obs[local_pos]
                action_masks[global_pos] = shard_masks[local_pos]
                action_biases[global_pos] = shard_biases[local_pos]
                action_scores[global_pos] = shard_scores[local_pos]
                rewards[global_pos] = shard_rewards[local_pos]
                dones[global_pos] = shard_dones[local_pos]
                infos[global_pos] = shard_infos[local_pos]
            shard_profile = result.get("profile", {})
            if isinstance(shard_profile, dict):
                for key in profile_keys:
                    value = shard_profile.get(key)
                    if isinstance(value, (int, float)):
                        profile[key] += float(value)
        return {
            "obs": obs,
            "action_masks": action_masks,
            "action_biases": action_biases,
            "action_scores": action_scores,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
            "profile": profile,
        }

    def init(
        self,
        mode_id: str = "practice",
        num_envs: int = 1,
        model_path: str = "public/models/model_v4.json",
        observation_space: str = "raw_v1",
        phase_context_enabled: bool = False,
        placement_execution_mode: str = "teleport",
        action_space_kind: str = "placement_full_v1",
        piece_source_profile: str = "bag7",
        queue_policy_id: str = "next_piece_v1",
        max_pieces_per_episode: int = 512,
        reward_function_from: str = "v1",
        reward_function_to: str = "v2",
        reward_blend_timesteps: int = 10_000_000,
        reward_blend_unit: str = "updates",
        reward_blend_start_step: int = 0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        env_counts = self._split_env_counts(max(1, int(num_envs)))
        self._ensure_active_clients(len(env_counts))
        self._env_to_shard.clear()
        self._next_pop_shard = 0
        base_seed = int(seed) if seed is not None else None
        calls: list[tuple[int, Any]] = []
        for shard_idx, env_count in enumerate(env_counts):
            payload: dict[str, Any] = {
                "modeId": mode_id,
                "numEnvs": env_count,
                "modelPath": model_path,
                "observationSpace": observation_space,
                "phaseContextEnabled": bool(phase_context_enabled),
                "placementExecutionMode": placement_execution_mode,
                "actionSpaceKind": str(action_space_kind).strip().lower(),
                "pieceSourceProfile": piece_source_profile,
                "queuePolicyId": queue_policy_id,
                "maxPiecesPerEpisode": max_pieces_per_episode,
                "rewardFunctionFrom": str(reward_function_from).strip().lower(),
                "rewardFunctionTo": str(reward_function_to).strip().lower(),
                "rewardBlendTimesteps": int(reward_blend_timesteps),
                "rewardBlendUnit": (
                    "timesteps"
                    if str(reward_blend_unit).strip().lower() == "timesteps"
                    else "updates"
                ),
                "rewardBlendStartStep": int(reward_blend_start_step),
            }
            if base_seed is not None:
                payload["seed"] = base_seed + shard_idx * 104729
            client = self._active_clients()[shard_idx]
            calls.append((shard_idx, lambda c=client, p=payload: c.init(p)))
        results = self._run_shard_calls(calls)

        global_env_ids: list[int] = []
        global_env_id = 0
        for shard_idx, env_count in enumerate(env_counts):
            result = results[shard_idx]
            local_env_ids = [int(v) for v in result.get("env_ids", [])]
            if len(local_env_ids) != env_count:
                raise RuntimeError(
                    f"Shard init env count mismatch. shard={shard_idx} expected={env_count} got={len(local_env_ids)}"
                )
            for local_env_id in local_env_ids:
                self._env_to_shard[global_env_id] = (shard_idx, int(local_env_id))
                global_env_ids.append(global_env_id)
                global_env_id += 1
        return {"env_ids": global_env_ids}

    def reset_many(
        self,
        env_ids: list[int],
        seeds: list[int] | None = None,
        initial_boards: list[list[list[int]] | None] | None = None,
    ) -> dict[str, Any]:
        if not env_ids:
            return {
                "obs": [],
                "action_masks": [],
                "action_biases": [],
                "action_scores": [],
                "rewards": [],
                "dones": [],
                "infos": [],
                "profile": {
                    "batch_total_s": 0.0,
                    "env_count": 0,
                    "reset_env_total_s": 0.0,
                    "reset_obs_s": 0.0,
                    "reset_choices_s": 0.0,
                },
            }
        partitions = self._partition_env_ids(
            env_ids,
            seeds=seeds,
            initial_boards=initial_boards,
        )
        calls: list[tuple[int, Any]] = []
        for shard_idx, part in partitions.items():
            payload: dict[str, Any] = {"envIds": list(part["env_ids"])}
            if seeds is not None:
                payload["seeds"] = list(part["seeds"])
            if initial_boards is not None:
                payload["initialBoards"] = list(part["initial_boards"])
            client = self._active_clients()[shard_idx]
            calls.append((shard_idx, lambda c=client, p=payload: c.reset_many(p)))
        wall_start = time.perf_counter()
        results = self._run_shard_calls(calls)
        wall_s = time.perf_counter() - wall_start
        return self._merge_step_like_results(
            env_ids,
            partitions,
            results,
            profile_keys=("reset_env_total_s", "reset_obs_s", "reset_choices_s"),
            profile_wall_s=wall_s,
        )

    def set_piece_source(self, piece_source_profile: str) -> dict[str, Any]:
        payload = {"pieceSourceProfile": str(piece_source_profile).strip().lower()}
        calls = [
            (
                shard_idx,
                lambda c=client, p=payload: c.set_piece_source(p),
            )
            for shard_idx, client in enumerate(self._active_clients())
        ]
        results = self._run_shard_calls(calls)
        piece_source = payload["pieceSourceProfile"]
        for result in results.values():
            value = result.get("piece_source_profile")
            if isinstance(value, str) and value:
                piece_source = value
                break
        return {"piece_source_profile": piece_source}

    def set_piece_sources(
        self,
        env_ids: list[int],
        piece_source_profiles: list[str],
    ) -> dict[str, Any]:
        if len(env_ids) != len(piece_source_profiles):
            raise ValueError("env_ids and piece_source_profiles must have matching lengths")
        partitions = self._partition_env_ids(
            env_ids,
            piece_source_profiles=piece_source_profiles,
        )
        calls: list[tuple[int, Any]] = []
        for shard_idx, part in partitions.items():
            payload = {
                "envIds": list(part["env_ids"]),
                "pieceSourceProfiles": list(part["piece_source_profiles"]),
            }
            client = self._active_clients()[shard_idx]
            calls.append(
                (shard_idx, lambda c=client, p=payload: c.set_piece_sources(p))
            )
        results = self._run_shard_calls(calls)
        assigned = 0
        counts: dict[str, int] = {}
        for result in results.values():
            assigned += int(result.get("assigned", 0))
            shard_counts = result.get("counts", {})
            if isinstance(shard_counts, dict):
                for key, value in shard_counts.items():
                    counts[str(key)] = counts.get(str(key), 0) + int(value)
        return {"assigned": assigned, "counts": counts}

    def set_curriculum(
        self,
        top_k: int | None = None,
        bias_strength: float | None = None,
        danger_height: int | None = None,
        compute_scores: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if top_k is not None:
            payload["topK"] = int(top_k)
        if bias_strength is not None:
            payload["biasStrength"] = float(bias_strength)
        if danger_height is not None:
            payload["dangerHeight"] = int(danger_height)
        if compute_scores is not None:
            payload["computeScores"] = bool(compute_scores)
        calls = [
            (shard_idx, lambda c=client, p=payload: c.set_curriculum(p))
            for shard_idx, client in enumerate(self._active_clients())
        ]
        results = self._run_shard_calls(calls)
        if results:
            first = results[min(results.keys())]
            return {
                "top_k": int(first.get("top_k", top_k or 0)),
                "bias_strength": float(first.get("bias_strength", bias_strength or 0.0)),
                "danger_height": int(first.get("danger_height", danger_height or 0)),
                "compute_scores": bool(
                    first.get(
                        "compute_scores",
                        True if compute_scores is None else bool(compute_scores),
                    )
                ),
            }
        return {
            "top_k": int(top_k or 0),
            "bias_strength": float(bias_strength or 0.0),
            "danger_height": int(danger_height or 0),
            "compute_scores": True if compute_scores is None else bool(compute_scores),
        }

    def set_reward_blend_step(self, transition_step: int) -> dict[str, Any]:
        payload = {"transitionStep": max(0, int(transition_step))}
        calls = [
            (shard_idx, lambda c=client, p=payload: c.set_reward_blend_step(p))
            for shard_idx, client in enumerate(self._active_clients())
        ]
        results = self._run_shard_calls(calls)
        transition = payload["transitionStep"]
        for result in results.values():
            value = result.get("transition_step")
            if isinstance(value, int):
                transition = value
                break
        return {"transition_step": transition}

    def step_many(self, env_ids: list[int], actions: list[int]) -> dict[str, Any]:
        if len(env_ids) != len(actions):
            raise ValueError("env_ids and actions must have matching lengths")
        if not env_ids:
            return {
                "obs": [],
                "action_masks": [],
                "action_biases": [],
                "action_scores": [],
                "rewards": [],
                "dones": [],
                "infos": [],
                "profile": {
                    "batch_total_s": 0.0,
                    "env_count": 0,
                    "step_env_total_s": 0.0,
                    "step_choices_current_s": 0.0,
                    "step_runner_s": 0.0,
                    "step_reward_s": 0.0,
                    "step_obs_s": 0.0,
                    "step_choices_next_s": 0.0,
                },
            }
        partitions = self._partition_env_ids(env_ids, actions=actions)
        calls: list[tuple[int, Any]] = []
        for shard_idx, part in partitions.items():
            payload = {"envIds": list(part["env_ids"]), "actions": list(part["actions"])}
            client = self._active_clients()[shard_idx]
            calls.append((shard_idx, lambda c=client, p=payload: c.step_many(p)))
        wall_start = time.perf_counter()
        results = self._run_shard_calls(calls)
        wall_s = time.perf_counter() - wall_start
        return self._merge_step_like_results(
            env_ids,
            partitions,
            results,
            profile_keys=(
                "step_env_total_s",
                "step_choices_current_s",
                "step_runner_s",
                "step_reward_s",
                "step_obs_s",
                "step_choices_next_s",
            ),
            profile_wall_s=wall_s,
        )

    def evaluate_hold_candidates_many(self, env_ids: list[int]) -> dict[str, Any]:
        partitions = self._partition_env_ids(env_ids)
        calls: list[tuple[int, Any]] = []
        for shard_idx, part in partitions.items():
            payload = {"envIds": list(part["env_ids"])}
            client = self._active_clients()[shard_idx]
            calls.append(
                (
                    shard_idx,
                    lambda c=client, p=payload: c.evaluate_hold_candidates_many(p),
                )
            )
        results = self._run_shard_calls(calls)
        candidates: list[Any] = [None] * len(env_ids)
        for shard_idx, part in partitions.items():
            shard_candidates = list(results[shard_idx].get("candidates", []))
            for local_pos, global_pos in enumerate(part["positions"]):
                candidates[global_pos] = shard_candidates[local_pos]
        return {"candidates": candidates}

    def probe_actions_many(self, env_ids: list[int]) -> dict[str, Any]:
        partitions = self._partition_env_ids(env_ids)
        calls: list[tuple[int, Any]] = []
        for shard_idx, part in partitions.items():
            payload = {"envIds": list(part["env_ids"])}
            client = self._active_clients()[shard_idx]
            calls.append((shard_idx, lambda c=client, p=payload: c.probe_actions_many(p)))
        results = self._run_shard_calls(calls)
        obs: list[Any] = [None] * len(env_ids)
        action_masks: list[Any] = [None] * len(env_ids)
        action_biases: list[Any] = [None] * len(env_ids)
        action_scores: list[Any] = [None] * len(env_ids)
        probes: list[Any] = [None] * len(env_ids)
        for shard_idx, part in partitions.items():
            result = results[shard_idx]
            shard_obs = list(result.get("obs", []))
            shard_masks = list(result.get("action_masks", []))
            shard_biases = list(result.get("action_biases", []))
            shard_scores = list(result.get("action_scores", []))
            shard_probes = list(result.get("probes", []))
            for local_pos, global_pos in enumerate(part["positions"]):
                obs[global_pos] = shard_obs[local_pos]
                action_masks[global_pos] = shard_masks[local_pos]
                action_biases[global_pos] = shard_biases[local_pos]
                action_scores[global_pos] = shard_scores[local_pos]
                probes[global_pos] = shard_probes[local_pos]
        return {
            "obs": obs,
            "action_masks": action_masks,
            "action_biases": action_biases,
            "action_scores": action_scores,
            "probes": probes,
        }

    def pop_trajectory(self) -> dict[str, Any]:
        clients = self._active_clients()
        if not clients:
            return {"trajectory": None}
        shard_count = len(clients)
        start_idx = self._next_pop_shard % shard_count
        for offset in range(shard_count):
            shard_idx = (start_idx + offset) % shard_count
            result = clients[shard_idx].pop_trajectory()
            trajectory = result.get("trajectory")
            self._next_pop_shard = (shard_idx + 1) % shard_count
            if isinstance(trajectory, dict):
                return result
        return {"trajectory": None}
