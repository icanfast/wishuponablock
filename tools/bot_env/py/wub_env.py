#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
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


class WubEnvBridge:
    def __init__(
        self,
        server_cmd: list[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        self.cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[3]
        self.server_cmd = (
            server_cmd
            if server_cmd is not None
            else _resolve_default_server_cmd()
        )
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

    def __enter__(self) -> "WubEnvBridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
        payload: dict[str, Any] = {
            "modeId": mode_id,
            "numEnvs": num_envs,
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
        if seed is not None:
            payload["seed"] = int(seed)
        return self._request("init", payload)

    def reset_many(
        self,
        env_ids: list[int],
        seeds: list[int] | None = None,
        initial_boards: list[list[list[int]] | None] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"envIds": [int(v) for v in env_ids]}
        if seeds is not None:
            payload["seeds"] = [int(v) for v in seeds]
        if initial_boards is not None:
            payload["initialBoards"] = initial_boards
        return self._request("reset_many", payload)

    def set_piece_source(self, piece_source_profile: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pieceSourceProfile": str(piece_source_profile).strip().lower(),
        }
        return self._request("set_piece_source", payload)

    def set_piece_sources(
        self,
        env_ids: list[int],
        piece_source_profiles: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "envIds": [int(v) for v in env_ids],
            "pieceSourceProfiles": [
                str(v).strip().lower() for v in piece_source_profiles
            ],
        }
        return self._request("set_piece_sources", payload)

    def set_curriculum(
        self,
        top_k: int | None = None,
        bias_strength: float | None = None,
        danger_height: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if top_k is not None:
            payload["topK"] = int(top_k)
        if bias_strength is not None:
            payload["biasStrength"] = float(bias_strength)
        if danger_height is not None:
            payload["dangerHeight"] = int(danger_height)
        return self._request("set_curriculum", payload)

    def set_reward_blend_step(self, transition_step: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transitionStep": max(0, int(transition_step)),
        }
        return self._request("set_reward_blend_step", payload)

    def step_many(self, env_ids: list[int], actions: list[int]) -> dict[str, Any]:
        payload = {
            "envIds": [int(v) for v in env_ids],
            "actions": [int(v) for v in actions],
        }
        return self._request("step_many", payload)

    def evaluate_hold_candidates_many(self, env_ids: list[int]) -> dict[str, Any]:
        payload = {
            "envIds": [int(v) for v in env_ids],
        }
        return self._request("evaluate_hold_candidates_many", payload)

    def probe_actions_many(self, env_ids: list[int]) -> dict[str, Any]:
        payload = {
            "envIds": [int(v) for v in env_ids],
        }
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
            raise RuntimeError(f"Mismatched response id. expected={request_id} got={resp.get('id')}")
        if not bool(resp.get("ok", False)):
            raise RuntimeError(str(resp.get("error", "Unknown bridge error.")))
        return dict(resp.get("result", {}))
