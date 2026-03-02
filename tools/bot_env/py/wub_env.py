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
        mode_id: str = "charcuterie",
        num_envs: int = 1,
        model_path: str = "public/models/model_v4.json",
        observation_space: str = "raw_v1",
        piece_source_profile: str = "bag7",
        queue_policy_id: str = "next_piece_v1",
        max_pieces_per_episode: int = 512,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modeId": mode_id,
            "numEnvs": num_envs,
            "modelPath": model_path,
            "observationSpace": observation_space,
            "pieceSourceProfile": piece_source_profile,
            "queuePolicyId": queue_policy_id,
            "maxPiecesPerEpisode": max_pieces_per_episode,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        return self._request("init", payload)

    def reset_many(self, env_ids: list[int], seeds: list[int] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"envIds": [int(v) for v in env_ids]}
        if seeds is not None:
            payload["seeds"] = [int(v) for v in seeds]
        return self._request("reset_many", payload)

    def step_many(self, env_ids: list[int], actions: list[int]) -> dict[str, Any]:
        payload = {
            "envIds": [int(v) for v in env_ids],
            "actions": [int(v) for v in actions],
        }
        return self._request("step_many", payload)

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
