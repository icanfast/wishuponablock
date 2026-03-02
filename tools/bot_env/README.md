# RL Bot Local Bridge (Phase 1)

This folder provides a minimal local bridge so Python (PyTorch) can drive the TS game logic as an RL environment.

## Components

- `tools/bot_env/ts/envServer.ts`
  - JSONL stdio server.
  - Commands: `init`, `reset_many`, `step_many`, `close`.
- `tools/bot_env/ts/envCore.ts`
  - Vectorized env pool.
  - One RL step = one piece-placement decision.
  - Action space: placement slots (executor-backed).
- `tools/bot_env/py/wub_env.py`
  - Python client wrapper around the stdio protocol.
- `tools/bot_env/py/smoke_test.py`
  - End-to-end smoke run (`reset_many` + `step_many` loop).
- `tools/bot_env/py/train_ppo.py`
  - Full PPO trainer in PyTorch (GAE, minibatches, clipping, KL early-stop).
  - Saves Torch checkpoints and app-compatible bot policy artifacts.

## Prerequisites

This setup expects a TS runtime for the server command:

- Default command in Python wrapper: `npx --yes tsx tools/bot_env/ts/envServer.ts`

Python dependencies for PPO:

```bash
python3 -m pip install -r tools/bot_env/py/requirements.txt
```

If `tsx` is not available locally, install it or override `server_cmd` in `WubEnvBridge`.

## Smoke test

From repo root:

```bash
python3 tools/bot_env/py/smoke_test.py
```

You should see:

- env ids returned from `init`
- non-empty observation/action dimensions
- step rewards/dones over multiple steps

## PPO training (local)

Minimal run:

```bash
python3 tools/bot_env/py/train_ppo.py \
  --mode-id charcuterie \
  --model-path public/models/model_v4.json \
  --num-envs 16 \
  --num-steps 256 \
  --total-timesteps 2000000 \
  --piece-source-profile bag7
```

Outputs:

- `tools/bot_env/output/<run_name>/checkpoints/*.pt` (PyTorch checkpoints)
- `tools/bot_env/output/<run_name>/artifacts/*.json` (WUB bot policy artifacts)
- `tools/bot_env/output/<run_name>/bot_policy_final.json` (final artifact)

Resume from checkpoint:

```bash
python3 tools/bot_env/py/train_ppo.py --resume-checkpoint <path/to/ppo_update_000100.pt>
```

Initialize from existing bot artifact:

```bash
python3 tools/bot_env/py/train_ppo.py --init-artifact <path/to/bot_policy.json>
```
