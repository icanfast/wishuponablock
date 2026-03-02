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
  - Optional behavior cloning warm-start from human replay dataset.
  - Saves Torch checkpoints and app-compatible bot policy artifacts.
- `tools/bot_env/py/download_recordings.py`
  - Downloads recordings in bulk from admin API export manifest/object routes.
  - Saves local `.json` files for BC dataset building.
- `tools/bot_env/ts/buildBcDataset.ts`
  - Builds BC dataset from local trajectory JSON/JSON.GZ recordings.
  - Aligns replay action to previous state (`state_{t-1} -> action_t`).
  - Silently skips incompatible recording samples.

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

### 0) Download recordings in bulk from dev/prod (optional helper)

Using a current session cookie:

```bash
python3 tools/bot_env/py/download_recordings.py \
  --base-url https://dev.wishuponablock.com \
  --cookie "<wub_session_cookie_value>" \
  --mode charcuterie \
  --min-samples 2 \
  --out-dir tools/bot_env/recordings
```

Using email login:

```bash
python3 tools/bot_env/py/download_recordings.py \
  --base-url https://dev.wishuponablock.com \
  --email "<your_email>" \
  --password "<your_password>" \
  --mode charcuterie \
  --min-samples 2 \
  --out-dir tools/bot_env/recordings
```

If you hit TLS trust errors on local Python:

- Use your own CA bundle: `--ca-file /path/to/cacert.pem`
- Or bypass verification for local testing only: `--insecure`

### 1) Build behavior-cloning dataset (optional warm-start)

```bash
npx --yes tsx tools/bot_env/ts/buildBcDataset.ts \
  --input tools/bot_env/recordings \
  --output tools/bot_env/output/bc_dataset_charcuterie.json \
  --mode charcuterie \
  --model-path public/models/model_v4.json \
  --return-gamma 0.995
```

You can pass `--input` multiple times (file or directory). Incompatible samples are skipped silently and only reflected in summary counters.

### 2) Train PPO

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

With BC warm-start:

```bash
python3 tools/bot_env/py/train_ppo.py \
  --mode-id charcuterie \
  --model-path public/models/model_v4.json \
  --num-envs 16 \
  --num-steps 256 \
  --total-timesteps 2000000 \
  --piece-source-profile bag7 \
  --bc-dataset tools/bot_env/output/bc_dataset_charcuterie.json \
  --bc-epochs 5 \
  --bc-batch-size 4096 \
  --bc-learning-rate 1e-3 \
  --bc-value-weight 0.25
```

Outputs:

- `tools/bot_env/output/<run_name>/checkpoints/*.pt` (PyTorch checkpoints)
- `tools/bot_env/output/<run_name>/artifacts/*.json` (WUB bot policy artifacts)
- `tools/bot_env/output/<run_name>/bot_policy_final.json` (final artifact)
- `tools/bot_env/output/<run_name>/bot_policy_best.json` (best-by-ret100 artifact)
- `tools/bot_env/output/<run_name>/bc_stats.json` (if BC was enabled)

Resume from checkpoint:

```bash
python3 tools/bot_env/py/train_ppo.py --resume-checkpoint <path/to/ppo_update_000100.pt>
```

Initialize from existing bot artifact:

```bash
python3 tools/bot_env/py/train_ppo.py --init-artifact <path/to/bot_policy.json>
```
