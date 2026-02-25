# Phase 0 Bootstrap (0.3.0 Dev)

This branch keeps production (`0.2.4`) untouched and deploys `0.3.0` to a separate dev worker.

## 1) Deploy dev worker (no new DB migrations yet)

```bash
npx wrangler login
npm run deploy:dev
```

This deploys `env.dev` (`wishuponablock-030-dev`) instead of the default worker.
The deploy uses `vite --mode dev030` with:

- `VITE_UPLOAD_MODE=local`
- `VITE_SHOW_PERF_OVERLAY=true`
- `VITE_SHOW_EXPERIMENTAL_GAMEPLAY_CONTROLS=false`
- `VITE_ENABLE_LEGACY_DATA_TOOLS=false`

So the online dev build does not depend on new D1/R2 yet.

## 2) Verify runtime flags endpoint

```bash
curl https://wishuponablock-030-dev.nikitin-maxim-94.workers.dev/api/runtime/flags
```

Expected response shape:

```json
{
  "channel": "dev",
  "flags": {
    "computeProfiler": false,
    "rlPipeline": false,
    "authV1": false
  }
}
```

## 3) (Optional) Attach a custom dev domain

Attach `dev.wishuponablock.com` to the dev worker in Cloudflare Dashboard.

## 4) Later (when Phase 2 starts)

At that point create clean-slate D1/R2 and swap `env.dev` bindings to the new resources.
