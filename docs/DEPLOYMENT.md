**Internal Notes**
This file is for deployment and ops details that don’t belong in `README.md`.

**Cloudflare Deploy**

1. Cloudflare runs `npm run build` and `npx wrangler deploy` automatically on push (per your project settings).
2. The Worker serves static assets from `dist` and exposes the API at `/api`.

**D1 Migrations**

1. Cloudflare does not apply D1 migrations automatically.
2. Run migrations manually from your machine:
   - `npx wrangler login`
   - `npx wrangler d1 migrations apply DB --remote`
3. SQL lives in `tools/db/migrations/`. The initial schema is `tools/db/migrations/0001_init.sql`.

**Wrangler Bindings**

1. `wrangler.jsonc` binds:
   - `DB` → D1 database
   - `ASSETS` → static asset directory (`dist`)
2. If you rename the D1 binding in `wrangler.jsonc`, update `src/worker/index.ts` accordingly.

**Upload Modes**

1. Local mode: snapshots/labels saved locally via folder picker (defaults under `data/snapshots/`).
2. Remote mode: snapshots/labels POST to the Worker.
3. Configure via env variables:
   - `VITE_UPLOAD_MODE=remote` (or `auto`)
   - `VITE_UPLOAD_URL=/api` (or full URL)

**ML Model Asset**

1. The client loads `ML_MODEL_URL`, default `/model.json`.
2. Place the model file at `public/model.json` so it is available after build.
