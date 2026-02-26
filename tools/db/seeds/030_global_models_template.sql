PRAGMA foreign_keys = ON;

-- Seed template for published global baselines used by:
--   GET  /api/models/global/list
--   POST /api/models/me/reset
--
-- How to use:
-- 1) Upload model bytes to the MODELS_BUCKET in R2 (one object per baseline).
-- 2) Replace the placeholder values below.
-- 3) Run this SQL against D1.
--
-- Notes:
-- - Keep one default per (mode_id, model_arch, reward_profile_id, queue_policy_id).
-- - Set retired_at_ms when deprecating a baseline instead of deleting rows.

-- Optional: clear default flag for a specific axis before setting a new default.
-- UPDATE global_models
-- SET is_default = 0, updated_at_ms = CAST(strftime('%s','now') AS INTEGER) * 1000
-- WHERE mode_id = 'practice'
--   AND model_arch = 'full'
--   AND reward_profile_id = 'default'
--   AND queue_policy_id = 'next_piece_v1'
--   AND retired_at_ms IS NULL;

INSERT INTO global_models (
  id,
  mode_id,
  model_arch,
  reward_profile_id,
  queue_policy_id,
  pipeline_id,
  label,
  r2_key,
  sha256,
  size_bytes,
  metrics_json,
  is_default,
  created_at_ms,
  updated_at_ms,
  retired_at_ms
) VALUES (
  'global_practice_full_default_next_piece_v1_v4',
  'practice',
  'full',
  'default',
  'next_piece_v1',
  'global_supervised_v4',
  'Global v4 (practice)',
  'global/practice/full/default/next_piece_v1/v4.bin',
  NULL,
  NULL,
  NULL,
  1,
  CAST(strftime('%s','now') AS INTEGER) * 1000,
  CAST(strftime('%s','now') AS INTEGER) * 1000,
  NULL
) ON CONFLICT(id) DO UPDATE SET
  mode_id = excluded.mode_id,
  model_arch = excluded.model_arch,
  reward_profile_id = excluded.reward_profile_id,
  queue_policy_id = excluded.queue_policy_id,
  pipeline_id = excluded.pipeline_id,
  label = excluded.label,
  r2_key = excluded.r2_key,
  sha256 = excluded.sha256,
  size_bytes = excluded.size_bytes,
  metrics_json = excluded.metrics_json,
  is_default = excluded.is_default,
  updated_at_ms = excluded.updated_at_ms,
  retired_at_ms = excluded.retired_at_ms;
