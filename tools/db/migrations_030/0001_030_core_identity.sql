PRAGMA foreign_keys = ON;

-- 0.3.0 core identity/auth schema (authoritative in-repo).
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  username TEXT NOT NULL,
  email_norm TEXT UNIQUE,
  email_verified_at_ms INTEGER,
  password_hash TEXT,
  google_sub TEXT UNIQUE,
  discord_sub TEXT UNIQUE,
  ui_settings TEXT,
  playstyle_vector TEXT,
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  CHECK (email_norm IS NULL OR length(email_norm) <= 320),
  CHECK (ui_settings IS NULL OR json_valid(ui_settings)),
  CHECK (playstyle_vector IS NULL OR json_valid(playstyle_vector))
);

CREATE INDEX IF NOT EXISTS idx_users_created_at_ms
  ON users (created_at_ms);

CREATE TABLE IF NOT EXISTS auth_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  expires_at_ms INTEGER NOT NULL,
  last_seen_at_ms INTEGER,
  revoked_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_exp
  ON auth_sessions (user_id, expires_at_ms);

CREATE TABLE IF NOT EXISTS auth_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('email_verify', 'password_reset')),
  token_hash TEXT NOT NULL UNIQUE,
  email_norm TEXT,
  expires_at_ms INTEGER NOT NULL,
  consumed_at_ms INTEGER,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_type_exp
  ON auth_tokens (user_id, type, expires_at_ms);

-- Trajectory recording index (R2 payload pointer + compact metadata).
CREATE TABLE IF NOT EXISTS recordings_index (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  game_mode TEXT NOT NULL,
  build_version TEXT NOT NULL,
  r2_key TEXT NOT NULL UNIQUE,
  started_at_ms INTEGER NOT NULL,
  ended_at_ms INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  snapshots_total INTEGER NOT NULL,
  model_arch TEXT,
  reward_profile_id TEXT,
  queue_policy_id TEXT,
  model_source TEXT,
  reward_policy_id TEXT,
  pipeline_id TEXT,
  meta_json TEXT,
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  CHECK (meta_json IS NULL OR json_valid(meta_json)),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_recordings_started_id
  ON recordings_index (started_at_ms DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_recordings_mode_started
  ON recordings_index (game_mode, started_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_recordings_user_started
  ON recordings_index (user_id, started_at_ms DESC);

-- Registry of immutable global baselines (publishable presets).
CREATE TABLE IF NOT EXISTS global_models (
  id TEXT PRIMARY KEY,
  mode_id TEXT NOT NULL,
  model_arch TEXT NOT NULL DEFAULT 'full',
  reward_profile_id TEXT NOT NULL DEFAULT 'default',
  queue_policy_id TEXT NOT NULL DEFAULT 'next_piece_v1',
  pipeline_id TEXT NOT NULL,
  label TEXT,
  r2_key TEXT NOT NULL UNIQUE,
  sha256 TEXT,
  size_bytes INTEGER,
  metrics_json TEXT,
  is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  retired_at_ms INTEGER,
  CHECK (metrics_json IS NULL OR json_valid(metrics_json))
);

CREATE INDEX IF NOT EXISTS idx_global_models_axes_default
  ON global_models (
    mode_id,
    model_arch,
    reward_profile_id,
    queue_policy_id,
    is_default,
    retired_at_ms,
    created_at_ms DESC
  );

-- Immutable personalized model lineage.
CREATE TABLE IF NOT EXISTS personal_model_versions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  game_mode TEXT NOT NULL,
  model_arch TEXT NOT NULL DEFAULT 'full',
  reward_profile_id TEXT NOT NULL DEFAULT 'default',
  queue_policy_id TEXT NOT NULL DEFAULT 'next_piece_v1',
  version_seq INTEGER NOT NULL,
  parent_version_id TEXT,
  base_global_model_id TEXT,
  source TEXT NOT NULL DEFAULT 'upload',
  pipeline_id TEXT,
  r2_key TEXT NOT NULL UNIQUE,
  sha256 TEXT,
  size_bytes INTEGER,
  metrics_json TEXT,
  created_at_ms INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (parent_version_id) REFERENCES personal_model_versions(id) ON DELETE SET NULL,
  FOREIGN KEY (base_global_model_id) REFERENCES global_models(id) ON DELETE SET NULL,
  UNIQUE (
    user_id,
    game_mode,
    model_arch,
    reward_profile_id,
    queue_policy_id,
    version_seq
  ),
  CHECK (source IN ('upload', 'train', 'warm_start', 'reset', 'auto_save')),
  CHECK (metrics_json IS NULL OR json_valid(metrics_json))
);

CREATE INDEX IF NOT EXISTS idx_personal_model_versions_user_created
  ON personal_model_versions (user_id, created_at_ms DESC);

CREATE INDEX IF NOT EXISTS idx_personal_model_versions_axes
  ON personal_model_versions (
    user_id,
    game_mode,
    model_arch,
    reward_profile_id,
    queue_policy_id,
    version_seq DESC
  );

-- Active personalized model pointer per user+axes.
CREATE TABLE IF NOT EXISTS personal_model_slots (
  user_id TEXT NOT NULL,
  game_mode TEXT NOT NULL,
  model_arch TEXT NOT NULL DEFAULT 'full',
  reward_profile_id TEXT NOT NULL DEFAULT 'default',
  queue_policy_id TEXT NOT NULL DEFAULT 'next_piece_v1',
  active_version_id TEXT,
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  PRIMARY KEY (user_id, game_mode, model_arch, reward_profile_id, queue_policy_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (active_version_id) REFERENCES personal_model_versions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_personal_model_slots_user_updated
  ON personal_model_slots (user_id, updated_at_ms DESC);
