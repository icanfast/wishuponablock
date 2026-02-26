PRAGMA foreign_keys = ON;

-- Durable bot policy registry (charcuterie-first, axis-aware).
CREATE TABLE IF NOT EXISTS bot_policies (
  id TEXT PRIMARY KEY,
  mode_id TEXT NOT NULL,
  arch_id TEXT NOT NULL DEFAULT 'full',
  queue_policy_id TEXT NOT NULL DEFAULT 'default',
  pipeline_id TEXT NOT NULL,
  piece_source_profile TEXT NOT NULL DEFAULT 'bag7',
  r2_key TEXT NOT NULL UNIQUE,
  version INTEGER NOT NULL,
  is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
  metrics_json TEXT,
  created_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  UNIQUE (mode_id, arch_id, queue_policy_id, version),
  CHECK (metrics_json IS NULL OR json_valid(metrics_json))
);

CREATE INDEX IF NOT EXISTS idx_bot_policies_axes_created
  ON bot_policies (mode_id, arch_id, queue_policy_id, created_at_ms DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_bot_policies_axes_pinned
  ON bot_policies (mode_id, arch_id, queue_policy_id, is_pinned, created_at_ms DESC, id DESC);

-- Active policy pointer per bot selector axis.
CREATE TABLE IF NOT EXISTS bot_policy_current (
  mode_id TEXT NOT NULL,
  arch_id TEXT NOT NULL DEFAULT 'full',
  queue_policy_id TEXT NOT NULL DEFAULT 'default',
  bot_policy_id TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000),
  PRIMARY KEY (mode_id, arch_id, queue_policy_id),
  FOREIGN KEY (bot_policy_id) REFERENCES bot_policies(id) ON DELETE CASCADE
);

