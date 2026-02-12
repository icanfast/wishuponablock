CREATE TABLE IF NOT EXISTS snapshots_idx (
  snapshot_id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  session_id TEXT,
  session_created_at TEXT,
  protocol_version INTEGER,
  rows INTEGER,
  cols INTEGER,
  mode_id TEXT,
  build_version TEXT,
  generator_type TEXT,
  generator_strategy TEXT,
  trigger TEXT,
  sample_index INTEGER,
  sample_time_ms INTEGER,
  sample_hold INTEGER,
  lines_left INTEGER,
  level INTEGER,
  score INTEGER,
  model_url TEXT,
  device_id TEXT,
  user_id TEXT,
  FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
);

CREATE INDEX IF NOT EXISTS snapshots_idx_filter_idx
  ON snapshots_idx(mode_id, trigger, build_version, snapshot_id);

CREATE INDEX IF NOT EXISTS snapshots_idx_build_idx
  ON snapshots_idx(build_version);

CREATE INDEX IF NOT EXISTS snapshots_idx_session_sample_idx
  ON snapshots_idx(session_id, sample_index);
