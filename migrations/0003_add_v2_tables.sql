CREATE TABLE IF NOT EXISTS snapshot_sessions_v2 (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  protocol_version INTEGER NOT NULL,
  rows INTEGER NOT NULL,
  cols INTEGER NOT NULL,
  settings TEXT NOT NULL,
  mode_id TEXT,
  mode_options TEXT,
  comment TEXT
);

CREATE TABLE IF NOT EXISTS snapshot_samples_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  sample_index INTEGER NOT NULL,
  time_ms INTEGER NOT NULL,
  board TEXT NOT NULL,
  hold INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, sample_index),
  FOREIGN KEY(session_id) REFERENCES snapshot_sessions_v2(id)
);

CREATE TABLE IF NOT EXISTS label_records_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  session_id TEXT,
  file_name TEXT,
  sample_index INTEGER,
  shown_count INTEGER,
  board TEXT NOT NULL,
  hold INTEGER,
  labels TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS label_records_v2_session_idx
  ON label_records_v2(session_id, sample_index);
