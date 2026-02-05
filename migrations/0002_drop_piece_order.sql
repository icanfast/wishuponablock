CREATE TABLE IF NOT EXISTS snapshot_sessions_new (
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

INSERT INTO snapshot_sessions_new (
  id,
  created_at,
  protocol_version,
  rows,
  cols,
  settings,
  mode_id,
  mode_options,
  comment
)
SELECT
  id,
  created_at,
  protocol_version,
  rows,
  cols,
  settings,
  mode_id,
  mode_options,
  comment
FROM snapshot_sessions;

DROP TABLE snapshot_sessions;
ALTER TABLE snapshot_sessions_new RENAME TO snapshot_sessions;

CREATE TABLE IF NOT EXISTS label_records_new (
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

INSERT INTO label_records_new (
  id,
  created_at,
  session_id,
  file_name,
  sample_index,
  shown_count,
  board,
  hold,
  labels
)
SELECT
  id,
  created_at,
  session_id,
  file_name,
  sample_index,
  shown_count,
  board,
  hold,
  labels
FROM label_records;

DROP TABLE label_records;
ALTER TABLE label_records_new RENAME TO label_records;

CREATE INDEX IF NOT EXISTS label_records_session_idx
  ON label_records(session_id, sample_index);
