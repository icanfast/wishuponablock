ALTER TABLE snapshots_idx
  ADD COLUMN batch_id INTEGER;

ALTER TABLE snapshots_idx
  ADD COLUMN label_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE snapshots_idx
  ADD COLUMN last_labeled_at TEXT;

CREATE INDEX IF NOT EXISTS snapshots_idx_label_priority_idx
  ON snapshots_idx(mode_id, trigger, build_version, label_count, snapshot_id);

CREATE INDEX IF NOT EXISTS snapshots_idx_session_batch_idx
  ON snapshots_idx(session_id, batch_id, snapshot_id);

ALTER TABLE labels
  ADD COLUMN snapshot_id INTEGER;

ALTER TABLE labels
  ADD COLUMN session_id TEXT;

ALTER TABLE labels
  ADD COLUMN sample_index INTEGER;

ALTER TABLE labels
  ADD COLUMN batch_id INTEGER;

CREATE INDEX IF NOT EXISTS labels_snapshot_idx
  ON labels(snapshot_id);

CREATE INDEX IF NOT EXISTS labels_session_sample_idx
  ON labels(session_id, sample_index);
