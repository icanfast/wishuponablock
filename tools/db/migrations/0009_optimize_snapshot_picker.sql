UPDATE snapshots_idx
SET label_count = 0
WHERE label_count IS NULL;

CREATE INDEX IF NOT EXISTS snapshots_idx_picker_mode_build_label_idx
  ON snapshots_idx(mode_id, build_version, label_count, snapshot_id);

CREATE INDEX IF NOT EXISTS snapshots_idx_picker_mode_trigger_label_idx
  ON snapshots_idx(mode_id, trigger, label_count, snapshot_id);

CREATE INDEX IF NOT EXISTS snapshots_idx_picker_build_label_idx
  ON snapshots_idx(build_version, label_count, snapshot_id);
