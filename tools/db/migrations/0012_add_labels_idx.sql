CREATE TABLE IF NOT EXISTS labels_idx (
  label_id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  snapshot_id INTEGER,
  session_id TEXT,
  sample_index INTEGER,
  batch_id INTEGER,
  source_build_version TEXT,
  source_mode_id TEXT,
  source_trigger TEXT,
  source_lines_left INTEGER,
  source_level INTEGER,
  source_score INTEGER,
  source_device_id TEXT,
  source_user_id TEXT,
  intent_playstyle TEXT,
  intent_mode_filter TEXT,
  intent_trigger_filter TEXT,
  intent_build_filter TEXT,
  label_count INTEGER NOT NULL DEFAULT 0,
  labels_csv TEXT,
  FOREIGN KEY(label_id) REFERENCES labels(id)
);

CREATE INDEX IF NOT EXISTS labels_idx_snapshot_idx
  ON labels_idx(snapshot_id);

CREATE INDEX IF NOT EXISTS labels_idx_session_sample_idx
  ON labels_idx(session_id, sample_index);

CREATE INDEX IF NOT EXISTS labels_idx_intent_filter_idx
  ON labels_idx(intent_playstyle, intent_mode_filter, intent_trigger_filter, intent_build_filter, created_at);

CREATE INDEX IF NOT EXISTS labels_idx_source_filter_idx
  ON labels_idx(source_build_version, source_mode_id, source_trigger, created_at);

INSERT OR REPLACE INTO labels_idx (
  label_id,
  created_at,
  snapshot_id,
  session_id,
  sample_index,
  batch_id,
  source_build_version,
  source_mode_id,
  source_trigger,
  source_lines_left,
  source_level,
  source_score,
  source_device_id,
  source_user_id,
  intent_playstyle,
  intent_mode_filter,
  intent_trigger_filter,
  intent_build_filter,
  label_count,
  labels_csv
)
SELECT
  l.id AS label_id,
  l.created_at,
  COALESCE(
    l.snapshot_id,
    CAST(json_extract(l.data, '$.source.snapshotId') AS INTEGER),
    CAST(json_extract(l.data, '$.source.snapshot_id') AS INTEGER)
  ) AS snapshot_id,
  COALESCE(
    l.session_id,
    json_extract(l.data, '$.source.sessionId')
  ) AS session_id,
  COALESCE(
    l.sample_index,
    CAST(json_extract(l.data, '$.source.sampleIndex') AS INTEGER)
  ) AS sample_index,
  COALESCE(
    l.batch_id,
    CAST(json_extract(l.data, '$.source.batchId') AS INTEGER),
    CAST(json_extract(l.data, '$.source.batch_id') AS INTEGER)
  ) AS batch_id,
  COALESCE(
    sidx.build_version,
    json_extract(l.data, '$.snapshot_meta.session.buildVersion')
  ) AS source_build_version,
  COALESCE(
    sidx.mode_id,
    json_extract(l.data, '$.snapshot_meta.session.mode.id')
  ) AS source_mode_id,
  COALESCE(
    sidx.trigger,
    json_extract(l.data, '$.snapshot_meta.sample.trigger')
  ) AS source_trigger,
  COALESCE(
    sidx.lines_left,
    CAST(json_extract(l.data, '$.snapshot_meta.sample.linesLeft') AS INTEGER)
  ) AS source_lines_left,
  COALESCE(
    sidx.level,
    CAST(json_extract(l.data, '$.snapshot_meta.sample.level') AS INTEGER)
  ) AS source_level,
  COALESCE(
    sidx.score,
    CAST(json_extract(l.data, '$.snapshot_meta.sample.score') AS INTEGER)
  ) AS source_score,
  COALESCE(
    sidx.device_id,
    json_extract(l.data, '$.snapshot_meta.session.device_id')
  ) AS source_device_id,
  COALESCE(
    sidx.user_id,
    json_extract(l.data, '$.snapshot_meta.session.user_id')
  ) AS source_user_id,
  json_extract(l.data, '$.label_context.playstyle') AS intent_playstyle,
  json_extract(l.data, '$.label_context.mode_filter') AS intent_mode_filter,
  json_extract(l.data, '$.label_context.trigger_filter') AS intent_trigger_filter,
  json_extract(l.data, '$.label_context.build_filter') AS intent_build_filter,
  COALESCE(json_array_length(json_extract(l.data, '$.labels')), 0) AS label_count,
  (
    SELECT group_concat(j.value, ',')
    FROM json_each(l.data, '$.labels') AS j
  ) AS labels_csv
FROM labels l
LEFT JOIN snapshots_idx sidx
  ON sidx.snapshot_id = COALESCE(
    l.snapshot_id,
    CAST(json_extract(l.data, '$.source.snapshotId') AS INTEGER),
    CAST(json_extract(l.data, '$.source.snapshot_id') AS INTEGER)
  );
