CREATE TABLE IF NOT EXISTS label_progress_build (
  build_key TEXT PRIMARY KEY,
  labeled_snapshot_count INTEGER NOT NULL DEFAULT 0
);

INSERT INTO label_progress_build (build_key, labeled_snapshot_count)
SELECT
  COALESCE(build_version, 'unknown') AS build_key,
  COUNT(*) AS labeled_snapshot_count
FROM snapshots_idx
WHERE COALESCE(label_count, 0) > 0
GROUP BY COALESCE(build_version, 'unknown')
ON CONFLICT(build_key) DO UPDATE
SET labeled_snapshot_count = excluded.labeled_snapshot_count;
