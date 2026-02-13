CREATE TABLE IF NOT EXISTS snapshot_build_counts (
  build_key TEXT PRIMARY KEY,
  snapshot_count INTEGER NOT NULL DEFAULT 0
);

INSERT INTO snapshot_build_counts (build_key, snapshot_count)
SELECT
  COALESCE(build_version, 'unknown') AS build_key,
  COUNT(*) AS snapshot_count
FROM snapshots_idx
GROUP BY COALESCE(build_version, 'unknown')
ON CONFLICT(build_key) DO UPDATE
SET snapshot_count = excluded.snapshot_count;
