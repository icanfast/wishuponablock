-- Conservative first pass to reduce write amplification with minimal read risk.
-- Keep snapshots_idx_filter_idx for now because some fallback/picker paths do
-- not constrain label_count and can still benefit from (mode,trigger,build,snapshot_id).
-- snapshots_idx_build_idx is likely redundant with build-prefixed picker indexes.

DROP INDEX IF EXISTS snapshots_idx_build_idx;
