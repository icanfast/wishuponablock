ALTER TABLE snapshots_idx
  ADD COLUMN sample_next_count INTEGER;

ALTER TABLE snapshots_idx
  ADD COLUMN sample_odds_count INTEGER;

ALTER TABLE snapshots_idx
  ADD COLUMN sample_odds_top_piece INTEGER;

ALTER TABLE snapshots_idx
  ADD COLUMN sample_odds_top_probability REAL;
