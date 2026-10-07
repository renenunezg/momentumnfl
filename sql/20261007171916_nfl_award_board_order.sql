-- Matches the site's equality filters and stable 250-row pagination order.
-- Run outside a transaction: CONCURRENTLY keeps snapshot publication available.
CREATE INDEX CONCURRENTLY award_boards_snapshot_rank_idx
  ON nfl.award_boards (season, week, award, predicted_rank, candidate_id);
