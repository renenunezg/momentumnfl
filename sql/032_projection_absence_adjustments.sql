begin;

-- Points each side's projected margin gives up to ruled-out regulars other
-- than the quarterback (backend/model/absence_layer.py), weighted, from that
-- team's perspective: home_margin moves by home minus away and the total does
-- not. Rows published before this migration keep NULL.
alter table nfl.game_projections add column home_absence_adjustment double precision;
alter table nfl.game_projections add column away_absence_adjustment double precision;

-- archive_pregame_projection copies new.* positionally, so the snapshot table
-- must gain the same columns in the same order in the same transaction.
alter table nfl.forecast_snapshots add column home_absence_adjustment double precision;
alter table nfl.forecast_snapshots add column away_absence_adjustment double precision;

commit;
