begin;

-- The published line and score margin already carry the market blend; the
-- published total did not. Store the forecast-time market total and the
-- total and scores blended toward it beside the model's own total, which
-- pick pricing keeps blending toward the decision-time market. Rows published
-- before this migration keep NULL.
alter table nfl.game_projections add column market_total double precision;
alter table nfl.game_projections add column market_informed_total double precision;
alter table nfl.game_projections add column market_informed_home_points double precision;
alter table nfl.game_projections add column market_informed_away_points double precision;

commit;
