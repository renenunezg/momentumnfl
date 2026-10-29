begin;

-- sql/025 copied the grading rules into the view as text, a second copy of
-- what backend/performance.py does. The view now carries the probability
-- formula each row was graded with (graded_games.probability_method) and
-- points at the code for the scoring rules instead of restating them.
create or replace view nfl.graded_game_ledger with (security_invoker = true) as
select g.game_id, g.season, g.week, g.start_date as kickoff,
  f.snapshot_id, g.forecast_recorded_at, g.forecast_as_of, g.model_version,
  g.pure_home_margin, g.market_informed_home_margin as published_home_margin,
  g.model_total, g.market_informed_total as published_total,
  g.margin_sd, g.degrees_of_freedom, g.home_win_probability,
  g.forecast_market_home_spread, f.market_total as forecast_market_total,
  g.closing_spread, g.closing_total, g.closing_source,
  g.home_points, g.away_points, g.score_source, g.source_fetched_at,
  r.run_id as input_run_id, r.recorded_at as input_run_recorded_at,
  'momentumnfl backend/performance.py compute_performance_metrics'::text
    as grading_rule,
  g.graded_at,
  g.probability_method
from nfl.graded_games g
join nfl.forecast_snapshots f
  on f.game_id = g.game_id and f.recorded_at = g.forecast_recorded_at
left join lateral (
  select run_id, recorded_at from nfl.forecast_input_runs i
  where i.season = g.season and i.forecast_as_of = g.forecast_as_of
  order by i.recorded_at, i.run_id limit 1
) r on true;

notify pgrst, 'reload schema';

commit;
