begin;

-- One row per graded game with everything needed to regrade it outside this
-- repository: the frozen snapshot that was graded and its server receipt
-- time, the market at forecast time, the nflverse close, the final score, the
-- archived input run, and the grading rule. Reads only public tables.
grant select (run_id, season, week, forecast_as_of, recorded_at)
  on nfl.forecast_input_runs to anon, authenticated;
create policy "anon read run ids" on nfl.forecast_input_runs
  for select to anon, authenticated using (true);

create view nfl.graded_game_ledger with (security_invoker = true) as
select g.game_id, g.season, g.week, g.start_date as kickoff,
  f.snapshot_id, g.forecast_recorded_at, g.forecast_as_of, g.model_version,
  g.pure_home_margin, g.market_informed_home_margin as published_home_margin,
  g.model_total, g.market_informed_total as published_total,
  g.margin_sd, g.degrees_of_freedom, g.home_win_probability,
  g.forecast_market_home_spread, f.market_total as forecast_market_total,
  g.closing_spread, g.closing_total, g.closing_source,
  g.home_points, g.away_points, g.score_source, g.source_fetched_at,
  r.run_id as input_run_id, r.recorded_at as input_run_recorded_at,
  'MAE on the home margin; Brier and log loss on P(home margin > 0) with a '
  'tie scored 0.5; closer means a strictly smaller absolute error than the '
  'nflverse closing spread' as grading_rule,
  g.graded_at
from nfl.graded_games g
join nfl.forecast_snapshots f
  on f.game_id = g.game_id and f.recorded_at = g.forecast_recorded_at
left join lateral (
  select run_id, recorded_at from nfl.forecast_input_runs i
  where i.season = g.season and i.forecast_as_of = g.forecast_as_of
  order by i.recorded_at, i.run_id limit 1
) r on true;

grant select on nfl.graded_game_ledger to anon, authenticated;

notify pgrst, 'reload schema';

commit;
