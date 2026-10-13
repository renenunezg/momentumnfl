begin;

-- Live-season grading record: the last forecast published before kickoff,
-- the nflverse closing line, and the final score, one row per completed
-- game. Rebuilt by the backend from immutable snapshots on every grade run.
create table nfl.graded_games (
  game_id text primary key,
  season integer not null,
  week integer not null,
  season_type text not null,
  forecast_week integer not null,
  start_date timestamptz not null,
  neutral_site boolean not null,
  div_game boolean,
  home_team_abbr text not null,
  away_team_abbr text not null,
  home_team text not null,
  away_team text not null,
  model_version text not null,
  forecast_as_of timestamptz not null,
  forecast_recorded_at timestamptz not null,
  home_qb_adjustment double precision,
  away_qb_adjustment double precision,
  rest_adjustment double precision,
  pure_home_margin double precision not null,
  market_informed_home_margin double precision not null,
  market_weight double precision,
  forecast_market_home_spread double precision,
  model_total double precision,
  market_informed_total double precision,
  margin_sd double precision not null,
  total_sd double precision,
  distribution text,
  degrees_of_freedom double precision not null,
  home_win_probability double precision not null,
  probability_method text not null,
  closing_spread double precision,
  closing_total double precision,
  closing_source text,
  home_points integer not null check (home_points >= 0),
  away_points integer not null check (away_points >= 0),
  actual_margin integer not null,
  actual_total integer not null,
  score_source text not null,
  source_fetched_at timestamptz not null,
  graded_at timestamptz not null,
  check (forecast_as_of < start_date and forecast_recorded_at < start_date)
);
create index graded_games_season_start on nfl.graded_games (season, start_date desc);

-- Precomputed aggregates per prediction source and segment; the site reads
-- these rows as published and never aggregates games itself.
create table nfl.performance_metrics (
  season integer not null,
  prediction_source text not null
    check (prediction_source in ('pure_model', 'market_informed', 'closing_market')),
  segment_kind text not null,
  segment text not null,
  segment_order integer not null,
  games integer not null,
  thin_sample boolean not null,
  margin_mae double precision,
  margin_rmse double precision,
  margin_bias double precision,
  total_games integer,
  total_mae double precision,
  total_rmse double precision,
  total_bias double precision,
  coverage_50 double precision,
  coverage_80 double precision,
  coverage_90 double precision,
  games_with_market integer,
  market_mae double precision,
  model_minus_market_mae double precision,
  closer_than_market_share double precision,
  probability_games integer,
  brier_score double precision,
  log_loss double precision,
  computed_at timestamptz not null,
  primary key (season, prediction_source, segment_kind, segment)
);

alter table nfl.graded_games enable row level security;
alter table nfl.performance_metrics enable row level security;
create policy "anon read" on nfl.graded_games
  for select to anon, authenticated using (true);
create policy "anon read" on nfl.performance_metrics
  for select to anon, authenticated using (true);
grant select on nfl.graded_games, nfl.performance_metrics to anon, authenticated;

create trigger site_revalidate
  after insert or delete or update or truncate on nfl.graded_games
  for each statement execute function public.site_revalidate();
create trigger site_revalidate
  after insert or delete or update or truncate on nfl.performance_metrics
  for each statement execute function public.site_revalidate();

notify pgrst, 'reload schema';

commit;
