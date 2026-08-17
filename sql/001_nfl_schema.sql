-- nfl schema v1: the sole interface between momentumnfl and momentumweb.
-- Applied to the shared momentum Supabase project. RLS with anon read on
-- every table, mirroring the cfb schema's policy shape.

create schema if not exists nfl;

create table nfl.teams (
  -- identity only, per the cfb pattern: conference/division live on
  -- team_ratings, which every consumer already loads
  team_abbr text primary key,
  team text not null,
  color text,
  alternate_color text,
  logo_light text,
  logo_dark text
);

create table nfl.team_ratings (
  season int not null,
  week int not null,
  as_of timestamptz not null,
  model_version text not null,
  team_abbr text not null,
  team text not null,
  conference text,
  division text,
  offense_points float8,
  defense_points float8,
  power_rating float8 not null,
  scoring_environment float8,
  expected_drives float8,
  power_rating_sd float8,
  missing_input_count int,
  primary key (season, week, team_abbr)
);

create table nfl.team_unit_ratings (
  season int not null,
  week int not null,
  as_of timestamptz not null,
  model_version text not null,
  team_abbr text not null,
  team text not null,
  rush_offense float8,
  pass_offense float8,
  rush_defense float8,
  pass_defense float8,
  pass_block float8,
  run_block float8,
  special_teams float8,
  primary key (season, week, team_abbr)
);

create table nfl.game_projections (
  game_id text primary key,
  season int not null,
  week int not null,
  as_of timestamptz not null,
  model_version text not null,
  start_date timestamptz,
  home_team_abbr text,
  home_team text not null,
  away_team_abbr text,
  away_team text not null,
  neutral_site bool,
  div_game bool,
  home_field_points float8,
  expected_home_points float8,
  expected_away_points float8,
  home_qb_adjustment float8,
  away_qb_adjustment float8,
  rest_adjustment float8,
  pure_home_margin float8,
  pure_home_spread float8,
  market_home_spread float8,
  market_weight float8,
  home_margin float8,
  home_spread float8,
  model_total float8,
  margin_sd float8,
  total_sd float8,
  margin_total_correlation float8,
  distribution text,
  degrees_of_freedom float8
);

create table nfl.market_comparisons (
  game_id text primary key,
  start_date timestamptz,
  home_team text,
  away_team text,
  model_home_spread float8,
  model_total float8,
  margin_sd float8,
  total_sd float8,
  model_as_of timestamptz,
  market_available bool,
  priced_offer_available bool,
  executable_offer_available bool,
  review_status text,
  recommendation_status text,
  best_offer_market text,
  best_offer_selection text,
  best_offer_point float8,
  best_offer_price float8,
  best_offer_provider text,
  best_offer_provider_key text,
  best_offer_provider_last_update timestamptz,
  best_offer_event_link text,
  best_offer_market_link text,
  best_offer_bet_link text,
  best_offer_edge_points float8,
  best_offer_edge_standardized float8,
  best_offer_model_cover_probability float8,
  best_offer_model_fair_price float8,
  best_offer_expected_value_per_unit float8
);

create table nfl.backtest_predictions (
  game_id text primary key,
  season int not null,
  week int not null,
  week_index int,
  season_type text,
  home_team text not null,
  away_team text not null,
  neutral_site bool,
  home_points int,
  away_points int,
  closing_spread float8,
  model_margin float8,
  pure_model_margin float8,
  actual_margin float8
);

alter table nfl.teams enable row level security;
alter table nfl.team_ratings enable row level security;
alter table nfl.team_unit_ratings enable row level security;
alter table nfl.game_projections enable row level security;
alter table nfl.market_comparisons enable row level security;
alter table nfl.backtest_predictions enable row level security;

create policy "anon read" on nfl.teams for select to anon, authenticated using (true);
create policy "anon read" on nfl.team_ratings for select to anon, authenticated using (true);
create policy "anon read" on nfl.team_unit_ratings for select to anon, authenticated using (true);
create policy "anon read" on nfl.game_projections for select to anon, authenticated using (true);
create policy "anon read" on nfl.market_comparisons for select to anon, authenticated using (true);
create policy "anon read" on nfl.backtest_predictions for select to anon, authenticated using (true);

grant usage on schema nfl to anon, authenticated;
grant select on all tables in schema nfl to anon, authenticated;
