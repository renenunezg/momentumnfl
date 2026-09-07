-- Latest regular-season forecast per team and season. The sportsbook line
-- is a frozen preseason input/comparison, never a fresh market quote.
create table nfl.season_win_totals (
  season int not null,
  as_of timestamptz not null,
  model_version text not null,
  team_abbr text not null,
  team text not null,
  conference text,
  division text,
  wins int not null check (wins >= 0),
  losses int not null check (losses >= 0),
  ties int not null check (ties >= 0),
  games_played int not null check (games_played = wins + losses + ties),
  games_remaining int not null check (games_remaining >= 0 and games_played + games_remaining = 17),
  projected_wins float8 not null check (projected_wins >= wins and projected_wins <= wins + games_remaining),
  remaining_expected_wins float8 not null check (remaining_expected_wins >= 0 and remaining_expected_wins <= games_remaining),
  wins_p10 int not null check (wins_p10 >= wins),
  wins_p50 int not null check (wins_p50 >= wins_p10),
  wins_p90 int not null check (wins_p90 >= wins_p50 and wins_p90 <= wins + games_remaining),
  simulation_count int not null,
  simulation_seed int not null,
  ratings_through_week int not null,
  ratings_through_date timestamptz,
  schedule_fetched_at timestamptz not null,
  depth_chart_as_of timestamptz,
  sportsbook_win_total float8,
  sportsbook_source_name text,
  sportsbook_source_date date,
  sportsbook_source_url text,
  primary key (season, team_abbr)
);

alter table nfl.season_win_totals enable row level security;
create policy "anon read" on nfl.season_win_totals for select to anon, authenticated using (true);
grant select on nfl.season_win_totals to anon, authenticated;
