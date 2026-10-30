begin;

-- The closing line captured before kickoff, one row per game, written once by
-- the live worker (backend/closing_capture.py) at the first poll where the
-- game is live or past its scheduled start. Shared contract across the
-- Momentum repositories; each sport keeps its own copy in its own schema.
--
-- source names what priced the row:
--   'odds_api:median'  The Odds API, one request per kickoff cluster about
--                      three minutes before kickoff. home_spread and total are
--                      the median offer across bookmakers; the prices are the
--                      DraftKings offer at that point when present, otherwise
--                      the bookmaker updated most recently at that point, and
--                      observed_at is that bookmaker's own last_update.
--   'espn:draftkings'  The DraftKings quote on ESPN's free scoreboard, held in
--                      memory from the scoreboard poll the worker already
--                      makes. ESPN gives no provider stamp, so observed_at is
--                      the fetch time.
-- Grading reads this table first and falls back to the nflverse schedule
-- value in game_results.closing_spread, labelled 'nflverse'. The nflverse
-- value is recorded after the game and may be revised; this row never is.
--
-- home_spread is the home line (negative when the home team is favored);
-- prices are American odds. n_offers counts the bookmakers whose spread
-- offers the median was taken from (1 for an ESPN row) and price_book names
-- the bookmaker the prices came from.
create table nfl.closing_lines (
  game_id text primary key,
  season integer not null,
  start_date timestamptz not null,
  source text not null,
  observed_at timestamptz not null,
  fetched_at timestamptz not null,
  home_spread float8,
  spread_home_price float8,
  spread_away_price float8,
  total float8,
  over_price float8,
  under_price float8,
  home_moneyline float8,
  away_moneyline float8,
  n_offers integer,
  price_book text
);

-- A row written after kickoff is not a closing line. A quote fetched at or
-- after the scheduled start, or an insert arriving more than ten minutes
-- after it, is dropped rather than recorded; the worker inserts with
-- on conflict do nothing and treats a zero row count as no capture.
create function nfl.guard_closing_line() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  if new.fetched_at >= new.start_date
      or clock_timestamp() > new.start_date + interval '10 minutes' then
    return null;
  end if;
  return new;
end;
$$;
create trigger guard_closing_line
before insert on nfl.closing_lines
for each row execute function nfl.guard_closing_line();

create function nfl.reject_closing_line_mutation() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  raise exception 'Closing lines are immutable once captured';
end;
$$;
create trigger immutable_closing_lines
before update or delete or truncate on nfl.closing_lines
for each statement execute function nfl.reject_closing_line_mutation();

alter table nfl.closing_lines enable row level security;
create policy "anon read" on nfl.closing_lines
  for select to anon, authenticated using (true);
grant select on nfl.closing_lines to anon, authenticated;
grant insert on nfl.closing_lines to service_role;

-- No site_revalidate trigger: the site sees the close through grading, which
-- already revalidates when game_results and graded_games are written.

-- The served closing spread prefers the captured row and falls back to the
-- nflverse value; closing_source says which one each game got. Column order
-- is unchanged so the view is replaced in place.
create or replace view nfl.live_predictions with (security_invoker = true) as
select r.game_id, r.season, r.week, r.week as week_index, r.season_type,
  r.home_team, r.away_team, r.neutral_site, r.home_points, r.away_points,
  coalesce(c.home_spread, r.closing_spread) as closing_spread,
  p.home_margin as model_margin,
  p.pure_home_margin as pure_model_margin,
  (r.home_points - r.away_points)::float8 as actual_margin,
  r.start_date, p.as_of as forecast_as_of, p.recorded_at as forecast_recorded_at,
  p.model_version,
  case when c.home_spread is not null then c.source
       when r.closing_spread is not null then 'nflverse' end as closing_source,
  r.source_fetched_at,
  abs(p.home_margin - (r.home_points - r.away_points)) as model_absolute_error,
  abs(p.pure_home_margin - (r.home_points - r.away_points)) as pure_absolute_error,
  abs(-coalesce(c.home_spread, r.closing_spread) - (r.home_points - r.away_points))
    as closing_absolute_error
from nfl.game_results r
left join nfl.closing_lines c on c.game_id = r.game_id
left join lateral (
  select f.* from nfl.forecast_snapshots f
  where f.game_id = r.game_id and f.season = r.season
    and f.home_team_abbr = r.home_team_abbr
    and f.away_team_abbr = r.away_team_abbr
    and f.recorded_at < r.start_date and f.as_of < r.start_date
    and f.recorded_at < f.start_date
  order by f.recorded_at desc, f.snapshot_id desc limit 1
) p on true
where r.start_date < now();

-- closingSources counts the benchmarked games per closing_source so the site
-- can show the population split instead of pooling captured and nflverse
-- closes. The backtest is graded on nflverse closes only.
create or replace function nfl.forecast_accuracy(p_source text default 'live') returns jsonb
language sql stable security invoker set search_path = pg_catalog, nfl as $$
  with selected as materialized (
    select season, model_margin, pure_model_margin, closing_spread, actual_margin,
      model_margin as primary_margin, closing_source
    from nfl.live_predictions where p_source = 'live'
    union all
    select season, model_margin, pure_model_margin, closing_spread, actual_margin,
      pure_model_margin as primary_margin, 'nflverse'::text as closing_source
    from nfl.backtest_predictions where p_source = 'backtest'
  ), paired as (
    select * from selected where model_margin is not null and pure_model_margin is not null
      and closing_spread is not null and actual_margin is not null
  ), metrics as (
    select season, grouping(season) as is_overall,
      coalesce(season::text, 'All seasons') as label, count(*) as games,
      avg(abs(primary_margin - actual_margin)) as "modelMae",
      avg(abs(pure_model_margin - actual_margin)) as "pureMae",
      case when p_source = 'backtest'
        then avg(abs(model_margin - actual_margin)) end as "closingBlendMae",
      avg(abs(-closing_spread - actual_margin)) as "marketMae",
      avg((abs(primary_margin - actual_margin) < abs(-closing_spread - actual_margin))::int) as "modelBeatsMarket",
      avg(primary_margin - actual_margin) as bias
    from paired group by grouping sets ((season), ()) having count(*) > 0
  ) select jsonb_build_object(
    'primarySource', case when p_source = 'backtest' then 'pure_model' else 'published' end,
    'overall', (select to_jsonb(m) from metrics m where is_overall = 1),
    'bySeason', (select coalesce(jsonb_agg(to_jsonb(m) order by season), '[]'::jsonb) from metrics m where is_overall = 0),
    'completed', (select count(*) from selected),
    'missingForecast', (select count(*) from selected where model_margin is null or pure_model_margin is null),
    'missingClose', (select count(*) from selected where closing_spread is null),
    'closingSources', (select coalesce(jsonb_object_agg(closing_source, games), '{}'::jsonb)
      from (select closing_source, count(*) as games from paired group by closing_source) s),
    'modelVersion', case when p_source = 'backtest' then
      (select string_agg(distinct model_version, ',') from nfl.backtest_predictions) end,
    'generatedAt', case when p_source = 'backtest' then
      (select max(generated_at) from nfl.backtest_predictions) end
  );
$$;

notify pgrst, 'reload schema';

commit;
