-- Only forecasts actually published before kickoff qualify for live grading.
-- Database receipt time is authoritative, never a caller's backdated as_of.
create table nfl.forecast_snapshots (
  snapshot_id bigint generated always as identity primary key,
  recorded_at timestamptz not null default clock_timestamp(),
  like nfl.game_projections including defaults
);
create index forecast_snapshots_game_time
  on nfl.forecast_snapshots (game_id, recorded_at desc, snapshot_id desc);

create function nfl.guard_pregame_projection() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  if TG_OP = 'DELETE' then
    raise exception 'Game projections cannot be deleted; publish with an upsert';
  end if;
  if new.start_date is null or new.as_of > clock_timestamp()
      or new.as_of >= new.start_date or new.start_date <= clock_timestamp() then
    return null;
  end if;
  if TG_OP = 'UPDATE' then
    if old.start_date is null or old.start_date <= clock_timestamp()
        or new.as_of < old.as_of then
      return null;
    end if;
    if new.game_id <> old.game_id or new.season <> old.season
        or new.home_team_abbr is distinct from old.home_team_abbr
        or new.away_team_abbr is distinct from old.away_team_abbr then
      raise exception 'Forecast game identity cannot change';
    end if;
    if new is not distinct from old then
      return null;
    end if;
  end if;
  return new;
end;
$$;

create trigger guard_pregame_projection
before insert or update or delete on nfl.game_projections
for each row execute function nfl.guard_pregame_projection();

create function nfl.archive_pregame_projection() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  insert into nfl.forecast_snapshots
    overriding system value
    select nextval(pg_get_serial_sequence('nfl.forecast_snapshots', 'snapshot_id')),
      clock_timestamp(), new.*;
  return new;
end;
$$;

create trigger archive_pregame_projection
after insert or update on nfl.game_projections
for each row execute function nfl.archive_pregame_projection();

-- A transaction that begins pregame but finishes after kickoff did not make
-- its forecast public in time. Roll it back instead of counting that forecast.
create function nfl.check_forecast_commit_cutoff() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  if new.start_date is null or clock_timestamp() >= new.start_date
      or new.recorded_at >= new.start_date or new.as_of >= new.start_date then
    raise exception 'Forecast publication crossed kickoff; retry to skip it';
  end if;
  return new;
end;
$$;
create constraint trigger forecast_commit_cutoff
after insert on nfl.forecast_snapshots
deferrable initially deferred
for each row execute function nfl.check_forecast_commit_cutoff();

create function nfl.reject_forecast_mutation() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  raise exception 'Published forecast snapshots are immutable';
end;
$$;
create trigger immutable_forecast_snapshots
before update or delete or truncate on nfl.forecast_snapshots
for each statement execute function nfl.reject_forecast_mutation();
create trigger no_truncate_game_projections
before truncate on nfl.game_projections
for each statement execute function nfl.reject_forecast_mutation();

create table nfl.game_results (
  game_id text primary key,
  season int not null,
  week int not null,
  season_type text not null,
  start_date timestamptz not null,
  home_team_abbr text not null,
  away_team_abbr text not null,
  home_team text not null,
  away_team text not null,
  neutral_site bool not null,
  home_points int not null check (home_points >= 0),
  away_points int not null check (away_points >= 0),
  closing_spread float8,
  source text not null,
  source_fetched_at timestamptz not null
);

-- Include missing forecasts and missing closes so coverage is visible.
-- A later correction to the result or closing line regrades automatically.
create view nfl.live_predictions with (security_invoker = true) as
select r.game_id, r.season, r.week, r.week as week_index, r.season_type,
  r.home_team, r.away_team, r.neutral_site, r.home_points, r.away_points,
  r.closing_spread, p.home_margin as model_margin,
  p.pure_home_margin as pure_model_margin,
  (r.home_points - r.away_points)::float8 as actual_margin,
  r.start_date, p.as_of as forecast_as_of, p.recorded_at as forecast_recorded_at,
  p.model_version, r.source as closing_source, r.source_fetched_at,
  abs(p.home_margin - (r.home_points - r.away_points)) as model_absolute_error,
  abs(p.pure_home_margin - (r.home_points - r.away_points)) as pure_absolute_error,
  abs(-r.closing_spread - (r.home_points - r.away_points)) as closing_absolute_error
from nfl.game_results r
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

alter table nfl.forecast_snapshots enable row level security;
alter table nfl.game_results enable row level security;
create policy "anon read" on nfl.forecast_snapshots
  for select to anon, authenticated using (true);
create policy "anon read" on nfl.game_results
  for select to anon, authenticated using (true);
grant select on nfl.forecast_snapshots, nfl.game_results, nfl.live_predictions
  to anon, authenticated;

-- Existing future forecasts were already public; preserve them at migration
-- receipt time. Never retroactively label an already-started game as live.
insert into nfl.forecast_snapshots
  overriding system value
  select nextval(pg_get_serial_sequence('nfl.forecast_snapshots', 'snapshot_id')),
    clock_timestamp(), p.*
  from nfl.game_projections p
  where p.start_date > clock_timestamp() and p.as_of <= clock_timestamp()
    and p.as_of < p.start_date;

notify pgrst, 'reload schema';
