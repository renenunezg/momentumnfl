begin;

-- Shared immutable bytes avoid duplicating historical features in every run.
create table nfl.forecast_input_objects (
  sha256 text primary key check (sha256 ~ '^[0-9a-f]{64}$'),
  compressed_content bytea not null,
  recorded_at timestamptz not null default clock_timestamp()
);
create table nfl.forecast_input_runs (
  run_id text primary key check (run_id ~ '^[0-9a-f]{64}$'),
  season integer not null,
  week integer not null,
  forecast_as_of timestamptz not null,
  manifest jsonb not null,
  recorded_at timestamptz not null default clock_timestamp(),
  check (forecast_as_of <= recorded_at)
);
create index forecast_input_runs_season on nfl.forecast_input_runs(season, week);
create function nfl.protect_forecast_inputs() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  raise exception 'Archived forecast inputs are immutable';
end $$;
create trigger immutable_forecast_input_objects
before update or delete or truncate on nfl.forecast_input_objects
for each statement execute function nfl.protect_forecast_inputs();
create trigger immutable_forecast_input_runs
before update or delete or truncate on nfl.forecast_input_runs
for each statement execute function nfl.protect_forecast_inputs();
alter table nfl.forecast_input_objects enable row level security;
alter table nfl.forecast_input_runs enable row level security;
grant select, insert on nfl.forecast_input_objects, nfl.forecast_input_runs
  to service_role;
commit;
