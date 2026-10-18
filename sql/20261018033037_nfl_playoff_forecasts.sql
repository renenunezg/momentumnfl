begin;
create table nfl.playoff_forecasts (
  season integer primary key check (season between 2026 and 2100),
  published_at timestamptz not null default now(),
  payload jsonb not null check (
    jsonb_typeof(payload) = 'object'
    and payload ?& array['schema_version','season','teams','bracket','path']
    and payload->>'schema_version' = '1'
    and (payload->>'season')::integer = season
    and jsonb_typeof(payload->'teams') = 'array'
    and jsonb_array_length(payload->'teams') = 32
    and jsonb_typeof(payload->'bracket') = 'array'
    and jsonb_array_length(payload->'bracket') = 14
    and jsonb_typeof(payload->'path') = 'array'
    and jsonb_array_length(payload->'path') = 13
  )
);
alter table nfl.playoff_forecasts enable row level security;
grant usage on schema nfl to anon, authenticated, service_role;
revoke all on nfl.playoff_forecasts from anon, authenticated;
grant select on nfl.playoff_forecasts to anon, authenticated;
grant all on nfl.playoff_forecasts to service_role;
create policy public_read on nfl.playoff_forecasts
  for select to anon, authenticated using (true);
create trigger site_revalidate
  after insert or update or delete or truncate on nfl.playoff_forecasts
  for each statement execute function public.site_revalidate();
commit;
