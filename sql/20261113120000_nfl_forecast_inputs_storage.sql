-- Forecast input bytes live in the private nfl-forecast-inputs Storage bucket
-- (backend/forecast_archive.py); run manifests stay in nfl.forecast_input_runs.
-- Apply only after every object in the table has been copied to the bucket
-- and verified against its SHA-256.
do $$
begin
  if to_regclass('storage.buckets') is not null then
    insert into storage.buckets (id, name, public)
    values ('nfl-forecast-inputs', 'nfl-forecast-inputs', false)
    on conflict (id) do nothing;
  end if;
end $$;

drop table nfl.forecast_input_objects;
