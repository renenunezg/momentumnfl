-- Separate live state from immutable pregame forecasts.
begin;

create table if not exists nfl.live_win_probability (
  game_id text primary key,
  updated_at timestamptz not null,
  payload jsonb not null check (
    payload->>'schema_version' = '1'
    and payload->>'game_id' = game_id
  )
);

alter table nfl.live_win_probability enable row level security;
grant select on nfl.live_win_probability to anon, authenticated;
drop policy if exists "anon read" on nfl.live_win_probability;
create policy "anon read" on nfl.live_win_probability
  for select to anon, authenticated using (true);

-- No revalidate trigger: the site reads this table through a short timed
-- cache, so a slate of writes costs no site function calls.

notify pgrst, 'reload schema';
commit;
