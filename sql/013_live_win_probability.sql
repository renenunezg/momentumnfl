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

-- Dispatch only around unfinished games, and only when no worker is alive.
-- A terminal row ('Final' or 'Off') ends a game's claim on the dispatcher, and
-- the six-hour bound ends it for a game no worker ever reached.
select cron.schedule('nfl-live-win-probability-dispatch', '*/5 * * * *', $dispatch$
  select net.http_post(
    url := 'https://api.github.com/repos/renenunezg/momentumnfl/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || (select decrypted_secret
        from vault.decrypted_secrets where name = 'github_dispatch_pat'),
      'Accept', 'application/vnd.github+json',
      'User-Agent', 'supabase-pg-cron',
      'X-GitHub-Api-Version', '2022-11-28'
    ),
    body := jsonb_build_object('event_type', 'nfl-live-win-probability'),
    timeout_milliseconds := 15000
  ) where exists (
    select 1 from nfl.game_projections as g
    left join nfl.live_win_probability as live using (game_id)
    where g.start_date between now() - interval '6 hours'
                           and now() + interval '5 minutes'
      and coalesce(live.payload->>'abstract_state', '') not in ('Final', 'Off')
  ) and not exists (
    -- Fresh writes are the heartbeat. Queue a handoff before the bounded
    -- worker expires, or a recovery when publication stops.
    select 1 from nfl.live_win_probability
    where updated_at > now() - interval '4 minutes'
      and payload->>'abstract_state' not in ('Final', 'Off')
      and coalesce((payload->>'worker_expires_at')::timestamptz,
                   'infinity'::timestamptz) > now() + interval '10 minutes'
  );
$dispatch$);

notify pgrst, 'reload schema';
commit;
