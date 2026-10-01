-- Run after the repository_dispatch workflows are on main.
-- Named schedules are updated in place, so rerunning creates no duplicates.
-- Times are UTC; Supabase schedules dispatch and GitHub still executes the jobs.
begin;
do $$
declare
  job record;
begin
  if not exists (select 1 from vault.secrets where name = 'github_dispatch_pat') then
    raise exception 'Missing github_dispatch_pat in Vault';
  end if;
  -- Awards now run after publication in the Tuesday workflow.
  for job in select jobid from cron.job where jobname = 'nfl-awards-dispatch'
  loop
    perform cron.unschedule(job.jobid);
  end loop;
  for job in select * from (values
    ('nfl-refresh-dispatch', '0 10 * 1,2,8-12 *', 'nfl-refresh'),
    ('nfl-weekly-dispatch', '0 16 * 1,2,8-12 2', 'nfl-weekly')
  ) as jobs(name, schedule, event_type)
  loop
    perform cron.schedule(job.name, job.schedule, format($command$
      select net.http_post(
        url := 'https://api.github.com/repos/renenunezg/momentumnfl/dispatches',
        headers := jsonb_build_object(
          'Authorization', 'Bearer ' || (select decrypted_secret
            from vault.decrypted_secrets where name = 'github_dispatch_pat'),
          'Accept', 'application/vnd.github+json',
          'User-Agent', 'supabase-pg-cron',
          'X-GitHub-Api-Version', '2022-11-28'
        ),
        body := jsonb_build_object('event_type', %L),
        timeout_milliseconds := 15000
      );
    $command$, job.event_type));
  end loop;
end $$;

-- Live win probability: dispatch only around unfinished games, and only when
-- no worker is alive.
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
commit;
