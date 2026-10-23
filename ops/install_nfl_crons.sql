-- Run after the repository_dispatch workflows are on main.
-- Named schedules are updated in place, so rerunning creates no duplicates.
-- Schedules are UTC; Supabase schedules dispatch and GitHub still executes the jobs.
begin;
do $$
declare
  job record;
  conditions text[];
begin
  if not exists (select 1 from vault.secrets where name = 'github_dispatch_pat') then
    raise exception 'Missing github_dispatch_pat in Vault';
  end if;
  -- Awards now run after publication in the Tuesday workflow.
  for job in select jobid from cron.job where jobname = 'nfl-awards-dispatch'
  loop
    perform cron.unschedule(job.jobid);
  end loop;
  -- A cron tick that never starts ('job startup timeout' while the database
  -- is starved, as on 2026-10-06 through 2026-10-08) sends nothing to GitHub,
  -- so each dispatch has retry ticks that post only while no tick for that
  -- event has posted today. Both workflows tolerate a late dispatch: the
  -- refresh is schedule-aware and the weekly run is a full republish.
  -- pg_cron runs in UTC, so the weekly jobs tick at both DST offsets and post
  -- only during their Pacific hours: 8:00 AM, with retries at 8:30 and 10:30.
  for job in select * from (values
    ('nfl-refresh-dispatch', '0 10 * 1,2,8-12 *', 'nfl-refresh', null, null),
    ('nfl-refresh-dispatch-retry', '30 10,12 * 1,2,8-12 *', 'nfl-refresh',
     'nfl-refresh-dispatch', null),
    ('nfl-weekly-dispatch', '0 15,16 * 1,2,8-12 2', 'nfl-weekly', null,
     array[8]),
    ('nfl-weekly-dispatch-retry', '30 15-18 * 1,2,8-12 2', 'nfl-weekly',
     'nfl-weekly-dispatch', array[8, 10])
  ) as jobs(name, schedule, event_type, retries_for, pacific_hours)
  loop
    conditions := '{}';
    if job.pacific_hours is not null then
      conditions := conditions || format(
        'extract(hour from now() at time zone %L) = any (%L::int[])',
        'America/Los_Angeles', job.pacific_hours);
    end if;
    if job.retries_for is not null then
      -- Ticks outside their Pacific hours succeed with '0 rows' and post
      -- nothing, so only a '1 row' run counts as today's dispatch.
      conditions := conditions || format($guard$not exists (
        select 1 from cron.job_run_details as run
        join cron.job using (jobid)
        where cron.job.jobname in (%L, %L)
          and run.status = 'succeeded'
          and run.return_message = '1 row'
          and run.start_time >= date_trunc('day', now())
      )$guard$, job.retries_for, job.name);
    end if;
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
      ) %s;
    $command$, job.event_type, case when cardinality(conditions) = 0 then ''
      else 'where ' || array_to_string(conditions, ' and ') end));
  end loop;
end $$;

-- Live win probability: dispatch only around unfinished games, and only when
-- no worker is alive.
-- A terminal row ('Final' or 'Off') ends a game's claim on the dispatcher, and
-- the six-hour bound ends it for a game no worker ever reached.
-- Minute offset 2: the NHL, CFB, and MLB dispatchers take offsets 0, 1, and 3
-- so the four never start on the same minute of the shared database.
select cron.schedule('nfl-live-win-probability-dispatch', '2-59/5 * * * *', $dispatch$
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
