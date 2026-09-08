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
  for job in select * from (values
    ('nfl-refresh-dispatch', '0 10 * 1,2,8-12 *', 'nfl-refresh'),
    ('nfl-weekly-dispatch', '0 16 * 1,2,8-12 2', 'nfl-weekly'),
    ('nfl-awards-dispatch', '0 18 * 1,9-12 3', 'nfl-awards')
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
commit;
