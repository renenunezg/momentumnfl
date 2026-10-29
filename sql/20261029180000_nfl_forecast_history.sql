begin;

-- The serving tables hold the latest snapshot only (the site reads them as
-- the current forecast). These append-only histories keep every published
-- snapshot so season wins, playoff odds and player grades can be graded at
-- season end. The backend copies each publish into them in the same
-- transaction with an explicit column list; rows can never be edited.
create function nfl.reject_history_mutation() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  raise exception 'Published forecast history is append-only';
end;
$$;

create table nfl.season_win_total_history (
  like nfl.season_win_totals including defaults including constraints,
  primary key (season, as_of, team_abbr)
);

create table nfl.playoff_forecast_history (
  like nfl.playoff_forecasts including defaults including constraints,
  primary key (season, published_at)
);

create table nfl.position_traits_history (
  like nfl.position_traits including defaults including constraints,
  primary key (season, week, as_of, athlete_id, trait)
);

do $$
declare
  history text;
begin
  foreach history in array array[
    'season_win_total_history', 'playoff_forecast_history', 'position_traits_history'
  ] loop
    execute format('alter table nfl.%I enable row level security', history);
    execute format(
      'create policy "anon read" on nfl.%I for select to anon, authenticated using (true)',
      history
    );
    execute format('grant select on nfl.%I to anon, authenticated', history);
    execute format('grant select, insert on nfl.%I to service_role', history);
    execute format(
      'create trigger append_only before update or delete or truncate on nfl.%I '
      'for each statement execute function nfl.reject_history_mutation()',
      history
    );
  end loop;
end $$;

-- Preserve the snapshots already published.
insert into nfl.season_win_total_history select * from nfl.season_win_totals;
insert into nfl.playoff_forecast_history select * from nfl.playoff_forecasts;
insert into nfl.position_traits_history select * from nfl.position_traits;

notify pgrst, 'reload schema';

commit;
