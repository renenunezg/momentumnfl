begin;

-- One raw season stat line per published athlete (every nfl.position_traits
-- athlete but kickers). ``stats`` holds the values in the key order and scale
-- backend/players/season_stats.py (STAT_KEYS) defines per position group;
-- momentumweb decodes it. The weekly publish replaces the season's rows with
-- one delete and one insert, in the position_traits transaction.
create table if not exists nfl.player_season_stats (
    season smallint not null,
    athlete_id text not null,
    position_group text not null,
    week smallint not null,
    stats smallint[] not null,
    updated_at timestamptz not null default now(),
    primary key (season, athlete_id)
);
alter table nfl.player_season_stats enable row level security;
create policy "Public player season stats read" on nfl.player_season_stats
    for select to anon, authenticated using (true);
grant select on nfl.player_season_stats to anon, authenticated;
grant all on nfl.player_season_stats to service_role;

-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
drop trigger if exists site_revalidate on nfl.player_season_stats;
create trigger site_revalidate
    after insert or delete or update or truncate on nfl.player_season_stats
    for each statement execute function public.site_revalidate();

commit;
