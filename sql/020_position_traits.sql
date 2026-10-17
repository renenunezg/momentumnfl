-- Latest weekly position trait snapshot per season: one row per player, trait
-- and the grade and war headline rows. The parquet artifact keeps the history.
create table if not exists nfl.position_traits (
    season integer not null,
    week integer not null,
    as_of timestamptz not null,
    model_version text not null,
    athlete_id text not null,
    athlete_name text,
    headshot_url text,
    team text,
    position_group text not null,
    games double precision,
    trait text not null,
    value double precision,
    percentile double precision,
    kind text,
    in_grade boolean not null,
    uncertainty double precision,
    reliability double precision,
    points_above_replacement double precision,
    primary key (season, athlete_id, trait)
);
create index if not exists position_traits_group_trait_idx
    on nfl.position_traits (season, position_group, trait);
alter table nfl.position_traits enable row level security;
create policy "Public position traits read" on nfl.position_traits
    for select to anon, authenticated using (true);
grant select on nfl.position_traits to anon, authenticated;
grant all on nfl.position_traits to service_role;

-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
drop trigger if exists site_revalidate on nfl.position_traits;
create trigger site_revalidate
    after insert or delete or update or truncate on nfl.position_traits
    for each statement execute function public.site_revalidate();
