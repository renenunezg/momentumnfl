-- Independent weekly AP awards snapshots, read-only to the public site.
create table nfl.award_boards (
    season integer not null,
    week integer not null,
    award text not null check (award in ('MVP','OPOY','OROY','DPOY','DROY','CPOY','COY')),
    as_of timestamptz not null,
    model_version text not null,
    candidate_id text not null,
    candidate_name text not null,
    headshot_url text,
    team text not null,
    position text not null,
    predicted_rank integer,
    performance_rank integer,
    performance_score double precision,
    win_probability double precision check (win_probability between 0 and 1),
    probability_status text not null,
    rank_change integer,
    games integer not null,
    projected_stats jsonb not null,
    drivers jsonb not null,
    context_source text,
    context_reason text,
    primary key (season, week, award, candidate_id)
);

create table nfl.award_model_meta (
    season integer not null,
    week integer not null,
    award text not null,
    as_of timestamptz not null,
    model_version text not null,
    status text not null,
    candidate_count integer not null,
    training_seasons jsonb not null,
    validation jsonb not null,
    provenance jsonb not null,
    primary key (season, week, award)
);

alter table nfl.award_boards enable row level security;
alter table nfl.award_model_meta enable row level security;
create policy "Public awards read" on nfl.award_boards for select using (true);
create policy "Public award metadata read" on nfl.award_model_meta for select using (true);
grant select on nfl.award_boards, nfl.award_model_meta to anon, authenticated;
grant all on nfl.award_boards, nfl.award_model_meta to service_role;
