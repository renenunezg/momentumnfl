begin;

-- Every configured bookmaker's quote from each pregame odds pull, append-only,
-- taken from requests already made (no extra Odds API credits). source
-- 'decision' rows come from the pulls that price picks; 'closing' rows from
-- the live worker's request about three minutes before kickoff. With
-- Pinnacle among the configured books it is the sharp reference: pick CLV
-- against Pinnacle at decision time and at the close, and soft-book quotes
-- against Pinnacle's devigged price at the same pull.
create table nfl.book_quotes (
  game_id text not null,
  source text not null check (source in ('decision', 'closing')),
  fetched_at timestamptz not null,
  provider_key text not null,
  market text not null check (market in ('h2h', 'spreads', 'totals')),
  selection text not null check (selection in ('home', 'away', 'over', 'under')),
  point float8,
  price float8 not null,
  provider_last_update timestamptz,
  primary key (game_id, fetched_at, provider_key, market, selection)
);

-- Paper-tracked six-point teaser legs on +1.5 to +2.5 underdogs, from the
-- median dog line across the configured books in a pull. Never staked
-- and kept out of nfl.recommendations, so the published pick record is
-- unchanged. One row per game, frozen at its first qualifying observation;
-- only grading writes afterwards. The criterion and sample size are fixed in
-- advance (two-team six-point teaser break-even at -120, n = 200 legs).
create table nfl.teaser_legs (
  game_id text primary key,
  season integer not null,
  week integer not null,
  start_date timestamptz not null,
  decision_at timestamptz not null check (decision_at < start_date),
  side text not null check (side in ('home', 'away')),
  team text not null,
  books integer not null check (books > 0),
  point float8 not null check (point in (1.5, 2.0, 2.5)),
  teased_point float8 not null check (teased_point = point + 6),
  recorded_at timestamptz not null default clock_timestamp()
    check (recorded_at < start_date),
  outcome text not null default 'pending'
    check (outcome in ('pending', 'win', 'loss', 'push', 'void')),
  home_points integer,
  away_points integer,
  graded_at timestamptz,
  check ((outcome = 'pending') = (graded_at is null))
);

create function nfl.protect_teaser_leg() returns trigger
language plpgsql set search_path = pg_catalog, nfl as $$
begin
  if tg_op <> 'UPDATE' or old.outcome <> 'pending'
      or (to_jsonb(new) - array['outcome','home_points','away_points','graded_at'])
         is distinct from
         (to_jsonb(old) - array['outcome','home_points','away_points','graded_at']) then
    raise exception 'Teaser legs are frozen once logged; only a pending leg is graded';
  end if;
  return new;
end;
$$;
create trigger protect_teaser_leg
before update or delete on nfl.teaser_legs
for each row execute function nfl.protect_teaser_leg();

-- Internal research records: the site does not read either table, so there is
-- no anon grant and no site_revalidate trigger.
alter table nfl.book_quotes enable row level security;
alter table nfl.teaser_legs enable row level security;
grant select, insert on nfl.book_quotes to service_role;
grant select, insert, update on nfl.teaser_legs to service_role;

commit;
