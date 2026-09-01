-- nfl schema v2: append-only archive of Odds API consensus lines, one row per
-- game per snapshot. nflverse overwrites spread_line in place, so without this
-- only the closing line survives; the Tuesday row is the early-week line the
-- published number actually blends against.

create table nfl.market_snapshots (
  game_id text not null,
  season int not null,
  week int not null,
  fetched_at timestamptz not null,
  home_spread float8,
  total float8,
  spread_books int,
  total_books int,
  primary key (game_id, fetched_at)
);

alter table nfl.market_snapshots enable row level security;
create policy "anon read" on nfl.market_snapshots for select to anon, authenticated using (true);
grant select on nfl.market_snapshots to anon, authenticated;
