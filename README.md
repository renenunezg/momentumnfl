# momentumnfl

NFL power ratings, game projections, prospective picks, and season forecasts.
Team strengths are fit weekly from [nflverse](https://github.com/nflverse) play-by-play and published to the `nfl` schema of the shared Momentum Supabase database.
The website at [renenunez.dev](https://renenunez.dev), in the separate `momentumweb` repository, reads that schema.

## Model

- Each week the engine refits from scratch on every completed game of the season: offense and defense points per drive fused with competitive-play EPA, a pace term, and a fitted home field, with a Student-t score distribution.
  Before week one, a preseason prior blends last season's final ratings, regressed toward average, with the season win-total market.
- The expected starting quarterback comes from a sourced announcement, the current depth chart, or a manual override, skipping anyone ruled out.
  The margin moves by the gap between that starter's rolling EPA strength and the quarterback mix already in the team's rating.
- Regulars ruled out at other positions cost their team the value the player ratings assign them, weighted by how much of the team's recent play they were part of.
- For totals, league scoring rate and pace carry preseason priors that fade through the season, and a fixed calibration sets the total's location.
- The published margin is shifted toward the market line at a capped weight.
  The pure-model margin is published beside it, and the closing line is kept as a separate accuracy benchmark, never an input.
- Opponent-adjusted unit ratings (rushing and passing on both sides, blocking, special teams) and a market-implied team rating are published for context; neither feeds the forecast.
- During games, nflfastR's spread-aware win probability model is anchored on the published pregame margin instead of the closing spread.

Parameters are selected on development seasons 2016 through 2021.
Later seasons are retrospective validation, not a fresh holdout.

## Picks

Picks are prospective and priced from the decision-time market: the devigged market number, moved toward the model only as far as the model has earned against the closing line.
Offers with positive expected value are picks, and a small weekly slate is filled with the best-priced sides as ranked leans, recorded under their own reason so the two tiers grade apart.
Stakes are a flat one unit.
Estimated expected value does not establish a real betting advantage.

An offer is eligible only from a configured bookmaker (`ODDS_API_BOOKMAKERS`), with a fresh two-sided quote matched to the exact game and kickoff.
A game without fresh schedule, depth-chart, injury, and expected-quarterback coverage is a No Play, never treated as healthy.
The rules live in `backend/recommendations.py`, `backend/source_inputs.py`, and `backend/availability.py`.

Grading uses official nflverse final scores at the frozen line and price.
The closing line is the one the live worker captures at kickoff, with the nflverse schedule line as the fallback.
Six-point teaser legs on small underdogs are paper-tracked in a separate record and never enter the pick ledger.

## Other pipelines

### Season wins

The engine and current expected quarterbacks are applied to the full regular-season schedule.
Reproducible simulated seasons give each team's expected wins and a 10th to 90th percentile range.
Because the preseason prior already uses sportsbook win totals, differences from those totals are not betting edges.

### Playoffs

`backend/playoffs/` simulates the rest of the season, applies the NFL tiebreakers through strength of schedule, and plays out the bracket, including reported quarterback return windows from `overrides/qb_return_windows.json`.
It writes a JSON snapshot offline, and a separate publish step stores it in `nfl.playoff_forecasts`.

### Awards

`backend/awards/` forecasts MVP, OPOY, OROY, DPOY, DROY, and COY with winner-choice models trained only on earlier seasons at the same week.
CPOY ranks a sourced list of comeback candidates until enough seasons exist under its current criteria to fit a model.
Win percentages stay hidden until a model passes its validation checks.

### Players

`backend/players/` scores weekly position traits, a 20 to 80 position grade, wins above replacement, each player's point-spread value, and contract surplus.
The absence layer reads these values, so the weekly job builds them before the team fit.

## Layout

```
backend/
  cli.py          python -m backend <command>
  etl/            nflverse pulls and the local parquet store
  features/       team-game, QB, and unit-channel aggregates
  model/          engine, preseason prior, layers, calibration, season wins, in-game model
  odds/           The Odds API client, ESPN fallback, and offer pricing
  players/        position traits, grades, player value
  playoffs/       playoff simulation and publication
  awards/         award forecasts
  publish.py      the database writer
  data_static/    hand-collected win totals, preseason QBs, award labels
overrides/        manual QB starters and return windows
sql/              nfl schema migrations, applied in filename order
ops/              pg_cron installation and the in-game model export
tests/            production-critical tests only
```

## Setup

Python 3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry sync
cp .env.example .env
poetry run pytest -q
```

| Variable | Needed for |
| --- | --- |
| `DATABASE_URL` | Anything that reads or writes Supabase |
| `MOMENTUMNFL_DB_WRITES` | Set to `1` to allow writes from a local shell |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | `publish`, which uploads forecast input files to Supabase Storage |
| `ODDS_API_KEY` | `odds` and the live worker's closing-line request |
| `ODDS_API_BOOKMAKERS` | Books whose quotes can become picks; without it no offer is eligible |
| `ODDS_API_REGIONS` | Optional, defaults to `us` |
| `ODDS_SOURCE` | Optional; `espn` switches to the ESPN DraftKings fallback when The Odds API quota runs out |
| `ODDS_API_CLOSING_RESERVE` | Optional credit floor for the live worker's closing-line request, defaults to 60 |

Model work runs offline once data is ingested into `backend/data/`, which is gitignored.

Core commands:

```bash
poetry run python -m backend ingest --seasons 2024 2025
poetry run python -m backend features --seasons 2024 2025
poetry run python -m backend fit --season 2026
poetry run python -m backend odds --season 2026
poetry run python -m backend publish --season 2026
poetry run python -m backend grade --season 2026
```

`fit`, `odds`, and `publish` infer the week from the next unplayed kickoff when it is omitted.
Run `poetry run python -m backend --help` for the full list, including the read-only `validate-*` commands and forecast replay.
Manual starter overrides go in `overrides/qb_starters.csv`.

## Production

Supabase pg_cron dispatches the GitHub Actions workflows through `repository_dispatch`, with retry ticks.
`ops/install_nfl_crons.sql` installs the schedules.

- `weekly` runs Tuesday morning Pacific: ingest, grade, rebuild features and player traits, fit, project season wins, snapshot odds, publish, then build the playoff forecast and, when enabled, awards and player grades.
- `refresh` runs daily in season, grades completed games, and when a kickoff falls in the next 30 hours reprojects from the Tuesday ratings with fresh inputs and odds.
- `live-win-probability` is dispatched around unfinished games, publishes in-game win probability, and captures closing lines.
- `awards` is a manual retry for the awards run.

Repository variables `NFL_PLAYOFFS_ENABLED`, `NFL_AWARDS_ENABLED`, and `NFL_PLAYER_GRADES_ENABLED` gate those publications.

## Rules that must hold

- Database writes are refused unless `GITHUB_ACTIONS=true` or `MOMENTUMNFL_DB_WRITES=1`; otherwise the session is also read-only on the server.
- Each pregame publication appends an immutable forecast snapshot, and the first qualifying pick is frozen; the database rejects post-kickoff replacements, late insertions, and changes to settled results.
- Every input to a forecast must have been observed before its cutoff, historical evaluation uses only information available at the time, and missing coverage fails or produces No Play instead of a guessed default.
- Each forecast archives its exact inputs, stored by content hash in Supabase Storage, so `replay-forecast` can reproduce it offline.

## Database

This repository owns the `nfl` schema.
The column lists in `backend/contracts.py` are the contract the website reads.
To rebuild an empty project, install `momentumweb/sql/001_site_revalidate.sql` first, then apply `sql/` in order.
Production schema changes require separate approval.

## Data sources

- Play-by-play, schedules, rosters, depth charts, injuries, contracts, PFR charting, and Next Gen Stats from nflverse via [nflreadpy](https://github.com/nflverse/nflreadpy), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Moneylines, spreads, and totals from [The Odds API](https://the-odds-api.com/), with ESPN's scoreboard as fallback and for live scores.
- Current injury news from ESPN.
- Preseason sportsbook win totals, collected by hand in `backend/data_static/`.

## License

MIT. See `LICENSE`.
