# momentumnfl

Bayesian NFL power ratings and point-spread projections. Team strengths are
fit weekly from [nflverse](https://github.com/nflverse) play-by-play (EPA
fused with points per drive) with a conjugate-Gaussian ridge, adjusted for
starting-quarterback changes and rest, blended with the market at a capped
weight, and published to a Postgres schema that a separate web frontend reads.

The model is not a betting product. Published spreads always carry the pure
model number alongside the blended one, the closing line is reported as the
honesty benchmark, and `recommendation_status` is always `not_recommended`.

## How it works

1. **Ingest.** Schedules, play-by-play, depth charts, PFR pass-protection
   charting, and Next Gen passing stats are pulled with
   [nflreadpy](https://github.com/nflverse/nflreadpy) into local parquet.
2. **Features.** Play-by-play is reduced to one row per team-game: points,
   drives, competitive-play EPA, QB dropbacks, and the unit channels used by
   the descriptive unit ratings.
3. **Fit.** Each week the engine refits from scratch on every prior game of
   the season: per-team offense and defense points per drive, a pace ridge,
   and a fitted home-field parameter, with a Student-t score distribution.
   Before week 1 a preseason prior blends mean reversion of last season's
   final ratings with the season win-total market.
4. **Layers.** The QB adjustment is `selected QB strength - team QB baseline`.
   Both sides use rolling EPA strengths on the same replacement-relative scale, with half-lives of 1,000 dropbacks and 52 calendar weeks.
   The estimate shrinks toward replacement using 200 prior dropbacks and the remaining effective sample, rather than treating lifetime dropbacks as current evidence.
   The team baseline weights every passer by recent dropbacks and the engine's time decay, carrying the previous season's mix into preseason.
   For a team that has not played yet, the win-total-funded share of that baseline uses the fixed preseason QB reference instead, so the market's offseason QB change is not added again.
   That reference stays fixed when a depth chart or manual starter override changes; an injury moves the pure spread by `backup strength - starter strength`.
   This separates a QB's strength from the change applied to a team that already contains his contribution, and permits starter-to-backup substitutions without changing stored team ratings.
   Expected points then receive the rest adjustment, and the published margin is shifted toward the market line by a weight capped at 0.5.
   QB memory and shrinkage were selected using development-season pure-model log loss with the engine held fixed; the subsequent layer search selected a 1.0 coefficient.
   The 2022-2025 evaluation is a repeatedly inspected retrospective validation window, not a fresh prospective test of this revision.
   `qbs --context` additionally fits separate QB, team-season, and opposing-defense effects, including hit/sack exposure and the QB's own tendency to invite pressure.
   Its conditional parameter uncertainty reflects limited recent samples and ambiguity between player and supporting cast; it is not a calibrated interval for game scores or an identified causal player value.
   These context estimates remain diagnostics because development testing did not support adding their means or uncertainty to published spreads.
5. **Calibrate.** A walk-forward backtest over 2016 onward selects every
   hyperparameter on development seasons (2016 to 2021) and reports holdout
   seasons (2022 to 2025) untouched. The backtest is republished as the
   model's track record.
6. **Publish.** Seven tables in the `nfl` schema, written in one transaction.
   The column lists in `backend/publish.py` are the contract with the
   frontend and are checked against `sql/` by a test.

Opponent-adjusted unit ratings (rush and pass offense and defense, pass
block, run block, special teams) are published as descriptive companions.
They do not feed the engine.

## Layout

```
backend/
  cli.py          python -m backend <command>
  pipeline.py     incremental rebuilds and schedule gates for CI
  etl/            nflreadpy pulls and the parquet store
  features/       team-game, QB, and unit-channel aggregates
  model/          engine, preseason prior, layers, calibration, unit ratings
  odds/           The Odds API client and offer pricing
  publish.py      the only database writer
  data_static/    win totals by season and the frozen margin distribution
sql/              DDL for the nfl schema, applied in order with psql
tests/            production-critical tests only
```

## Setup

Python 3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry sync
cp .env.example .env
```

Only `DATABASE_URL` (for `publish`) and `ODDS_API_KEY` (for `odds`) need
values. Everything else runs offline once data has been ingested. Data lands
in `backend/data/`, which is gitignored.

```bash
poetry run pytest -q
```

## Commands

Local development, oldest to newest:

```bash
poetry run python -m backend ingest --seasons 2024 2025
poetry run python -m backend features --seasons 2024 2025
poetry run python -m backend calibrate
poetry run python -m backend preseason --season 2025
poetry run python -m backend fit --season 2025 --week 3
```

For the current preseason forecast, after the historical feature cache exists:

```bash
poetry run python -m backend refresh-inputs --season 2026
poetry run python -m backend fit --season 2026
poetry run python -m backend season-wins --season 2026
poetry run python -m backend odds --season 2026
poetry run python -m backend publish --season 2026 --skip-backtest
```

`fit`, `odds`, and `publish` infer the season and week from the next unplayed
kickoff when the flags are omitted. `fit` falls back to `preseason` when no
game of the season has been played yet.

| Command | What it does |
| --- | --- |
| `ingest --seasons` | Pull raw nflverse sources for the given seasons. |
| `bootstrap-history --through` | Rebuild cached team-game and QB features for any missing historical season without keeping raw play-by-play. |
| `refresh-inputs --season` | Pull only the inputs that can change an unplayed projection: schedules, teams, depth charts. |
| `features --seasons [--incremental --lookback-weeks N]` | Build feature parquet; incremental mode rebuilds only recent and missing games. |
| `fit [--season --week] [--projections-only]` | Fit ratings and unit ratings and project the week. |
| `preseason --season` | Build the week-1 prior, ratings, and projections. |
| `season-wins --season` | Project full regular-season wins for all 32 teams and write a per-game probability audit. |
| `calibrate` | Run the walk-forward search and freeze the margin distribution. |
| `validate-qb [--memory] [--context]` | Reproduce the development memory search, select the QB layer, report retrospective validation, and optionally test the context promotion gate, without changing artifacts. |
| `qbs --season --week --team BUF [--context]` | Inspect starter/backup strengths, recorded and effective samples, and substitutions; optionally show context effects and model uncertainty. |
| `odds [--season --week]` | Snapshot Odds API offers and price them against the projections. |
| `upcoming --season [--hours]` | Exit 0 when an unplayed game kicks off within the window, 3 otherwise. |
| `publish [--season --week] [--skip-backtest] [--projections-only]` | Write the week to the `nfl` schema. |

Publishing refuses to touch the database unless `MOMENTUMNFL_DB_WRITES=1` is
set; GitHub Actions opts in automatically. Without it the session is opened
read-only server-side as well.

Manual starter overrides can be placed in `overrides/qb_starters.csv` with
columns `season, week, team_abbr, gsis_id`; they take priority over the depth
chart.

## Season win projections

`season-wins` applies the existing engine and current expected QB adjustments to the complete regular-season schedule, with no game-line blend.
Preseason ratings already use the frozen sportsbook win totals, so these outputs are not market-independent and their differences from that input are not betting edges.
The sportsbook source and date live in `backend/data_static/win_total_sources.json`; missing provenance stays missing.

Expected wins are completed wins plus the sum of remaining Student-t win probabilities.
Completed ties contribute zero wins; future ties are not simulated.
The command requires 272 unique games and 17 appearances per team (2021 onward), so incomplete schedules fail closed.
Canceled games require explicit schedule support before a shortened season can be published.

The 10th, 50th, and 90th percentiles come from 100,000 reproducible season draws using seed 20260907.
A Gaussian copula combines shared draws from the engine's team-strength/HFA covariance with independent game residuals, preserving each game's Student-t win probability without counting parameter uncertainty twice.
Current strength means and expected QBs stay fixed through the remaining schedule; future injuries, roster changes, and strength evolution are not modeled.
The season ranges have not been coverage-calibrated and are not calibrated confidence intervals for the mean.
Every draw awards exactly one win per remaining game, and analytic league expected wins equal 272 minus completed tied games.
A retrospective preseason replay of 2023-2025 (96 team-seasons) had 2.53 wins MAE versus 2.39 for the frozen sportsbook input, with 81.25% of actual win totals inside the nominal 80% ranges.
This is a descriptive check, not a tuning step or evidence of an independent market edge.

`nfl.season_win_totals` contains the latest 32-team snapshot for each season, replaced in the same transaction as game projections.
Apply `sql/003_season_win_totals.sql` before publishing this contract.
The weekly and projection-refresh workflows regenerate season wins before publication.
The `/nfl/season-wins` frontend reads only that schema and displays completed records, full-season expected wins, remaining wins, model ranges, and the dated sportsbook comparison.

## Production

Two GitHub Actions workflows drive the season, serialized under one
concurrency group:

- `weekly` runs Tuesday after Monday Night Football: restore cached history,
  ingest the active season, rebuild the last two weeks of features, fit, take
  an odds snapshot, publish.
- `refresh` runs daily and, only when a kickoff falls in the next 30 hours,
  refreshes schedules and depth charts, reprojects from the Tuesday ratings,
  takes an odds snapshot, and republishes the projections. Ratings stay the
  Tuesday snapshot.

Secrets: `DATABASE_URL`, `ODDS_API_KEY`.

## Data sources

- Play-by-play, schedules, teams, depth charts, injuries, PFR advanced stats,
  and Next Gen Stats come from nflverse via nflreadpy. nflverse data is
  released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The one-game fixtures under `tests/fixtures/` are nflverse excerpts.
- Live spreads and totals come from [The Odds API](https://the-odds-api.com/).
- `backend/data_static/win_totals.csv` holds preseason sportsbook win totals
  by team and season, collected by hand.
  The 2026 inputs are BetMGM's September 1, 2026 [regular-season win totals](https://sports.betmgm.com/en/blog/nfl/nfl-over-under-wins-2026-win-totals-all-32-teams-bm16/), retrieved September 7 before kickoff and frozen for the season.
- `backend/data_static/preseason_qbs.csv` freezes the market input's assumed QB lineup using [nflverse depth charts](https://github.com/nflverse/nflverse-data/releases/tag/depth_charts).
  The 2026 references use September 1 snapshots, and 2025 uses the last snapshot before the season opener.
  The 2016-2024 references are week-one lineup proxies with no intraday timestamps; missing references, including the postponed Miami and Tampa Bay openers in 2017, retain the empirical baseline.
  These references approximate the lineup priced into historical win totals; the historical evaluation remains conditional on the realized majority-dropback QB and does not validate injury-news timing.
  QB strengths are rolling EPA estimates, not identified causal player values or a forced 5-to-8-point scale; unobserved QBs receive a replacement prior, and `qbs` exposes recorded dropbacks since 2015 alongside each estimate.
- Context diagnostics require local historical raw PBP and make no network requests or artifact writes.
  Their separate observations include competitive scrambles and [nflfastR QB EPA](https://nflfastr.com/articles/beginners_guide.html), which avoids charging the passer for a receiver's lost fumble.
  Hits and sacks are a consistently available protection proxy, not all charted pressures or an offensive-line grade.
  The ridge separates supporting-cast and opponent effects while retaining estimated QB-specific pressure liability.
  Its penalty sizes and residual-variance floors define conditional uncertainty; the context model is not used by the ratings engine or published projection distributions.
- `backend/data_static/margin_distribution.csv` is generated by `calibrate`.

## License

MIT. See `LICENSE`.
