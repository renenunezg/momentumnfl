# momentumnfl

NFL power ratings and point-spread projections, modeled after
[momentumcfb](../momentumcfb)'s Bayesian joint-scoring engine. Ratings are fit
weekly from nflverse play-by-play (EPA fused with points per drive), adjusted
for starting QB changes and rest, blended lightly with the market, and
published to the `nfl` schema of the shared `momentum` Supabase project. The
frontend lives in the momentumweb repo at renenunez.dev/nfl.

## Stack

Python 3.12, Poetry (script project, `package-mode = false`), pandas/numpy/scipy,
nflreadpy for nflverse data, SQLAlchemy + psycopg2 for Supabase Postgres,
The Odds API for live market lines.

## Layout

```
backend/          CLI, ETL, features, model, publish
sql/              checked-in DDL for the nfl schema
tests/            production-critical tests only
docs/superpowers/ design specs and implementation plans
```

## Weekly production path

```
python -m backend ingest    --seasons <Y>
python -m backend features  --seasons <Y>
python -m backend fit       --season <Y> --week <W>    # `preseason` for week 1
python -m backend publish   --season <Y> --week <W> --source fit
```

Publishing refuses to touch production unless `MOMENTUMNFL_DB_WRITES=1` is set.

## Environment

Copy `.env.example` to `.env` and fill in `DATABASE_URL`, `ODDS_API_KEY`
(plus optional `ODDS_API_REGIONS`, `ODDS_API_BOOKMAKERS`). nflverse data needs
no key.

## Design

The design of record is
[docs/superpowers/specs/2026-08-17-nfl-model-design.md](docs/superpowers/specs/2026-08-17-nfl-model-design.md):
engine math, nfelo idea adoption audit, unit ratings, schema DDL, publish
contract, and frontend plan.
