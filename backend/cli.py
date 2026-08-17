"""momentumnfl command line. Heavy imports stay inside each subcommand so
startup is instant and a broken optional dependency only breaks its own
command. Commands collect problems and exit 1 at the end rather than
aborting on the first."""

import argparse

from backend.config import SEASONS


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="pull raw nflverse data")
    ingest_parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)

    features_parser = subparsers.add_parser(
        "features", help="build team_games and qb_games parquet"
    )
    features_parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)

    fit_parser = subparsers.add_parser(
        "fit", help="fit in-season ratings and project a week"
    )
    fit_parser.add_argument("--season", type=int)
    fit_parser.add_argument("--week", type=int)

    preseason_parser = subparsers.add_parser(
        "preseason", help="build the week-1 prior, ratings, and projections"
    )
    preseason_parser.add_argument("--season", type=int, required=True)

    subparsers.add_parser(
        "calibrate", help="walk-forward hyperparameter search"
    )

    odds_parser = subparsers.add_parser(
        "odds", help="snapshot Odds API offers for a week"
    )
    odds_parser.add_argument("--season", type=int)
    odds_parser.add_argument("--week", type=int)

    publish_parser = subparsers.add_parser(
        "publish", help="publish a week to the nfl schema"
    )
    publish_parser.add_argument("--season", type=int)
    publish_parser.add_argument("--week", type=int)
    publish_parser.add_argument("--skip-backtest", action="store_true")
    publish_parser.add_argument("--projections-only", action="store_true")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "features":
        run_features(args)
    elif args.command == "fit":
        run_fit(args)
    elif args.command == "preseason":
        run_preseason(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "odds":
        run_odds(args)
    elif args.command == "publish":
        run_publish(args)


def run_ingest(args) -> None:
    from backend.etl import ingest

    problems: list[str] = []
    seasons = sorted(args.seasons)
    ingest.ingest_shared(seasons)
    print(f"ingested shared (schedules, teams, ngs) for {seasons[0]}-{seasons[-1]}")
    for season in seasons:
        problems.extend(ingest.ingest_season(season))
        print(f"ingested {season}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)


def resolve_week(args) -> tuple[int, int]:
    """(season, week) from flags, or the first week with unplayed games."""
    if args.season is not None and args.week is not None:
        return args.season, args.week
    from backend.etl import store

    schedules = store.read_raw("schedules.parquet")
    unplayed = schedules[schedules["home_score"].isna()]
    if unplayed.empty:
        raise SystemExit("PROBLEM: no unplayed games to infer a week from")
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    season = args.season if args.season is not None else int(first["season"])
    week = args.week if args.week is not None else int(first["week"])
    return season, week


def run_fit(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.model.fit_week import fit_and_project

    args.season, args.week = resolve_week(args)
    # Before any current-season game has been played there is nothing to fit;
    # the preseason prior is the correct producer for that window, so the
    # weekly cron stays valid across the season boundary.
    played = str(args.season) in store.processed_names("team_games")
    if not played:
        print(f"fit: no played {args.season} games yet, running preseason")
        run_preseason(args)
        return

    fit, ratings, projections = fit_and_project(args.season, args.week)
    ratings_df = pd.DataFrame([rating.to_record() for rating in ratings])
    projections_df = pd.DataFrame(
        [projection.to_record() for projection in projections]
    )
    store.write_processed(
        ratings_df, "ratings", f"{args.season}_{args.week:02d}.parquet"
    )
    store.write_processed(
        projections_df, "projections", f"{args.season}_{args.week:02d}.parquet"
    )
    from backend.model.fit_week import load_season_games, load_team_names
    from backend.model.fit_week import recency_by_game as recency_map
    from backend.model.unit_ratings import fit_unit_ratings

    games = load_season_games(args.season)
    unit_games = store.read_processed("unit_games", f"{args.season}.parquet")
    units = fit_unit_ratings(
        unit_games,
        games,
        args.week,
        fit.as_of,
        recency_map(games, args.week, fit.config),
    )
    units_df = pd.DataFrame(units.to_records(load_team_names()))
    store.write_processed(
        units_df, "unit_ratings", f"{args.season}_{args.week:02d}.parquet"
    )
    print(
        f"fit {args.season} week {args.week}: {len(ratings_df)} ratings, "
        f"{len(projections_df)} projections, {len(units_df)} unit ratings, "
        f"hfa {fit.hfa_points:.2f} pts, base drives {fit.base_drives:.1f}"
    )
    top = ratings_df.head(5)[["team_abbr", "power_rating"]]
    print(top.to_string(index=False))


def run_preseason(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.features.drives import _kickoff_utc
    from backend.model.fit_week import (
        compute_qb_adjustments,
        load_qb_games,
        load_team_names,
    )
    from backend.model.preseason import MODEL_VERSION, build_preseason_prior
    from backend.model.projections import LayerConfig, assemble_projections

    prior = build_preseason_prior(args.season)
    team_names = load_team_names()
    ratings = prior.ratings(team_names)
    ratings_df = pd.DataFrame([rating.to_record() for rating in ratings])
    ratings_df["model_version"] = MODEL_VERSION
    ratings_df["missing_input_count"] = ratings_df["team_abbr"].map(
        prior.ratings_frame.set_index("team_abbr")["missing_input_count"]
    )
    store.write_processed(ratings_df, "ratings", f"{args.season}_01.parquet")

    schedules = store.read_raw("schedules.parquet")
    slate = schedules[
        schedules["season"].eq(args.season) & schedules["week"].eq(1)
    ].copy()
    if slate.empty:
        print(f"preseason {args.season}: no week-1 schedule yet; ratings only")
        return
    slate["neutral_site"] = slate["location"].eq("Neutral")
    slate["start_date"] = _kickoff_utc(slate)
    week1_fit = prior.week1_fit()
    qb_games = load_qb_games(list(range(2015, args.season)))
    try:
        depth_charts = store.read_raw(
            "depth_charts", f"{args.season}.parquet"
        )
    except FileNotFoundError:
        depth_charts = pd.DataFrame(
            columns=["season", "week", "position", "team", "gsis_id"]
        )
    previous_games = store.read_processed(
        "team_games", f"{args.season - 1}.parquet"
    )
    qb_adjustments = compute_qb_adjustments(
        args.season, 1, slate,
        previous_games.assign(model_week=0), qb_games, depth_charts,
        week1_fit.config, LayerConfig(),
    )
    market = {
        str(row.game_id): -float(row.spread_line)
        for row in slate.itertuples()
        if pd.notna(getattr(row, "spread_line", None))
    }
    projections = assemble_projections(
        week1_fit, slate, prior.as_of, team_names, qb_adjustments, market
    )
    projections_df = pd.DataFrame(
        [projection.to_record() for projection in projections]
    )
    projections_df["model_version"] = MODEL_VERSION
    store.write_processed(
        projections_df, "projections", f"{args.season}_01.parquet"
    )
    print(
        f"preseason {args.season}: {len(ratings_df)} ratings, "
        f"{len(projections_df)} week-1 projections, "
        f"points/win slope {prior.slope:.2f}"
    )
    print(
        ratings_df.head(5)[["team_abbr", "power_rating"]].to_string(index=False)
    )


def run_calibrate(args) -> None:
    from backend.model.calibration import run_calibration

    summary = run_calibration()
    print("\n=== selected configuration ===")
    print(summary["engine_config"])
    print(summary["layer_params"])
    print(summary["preseason_config"])
    print(f"use_prior_means: {summary['use_prior_means']}")
    print(f"dev margin log loss: {summary['dev_margin_log_loss']:.4f}")
    print(f"holdout margin log loss: {summary['holdout_margin_log_loss']:.4f}")
    print(f"holdout coverage: {summary['holdout_coverage']}")
    print("\n=== honesty report (dev) ===")
    print(summary["dev_honesty"].to_string())
    print("\n=== honesty report (holdout, untouched) ===")
    print(summary["holdout_honesty"].to_string())


def run_odds(args) -> None:
    from datetime import timedelta

    import pandas as pd

    from backend.etl import store
    from backend.features.drives import _kickoff_utc
    from backend.model.market_blend import fit_margin_residual_distribution
    from backend.odds.client import OddsAPIClient
    from backend.odds.markets import compare_priced_offers, flatten_offers

    season, week = resolve_week(args)
    projections = store.read_processed(
        "projections", f"{season}_{week:02d}.parquet"
    )
    schedules = store.read_raw("schedules.parquet")
    slate = schedules[
        schedules["season"].eq(season) & schedules["week"].eq(week)
    ].copy()
    slate["start_date"] = _kickoff_utc(slate)
    # Odds API events carry full team names; map abbrs for matching.
    from backend.model.fit_week import load_team_names

    names = load_team_names()
    match_frame = pd.DataFrame(
        {
            "game_id": slate["game_id"],
            "start_date": slate["start_date"],
            "home_team": slate["home_team"].map(names),
            "away_team": slate["away_team"].map(names),
        }
    )
    client = OddsAPIClient()
    window_from = pd.to_datetime(slate["start_date"]).min().to_pydatetime()
    window_to = pd.to_datetime(
        slate["start_date"]
    ).max().to_pydatetime() + timedelta(hours=6)
    snapshot = client.get_nfl_odds(window_from, window_to)
    offers = flatten_offers(
        snapshot.events,
        match_frame,
        snapshot.fetched_at,
        bool(snapshot.configured_bookmakers),
    )
    store.write_processed(
        offers, "market_offers", f"{season}_{week:02d}.parquet"
    )
    backtest = store.read_processed("calibration", "predictions.parquet")
    distribution = fit_margin_residual_distribution(
        (backtest["actual_margin"] - backtest["model_margin"]).to_numpy()
    )
    comparisons = compare_priced_offers(projections, offers, distribution)
    store.write_processed(
        comparisons, "market_comparisons", f"{season}_{week:02d}.parquet"
    )
    print(
        f"odds {season} week {week}: {len(offers)} offers, "
        f"{len(comparisons)} comparisons, "
        f"requests remaining {snapshot.requests_remaining}"
    )


def run_publish(args) -> None:
    from backend import db, publish
    from backend.config import BACKTEST_PUBLISH_FLOOR
    from backend.etl import store

    season, week = resolve_week(args)
    stem = f"{season}_{week:02d}.parquet"
    projections = store.read_processed("projections", stem)
    try:
        market = store.read_processed("market_comparisons", stem)
    except FileNotFoundError:
        market = publish.fallback_market_comparisons(projections)
        print("no odds snapshot; publishing nflverse-line market comparisons")

    if args.projections_only:
        counts = publish.publish_week(
            db.engine, season, week,
            ratings=None,
            unit_ratings=None,
            projections=projections,
            market_comparisons=market,
            backtest=None,
        )
    else:
        ratings = store.read_processed("ratings", stem)
        try:
            unit_ratings = store.read_processed("unit_ratings", stem)
        except FileNotFoundError:
            unit_ratings = None
        backtest = (
            None
            if args.skip_backtest
            else publish.build_backtest_frame(BACKTEST_PUBLISH_FLOOR)
        )
        counts = publish.publish_week(
            db.engine, season, week,
            ratings=ratings,
            unit_ratings=unit_ratings,
            projections=projections,
            market_comparisons=market,
            backtest=backtest,
        )
    for table, count in counts.items():
        print(f"{table}: {count} rows")


def run_features(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.features.drives import build_team_games
    from backend.features.qb import build_qb_games
    from backend.features.scoring import build_model_games
    from backend.features.units import build_unit_games

    problems: list[str] = []
    schedules = store.read_raw("schedules.parquet")
    try:
        ngs_passing = store.read_raw("ngs_passing.parquet")
    except FileNotFoundError:
        ngs_passing = None
    for season in sorted(args.seasons):
        try:
            try:
                pbp = store.read_raw("pbp", f"{season}.parquet")
            except FileNotFoundError:
                print(f"features {season}: no pbp yet, skipping")
                continue
            season_schedules = schedules[schedules["season"].eq(season)]
            team_games = build_team_games(pbp, season_schedules)
            model_games = build_model_games(team_games, season_schedules)
            store.write_processed(model_games, "team_games", f"{season}.parquet")
            store.write_processed(
                build_qb_games(pbp), "qb_games", f"{season}.parquet"
            )
            try:
                pfr_pass = store.read_raw("pfr_pass", f"{season}.parquet")
            except FileNotFoundError:
                pfr_pass = None
            store.write_processed(
                build_unit_games(pbp, pfr_pass, ngs_passing),
                "unit_games",
                f"{season}.parquet",
            )
            print(f"features {season}: {len(model_games)} games")
        except Exception as error:  # noqa: BLE001 - report and continue
            problems.append(f"{season}: {error}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)
