"""momentumnfl command line. Heavy imports stay inside each subcommand so
startup is instant and a broken optional dependency only breaks its own
command. Commands collect problems and exit 1 at the end rather than
aborting on the first."""

import argparse

from backend.config import HISTORY_START_SEASON, SEASONS


def main() -> None:
    parser = argparse.ArgumentParser(prog="backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="pull raw nflverse data")
    ingest_parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)

    history_parser = subparsers.add_parser(
        "bootstrap-history", help="build missing cached historical core features"
    )
    history_parser.add_argument("--through", type=int, required=True)

    refresh_inputs_parser = subparsers.add_parser(
        "refresh-inputs", help="pull schedules, teams, and current depth charts"
    )
    refresh_inputs_parser.add_argument("--season", type=int, required=True)

    features_parser = subparsers.add_parser(
        "features", help="build team_games and qb_games parquet"
    )
    features_parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    features_parser.add_argument("--incremental", action="store_true")
    features_parser.add_argument("--lookback-weeks", type=int, default=2)

    fit_parser = subparsers.add_parser(
        "fit", help="fit in-season ratings and project a week"
    )
    fit_parser.add_argument("--season", type=int)
    fit_parser.add_argument("--week", type=int)
    fit_parser.add_argument("--projections-only", action="store_true")

    preseason_parser = subparsers.add_parser(
        "preseason", help="build the week-1 prior, ratings, and projections"
    )
    preseason_parser.add_argument("--season", type=int, required=True)

    season_wins_parser = subparsers.add_parser(
        "season-wins", help="project full regular-season wins"
    )
    season_wins_parser.add_argument("--season", type=int, required=True)

    subparsers.add_parser("calibrate", help="walk-forward hyperparameter search")
    validate_qb_parser = subparsers.add_parser(
        "validate-qb",
        help="compare QB layers on frozen engine forecasts without writes",
    )
    validate_qb_parser.add_argument("--context", action="store_true")
    validate_qb_parser.add_argument("--memory", action="store_true")
    qbs_parser = subparsers.add_parser(
        "qbs", help="inspect quarterback strengths and starter/backup swap adjustments"
    )
    qbs_parser.add_argument("--season", type=int)
    qbs_parser.add_argument("--week", type=int)
    qbs_parser.add_argument("--team", required=True)
    qbs_parser.add_argument("--context", action="store_true")

    odds_parser = subparsers.add_parser(
        "odds", help="snapshot Odds API offers for a week"
    )
    odds_parser.add_argument("--season", type=int)
    odds_parser.add_argument("--week", type=int)

    upcoming_parser = subparsers.add_parser(
        "upcoming", help="succeed when an unplayed game is inside a time window"
    )
    upcoming_parser.add_argument("--season", type=int, required=True)
    upcoming_parser.add_argument("--hours", type=float, default=30.0)

    publish_parser = subparsers.add_parser(
        "publish", help="publish a week to the nfl schema"
    )
    publish_parser.add_argument("--season", type=int)
    publish_parser.add_argument("--week", type=int)
    publish_parser.add_argument("--skip-backtest", action="store_true")
    publish_parser.add_argument("--projections-only", action="store_true")

    args = parser.parse_args()
    COMMANDS[args.command](args)


def run_ingest(args) -> None:
    from backend.etl import ingest

    problems: list[str] = []
    seasons = sorted(args.seasons)
    ingest.ingest_shared(seasons)
    print(f"ingested shared (schedules, teams, ngs) for {seasons[0]}-{seasons[-1]}")
    for season in seasons:
        problems.extend(ingest.ingest_season(season))
        print(f"ingested {season}")
    _exit_on_problems(problems)


def _exit_on_problems(problems: list[str]) -> None:
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        raise SystemExit(1)


def run_bootstrap_history(args) -> None:
    from backend.pipeline import bootstrap_history

    bootstrap_history(args.through)


def run_refresh_inputs(args) -> None:
    from backend.etl.ingest import ingest_projection_inputs

    _exit_on_problems(ingest_projection_inputs(args.season))
    print(f"refreshed {args.season} schedules, teams, and depth charts")


def resolve_week(args) -> tuple[int, int]:
    """(season, week) from flags, or the week of the next unplayed kickoff.
    Ordering by kickoff rather than week number keeps a postponed game from
    dragging the slate back to a week that has otherwise been played."""
    if args.season is not None and args.week is not None:
        return args.season, args.week
    from backend.etl import store
    from backend.features.drives import kickoff_utc

    schedules = store.read_raw("schedules.parquet")
    unplayed = schedules[schedules["home_score"].isna()]
    if args.season is not None:
        unplayed = unplayed[unplayed["season"].eq(args.season)]
    if unplayed.empty:
        raise SystemExit("PROBLEM: no unplayed games to infer a week from")
    first = unplayed.loc[kickoff_utc(unplayed).idxmin()]
    season = args.season if args.season is not None else int(first["season"])
    week = args.week if args.week is not None else int(first["week"])
    return season, week


def _write_week(frame, directory: str, season: int, week: int) -> None:
    from backend.etl import store

    store.write_processed(frame, directory, f"{season}_{week:02d}.parquet")


def run_fit(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.model.fit_week import fit_and_project, recency_by_game
    from backend.model.unit_ratings import fit_unit_ratings

    args.season, args.week = resolve_week(args)
    # Before any current-season game has been played there is nothing to fit;
    # the preseason prior is the correct producer for that window, so the
    # weekly cron stays valid across the season boundary.
    if str(args.season) not in store.processed_names("team_games"):
        print(f"fit: no played {args.season} games yet, running preseason")
        run_preseason(args)
        return

    fit, ratings, projections = fit_and_project(args.season, args.week)
    projections_df = pd.DataFrame(
        [projection.to_record() for projection in projections]
    )
    _write_week(projections_df, "projections", args.season, args.week)
    if args.projections_only:
        print(
            f"projected {args.season} week {args.week}: "
            f"{len(projections_df)} games from the cached weekly state"
        )
        return

    ratings_df = pd.DataFrame([rating.to_record() for rating in ratings])
    _write_week(ratings_df, "ratings", args.season, args.week)
    games = store.season_games(args.season)
    unit_games = store.read_processed("unit_games", f"{args.season}.parquet")
    units = fit_unit_ratings(
        unit_games,
        games,
        args.week,
        fit.as_of,
        recency_by_game(games, args.week, fit.config),
    )
    units_df = pd.DataFrame(units.to_records(store.team_names()))
    _write_week(units_df, "unit_ratings", args.season, args.week)
    print(
        f"fit {args.season} week {args.week}: {len(ratings_df)} ratings, "
        f"{len(projections_df)} projections, {len(units_df)} unit ratings, "
        f"hfa {fit.hfa_points:.2f} pts, base drives {fit.base_drives:.1f}"
    )
    print(ratings_df.head(5)[["team_abbr", "power_rating"]].to_string(index=False))


def run_preseason(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.model.fit_week import (
        compute_qb_adjustments,
        load_depth_charts,
        market_home_spreads,
        week_slate,
    )
    from backend.model.preseason import MODEL_VERSION, build_preseason_prior
    from backend.model.projections import LayerConfig, assemble_projections

    prior = build_preseason_prior(args.season)
    team_names = store.team_names()
    ratings = prior.ratings(team_names)
    ratings_df = pd.DataFrame([rating.to_record() for rating in ratings])
    ratings_df["model_version"] = MODEL_VERSION
    ratings_df["missing_input_count"] = ratings_df["team_abbr"].map(
        prior.ratings_frame.set_index("team_abbr")["missing_input_count"]
    )
    if not getattr(args, "projections_only", False):
        _write_week(ratings_df, "ratings", args.season, 1)

    slate = week_slate(args.season, 1)
    if slate.empty:
        print(f"preseason {args.season}: no week-1 schedule yet; ratings only")
        return
    week1_fit = prior.week1_fit()
    qb_games = store.qb_games(list(range(HISTORY_START_SEASON, args.season)))
    qb_adjustments = compute_qb_adjustments(
        args.season,
        1,
        slate,
        qb_games,
        load_depth_charts(args.season),
        week1_fit.config,
        LayerConfig(),
    )
    projections = assemble_projections(
        week1_fit,
        slate,
        prior.as_of,
        team_names,
        qb_adjustments,
        market_home_spreads(slate),
    )
    projections_df = pd.DataFrame(
        [projection.to_record() for projection in projections]
    )
    projections_df["model_version"] = f"{MODEL_VERSION}_qb_v3"
    _write_week(projections_df, "projections", args.season, 1)
    print(
        f"preseason {args.season}: {len(ratings_df)} ratings, "
        f"{len(projections_df)} week-1 projections, "
        f"points/win slope {prior.slope:.2f}"
    )
    print(ratings_df.head(5)[["team_abbr", "power_rating"]].to_string(index=False))


def run_season_wins(args) -> None:
    from backend.etl import store
    from backend.model.season_wins import build_season_forecast

    totals, games = build_season_forecast(args.season)
    store.write_processed(totals, "season_win_totals", f"{args.season}.parquet")
    store.write_processed(games, "season_win_games", f"{args.season}.parquet")
    print(
        f"season wins {args.season}: {len(totals)} teams, {len(games)} remaining games"
    )
    print(
        totals[
            [
                "team_abbr",
                "projected_wins",
                "wins_p10",
                "wins_p90",
                "sportsbook_win_total",
            ]
        ].to_string(index=False)
    )


def run_upcoming(args) -> None:
    from backend.etl import store
    from backend.pipeline import upcoming_games

    schedules = store.read_raw("schedules.parquet")
    games = upcoming_games(schedules, args.season, args.hours)
    if games.empty:
        print(f"no {args.season} kickoff in the next {args.hours:g} hours")
        raise SystemExit(3)
    first = games.iloc[0]
    print(
        f"{len(games)} upcoming game(s); first is {first.game_id} "
        f"at {first.start_date.isoformat()}"
    )


def run_calibrate(args) -> None:
    from backend.model.calibration import run_calibration

    summary = run_calibration()
    print("\n=== selected configuration ===")
    print(summary["engine_config"])
    print(summary["layer_config"])
    print(summary["preseason_config"])
    print(f"use_prior_means: {summary['use_prior_means']}")
    print(f"dev margin log loss: {summary['dev_margin_log_loss']:.4f}")
    print(f"holdout margin log loss: {summary['holdout_margin_log_loss']:.4f}")
    print(f"holdout coverage: {summary['holdout_coverage']}")
    print("\n=== honesty report (dev) ===")
    print(summary["dev_honesty"].to_string())
    print("\n=== honesty report (holdout, untouched) ===")
    print(summary["holdout_honesty"].to_string())


def run_validate_qb(args) -> None:
    from dataclasses import replace

    from backend.etl import store
    from backend.model.calibration import (
        apply_layers,
        margin_log_loss,
        rescore_qb_layer,
        select_qb_layer,
    )
    from backend.model.projections import DEFAULT_MARKET_WEIGHT, LayerConfig

    frozen = store.read_processed("calibration", "predictions.parquet")
    if getattr(args, "memory", False):
        from backend.model.calibration import select_qb_memory

        print(f"Development memory selection: {select_qb_memory(frozen)}")
    rescored = rescore_qb_layer(frozen)
    previous = apply_layers(
        rescore_qb_layer(frozen, anchor_preseason=False), LayerConfig()
    )
    selected = replace(select_qb_layer(rescored), market_weight=DEFAULT_MARKET_WEIGHT)
    candidate = apply_layers(rescored, selected)
    print(f"Development-selected QB layer: {selected}")
    print("Conditional on majority-dropback QBs; engine/artifacts unchanged.")
    print("Historical market QB references include weekly lineup proxies.")
    for label, frame in (
        ("frozen", frozen), ("unanchored", previous), ("candidate", candidate)
    ):
        for split, mask in (
            ("development", frame["season"].between(2016, 2021)),
            ("holdout", frame["season"].between(2022, 2025)),
        ):
            subset = frame[mask]
            actual = subset["actual_margin"]
            pure_mae = (actual - subset["pure_model_margin"]).abs().mean()
            blended_mae = (actual - subset["model_margin"]).abs().mean()
            market_mae = (actual - subset["market_margin"]).abs().mean()
            print(
                f"{label} {split}: {len(subset)} games, pure MAE {pure_mae:.4f}, "
                f"blended MAE {blended_mae:.4f}, closing MAE {market_mae:.4f}, "
                f"blended NLL {margin_log_loss(subset, 1.0, 7.0):.5f}"
            )
    if getattr(args, "context", False):
        from backend.model.calibration import evaluate_qb_context

        print("Context gate (development-selected, no artifact writes):")
        for key, value in evaluate_qb_context(frozen).items():
            print(f"  {key}: {value:.6f}")


def run_qbs(args) -> None:
    import pandas as pd

    from backend.etl import store
    from backend.features.qb import expected_starters
    from backend.model.fit_week import load_depth_charts
    from backend.model.joint_scoring import DEFAULT_CONFIG
    from backend.model.preseason import WIN_TOTAL_BLEND, load_qb_references
    from backend.model.projections import LayerConfig
    from backend.model.qb_adjustment import context_before_week, strength_history

    season, week = resolve_week(args)
    team = args.team.upper()
    index = store.game_index(list(range(HISTORY_START_SEASON, season + 1)))
    games = store.qb_games(list(range(HISTORY_START_SEASON, season + 1)))
    layer = LayerConfig()
    eligible_ids = index.loc[
        (index["season"] < season)
        | (index["season"].eq(season) & (index["model_week"] < week)), "game_id"
    ]
    prior_games = games[games["game_id"].isin(eligible_ids)]
    dropbacks = prior_games.groupby("passer_player_id")["dropbacks"].sum()
    context = context_before_week(
        strength_history(games, index, layer.qb_span_dropbacks),
        season,
        week,
        DEFAULT_CONFIG.rating_half_life_weeks,
        load_qb_references(season),
        WIN_TOTAL_BLEND,
    )
    depth = load_depth_charts(season)
    if "pos_abb" not in depth or team not in depth["team"].values:
        raise SystemExit("Current snapshot depth charts are required for this team")
    roster = depth[depth["team"].eq(team) & depth["pos_abb"].eq("QB")]
    roster = roster[roster["dt"].eq(roster["dt"].max())].sort_values("pos_rank")
    starter = expected_starters(games, index, depth, season, week).get(team)
    rows = []
    for row in roster.itertuples():
        if pd.isna(row.gsis_id):
            continue
        adjustment = context.adjustment(team, row.gsis_id, layer.qb_adjustment_weight)
        rows.append(
            {
                "quarterback": row.player_name,
                "gsis_id": row.gsis_id,
                "starter": row.gsis_id == starter,
                "recorded_dropbacks": int(dropbacks.get(row.gsis_id, 0)),
                "effective_dropbacks": context.effective_dropbacks.get(
                    row.gsis_id, 0.0
                ),
                "estimate_basis": (
                    "observed EPA" if row.gsis_id in dropbacks else "replacement prior"
                ),
                "strength": context.strengths.get(row.gsis_id, 0.0),
                "team_baseline": context.baselines.get(team),
                "adjustment": adjustment,
                "change_from_starter": adjustment - context.adjustment(
                    team, starter, layer.qb_adjustment_weight
                ),
            }
        )
    print(f"{team}, {season} week {week}; depth chart {roster['dt'].max()}")
    print(f"Recorded history begins in {HISTORY_START_SEASON}.")
    print(
        "Strength/baseline: raw points above replacement; "
        f"adjustment = {layer.qb_adjustment_weight:g} * (strength - baseline)."
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if getattr(args, "context", False):
        from backend.features.qb_context import load_context_games
        from backend.model.qb_context import fit_context

        fitted = fit_context(load_context_games(season), season, week)
        comparisons = []
        for row in roster.itertuples():
            if pd.isna(row.gsis_id):
                continue
            contrast = {row.gsis_id: 1.0}
            if starter is not None:
                contrast[starter] = contrast.get(starter, 0.0) - 1.0
            comparisons.append({
                "quarterback": row.player_name,
                "context_vs_average": fitted.strengths.get(row.gsis_id, 0.0),
                "supporting_cast": fitted.supporting_cast.get(row.gsis_id),
                "opponents": fitted.opponents.get(row.gsis_id),
                "hit_sack_pct": 100 * fitted.hit_sack_rates.get(
                    row.gsis_id, float("nan")
                ),
                "swap_vs_starter": sum(
                    w * fitted.strengths.get(q, 0.0) for q, w in contrast.items()
                ),
                "swap_model_sd": fitted.contrast_sd([contrast])[0],
            })
        print("Context diagnostics only; not used in published spreads.")
        print("Positive context means a helpful environment; hits/sacks are a proxy.")
        print("Model SD is parameter uncertainty, not a game prediction interval.")
        print(pd.DataFrame(comparisons).round(3).to_string(index=False))


def run_odds(args) -> None:
    from datetime import timedelta

    import pandas as pd

    from backend.config import STATIC_DIR
    from backend.etl import store
    from backend.features.drives import kickoff_utc
    from backend.odds.client import OddsAPIClient
    from backend.odds.markets import (
        compare_priced_offers,
        consensus_lines,
        flatten_offers,
    )

    season, week = resolve_week(args)
    projections = store.read_processed("projections", f"{season}_{week:02d}.parquet")
    schedules = store.read_raw("schedules.parquet")
    slate = schedules[
        schedules["season"].eq(season) & schedules["week"].eq(week)
    ].copy()
    slate["start_date"] = kickoff_utc(slate)
    # Odds API events carry full team names; map abbrs for matching.
    names = store.team_names()
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
    window_to = pd.to_datetime(slate["start_date"]).max().to_pydatetime() + timedelta(
        hours=6
    )
    snapshot = client.get_nfl_odds(window_from, window_to)
    offers = flatten_offers(
        snapshot.events,
        match_frame,
        snapshot.fetched_at,
        bool(snapshot.configured_bookmakers),
    )
    _write_week(offers, "market_offers", season, week)
    _write_week(consensus_lines(offers, projections), "market_snapshots", season, week)
    distribution = pd.read_csv(STATIC_DIR / "margin_distribution.csv")[
        "probability"
    ].to_numpy()
    comparisons = compare_priced_offers(projections, offers, distribution)
    _write_week(comparisons, "market_comparisons", season, week)
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

    def read_optional(directory: str):
        try:
            return store.read_processed(directory, stem)
        except FileNotFoundError:
            return None

    projections = store.read_processed("projections", stem)
    market = read_optional("market_comparisons")
    if market is None:
        market = publish.fallback_market_comparisons(projections)
        print("no odds snapshot; publishing nflverse-line market comparisons")
    snapshot = read_optional("market_snapshots")

    if args.projections_only:
        ratings = unit_ratings = backtest = None
    else:
        ratings = store.read_processed("ratings", stem)
        unit_ratings = read_optional("unit_ratings")
        backtest = (
            None
            if args.skip_backtest
            else publish.build_backtest_frame(BACKTEST_PUBLISH_FLOOR)
        )
    counts = publish.publish_week(
        db.engine,
        season,
        week,
        ratings=ratings,
        unit_ratings=unit_ratings,
        projections=projections,
        market_comparisons=market,
        backtest=backtest,
        market_snapshot=snapshot,
        season_win_totals=store.read_processed(
            "season_win_totals", f"{season}.parquet"
        ),
    )
    for table, count in counts.items():
        print(f"{table}: {count} rows")


def run_features(args) -> None:
    from backend.etl import store
    from backend.features.drives import build_team_games
    from backend.features.qb import build_qb_games
    from backend.features.scoring import build_model_games
    from backend.features.units import build_unit_games
    from backend.pipeline import incremental_game_ids, write_incremental_features

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
            if season_schedules.empty:
                problems.append(f"{season}: no schedule data")
                continue
            selected_schedules = season_schedules
            if args.incremental:
                existing_ids = []
                for directory in ("team_games", "qb_games", "unit_games"):
                    try:
                        existing = store.read_processed(
                            directory,
                            f"{season}.parquet",
                            columns=["game_id"],
                        )
                        existing_ids.append(set(existing["game_id"].astype(str)))
                    except FileNotFoundError:
                        existing_ids.append(set())
                game_ids = incremental_game_ids(
                    season_schedules, existing_ids, args.lookback_weeks
                )
                if not game_ids:
                    print(f"features {season}: no completed games to update")
                    continue
                pbp = pbp[pbp["game_id"].astype(str).isin(game_ids)]
                selected_schedules = season_schedules[
                    season_schedules["game_id"].astype(str).isin(game_ids)
                ]

            team_games = build_team_games(pbp, selected_schedules)
            model_games = build_model_games(team_games, season_schedules)
            qb_games = build_qb_games(pbp)
            try:
                pfr_pass = store.read_raw("pfr_pass", f"{season}.parquet")
            except FileNotFoundError:
                pfr_pass = None
            if pfr_pass is not None and args.incremental:
                pfr_pass = pfr_pass[pfr_pass["game_id"].astype(str).isin(game_ids)]
            unit_games = build_unit_games(pbp, pfr_pass, ngs_passing)

            if args.incremental:
                write_incremental_features(
                    model_games, "team_games", season, ["start_date", "game_id"]
                )
                write_incremental_features(
                    qb_games,
                    "qb_games",
                    season,
                    ["game_id", "team", "passer_player_id"],
                )
                write_incremental_features(
                    unit_games, "unit_games", season, ["game_id", "team"]
                )
                print(
                    f"features {season}: rebuilt {len(model_games)} games "
                    f"across {len(set(selected_schedules['week']))} week(s)"
                )
            else:
                store.write_processed(model_games, "team_games", f"{season}.parquet")
                store.write_processed(qb_games, "qb_games", f"{season}.parquet")
                store.write_processed(unit_games, "unit_games", f"{season}.parquet")
                print(f"features {season}: {len(model_games)} games")
        except Exception as error:  # noqa: BLE001 - report and continue
            problems.append(f"{season}: {error}")
    _exit_on_problems(problems)


COMMANDS = {
    "ingest": run_ingest,
    "bootstrap-history": run_bootstrap_history,
    "refresh-inputs": run_refresh_inputs,
    "features": run_features,
    "fit": run_fit,
    "preseason": run_preseason,
    "calibrate": run_calibrate,
    "season-wins": run_season_wins,
    "validate-qb": run_validate_qb,
    "qbs": run_qbs,
    "odds": run_odds,
    "upcoming": run_upcoming,
    "publish": run_publish,
}
