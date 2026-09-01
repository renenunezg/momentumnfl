"""The only Supabase writer. Writes use per-key DELETE + append (TRUNCATE +
append for the full backtest refresh) so RLS, policies, and indexes survive;
to_sql(if_exists="replace") would drop them. Timestamps become tz-aware
datetimes and every NaN becomes None: Postgres double precision would accept
a literal NaN, but PostgREST cannot serialize it to JSON, so NULL is the only
safe missing value."""

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.etl import store

TEAMS_COLUMNS = [
    "team_abbr", "team", "color", "alternate_color", "logo_light", "logo_dark",
]
TEAM_RATINGS_COLUMNS = [
    "season", "week", "as_of", "model_version", "team_abbr", "team",
    "conference", "division", "offense_points", "defense_points",
    "power_rating", "scoring_environment", "expected_drives",
    "power_rating_sd", "missing_input_count",
]
TEAM_UNIT_RATINGS_COLUMNS = [
    "season", "week", "as_of", "model_version", "team_abbr", "team",
    "rush_offense", "pass_offense", "rush_defense", "pass_defense",
    "pass_block", "run_block", "special_teams",
]
GAME_PROJECTIONS_COLUMNS = [
    "game_id", "season", "week", "as_of", "model_version", "start_date",
    "home_team_abbr", "home_team", "away_team_abbr", "away_team",
    "neutral_site", "div_game", "home_field_points", "expected_home_points",
    "expected_away_points", "home_qb_adjustment", "away_qb_adjustment",
    "rest_adjustment", "pure_home_margin", "pure_home_spread",
    "market_home_spread", "market_weight", "home_margin", "home_spread",
    "model_total", "margin_sd", "total_sd", "margin_total_correlation",
    "distribution", "degrees_of_freedom",
]
MARKET_COMPARISONS_COLUMNS = [
    "game_id", "start_date", "home_team", "away_team", "model_home_spread",
    "model_total", "margin_sd", "total_sd", "model_as_of", "market_available",
    "priced_offer_available", "executable_offer_available", "review_status",
    "recommendation_status", "best_offer_market", "best_offer_selection",
    "best_offer_point", "best_offer_price", "best_offer_provider",
    "best_offer_provider_key", "best_offer_provider_last_update",
    "best_offer_event_link", "best_offer_market_link", "best_offer_bet_link",
    "best_offer_edge_points", "best_offer_edge_standardized",
    "best_offer_model_cover_probability", "best_offer_model_fair_price",
    "best_offer_expected_value_per_unit",
]
MARKET_SNAPSHOTS_COLUMNS = [
    "game_id", "season", "week", "fetched_at", "home_spread", "total",
    "spread_books", "total_books",
]
BACKTEST_COLUMNS = [
    "game_id", "season", "week", "week_index", "season_type", "home_team",
    "away_team", "neutral_site", "home_points", "away_points",
    "closing_spread", "model_margin", "pure_model_margin", "actual_margin",
]

_TIMESTAMP_COLUMNS = {
    "as_of", "start_date", "model_as_of", "best_offer_provider_last_update",
    "fetched_at",
}

LEGACY_TEAM_NAMES = {
    "Oakland Raiders",
    "San Diego Chargers",
    "St. Louis Rams",
    "Washington Redskins",
    "Washington Football Team",
}


def _prepare(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.reindex(columns=columns)
    for column in columns:
        if column in _TIMESTAMP_COLUMNS:
            out[column] = pd.to_datetime(out[column], utc=True, format="ISO8601")
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    return out


def build_teams_frame() -> pd.DataFrame:
    raw = store.read_raw("teams.parquet")
    raw = raw[~raw["team_name"].isin(LEGACY_TEAM_NAMES)]
    raw = raw.drop_duplicates("team_abbr")
    return pd.DataFrame(
        {
            "team_abbr": raw["team_abbr"],
            "team": raw["team_name"],
            "color": raw["team_color"],
            "alternate_color": raw["team_color2"],
            "logo_light": raw["team_logo_espn"],
            "logo_dark": raw["team_logo_espn"],
        }
    )


def team_metadata() -> pd.DataFrame:
    raw = store.read_raw("teams.parquet")
    raw = raw[~raw["team_name"].isin(LEGACY_TEAM_NAMES)]
    raw = raw.drop_duplicates("team_abbr")
    return raw.set_index("team_abbr")[["team_conf", "team_division"]]


def publish_week(
    engine,
    season: int,
    week: int,
    ratings: pd.DataFrame | None,
    unit_ratings: pd.DataFrame | None,
    projections: pd.DataFrame,
    market_comparisons: pd.DataFrame | None,
    backtest: pd.DataFrame | None,
    market_snapshot: pd.DataFrame | None = None,
) -> dict[str, int]:
    """One transaction; read-back counts returned for the caller to print.
    ratings=None (the projections-only refresh) leaves teams and both ratings
    tables untouched. market_snapshot appends to the archive keyed by
    (game_id, fetched_at), so a re-run of the same snapshot is idempotent
    and earlier snapshots are never touched."""
    if ratings is not None:
        teams = build_teams_frame()
        metadata = team_metadata()
        ratings = ratings.copy()
        ratings["conference"] = ratings["team_abbr"].map(metadata["team_conf"])
        ratings["division"] = ratings["team_abbr"].map(
            metadata["team_division"]
        )

    game_ids = projections["game_id"].astype(str).tolist()
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        if ratings is not None:
            conn.execute(text("DELETE FROM teams"))
            _prepare(teams, TEAMS_COLUMNS).to_sql(
                "teams", con=conn, if_exists="append", index=False
            )
            conn.execute(
                text(
                    "DELETE FROM team_ratings WHERE season = :s AND week = :w"
                ),
                {"s": season, "w": week},
            )
            _prepare(ratings, TEAM_RATINGS_COLUMNS).to_sql(
                "team_ratings", con=conn, if_exists="append", index=False
            )
        if unit_ratings is not None:
            conn.execute(
                text(
                    "DELETE FROM team_unit_ratings "
                    "WHERE season = :s AND week = :w"
                ),
                {"s": season, "w": week},
            )
            _prepare(unit_ratings, TEAM_UNIT_RATINGS_COLUMNS).to_sql(
                "team_unit_ratings", con=conn, if_exists="append", index=False
            )
        # Delete projections by game id, not by week, so a game that moved
        # weeks between publishes cannot survive as a duplicate row.
        conn.execute(
            text("DELETE FROM game_projections WHERE game_id = ANY(:ids)"),
            {"ids": game_ids},
        )
        _prepare(projections, GAME_PROJECTIONS_COLUMNS).to_sql(
            "game_projections", con=conn, if_exists="append", index=False
        )
        if market_comparisons is not None:
            conn.execute(text("DELETE FROM market_comparisons"))
            _prepare(market_comparisons, MARKET_COMPARISONS_COLUMNS).to_sql(
                "market_comparisons", con=conn, if_exists="append", index=False
            )
        if market_snapshot is not None and not market_snapshot.empty:
            snapshot = _prepare(market_snapshot, MARKET_SNAPSHOTS_COLUMNS)
            conn.execute(
                text(
                    "DELETE FROM market_snapshots "
                    "WHERE game_id = ANY(:ids) AND fetched_at = ANY(:ts)"
                ),
                {
                    "ids": snapshot["game_id"].tolist(),
                    "ts": [
                        t.to_pydatetime()
                        for t in snapshot["fetched_at"].drop_duplicates()
                    ],
                },
            )
            snapshot.to_sql(
                "market_snapshots", con=conn, if_exists="append", index=False
            )
        if backtest is not None:
            conn.execute(text("TRUNCATE TABLE backtest_predictions"))
            _prepare(backtest, BACKTEST_COLUMNS).to_sql(
                "backtest_predictions", con=conn, if_exists="append",
                index=False, chunksize=1000,
            )
        for table in (
            "teams", "team_ratings", "team_unit_ratings", "game_projections",
            "market_comparisons", "backtest_predictions", "market_snapshots",
        ):
            counts[table] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()
    return counts


def fallback_market_comparisons(projections: pd.DataFrame) -> pd.DataFrame:
    """Reduced market_comparisons from nflverse schedule lines when no Odds
    API snapshot exists: model vs line, no prices or links."""
    out = pd.DataFrame(
        {
            "game_id": projections["game_id"],
            "start_date": projections["start_date"],
            "home_team": projections["home_team"],
            "away_team": projections["away_team"],
            "model_home_spread": projections["home_spread"],
            "model_total": projections["model_total"],
            "margin_sd": projections["margin_sd"],
            "total_sd": projections["total_sd"],
            "model_as_of": projections["as_of"],
            "market_available": projections["market_home_spread"].notna(),
            "priced_offer_available": False,
            "executable_offer_available": False,
            "review_status": "no_priced_offer",
            "recommendation_status": "not_recommended",
        }
    )
    return out


def build_backtest_frame(floor_season: int) -> pd.DataFrame:
    predictions = store.read_processed("calibration", "predictions.parquet")
    predictions = predictions[predictions["season"].ge(floor_season)].copy()
    return pd.DataFrame(
        {
            "game_id": predictions["game_id"],
            "season": predictions["season"],
            "week": predictions["week"],
            "week_index": predictions["model_week"],
            "season_type": predictions["season_type"],
            "home_team": predictions["home_team"],
            "away_team": predictions["away_team"],
            "neutral_site": predictions["neutral_site"],
            "home_points": predictions["home_points"].astype(int),
            "away_points": predictions["away_points"].astype(int),
            "closing_spread": predictions["closing_spread"],
            "model_margin": predictions["model_margin"],
            "pure_model_margin": predictions["pure_model_margin"],
            "actual_margin": predictions["actual_margin"],
        }
    )
