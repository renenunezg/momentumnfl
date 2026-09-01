"""One production week: engine fit on games before the week, QB and rest
layers for the slate, market blend, ratings and projections out."""

from datetime import UTC, datetime

import pandas as pd

from backend.config import HISTORY_START_SEASON
from backend.etl import store
from backend.features import qb as qb_features
from backend.features.drives import kickoff_utc
from backend.model import qb_adjustment as qb_layer
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    JointScoringConfig,
    JointScoringFit,
    fit_joint_scoring,
)
from backend.model.outputs import GameProjection, TeamRating
from backend.model.preseason import build_preseason_prior
from backend.model.projections import LayerConfig, assemble_projections


def recency_by_game(
    games: pd.DataFrame, forecast_week: int, config: JointScoringConfig
) -> dict[str, float]:
    training = games[games["model_week"] < forecast_week]
    if training.empty:
        return {}
    latest = int(training["model_week"].max())
    return {
        str(row.game_id): 0.5
        ** ((latest - row.model_week) / config.rating_half_life_weeks)
        for row in training.itertuples()
    }


def compute_qb_adjustments(
    season: int,
    week: int,
    slate: pd.DataFrame,
    games: pd.DataFrame,
    qb_games: pd.DataFrame,
    depth_charts: pd.DataFrame,
    config: JointScoringConfig,
    layer_config: LayerConfig,
) -> dict[str, tuple[float, float]]:
    """game_id -> (home_adj, away_adj) for the expected starters.

    QB values use career history across seasons (game_index); baselines use
    only the current season's training window, matching the rating fit."""
    game_index = store.game_index(list(range(HISTORY_START_SEASON, season + 1)))
    eligible = game_index[
        (game_index["season"] < season)
        | (game_index["season"].eq(season) & (game_index["model_week"] < week))
    ]
    played_ids = games.loc[games["model_week"] < week, "game_id"]
    qb_history = qb_games[qb_games["game_id"].isin(set(eligible["game_id"]))]
    span = layer_config.qb_span_dropbacks
    values = qb_layer.qb_values(qb_history, eligible, span)
    weights = recency_by_game(games, week, config)
    baselines = qb_layer.team_baseline_values(values, qb_history, played_ids, weights)
    latest, replacement = qb_layer.latest_qb_values(qb_history, eligible, span)
    expected = qb_features.expected_starters(
        qb_history, eligible, depth_charts, season, week
    )

    adjustments: dict[str, tuple[float, float]] = {}
    for game in slate.itertuples():
        sides = []
        for team in (str(game.home_team), str(game.away_team)):
            passer = expected.get(team)
            if passer is None:
                sides.append(0.0)
                continue
            value = latest.get(passer, replacement)
            sides.append(value - float(baselines.get(team, 0.0)))
        adjustments[str(game.game_id)] = (sides[0], sides[1])
    return adjustments


def build_prior_means(season: int) -> dict[str, tuple[float, float]] | None:
    """Preseason prior means built fresh from the previous season, so runs
    on a clean checkout carry the offseason prior without stored state."""
    try:
        return build_preseason_prior(season).strength_prior_means()
    except FileNotFoundError:
        return None


def load_depth_charts(season: int) -> pd.DataFrame:
    try:
        return store.read_raw("depth_charts", f"{season}.parquet")
    except FileNotFoundError:
        return pd.DataFrame(columns=["season", "week", "position", "team", "gsis_id"])


def week_slate(season: int, week: int) -> pd.DataFrame:
    schedules = store.read_raw("schedules.parquet")
    slate = schedules[
        schedules["season"].eq(season) & schedules["week"].eq(week)
    ].copy()
    slate["neutral_site"] = slate["location"].eq("Neutral")
    slate["start_date"] = kickoff_utc(slate)
    return slate


def market_home_spreads(slate: pd.DataFrame) -> dict[str, float]:
    """game_id -> sportsbook home line from the nflverse schedule."""
    return {
        str(row.game_id): -float(row.spread_line)
        for row in slate.itertuples()
        if pd.notna(getattr(row, "spread_line", None))
    }


def fit_and_project(
    season: int,
    week: int,
    as_of: datetime | None = None,
    config: JointScoringConfig = DEFAULT_CONFIG,
    layer_config: LayerConfig = LayerConfig(),
) -> tuple[JointScoringFit, list[TeamRating], list[GameProjection]]:
    as_of = as_of or datetime.now(UTC)
    games = store.season_games(season)
    slate = week_slate(season, week)
    fit = fit_joint_scoring(games, week, as_of, config, build_prior_means(season))
    team_names = store.team_names()
    qb_games = store.qb_games(list(range(HISTORY_START_SEASON, season + 1)))
    qb_adjustments = compute_qb_adjustments(
        season,
        week,
        slate,
        games,
        qb_games,
        load_depth_charts(season),
        config,
        layer_config,
    )
    projections = assemble_projections(
        fit,
        slate,
        as_of,
        team_names,
        qb_adjustments,
        market_home_spreads(slate),
        layer_config,
    )
    return fit, fit.ratings(team_names), projections
