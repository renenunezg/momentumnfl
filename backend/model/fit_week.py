"""One production week: engine fit on games before the week, QB and rest
layers for the slate, market blend, ratings and projections out. Shared by
the fit CLI and the calibration walk-forward."""

from datetime import datetime, timezone

import pandas as pd

from backend.etl import store
from backend.features import qb as qb_features
from backend.model import qb_adjustment as qb_layer
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    JointScoringConfig,
    JointScoringFit,
    fit_joint_scoring,
)
from backend.model.outputs import GameProjection, TeamRating
from backend.model.projections import LayerConfig, assemble_projections


def load_team_names() -> dict[str, str]:
    teams = store.read_raw("teams.parquet")
    name_column = "team_name" if "team_name" in teams.columns else "team"
    return dict(zip(teams["team_abbr"], teams[name_column]))


def load_season_games(season: int) -> pd.DataFrame:
    return store.read_processed("team_games", f"{season}.parquet")


def load_qb_games(seasons: list[int]) -> pd.DataFrame:
    frames = [
        store.read_processed("qb_games", f"{season}.parquet")
        for season in seasons
        if f"{season}" in [n for n in store.processed_names("qb_games")]
    ]
    return pd.concat(frames, ignore_index=True)


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


def load_game_index(seasons: list[int]) -> pd.DataFrame:
    """game_id, season, model_week, start_date across seasons with features."""
    available = set(store.processed_names("team_games"))
    frames = [
        store.read_processed("team_games", f"{season}.parquet")[
            ["game_id", "season", "model_week", "start_date"]
        ]
        for season in seasons
        if str(season) in available
    ]
    return pd.concat(frames, ignore_index=True)


def compute_qb_adjustments(
    season: int,
    week: int,
    slate: pd.DataFrame,
    games: pd.DataFrame,
    qb_games: pd.DataFrame,
    depth_charts: pd.DataFrame,
    config: JointScoringConfig,
    layer_config: LayerConfig,
    actual_starters: bool = False,
    game_index: pd.DataFrame | None = None,
) -> dict[str, tuple[float, float]]:
    """game_id -> (home_adj, away_adj). With actual_starters (backtest), the
    starter is who actually started; otherwise the expected starter.

    QB values use career history across seasons (game_index); baselines use
    only the current season's training window, matching the rating fit."""
    if game_index is None:
        game_index = load_game_index(list(range(2015, season + 1)))
    eligible = game_index[
        (game_index["season"] < season)
        | (
            game_index["season"].eq(season)
            & (game_index["model_week"] < week)
        )
    ]
    all_games = eligible[["game_id", "season", "model_week", "start_date"]]
    played_ids = games.loc[games["model_week"] < week, "game_id"]
    qb_history = qb_games[qb_games["game_id"].isin(set(all_games["game_id"]))]
    values = qb_layer.qb_values(
        qb_history, all_games, layer_config.qb_span_dropbacks
    )
    weights = recency_by_game(games, week, config)
    baselines = qb_layer.team_baseline_values(
        values, qb_history, played_ids, weights
    )

    history_for_value = qb_history
    if actual_starters:
        slate_qb = qb_games[
            qb_games["game_id"].isin(set(slate["game_id"].astype(str)))
            & qb_games["started"]
        ]
        starter_of = {
            (row.game_id, row.team): row.passer_player_id
            for row in slate_qb.itertuples()
        }

        def starter(game_id: str, team: str) -> str | None:
            return starter_of.get((game_id, team))

    else:
        expected = qb_features.expected_starters(
            qb_history, all_games, depth_charts, season, week
        )

        def starter(game_id: str, team: str) -> str | None:
            return expected.get(team)

    adjustments: dict[str, tuple[float, float]] = {}
    for game in slate.itertuples():
        game_id = str(game.game_id)
        side_adjustments = []
        for team in (str(game.home_team), str(game.away_team)):
            passer = starter(game_id, team)
            baseline = float(baselines.get(team, 0.0))
            if passer is None:
                side_adjustments.append(0.0)
                continue
            value = qb_layer.latest_qb_value(
                history_for_value,
                all_games,
                passer,
                layer_config.qb_span_dropbacks,
            )
            side_adjustments.append(
                qb_layer.game_qb_adjustment(value, baseline)
            )
        adjustments[game_id] = (side_adjustments[0], side_adjustments[1])
    return adjustments


def build_prior_means(season: int) -> dict[str, tuple[float, float]] | None:
    """Preseason prior means built fresh from the previous season, so runs
    on a clean checkout carry the offseason prior without stored state."""
    from backend.model.preseason import build_preseason_prior

    try:
        return build_preseason_prior(season).strength_prior_means()
    except FileNotFoundError:
        return None


def fit_and_project(
    season: int,
    week: int,
    as_of: datetime | None = None,
    config: JointScoringConfig = DEFAULT_CONFIG,
    layer_config: LayerConfig = LayerConfig(),
    actual_starters: bool = False,
    strength_prior_means: dict[str, tuple[float, float]] | None = None,
    slate: pd.DataFrame | None = None,
    use_preseason_prior: bool = True,
) -> tuple[JointScoringFit, list[TeamRating], list[GameProjection]]:
    as_of = as_of or datetime.now(timezone.utc)
    if strength_prior_means is None and use_preseason_prior:
        strength_prior_means = build_prior_means(season)
    games = load_season_games(season)
    if slate is None:
        schedules = store.read_raw("schedules.parquet")
        schedules = schedules[
            schedules["season"].eq(season) & schedules["week"].eq(week)
        ].copy()
        schedules["neutral_site"] = schedules["location"].eq("Neutral")
        from backend.features.drives import _kickoff_utc

        schedules["start_date"] = _kickoff_utc(schedules)
        slate = schedules
    fit = fit_joint_scoring(
        games, week, as_of, config, strength_prior_means
    )
    team_names = load_team_names()
    qb_games = load_qb_games(list(range(2015, season + 1)))
    try:
        depth_charts = store.read_raw("depth_charts", f"{season}.parquet")
    except FileNotFoundError:
        depth_charts = pd.DataFrame(
            columns=["season", "week", "position", "team", "gsis_id"]
        )
    qb_adjustments = compute_qb_adjustments(
        season, week, slate, games, qb_games, depth_charts,
        config, layer_config, actual_starters,
    )
    market = {
        str(row.game_id): -float(row.spread_line)
        for row in slate.itertuples()
        if pd.notna(getattr(row, "spread_line", None))
    }
    projections = assemble_projections(
        fit, slate, as_of, team_names, qb_adjustments, market, layer_config
    )
    return fit, fit.ratings(team_names), projections
