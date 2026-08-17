"""Assemble published game projections: engine numbers, then the QB and rest
layers on expected points, then the market blend (total-invariant shift)."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from backend.model.joint_scoring import MODEL_VERSION, JointScoringFit
from backend.model.market_blend import blend_margin
from backend.model.outputs import GameProjection

# Selected by the calibrate walk-forward (dev 2016-2021): the rest signal is
# already priced into the market line at this blend weight, so its own
# coefficient tuned to zero; the market weight tuned to the 0.5 cap.
REST_POINTS_PER_DAY = 0.0
REST_CLIP_DAYS = 7.0
DEFAULT_MARKET_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class LayerConfig:
    market_weight: float = DEFAULT_MARKET_WEIGHT
    rest_points_per_day: float = REST_POINTS_PER_DAY
    qb_span_dropbacks: float = 250.0


def rest_adjustment(
    home_rest: float | None, away_rest: float | None, points_per_day: float
) -> float:
    if home_rest is None or away_rest is None:
        return 0.0
    if np.isnan(home_rest) or np.isnan(away_rest):
        return 0.0
    return points_per_day * float(
        np.clip(home_rest - away_rest, -REST_CLIP_DAYS, REST_CLIP_DAYS)
    )


def assemble_projections(
    fit: JointScoringFit,
    schedule: pd.DataFrame,
    as_of: datetime,
    team_names: dict[str, str],
    qb_adjustments: dict[str, tuple[float, float]] | None = None,
    market_home_spreads: dict[str, float] | None = None,
    config: LayerConfig = LayerConfig(),
) -> list[GameProjection]:
    """schedule rows need: game_id, season, week, home_team, away_team,
    neutral_site, and optionally start_date, div_game, home_rest, away_rest.
    qb_adjustments maps game_id -> (home_adj, away_adj) in points.
    market_home_spreads maps game_id -> sportsbook home line."""
    qb_adjustments = qb_adjustments or {}
    market_home_spreads = market_home_spreads or {}
    projections = []
    for game in schedule.itertuples():
        game_id = str(game.game_id)
        (
            expected_home,
            expected_away,
            home_field,
            margin_sd,
            total_sd,
            correlation,
        ) = fit.engine_projection(game)

        home_qb, away_qb = qb_adjustments.get(game_id, (0.0, 0.0))
        rest = rest_adjustment(
            getattr(game, "home_rest", None),
            getattr(game, "away_rest", None),
            config.rest_points_per_day,
        )
        expected_home += home_qb + 0.5 * rest
        expected_away += away_qb - 0.5 * rest
        expected_home = max(expected_home, 0.0)
        expected_away = max(expected_away, 0.0)
        pure_margin = expected_home - expected_away

        market_spread = market_home_spreads.get(game_id)
        market_margin = None if market_spread is None else -market_spread
        published_margin = blend_margin(
            pure_margin, market_margin, config.market_weight
        )
        shift = 0.5 * (published_margin - pure_margin)
        expected_home += shift
        expected_away -= shift

        start_date = getattr(game, "start_date", None)
        if pd.isna(start_date):
            start_date = None
        div_game = getattr(game, "div_game", None)
        projections.append(
            GameProjection(
                season=int(game.season),
                week=int(game.week),
                as_of=as_of,
                model_version=MODEL_VERSION,
                game_id=game_id,
                start_date=start_date,
                home_team_abbr=str(game.home_team),
                home_team=team_names.get(str(game.home_team), str(game.home_team)),
                away_team_abbr=str(game.away_team),
                away_team=team_names.get(str(game.away_team), str(game.away_team)),
                neutral_site=bool(game.neutral_site),
                div_game=None if div_game is None else bool(div_game),
                home_field_points=home_field,
                expected_home_points=float(expected_home),
                expected_away_points=float(expected_away),
                home_qb_adjustment=float(home_qb),
                away_qb_adjustment=float(away_qb),
                rest_adjustment=float(rest),
                pure_home_margin=float(pure_margin),
                market_home_spread=(
                    None if market_spread is None else float(market_spread)
                ),
                market_weight=(
                    0.0 if market_margin is None else float(
                        min(config.market_weight, 0.5)
                    )
                ),
                margin_sd=margin_sd,
                total_sd=total_sd,
                margin_total_correlation=correlation,
                degrees_of_freedom=fit.config.student_t_degrees_of_freedom,
            )
        )
    return projections
