"""Assemble published game projections: engine numbers, then the QB and rest
layers on expected points, then the market blend (total-invariant shift)."""

from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite

import numpy as np
import pandas as pd

from backend.model.joint_scoring import JointScoringFit
from backend.model.market_blend import blend_margin, capped_weight
from backend.model.outputs import GameProjection, TeamRating
from backend.model.qb_adjustment import DEFAULT_SPAN_DROPBACKS
from backend.model.totals import TOTALS_LAYER_VERSION, calibrate_total

# Selected by the calibrate walk-forward (dev 2016-2021): the rest signal is
# already priced into the market line at this blend weight, so its own
# coefficient tuned to zero; the market weight tuned to the 0.5 cap.
REST_POINTS_PER_DAY = 0.0
REST_CLIP_DAYS = 7.0
DEFAULT_MARKET_WEIGHT = 0.5
# Selected on development seasons 2016-2021 with the engine held fixed and
# injury-gated starter identification. The QB selection gate scores
# pure-model log loss before the market blend; the 2022-2025 retrospective
# window could not separate 0.25 from 0.75 and did not reject this choice.
DEFAULT_QB_SPAN_DROPBACKS = DEFAULT_SPAN_DROPBACKS
DEFAULT_QB_ADJUSTMENT_WEIGHT = 0.25
# Selected on 2016-2021 pregame confirmed-absence cases with team ratings fixed.
# The locked 2022-2025 retrospective check improved that cohort's margin MAE.
DEFAULT_QB_ABSENCE_WEIGHT = 0.5
# The QB layer version names the starter-identification rule and its weight;
# the preseason producer appends the same suffix to its own version.
QB_LAYER_VERSION = "qb_v7"
MODEL_VERSION = f"nfl_joint_scoring_{QB_LAYER_VERSION}_{TOTALS_LAYER_VERSION}"


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Output-layer parameters shared by production and the calibration
    search, so the backtest scores exactly what gets published."""

    market_weight: float = DEFAULT_MARKET_WEIGHT
    rest_points_per_day: float = REST_POINTS_PER_DAY
    qb_span_dropbacks: float = DEFAULT_QB_SPAN_DROPBACKS
    qb_adjustment_weight: float = DEFAULT_QB_ADJUSTMENT_WEIGHT
    qb_absence_weight: float = DEFAULT_QB_ABSENCE_WEIGHT

    def __post_init__(self) -> None:
        if not isfinite(self.qb_span_dropbacks) or self.qb_span_dropbacks <= 0:
            raise ValueError("qb_span_dropbacks must be finite and positive")
        if not 0 <= self.qb_adjustment_weight <= 1:
            raise ValueError("qb_adjustment_weight must be between 0 and 1")
        if not 0 <= self.qb_absence_weight <= 1:
            raise ValueError("qb_absence_weight must be between 0 and 1")


def rest_adjustment(
    home_rest: float | None, away_rest: float | None, points_per_day: float
) -> float:
    if pd.isna(home_rest) or pd.isna(away_rest):
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
    market_totals: dict[str, float] | None = None,
) -> list[GameProjection]:
    """schedule rows need: game_id, season, week, home_team, away_team,
    neutral_site, and optionally start_date, div_game, home_rest, away_rest.
    qb_adjustments maps game_id -> (home_adj, away_adj) in points.
    market_home_spreads maps game_id -> sportsbook home line.
    market_totals maps game_id -> sportsbook total."""
    qb_adjustments = qb_adjustments or {}
    market_home_spreads = market_home_spreads or {}
    market_totals = market_totals or {}
    projections = []
    for game in schedule.itertuples():
        game_id = str(game.game_id)
        # Preserve the incumbent QB/clipping and margin calculations exactly;
        # apply the common totals correction once after those layers.
        engine = fit.engine_projection(game, stabilize_totals=False)

        home_qb, away_qb = qb_adjustments.get(game_id, (0.0, 0.0))
        rest = rest_adjustment(
            getattr(game, "home_rest", None),
            getattr(game, "away_rest", None),
            config.rest_points_per_day,
        )
        expected_home = max(engine.expected_home + home_qb + 0.5 * rest, 0.0)
        expected_away = max(engine.expected_away + away_qb - 0.5 * rest, 0.0)
        pure_margin = expected_home - expected_away

        market_spread = market_home_spreads.get(game_id)
        market_margin = None if market_spread is None else -market_spread
        published_margin = blend_margin(
            pure_margin, market_margin, config.market_weight
        )
        shift = 0.5 * (published_margin - pure_margin)
        expected_home += shift
        expected_away -= shift
        expected_home, expected_away = fit.stabilize_scores(
            game, expected_home, expected_away, minimum_total=abs(pure_margin)
        )
        # Calibrate only after QB and environment pooling. Protect both the
        # published margin and the pure-margin score reconstruction at zero.
        total = max(
            calibrate_total(expected_home + expected_away, fit.totals_config),
            abs(published_margin),
            abs(pure_margin),
        )
        expected_home = max(0.0, 0.5 * (total + published_margin))
        expected_away = max(0.0, 0.5 * (total - published_margin))

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
                home_field_points=engine.home_field,
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
                    0.0
                    if market_margin is None
                    else capped_weight(config.market_weight)
                ),
                market_total=market_totals.get(game_id),
                margin_sd=engine.margin_sd,
                total_sd=engine.total_sd,
                margin_total_correlation=engine.correlation,
                degrees_of_freedom=fit.config.student_t_degrees_of_freedom,
            )
        )
    return projections


def align_ratings_to_forecast(
    ratings: list[TeamRating], projections: list[GameProjection]
) -> list[TeamRating]:
    """Shift team ratings so they reproduce the week's published lines.

    The fitted ratings carry no market, quarterback, or rest information, so
    their difference plus home field can sit points away from the published
    line. Each game's gap is split evenly between its two teams (the
    minimum-norm solution, which also handles a team with two games), leaving
    teams on a bye unchanged. Offense and defense each take half of a team's
    shift, so the scoring environment is unchanged.
    """
    index = {rating.team_abbr: row for row, rating in enumerate(ratings)}
    games = np.zeros((len(projections), len(ratings)))
    gaps = np.zeros(len(projections))
    for row, game in enumerate(projections):
        home, away = index[game.home_team_abbr], index[game.away_team_abbr]
        games[row, home], games[row, away] = 1.0, -1.0
        gaps[row] = game.home_margin - (
            ratings[home].power_rating
            - ratings[away].power_rating
            + game.home_field_points
        )
    shifts = np.linalg.lstsq(games, gaps, rcond=None)[0] if projections else gaps
    return [
        replace(
            rating,
            offense_points=rating.offense_points + shift / 2.0,
            defense_points=rating.defense_points + shift / 2.0,
            forecast_alignment_points=rating.forecast_alignment_points + shift,
        )
        for rating, shift in zip(ratings, shifts, strict=True)
    ]
