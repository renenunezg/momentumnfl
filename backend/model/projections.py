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
from backend.model.totals import (
    DEFAULT_TOTALS_CONFIG,
    TOTALS_LAYER_VERSION,
    calibrate_total,
)

# Selected by the calibrate walk-forward (dev 2016-2021): the rest signal is
# already priced into the market line at this blend weight, so its own
# coefficient tuned to zero; the market weight tuned to the 0.5 cap.
REST_POINTS_PER_DAY = 0.0
REST_CLIP_DAYS = 7.0
DEFAULT_MARKET_WEIGHT = 0.5
DEFAULT_QB_SPAN_DROPBACKS = DEFAULT_SPAN_DROPBACKS
# How the expected starter was identified before kickoff. The weight on a QB
# delta follows the reliability of that identification: a misidentified
# starter produces a large delta that carries no information.
QB_BASIS_DATED = "dated_snapshot"
QB_BASIS_PROXY = "prior_week_proxy"
# Selected on development seasons 2016-2021 by least squares of the closing
# line's move away from the engine margin on the raw QB delta, with the
# outcome regression as the independent check (calibration.select_qb_weights).
# Dated: development games whose pregame starters proved correct, standing in
# for dated snapshot charts (2025: 95% correct); the only basis production
# uses. Proxy: every development game under undated prior-week charts (83%
# correct), used by the historical backtest only. Proxy absence: proxy games
# with the incumbent confirmed Out or Doubtful. The earlier grid-selected
# 0.25 measured the proxy era's misidentification, not the value of a QB.
DEFAULT_QB_WEIGHT_DATED = 0.90
DEFAULT_QB_WEIGHT_PROXY = 0.45
DEFAULT_QB_WEIGHT_PROXY_ABSENCE = 0.89
# The QB layer version names the starter-identification rule and its weight;
# the preseason producer appends the same suffix to its own version.
QB_LAYER_VERSION = "qb_v8"
MODEL_VERSION = f"nfl_joint_scoring_{QB_LAYER_VERSION}_{TOTALS_LAYER_VERSION}"


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """Output-layer parameters shared by production and the calibration
    search, so the backtest scores exactly what gets published."""

    market_weight: float = DEFAULT_MARKET_WEIGHT
    rest_points_per_day: float = REST_POINTS_PER_DAY
    qb_span_dropbacks: float = DEFAULT_QB_SPAN_DROPBACKS
    qb_weight_dated: float = DEFAULT_QB_WEIGHT_DATED
    qb_weight_proxy: float = DEFAULT_QB_WEIGHT_PROXY
    qb_weight_proxy_absence: float = DEFAULT_QB_WEIGHT_PROXY_ABSENCE

    def __post_init__(self) -> None:
        if not isfinite(self.qb_span_dropbacks) or self.qb_span_dropbacks <= 0:
            raise ValueError("qb_span_dropbacks must be finite and positive")
        for name in ("qb_weight_dated", "qb_weight_proxy", "qb_weight_proxy_absence"):
            if not 0 <= getattr(self, name) <= 1.5:
                raise ValueError(f"{name} must be between 0 and 1.5")

    def qb_weight(self, basis: str, confirmed_absence: bool) -> float:
        """Weight on a QB delta given how its starter was identified."""
        if basis == QB_BASIS_DATED:
            return self.qb_weight_dated
        if basis != QB_BASIS_PROXY:
            raise ValueError(f"Unknown QB identification basis: {basis}")
        if confirmed_absence:
            return self.qb_weight_proxy_absence
        return self.qb_weight_proxy


def rest_adjustment(
    home_rest: float | None, away_rest: float | None, points_per_day: float
) -> float:
    if pd.isna(home_rest) or pd.isna(away_rest):
        return 0.0
    return points_per_day * float(
        np.clip(home_rest - away_rest, -REST_CLIP_DAYS, REST_CLIP_DAYS)
    )


def published_total(
    engine_total, pure_margin, published_margin, config=DEFAULT_TOTALS_CONFIG
):
    """Calibrate once and preserve nonnegative scores for both output margins."""
    return np.maximum(
        calibrate_total(engine_total, config),
        np.maximum(np.abs(pure_margin), np.abs(published_margin)),
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
        total = float(
            published_total(
                expected_home + expected_away,
                pure_margin,
                published_margin,
                fit.totals_config,
            )
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
    frame = pd.DataFrame([rating.to_record() for rating in ratings])
    games = pd.DataFrame([game.to_record() for game in projections])
    aligned = align_published_ratings(frame, games)
    return [
        replace(
            rating,
            offense_points=row.offense_points,
            defense_points=row.defense_points,
            forecast_alignment_points=row.forecast_alignment_points,
        )
        for rating, row in zip(ratings, aligned.itertuples(), strict=True)
    ]


def align_published_ratings(
    ratings: pd.DataFrame, projections: pd.DataFrame
) -> pd.DataFrame:
    """Align ratings to accepted forecasts, including frozen games."""
    out = ratings.copy()
    if out.empty or projections.empty:
        return out
    previous = out["forecast_alignment_points"].fillna(0.0).to_numpy(dtype=float)
    pure = out["power_rating"].to_numpy(dtype=float) - previous
    index = {team: i for i, team in enumerate(out["team_abbr"])}
    games = np.zeros((len(projections), len(out)))
    gaps = np.zeros(len(projections))
    for row, game in enumerate(projections.itertuples()):
        home, away = index[game.home_team_abbr], index[game.away_team_abbr]
        games[row, home], games[row, away] = 1.0, -1.0
        gaps[row] = game.home_margin - (
            pure[home] - pure[away] + game.home_field_points
        )
    if not np.isfinite(gaps).all():
        raise ValueError(
            "Rating alignment requires finite accepted margins and home fields"
        )
    shifts = np.linalg.lstsq(games, gaps, rcond=None)[0]
    if not np.allclose(games @ shifts, gaps, atol=1e-8):
        raise ValueError(
            "Accepted forecasts cannot be represented by one set of team ratings"
        )
    delta = shifts - previous
    out["offense_points"] += delta / 2
    out["defense_points"] += delta / 2
    out["power_rating"] = pure + shifts
    out["forecast_alignment_points"] = shifts
    return out
