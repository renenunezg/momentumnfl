"""Week-1 preseason prior: mean reversion of last season's final ratings
blended with a rating implied by the Vegas season win-total market. The win
total prices offseason change (QB moves, coaching, roster) that reversion
cannot see."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.config import STATIC_DIR
from backend.etl import store
from backend.model.fit_week import load_season_games, load_team_names
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    JointScoringConfig,
    JointScoringFit,
    fit_joint_scoring,
)
from backend.model.outputs import TeamRating

MODEL_VERSION = "nfl_preseason_v1"

# Starting values; calibration owns the final ones.
CARRYOVER = 0.60
WIN_TOTAL_BLEND = 0.35  # weight q on the win-total implied rating
ENVIRONMENT_CARRYOVER = 0.40
PACE_CARRYOVER = 0.35
BASE_OFFSEASON_SD = 4.5
WIN_TOTAL_MISSING_SD = 3.0
FALLBACK_POINTS_PER_WIN = 2.7


@dataclass(frozen=True, slots=True)
class PreseasonConfig:
    carryover: float = CARRYOVER
    win_total_blend: float = WIN_TOTAL_BLEND
    environment_carryover: float = ENVIRONMENT_CARRYOVER
    pace_carryover: float = PACE_CARRYOVER
    base_offseason_sd: float = BASE_OFFSEASON_SD


def load_win_totals(season: int) -> pd.Series:
    totals = pd.read_csv(STATIC_DIR / "win_totals.csv")
    totals = totals[totals["season"].eq(season)]
    return totals.set_index("team_abbr")["win_total"]


def points_per_win(previous_seasons: list[int]) -> float:
    """OLS slope of centered win totals onto same-season final power ratings,
    fit on history. Falls back to a fixed constant when degenerate."""
    xs, ys = [], []
    for season in previous_seasons:
        try:
            totals = load_win_totals(season)
            games = load_season_games(season)
        except (FileNotFoundError, KeyError):
            continue
        if totals.empty or games.empty:
            continue
        final_week = int(games["model_week"].max()) + 1
        fit = fit_joint_scoring(
            games,
            final_week,
            datetime.now(timezone.utc),
            DEFAULT_CONFIG,
        )
        power = {
            team: float(
                fit.base_drives
                * (
                    fit.offense_ppd[fit.team_index[team]]
                    + fit.defense_ppd[fit.team_index[team]]
                )
            )
            for team in fit.teams
        }
        centered = totals - totals.mean()
        for team, total in centered.items():
            if team in power:
                xs.append(float(total))
                ys.append(power[team])
    if len(xs) < 64:
        return FALLBACK_POINTS_PER_WIN
    xs_array = np.asarray(xs)
    ys_array = np.asarray(ys)
    denominator = float(np.sum(np.square(xs_array)))
    if denominator <= 0:
        return FALLBACK_POINTS_PER_WIN
    return float(np.sum(xs_array * ys_array) / denominator)


@dataclass(slots=True)
class PreseasonPrior:
    season: int
    as_of: datetime
    previous_fit: JointScoringFit
    ratings_frame: pd.DataFrame  # team_abbr, power, environment, pace, sd, missing
    slope: float

    def strength_prior_means(self) -> dict[str, tuple[float, float]]:
        """team -> (offense_ppd, defense_ppd) prior means for in-season fits."""
        means = {}
        base = self.previous_fit.base_drives
        for row in self.ratings_frame.itertuples():
            offense_points = 0.5 * (row.power_rating + row.scoring_environment)
            defense_points = 0.5 * (row.power_rating - row.scoring_environment)
            means[row.team_abbr] = (offense_points / base, defense_points / base)
        return means

    def week1_fit(self) -> JointScoringFit:
        """A synthetic engine fit for week-1 projections: preseason strengths
        with last season's base rates, HFA, and score covariance; parameter
        covariance is diagonal from the preseason power sd so projection
        margin_sd inherits offseason uncertainty."""
        base = self.previous_fit.base_drives
        teams = list(self.ratings_frame["team_abbr"])
        n_teams = len(teams)
        offense = np.empty(n_teams)
        defense = np.empty(n_teams)
        pace = np.empty(n_teams)
        variance = np.zeros(2 * n_teams + 1)
        for position, row in enumerate(self.ratings_frame.itertuples()):
            offense[position] = (
                0.5 * (row.power_rating + row.scoring_environment) / base
            )
            defense[position] = (
                0.5 * (row.power_rating - row.scoring_environment) / base
            )
            pace[position] = row.expected_drives - base
            half_variance = 0.5 * (row.power_rating_sd**2) / base**2
            variance[position] = half_variance
            variance[n_teams + position] = half_variance
        variance[-1] = float(self.previous_fit.parameter_covariance[-1, -1])
        return JointScoringFit(
            season=self.season,
            week=1,
            as_of=self.as_of,
            teams=teams,
            offense_ppd=offense,
            defense_ppd=defense,
            pace=pace,
            base_ppd=self.previous_fit.base_ppd,
            base_drives=base,
            hfa_ppd=self.previous_fit.hfa_ppd,
            parameter_covariance=np.diag(variance),
            score_residual_covariance=self.previous_fit.score_residual_covariance,
            config=self.previous_fit.config,
        )

    def ratings(self, team_names: dict[str, str]) -> list[TeamRating]:
        ratings = []
        for row in self.ratings_frame.itertuples():
            offense_points = 0.5 * (row.power_rating + row.scoring_environment)
            defense_points = 0.5 * (row.power_rating - row.scoring_environment)
            ratings.append(
                TeamRating(
                    season=self.season,
                    week=1,
                    as_of=self.as_of,
                    model_version=MODEL_VERSION,
                    team_abbr=row.team_abbr,
                    team=team_names.get(row.team_abbr, row.team_abbr),
                    offense_points=float(offense_points),
                    defense_points=float(defense_points),
                    expected_drives=float(row.expected_drives),
                    power_rating_sd=float(row.power_rating_sd),
                )
            )
        return sorted(
            ratings, key=lambda rating: rating.power_rating, reverse=True
        )


def build_preseason_prior(
    season: int,
    as_of: datetime | None = None,
    config: PreseasonConfig = PreseasonConfig(),
    engine_config: JointScoringConfig = DEFAULT_CONFIG,
) -> PreseasonPrior:
    as_of = as_of or datetime.now(timezone.utc)
    previous_games = load_season_games(season - 1)
    final_week = int(previous_games["model_week"].max()) + 1
    previous_fit = fit_joint_scoring(
        previous_games, final_week, as_of, engine_config
    )
    index = previous_fit.team_index
    base = previous_fit.base_drives

    try:
        win_totals = load_win_totals(season)
    except (FileNotFoundError, KeyError):
        win_totals = pd.Series(dtype=float)
    win_total_missing = win_totals.empty
    slope = points_per_win(list(range(2015, season)))

    rows = []
    centered_totals = win_totals - win_totals.mean() if not win_total_missing else win_totals
    for team in previous_fit.teams:
        team_idx = index[team]
        previous_power = float(
            base
            * (previous_fit.offense_ppd[team_idx] + previous_fit.defense_ppd[team_idx])
        )
        previous_environment = float(
            base
            * (previous_fit.offense_ppd[team_idx] - previous_fit.defense_ppd[team_idx])
        )
        reverted = config.carryover * previous_power
        if win_total_missing or team not in centered_totals:
            power = reverted
            missing = 1
            sd = float(
                np.sqrt(config.base_offseason_sd**2 + WIN_TOTAL_MISSING_SD**2)
            )
        else:
            implied = slope * float(centered_totals[team])
            power = (
                1 - config.win_total_blend
            ) * reverted + config.win_total_blend * implied
            missing = 0
            sd = config.base_offseason_sd
        rows.append(
            {
                "team_abbr": team,
                "power_rating": power,
                "scoring_environment": np.clip(
                    config.environment_carryover * previous_environment,
                    -8.0,
                    8.0,
                ),
                "expected_drives": base
                + config.pace_carryover * float(previous_fit.pace[team_idx]),
                "power_rating_sd": sd,
                "missing_input_count": missing,
            }
        )
    frame = pd.DataFrame(rows)
    frame["power_rating"] -= frame["power_rating"].mean()
    return PreseasonPrior(
        season=season,
        as_of=as_of,
        previous_fit=previous_fit,
        ratings_frame=frame,
        slope=slope,
    )
