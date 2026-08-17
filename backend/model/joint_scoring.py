"""NFL port of momentumcfb's joint scoring engine: a closed-form conjugate
Gaussian ridge over per-team offense/defense points-per-drive, fused with EPA
per drive through a two-pass GLS step, with a separate pace ridge and a fitted
home-field parameter. Refit from scratch each week on all prior games."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, isnan

import numpy as np
import pandas as pd

from backend.model.outputs import GameProjection, TeamRating

MODEL_VERSION = "nfl_joint_scoring_v1"
PACE_PRIOR_SD = 1.0
HFA_PRIOR_POINTS = 2.0
HFA_PRIOR_SD_POINTS = 1.0


@dataclass(frozen=True, slots=True)
class JointScoringConfig:
    rating_half_life_weeks: float = 6.0
    strength_prior_sd_ppd: float = 0.35
    covariance_shrinkage: float = 0.1
    student_t_degrees_of_freedom: float = 7.0
    score_covariance_scale: float = 1.0

    def __post_init__(self) -> None:
        if isnan(self.rating_half_life_weeks) or self.rating_half_life_weeks <= 0:
            raise ValueError("rating_half_life_weeks must be positive")
        if (
            not isfinite(self.strength_prior_sd_ppd)
            or self.strength_prior_sd_ppd <= 0
        ):
            raise ValueError("strength_prior_sd_ppd must be positive")
        if not 0 <= self.covariance_shrinkage < 1:
            raise ValueError("covariance_shrinkage must be in [0, 1)")
        if (
            not isfinite(self.student_t_degrees_of_freedom)
            or self.student_t_degrees_of_freedom <= 2
        ):
            raise ValueError("student_t_degrees_of_freedom must exceed 2")
        if (
            not isfinite(self.score_covariance_scale)
            or self.score_covariance_scale <= 0
        ):
            raise ValueError("score_covariance_scale must be positive")


# Selected by the calibrate walk-forward on development seasons 2016-2021
# (margin log loss); holdout 2022-2025 untouched. See calibration.py.
DEFAULT_CONFIG = JointScoringConfig(
    rating_half_life_weeks=12.0,
    strength_prior_sd_ppd=0.25,
    covariance_shrinkage=0.1,
    student_t_degrees_of_freedom=7.0,
    score_covariance_scale=0.85,
)


def _solve_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    precision = 1.0 / np.square(prior_sd)
    normal = design.T @ (weights[:, None] * design) + np.diag(precision)
    rhs = design.T @ (weights * target) + precision * prior_mean
    covariance = np.linalg.inv(normal)
    return covariance @ rhs, covariance


def _regularized_covariance(
    residuals: np.ndarray, floor: float, shrinkage: float
) -> np.ndarray:
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    diagonal = np.diag(np.diag(covariance))
    return (1 - shrinkage) * covariance + shrinkage * diagonal + np.eye(2) * floor


def _team_catalog(games: pd.DataFrame) -> list[str]:
    teams = sorted(
        set(games["home_team"].astype(str)) | set(games["away_team"].astype(str))
    )
    if not teams:
        raise ValueError("no teams in games")
    return teams


@dataclass(slots=True)
class JointScoringFit:
    season: int
    week: int
    as_of: datetime
    teams: list[str]
    offense_ppd: np.ndarray
    defense_ppd: np.ndarray
    pace: np.ndarray
    base_ppd: float
    base_drives: float
    hfa_ppd: float
    parameter_covariance: np.ndarray
    score_residual_covariance: np.ndarray
    config: JointScoringConfig

    @property
    def team_index(self) -> dict[str, int]:
        return {team: index for index, team in enumerate(self.teams)}

    @property
    def hfa_points(self) -> float:
        return self.base_drives * self.hfa_ppd

    def ratings(self, team_names: dict[str, str]) -> list[TeamRating]:
        ratings = []
        index_of = self.team_index
        for team in self.teams:
            index = index_of[team]
            vector = np.zeros(len(self.parameter_covariance))
            vector[index] = self.base_drives
            vector[len(self.teams) + index] = self.base_drives
            variance = float(vector @ self.parameter_covariance @ vector)
            ratings.append(
                TeamRating(
                    season=self.season,
                    week=self.week,
                    as_of=self.as_of,
                    model_version=MODEL_VERSION,
                    team_abbr=team,
                    team=team_names.get(team, team),
                    offense_points=float(self.base_drives * self.offense_ppd[index]),
                    defense_points=float(self.base_drives * self.defense_ppd[index]),
                    expected_drives=float(self.base_drives + 0.5 * self.pace[index]),
                    power_rating_sd=float(np.sqrt(max(variance, 0.0))),
                )
            )
        return sorted(ratings, key=lambda rating: rating.power_rating, reverse=True)

    def engine_projection(
        self, game
    ) -> tuple[float, float, float, float, float, float]:
        """Engine-only numbers for one schedule row: expected home/away points,
        home-field points, margin_sd, total_sd, correlation."""
        index = self.team_index
        n_teams = len(self.teams)
        home = index[str(game.home_team)]
        away = index[str(game.away_team)]
        home_field = 0.0 if bool(game.neutral_site) else self.hfa_points
        drives = self.base_drives + 0.5 * (self.pace[home] + self.pace[away])
        base_points = self.base_ppd * drives
        expected_home = (
            base_points
            + self.base_drives * (self.offense_ppd[home] - self.defense_ppd[away])
            + 0.5 * home_field
        )
        expected_away = (
            base_points
            + self.base_drives * (self.offense_ppd[away] - self.defense_ppd[home])
            - 0.5 * home_field
        )
        score_design = np.zeros((2, 2 * n_teams + 1))
        score_design[0, home] = self.base_drives
        score_design[0, n_teams + away] = -self.base_drives
        score_design[1, away] = self.base_drives
        score_design[1, n_teams + home] = -self.base_drives
        if not bool(game.neutral_site):
            score_design[:, -1] = [0.5 * self.base_drives, -0.5 * self.base_drives]
        score_covariance = self.score_residual_covariance + (
            score_design @ self.parameter_covariance @ score_design.T
        )
        score_covariance *= self.config.score_covariance_scale**2
        transform = np.array([[1.0, -1.0], [1.0, 1.0]])
        margin_total = transform @ score_covariance @ transform.T
        margin_sd = float(np.sqrt(margin_total[0, 0]))
        total_sd = float(np.sqrt(margin_total[1, 1]))
        correlation = float(margin_total[0, 1] / (margin_sd * total_sd))
        return (
            float(expected_home),
            float(expected_away),
            float(home_field),
            margin_sd,
            total_sd,
            float(np.clip(correlation, -0.999, 0.999)),
        )


def fit_joint_scoring(
    games: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
    config: JointScoringConfig = DEFAULT_CONFIG,
    strength_prior_means: dict[str, tuple[float, float]] | None = None,
) -> JointScoringFit:
    """Fit ratings using only games strictly before the requested model week.

    strength_prior_means optionally maps team -> (offense_ppd, defense_ppd)
    prior means (e.g. from the preseason prior), so early-season fits start
    from carried-over beliefs instead of zero. Missing teams default to 0."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    training = games[games["model_week"] < forecast_week].copy()
    if training.empty:
        raise ValueError("at least one prior model week is required")
    if "start_date" in training:
        latest_training_start = pd.to_datetime(
            training["start_date"], utc=True
        ).max()
        if latest_training_start.to_pydatetime() >= as_of:
            raise ValueError("training games must start before as_of")
    teams = _team_catalog(games)
    team_index = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(training)
    design = np.zeros((2 * n_games, 2 * n_teams + 1))
    points_per_drive = np.empty(2 * n_games)
    epa_per_drive = np.empty(2 * n_games)
    latest_week = int(training["model_week"].max())
    recency = np.empty(2 * n_games)

    for game_number, game in enumerate(training.itertuples()):
        home = team_index[str(game.home_team)]
        away = team_index[str(game.away_team)]
        home_row = 2 * game_number
        away_row = home_row + 1
        design[home_row, home] = 1.0
        design[home_row, n_teams + away] = -1.0
        design[away_row, away] = 1.0
        design[away_row, n_teams + home] = -1.0
        if not bool(game.neutral_site):
            design[home_row, -1] = 0.5
            design[away_row, -1] = -0.5
        points_per_drive[[home_row, away_row]] = [
            game.home_points / game.game_drives,
            game.away_points / game.game_drives,
        ]
        epa_per_drive[[home_row, away_row]] = [
            game.home_epa_per_drive,
            game.away_epa_per_drive,
        ]
        weight = 0.5 ** (
            (latest_week - game.model_week) / config.rating_half_life_weeks
        )
        recency[[home_row, away_row]] = weight

    base_ppd = float(np.average(points_per_drive, weights=recency))
    centered_points = points_per_drive - base_ppd
    centered_epa = epa_per_drive - np.average(epa_per_drive, weights=recency)
    epa_variance = np.average(np.square(centered_epa), weights=recency)
    epa_scale = float(
        np.clip(
            np.average(centered_epa * centered_points, weights=recency)
            / max(epa_variance, 1e-9),
            0.1,
            2.0,
        )
    )
    process_points = epa_scale * centered_epa
    target = 0.5 * (centered_points + process_points)
    base_drives = float(np.average(training["game_drives"]))
    prior_mean = np.zeros(2 * n_teams + 1)
    if strength_prior_means:
        for team, (offense_prior, defense_prior) in strength_prior_means.items():
            if team in team_index:
                prior_mean[team_index[team]] = offense_prior
                prior_mean[n_teams + team_index[team]] = defense_prior
    prior_mean[-1] = HFA_PRIOR_POINTS / base_drives
    prior_sd = np.full(2 * n_teams + 1, config.strength_prior_sd_ppd)
    prior_sd[-1] = HFA_PRIOR_SD_POINTS / base_drives
    parameters, covariance = _solve_ridge(
        design, target, recency, prior_mean, prior_sd
    )
    paired_residuals = np.column_stack(
        [centered_points - design @ parameters, process_points - design @ parameters]
    )
    process_covariance = _regularized_covariance(
        paired_residuals,
        floor=0.01,
        shrinkage=config.covariance_shrinkage,
    )
    inverse_process_covariance = np.linalg.inv(process_covariance)
    ones = np.ones(2)
    information = float(ones @ inverse_process_covariance @ ones)
    target = (
        np.column_stack([centered_points, process_points])
        @ inverse_process_covariance
        @ ones
        / information
    )
    parameters, covariance = _solve_ridge(
        design, target, recency * information, prior_mean, prior_sd
    )

    offense = parameters[:n_teams]
    defense = parameters[n_teams : 2 * n_teams]
    offense_mean = float(offense.mean())
    defense_mean = float(defense.mean())
    offense = offense - offense_mean
    defense = defense - defense_mean
    base_ppd += offense_mean - defense_mean
    centering = np.eye(n_teams) - np.full((n_teams, n_teams), 1.0 / n_teams)
    covariance_transform = np.zeros_like(covariance)
    covariance_transform[:n_teams, :n_teams] = centering
    covariance_transform[n_teams : 2 * n_teams, n_teams : 2 * n_teams] = centering
    covariance_transform[-1, -1] = 1.0
    covariance = covariance_transform @ covariance @ covariance_transform.T

    pace_design = np.zeros((n_games, n_teams))
    pace_target = training["game_drives"].to_numpy(float)
    for row, game in enumerate(training.itertuples()):
        pace_design[row, team_index[str(game.home_team)]] = 0.5
        pace_design[row, team_index[str(game.away_team)]] = 0.5
    base_drives = float(np.average(pace_target))
    pace, _ = _solve_ridge(
        pace_design,
        pace_target - base_drives,
        recency[::2],
        np.zeros(n_teams),
        np.full(n_teams, PACE_PRIOR_SD),
    )
    base_drives += float(pace.mean())
    pace -= pace.mean()

    home_index = training["home_team"].map(team_index).to_numpy(int)
    away_index = training["away_team"].map(team_index).to_numpy(int)
    expected_drives = base_drives + 0.5 * (pace[home_index] + pace[away_index])
    home_field = (~training["neutral_site"].astype(bool)).to_numpy(float)
    hfa_points = base_drives * parameters[-1]
    predicted_home = (
        base_ppd * expected_drives
        + base_drives * (offense[home_index] - defense[away_index])
        + 0.5 * hfa_points * home_field
    )
    predicted_away = (
        base_ppd * expected_drives
        + base_drives * (offense[away_index] - defense[home_index])
        - 0.5 * hfa_points * home_field
    )
    score_residuals = np.column_stack(
        [
            training["home_points"].to_numpy(float) - predicted_home,
            training["away_points"].to_numpy(float) - predicted_away,
        ]
    )
    score_covariance = _regularized_covariance(
        score_residuals,
        floor=4.0,
        shrinkage=config.covariance_shrinkage,
    )
    return JointScoringFit(
        season=int(training["season"].iloc[-1]),
        week=forecast_week,
        as_of=as_of,
        teams=teams,
        offense_ppd=offense,
        defense_ppd=defense,
        pace=pace,
        base_ppd=base_ppd,
        base_drives=base_drives,
        hfa_ppd=float(parameters[-1]),
        parameter_covariance=covariance,
        score_residual_covariance=score_covariance,
        config=config,
    )
