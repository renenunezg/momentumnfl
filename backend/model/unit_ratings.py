"""Descriptive, opponent-adjusted unit ratings. Companions to the engine's
offense/defense ratings, never inputs to it: each channel is its own small
conjugate ridge in points per game above league average. Line channels are
attribution proxies (no snap-level film data exists publicly)."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from backend.features.units import EPA_PER_PRESSURE, POINTS_PER_LINE_YARD
from backend.model.joint_scoring import _solve_ridge

MODEL_VERSION = "nfl_unit_ratings_v1"
CHANNEL_PRIOR_SD = 3.0  # points per game

COLUMNS = (
    "rush_offense",
    "pass_offense",
    "rush_defense",
    "pass_defense",
    "pass_block",
    "run_block",
    "special_teams",
)


def _two_sided_ridge(
    observations: pd.DataFrame, teams: list[str], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """obs = unit[team] - counter_unit[opponent] + noise. Returns (unit,
    counter_unit) posterior means, both centered."""
    index = {team: position for position, team in enumerate(teams)}
    n_teams = len(teams)
    design = np.zeros((len(observations), 2 * n_teams))
    for row, record in enumerate(observations.itertuples()):
        design[row, index[record.team]] = 1.0
        design[row, n_teams + index[record.opponent]] = -1.0
    target = observations["value"].to_numpy(float)
    center = float(np.average(target, weights=weights))
    parameters, _ = _solve_ridge(
        design,
        target - center,
        weights,
        np.zeros(2 * n_teams),
        np.full(2 * n_teams, CHANNEL_PRIOR_SD),
    )
    unit = parameters[:n_teams] - parameters[:n_teams].mean()
    counter = parameters[n_teams:] - parameters[n_teams:].mean()
    return unit, counter


def _channel_frame(
    unit_games: pd.DataFrame, game_map: pd.DataFrame, column: str
) -> pd.DataFrame:
    frame = unit_games.merge(game_map, on=["game_id", "team"])
    frame["value"] = frame[column]
    return frame[["game_id", "team", "opponent", "value", "weight"]]


def _pass_block_values(unit_games: pd.DataFrame) -> pd.Series:
    """Protection quality in points per game, higher is better. Uses charted
    pressures (2018+) corrected for QB time to throw; falls back to sack and
    hit EPA allowed."""
    charted = unit_games["pressures_allowed"].notna() & unit_games[
        "dropbacks"
    ].gt(0)
    values = pd.Series(np.nan, index=unit_games.index)
    if charted.any():
        subset = unit_games[charted]
        rate = subset["pressures_allowed"] / subset["dropbacks"]
        ttt = subset["avg_time_to_throw"]
        valid_ttt = ttt.notna()
        adjusted = rate.copy()
        if valid_ttt.sum() >= 100:
            centered_ttt = ttt[valid_ttt] - ttt[valid_ttt].mean()
            centered_rate = rate[valid_ttt] - rate[valid_ttt].mean()
            beta = float(
                (centered_ttt * centered_rate).sum()
                / max((centered_ttt**2).sum(), 1e-9)
            )
            adjusted.loc[valid_ttt] = rate[valid_ttt] - beta * centered_ttt
        values.loc[charted] = (
            (adjusted - adjusted.mean())
            * subset["dropbacks"]
            * EPA_PER_PRESSURE
        )
    fallback = values.isna()
    values.loc[fallback] = unit_games.loc[fallback, "protection_epa_allowed"]
    return values


def _run_block_values(unit_games: pd.DataFrame) -> pd.Series:
    """Adjusted line yards over a per-carry league baseline, in points."""
    with_carries = unit_games["carries"].gt(0)
    league_per_carry = float(
        unit_games.loc[with_carries, "line_yards"].sum()
        / unit_games.loc[with_carries, "carries"].sum()
    )
    values = (
        unit_games["line_yards"]
        - league_per_carry * unit_games["carries"]
    ) * POINTS_PER_LINE_YARD
    return values.where(with_carries, 0.0)


@dataclass(frozen=True, slots=True)
class UnitRatings:
    season: int
    week: int
    as_of: datetime
    frame: pd.DataFrame  # team_abbr + COLUMNS

    def to_records(self, team_names: dict[str, str]) -> list[dict]:
        records = []
        for row in self.frame.itertuples():
            record = {
                "season": self.season,
                "week": self.week,
                "as_of": self.as_of.isoformat(),
                "model_version": MODEL_VERSION,
                "team_abbr": row.team_abbr,
                "team": team_names.get(row.team_abbr, row.team_abbr),
            }
            for column in COLUMNS:
                record[column] = float(getattr(row, column))
            records.append(record)
        return records


def fit_unit_ratings(
    unit_games: pd.DataFrame,
    games: pd.DataFrame,
    forecast_week: int,
    as_of: datetime,
    recency_by_game: dict[str, float],
) -> UnitRatings:
    """Fit all channels on games strictly before forecast_week."""
    training = games[games["model_week"] < forecast_week]
    if training.empty:
        raise ValueError("at least one prior model week is required")
    home_map = training[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away_map = training[["game_id", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    game_map = pd.concat([home_map, away_map], ignore_index=True)
    game_map["weight"] = game_map["game_id"].map(recency_by_game).fillna(0.0)

    window = unit_games[unit_games["game_id"].isin(set(training["game_id"]))]
    window = window.assign(
        pass_block_value=_pass_block_values(window),
        run_block_value=_run_block_values(window),
    )
    teams = sorted(
        set(training["home_team"].astype(str))
        | set(training["away_team"].astype(str))
    )

    channels = {
        "rush": "rush_epa",
        "pass": "pass_epa",
        "pass_block": "pass_block_value",
        "run_block": "run_block_value",
        "special_teams": "st_epa",
    }
    ratings: dict[str, np.ndarray] = {}
    for name, column in channels.items():
        frame = _channel_frame(window, game_map, column)
        unit, counter = _two_sided_ridge(
            frame, teams, frame["weight"].to_numpy(float)
        )
        if name == "rush":
            ratings["rush_offense"], ratings["rush_defense"] = unit, counter
        elif name == "pass":
            ratings["pass_offense"], ratings["pass_defense"] = unit, counter
        else:
            ratings[name] = unit

    frame = pd.DataFrame({"team_abbr": teams})
    for column in COLUMNS:
        frame[column] = ratings[column]
    return UnitRatings(
        season=int(training["season"].iloc[-1]),
        week=forecast_week,
        as_of=as_of,
        frame=frame,
    )
