"""Ruled-out regulars other than the quarterback, priced in points per game.

A player's value is his shrunk ``epa.credit`` per credited game above the
week's replacement level at his position group (``war.shrunk_credit``), from
the latest position-trait snapshot strictly before the forecast week, else the
previous season's final week. His share is the recency-weighted fraction of
his team's games he was credited in before the forecast week, with the
engine's rating half-life, so a player who has been out for weeks is already
absent from the rating and costs nothing more. Value times share is the credit
per team game the rating carries and the backup does not. A team's points lost
sum that over its players listed Out or Doubtful at the forecast cutoff
(``ruled_out``). Quarterbacks have their own layer and the offensive line is
graded only as a unit, so neither is priced here.

Inputs are the ``snapshots_{season}`` and ``credited_{season}`` artifacts the
position-trait job writes. Absent inputs raise; nothing defaults to zero.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.features.qb import ruled_out
from backend.players import EPA_TRAIT, war

TRAIT_DIR = PROCESSED_DIR / "players" / "position_traits"
GROUPS = ("DB", "DL", "LB", "WR", "TE", "RB")
# Position traits cover the regular season only; playoff forecasts read its
# final week.
REGULAR_SEASON_WEEKS = 18
# Games at which pooled split-half reliability of ``epa.credit`` reaches one
# half on development seasons 2016-2021 (``backtest.epa_shrink_games`` over
# ``backtest.pooled_trait_reliability`` of those seasons). Every group but DL
# sits at the grade's 12-game cap. The published grade model's own shrinkage
# also saw 2022-2024, which the team holdout covers.
SHRINK_GAMES = {
    "DB": 12.0,
    "DL": 11.97,
    "LB": 12.0,
    "RB": 12.0,
    "TE": 12.0,
    "WR": 12.0,
}


@dataclass(frozen=True)
class AbsenceInputs:
    """``values``: season, week, athlete_id, value. ``credited``: season,
    game_id, model_week, team, athlete_id, one row per side of each game."""

    values: pd.DataFrame
    credited: pd.DataFrame


def input_names(seasons) -> list[str]:
    """The trait artifacts a forecast reads, relative to ``PROCESSED_DIR``."""
    return [
        f"{TRAIT_DIR.relative_to(PROCESSED_DIR)}/{kind}_{season}.parquet"
        for season in seasons
        for kind in ("snapshots", "credited")
    ]


def input_paths(seasons) -> list:
    return [PROCESSED_DIR / name for name in input_names(seasons)]


def load_inputs(seasons) -> AbsenceInputs:
    missing = [str(path) for path in input_paths(seasons) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run position-traits before pricing absences: " + ", ".join(missing)
        )
    values, credited = [], []
    for season in seasons:
        snapshot = pd.read_parquet(
            TRAIT_DIR / f"snapshots_{season}.parquet",
            filters=[("trait", "==", EPA_TRAIT)],
        )
        rows = war.shrunk_credit(snapshot, SHRINK_GAMES)
        rows = rows[rows["position_group"].isin(GROUPS)]
        values.append(
            rows.assign(
                season=season,
                value=rows["shrunk_per_game"] - rows["replacement_per_game"],
            )[["season", "week", "athlete_id", "value"]]
        )
        games = pd.read_parquet(TRAIT_DIR / f"credited_{season}.parquet").astype(
            {"athlete_id": str}
        )
        teams = games["game_id"].str.split("_", expand=True)
        for column in (2, 3):
            credited.append(
                games.assign(season=season, team=teams[column])[
                    ["season", "game_id", "model_week", "team", "athlete_id"]
                ]
            )
    return AbsenceInputs(
        pd.concat(values, ignore_index=True), pd.concat(credited, ignore_index=True)
    )


def require_coverage(inputs: AbsenceInputs, season: int, week: int) -> None:
    """The season's trait artifacts must reach the last week before ``week``."""
    needed = min(week - 1, REGULAR_SEASON_WEEKS)
    if needed < 1:
        return
    played = inputs.credited.loc[
        inputs.credited["season"].eq(season), "model_week"
    ].max()
    if pd.isna(played) or played < needed:
        raise ValueError(
            f"Position traits for {season} end at week {played}; "
            f"week {week} needs week {needed}"
        )


def shares(
    inputs: AbsenceInputs, season: int, week: int, half_life: float
) -> pd.DataFrame:
    """``team, athlete_id, share`` of recency-weighted team games before
    ``week``, from the previous season for a team that has not played yet."""
    credited = inputs.credited
    current = credited[credited["season"].eq(season) & credited["model_week"].lt(week)]
    previous = credited[
        credited["season"].eq(season - 1) & ~credited["team"].isin(set(current["team"]))
    ]
    window = pd.concat([current, previous], ignore_index=True)
    if window.empty:
        return pd.DataFrame(columns=["team", "athlete_id", "share"])
    games = window.drop_duplicates(["team", "game_id"])[
        ["team", "game_id", "season", "model_week"]
    ]
    clock = games["season"] * 52 + games["model_week"]
    latest = clock.groupby(games["team"]).transform("max")
    games = games.assign(weight=0.5 ** ((latest - clock) / half_life))
    totals = games.groupby("team")["weight"].sum()
    played = window.merge(games[["team", "game_id", "weight"]], on=["team", "game_id"])
    out = played.groupby(["team", "athlete_id"], as_index=False)["weight"].sum()
    out["share"] = out["weight"] / out["team"].map(totals)
    return out[["team", "athlete_id", "share"]]


def latest_values(inputs: AbsenceInputs, season: int, week: int) -> pd.DataFrame:
    values = inputs.values
    eligible = values[
        (values["season"].eq(season) & values["week"].lt(week))
        | values["season"].eq(season - 1)
    ]
    return (
        eligible.sort_values(["season", "week"])
        .drop_duplicates("athlete_id", keep="last")[["athlete_id", "value"]]
        .astype({"athlete_id": str})
    )


def points_lost(
    inputs: AbsenceInputs,
    slate: pd.DataFrame,
    season: int,
    model_week: int,
    injuries: pd.DataFrame | None,
    half_life: float,
) -> dict[str, tuple[float, float]]:
    """game_id -> (home, away) unweighted points per game lost to absences.

    ``slate`` rows need game_id, week, home_team, away_team and
    forecast_cutoff, the moment the injury report is read at.
    """
    team_shares = shares(inputs, season, model_week, half_life)
    priced = latest_values(inputs, season, model_week).merge(
        team_shares, on="athlete_id"
    )
    priced["points"] = priced["value"] * priced["share"]
    out = {}
    for game in slate.itertuples():
        lost = []
        unavailable = ruled_out(injuries, season, game.week, game.forecast_cutoff)
        listed = (
            injuries[
                injuries["season"].eq(season)
                & injuries["week"].eq(game.week)
                & injuries["gsis_id"].isin(unavailable)
            ]
            if unavailable
            else None
        )
        for team in (game.home_team, game.away_team):
            if listed is None:
                lost.append(0.0)
                continue
            ids = set(listed.loc[listed["team"].eq(team), "gsis_id"].astype(str))
            rows = priced[priced["team"].eq(team) & priced["athlete_id"].isin(ids)]
            lost.append(float(np.sum(rows["points"])))
        out[str(game.game_id)] = (lost[0], lost[1])
    return out
