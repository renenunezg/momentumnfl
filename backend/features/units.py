"""Per-team-game unit channel aggregates for the descriptive unit ratings.

Channels are EPA- or points-scaled per game so the ridges publish directly
in points per game above average. Line channels are proxies: pressures and
sacks allowed (PFR charted 2018+, pbp sack/hit fallback before), and
adjusted line yards by gap for run blocking."""

import numpy as np
import pandas as pd

from backend.features.drives import SCRIMMAGE_PLAY_TYPES, _competitive

EPA_PER_PRESSURE = -0.45
POINTS_PER_LINE_YARD = 0.08


def _adjusted_line_yards(yards: pd.Series) -> pd.Series:
    """Football Outsiders style credit to the line for each carry."""
    return pd.Series(
        np.select(
            [yards < 0, yards <= 4, yards <= 10],
            [1.2 * yards, yards, 4 + 0.5 * (yards - 4)],
            default=7.0,
        ),
        index=yards.index,
    )


def build_unit_games(
    pbp: pd.DataFrame,
    pfr_pass: pd.DataFrame | None,
    ngs_passing: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per (game_id, team) with unit channel observations."""
    scrimmage = pbp[
        pbp["posteam"].notna()
        & pbp["epa"].notna()
        & pbp["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    ]
    scrimmage = scrimmage[_competitive(scrimmage)]

    rush = scrimmage[scrimmage["rush_attempt"].eq(1)]
    dropback = scrimmage[scrimmage["qb_dropback"].eq(1)]
    protection = dropback[
        dropback["sack"].eq(1) | dropback["qb_hit"].eq(1)
    ]
    special = pbp[pbp["special"].eq(1) & pbp["epa"].notna() & pbp["posteam"].notna()]

    def per_team(frame: pd.DataFrame, name: str, statistic: str = "sum"):
        grouped = frame.groupby(["game_id", "posteam"])["epa"].agg(statistic)
        return grouped.rename(name)

    aly = rush.assign(line_yards=_adjusted_line_yards(rush["yards_gained"]))
    aly_team = aly.groupby(["game_id", "posteam"]).agg(
        line_yards=("line_yards", "sum"), carries=("line_yards", "size")
    )

    out = pd.concat(
        [
            per_team(rush, "rush_epa"),
            per_team(dropback, "pass_epa"),
            per_team(protection, "protection_epa_allowed"),
            per_team(special, "st_epa"),
            aly_team,
            dropback.groupby(["game_id", "posteam"])["epa"]
            .size()
            .rename("dropbacks"),
        ],
        axis=1,
    ).reset_index().rename(columns={"posteam": "team"})
    out = out.fillna(
        {
            "rush_epa": 0.0,
            "pass_epa": 0.0,
            "protection_epa_allowed": 0.0,
            "st_epa": 0.0,
            "line_yards": 0.0,
            "carries": 0.0,
            "dropbacks": 0.0,
        }
    )

    if pfr_pass is not None and not pfr_pass.empty:
        pressures = (
            pfr_pass.groupby(["game_id", "team"])
            .agg(
                pressures_allowed=("times_pressured", "sum"),
                sacks_allowed=("times_sacked", "sum"),
            )
            .reset_index()
        )
        out = out.merge(pressures, on=["game_id", "team"], how="left")
    else:
        out["pressures_allowed"] = np.nan
        out["sacks_allowed"] = np.nan

    if ngs_passing is not None and not ngs_passing.empty:
        weekly = ngs_passing[ngs_passing["week"].gt(0)]
        ttt = (
            weekly.groupby(["season", "week", "team_abbr"])[
                "avg_time_to_throw"
            ]
            .mean()
            .reset_index()
            .rename(columns={"team_abbr": "team"})
        )
        season_week = out["game_id"].str.extract(
            r"^(?P<season>\d{4})_(?P<week>\d{2})_"
        ).astype(int)
        out[["season", "week"]] = season_week
        out = out.merge(ttt, on=["season", "week", "team"], how="left")
    else:
        out["avg_time_to_throw"] = np.nan

    return out
