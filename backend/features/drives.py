"""Team-game aggregates from pbp. Drives come from nflverse fixed_drive."""

import pandas as pd

# NFL-scaled garbage-time margins by quarter; final values owned by calibration.
GARBAGE_MARGIN = {1: 28, 2: 21, 3: 16}
GARBAGE_MARGIN_LATE = 14

SCRIMMAGE_PLAY_TYPES = ("pass", "run", "qb_kneel", "qb_spike")


def competitive_plays(pbp: pd.DataFrame) -> pd.Series:
    margin = pbp["score_differential"].abs()
    threshold = pbp["qtr"].clip(upper=4).map(GARBAGE_MARGIN).fillna(GARBAGE_MARGIN_LATE)
    return margin.le(threshold) | margin.isna()


def kickoff_utc(schedules: pd.DataFrame) -> pd.Series:
    stamp = pd.to_datetime(
        schedules["gameday"] + " " + schedules["gametime"].fillna("13:00"),
        errors="coerce",
    )
    return stamp.dt.tz_localize(
        "America/New_York", nonexistent="shift_forward", ambiguous=True
    ).dt.tz_convert("UTC")


def build_team_games(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per played game with home/away points, drives, and EPA."""
    plays = pbp[
        pbp["posteam"].notna()
        & pbp["epa"].notna()
        & pbp["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    ]
    plays = plays[competitive_plays(plays)]
    grouped = (
        plays.groupby(["game_id", "posteam"])
        .agg(
            offense_epa=("epa", "sum"),
            offense_plays=("epa", "size"),
            drives=("fixed_drive", "nunique"),
        )
        .reset_index()
    )

    played = schedules[schedules["home_score"].notna()].copy()
    played["neutral_site"] = played["location"].eq("Neutral")
    played["start_date"] = kickoff_utc(played)
    out = played[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral_site",
            "start_date",
            "home_rest",
            "away_rest",
            "div_game",
            "spread_line",
            "total_line",
        ]
    ].rename(
        columns={
            "game_type": "season_type",
            "home_score": "home_points",
            "away_score": "away_points",
        }
    )
    for side in ("home", "away"):
        out = out.merge(
            grouped.rename(
                columns={
                    "posteam": f"{side}_team",
                    "offense_epa": f"{side}_epa",
                    "offense_plays": f"{side}_plays",
                    "drives": f"{side}_drives",
                }
            ),
            on=["game_id", f"{side}_team"],
            how="inner",
        )
    out["game_drives"] = out["home_drives"] + out["away_drives"]
    out["home_epa_per_drive"] = out["home_epa"] / out["home_drives"]
    out["away_epa_per_drive"] = out["away_epa"] / out["away_drives"]
    numeric = ["home_points", "away_points", "spread_line", "total_line"]
    out[numeric] = out[numeric].astype(float)
    return out.sort_values(["start_date", "game_id"]).reset_index(drop=True)
