"""nflreadpy wrappers returning pandas with current-franchise team codes."""

import pandas as pd
import nflreadpy

# Historical to current franchise codes so 32 teams are continuous 2015+.
TEAM_NORMALIZATION = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEAM_COLUMNS = (
    "home_team",
    "away_team",
    "team",
    "posteam",
    "defteam",
    "recent_team",
    "club_code",
    "team_abbr",
    "opponent_team",
)


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for column in TEAM_COLUMNS:
        if column in df.columns:
            df[column] = df[column].replace(TEAM_NORMALIZATION)
    return df


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    return normalize_teams(nflreadpy.load_pbp(seasons).to_pandas())


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    df = normalize_teams(nflreadpy.load_schedules().to_pandas())
    return df[df["season"].isin(seasons)].copy()


def load_depth_charts(seasons: list[int]) -> pd.DataFrame:
    return normalize_teams(nflreadpy.load_depth_charts(seasons).to_pandas())


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    return normalize_teams(nflreadpy.load_injuries(seasons).to_pandas())


def load_teams() -> pd.DataFrame:
    return normalize_teams(nflreadpy.load_teams().to_pandas())


def load_pfr_advstats(seasons: list[int], stat_type: str) -> pd.DataFrame:
    return normalize_teams(
        nflreadpy.load_pfr_advstats(seasons, stat_type=stat_type).to_pandas()
    )


def load_nextgen_stats(stat_type: str) -> pd.DataFrame:
    return normalize_teams(
        nflreadpy.load_nextgen_stats(stat_type=stat_type).to_pandas()
    )
