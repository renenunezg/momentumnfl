"""nflreadpy wrappers returning pandas with current-franchise team codes."""

import nflreadpy
import pandas as pd

# Historical to current franchise codes so 32 teams are continuous 2015+.
# LAR is the Rams code some nflverse datasets (teams, NGS, PFR) use where
# pbp and schedules use LA.
TEAM_NORMALIZATION = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
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
    return normalize_teams(nflreadpy.load_schedules(seasons).to_pandas())


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


def load_nextgen_stats(seasons: list[int], stat_type: str) -> pd.DataFrame:
    return normalize_teams(
        nflreadpy.load_nextgen_stats(seasons=seasons, stat_type=stat_type).to_pandas()
    )


RELEASES_URL = "https://github.com/nflverse/nflverse-data/releases/download/"
PARTICIPATION_URL = (
    RELEASES_URL + "pbp_participation/pbp_participation_{season}.parquet"
)
ESPN_QBR_URL = RELEASES_URL + "espn_data/qbr_week_level.parquet"


def _release_parquet(url: str) -> pd.DataFrame:
    """A parquet file read straight from an nflverse release, for the files
    nflreadpy has no loader for or fails on. Raises when it is not published."""
    import io

    import requests

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def load_participation(season: int) -> pd.DataFrame:
    """Players on the field for every play, with pass rushers, pressure and
    coverage."""
    return normalize_teams(_release_parquet(PARTICIPATION_URL.format(season=season)))


def load_espn_qbr() -> pd.DataFrame:
    """ESPN's weekly Total QBR (raw and opponent adjusted, with its plays and
    points added) for every season, one file."""
    return _release_parquet(ESPN_QBR_URL)


def load_ftn_charting(seasons: list[int]) -> pd.DataFrame:
    return nflreadpy.load_ftn_charting(seasons).to_pandas()


def load_players() -> pd.DataFrame:
    return nflreadpy.load_players().to_pandas()
