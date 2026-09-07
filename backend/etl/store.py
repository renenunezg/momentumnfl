"""Parquet store under backend/data plus the loaders every model stage shares."""

import os

import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR

LEGACY_TEAM_NAMES = {
    "Oakland Raiders",
    "San Diego Chargers",
    "St. Louis Rams",
    "Washington Redskins",
    "Washington Football Team",
}


def write_parquet(df: pd.DataFrame, path) -> None:
    """Atomic parquet write: tmp file then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_raw(*parts: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR.joinpath(*parts))


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    write_parquet(df, PROCESSED_DIR.joinpath(*parts))


def read_processed(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts), columns=columns)


def processed_names(*parts: str) -> list[str]:
    """Parquet file stems stored under a processed directory, if it exists."""
    directory = PROCESSED_DIR.joinpath(*parts)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def _read_seasons(
    directory: str, seasons: list[int], columns: list[str] | None = None
) -> pd.DataFrame:
    available = set(processed_names(directory))
    frames = [
        read_processed(directory, f"{season}.parquet", columns=columns)
        for season in seasons
        if str(season) in available
    ]
    return pd.concat(frames, ignore_index=True)


def current_teams() -> pd.DataFrame:
    """One identity per current franchise, excluding normalized legacy rows."""
    teams = read_raw("teams.parquet")
    return teams[~teams["team_name"].isin(LEGACY_TEAM_NAMES)].drop_duplicates(
        "team_abbr"
    )


def team_names() -> dict[str, str]:
    teams = current_teams()
    return dict(zip(teams["team_abbr"], teams["team_name"]))


def season_games(season: int) -> pd.DataFrame:
    return read_processed("team_games", f"{season}.parquet")


def qb_games(seasons: list[int]) -> pd.DataFrame:
    return _read_seasons("qb_games", seasons)


def game_index(seasons: list[int]) -> pd.DataFrame:
    """game_id, season, model_week, start_date across seasons with features."""
    return _read_seasons(
        "team_games", seasons, ["game_id", "season", "model_week", "start_date"]
    )
