"""Production-pipeline state and schedule gates.

Normal runs restore the small team-game and QB-game history from the Actions
cache. A cold cache rebuilds only those two durable feature sets, processing
one historical season at a time without storing old raw play-by-play.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.config import HISTORY_START_SEASON
from backend.etl import store
from backend.features.drives import build_team_games, kickoff_utc
from backend.features.qb import build_qb_games
from backend.features.scoring import build_model_games
from backend.nflverse import data


def missing_core_seasons(through_season: int) -> list[int]:
    """Historical seasons missing either required cached feature table."""
    required = range(HISTORY_START_SEASON, through_season + 1)
    team_games = set(store.processed_names("team_games"))
    qb_games = set(store.processed_names("qb_games"))
    return [
        season
        for season in required
        if str(season) not in team_games or str(season) not in qb_games
    ]


def bootstrap_history(through_season: int) -> list[int]:
    """Build missing historical core features without retaining raw PBP."""
    missing = missing_core_seasons(through_season)
    if not missing:
        print(f"history cache complete through {through_season}")
        return []

    schedules = data.load_schedules(missing)
    for season in missing:
        pbp = data.load_pbp([season])
        season_schedules = schedules[schedules["season"].eq(season)]
        team_games = build_team_games(pbp, season_schedules)
        model_games = build_model_games(team_games, season_schedules)
        store.write_processed(model_games, "team_games", f"{season}.parquet")
        store.write_processed(build_qb_games(pbp), "qb_games", f"{season}.parquet")
        print(f"bootstrapped {season}: {len(model_games)} games; raw PBP discarded")
    return missing


def upcoming_games(
    schedules: pd.DataFrame,
    season: int,
    hours: float,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Unplayed games beginning in the next window, independent of weekday."""
    now = now or datetime.now(UTC)
    slate = schedules[
        schedules["season"].eq(season) & schedules["home_score"].isna()
    ].copy()
    if slate.empty:
        return slate
    slate["start_date"] = kickoff_utc(slate)
    end = now + timedelta(hours=hours)
    return slate[slate["start_date"].ge(now) & slate["start_date"].le(end)].sort_values(
        "start_date"
    )


def incremental_game_ids(
    schedules: pd.DataFrame,
    existing_game_ids: list[set[str]],
    lookback_weeks: int,
) -> set[str]:
    """Recent played games plus any game absent from a feature artifact."""
    if lookback_weeks < 1:
        raise ValueError("lookback_weeks must be at least 1")
    played = schedules[schedules["home_score"].notna()].copy()
    if played.empty:
        return set()
    played["game_id"] = played["game_id"].astype(str)
    weeks = sorted(played["week"].astype(int).unique())
    recent_weeks = set(weeks[-lookback_weeks:])
    recent = set(played.loc[played["week"].isin(recent_weeks), "game_id"])
    complete = set.intersection(*existing_game_ids) if existing_game_ids else set()
    missing = set(played["game_id"]) - complete
    return recent | missing


def write_incremental_features(
    frame: pd.DataFrame,
    directory: str,
    season: int,
    sort_columns: list[str],
) -> None:
    """Replace every row for rebuilt game IDs and retain all other games."""
    if frame.empty:
        return
    game_ids = set(frame["game_id"].astype(str))
    try:
        existing = store.read_processed(directory, f"{season}.parquet")
        retained = existing[~existing["game_id"].astype(str).isin(game_ids)]
        combined = pd.concat([retained, frame], ignore_index=True)
    except FileNotFoundError:
        combined = frame
    combined = combined.sort_values(sort_columns).reset_index(drop=True)
    store.write_processed(combined, directory, f"{season}.parquet")
