"""The charting and tracking feeds the player ratings read beyond play-by-play.

FTN charting (reads, blitzers, play action, drops), PFR advanced stats
(pressures, yards after contact, broken tackles), Next Gen weekly stats and
the player id crosswalk all publish during the season and are stored under
the raw directory by season. The participation feed (players on the field per
play) is research only: it is published after the fact, so nothing served
reads it.
"""

import logging

import pandas as pd

from backend.config import RAW_DIR
from backend.etl.store import write_parquet
from backend.nflverse import data

log = logging.getLogger(__name__)

FTN_FIRST_SEASON = 2022
PFR_FIRST_SEASON = 2018
NGS_FIRST_SEASON = 2016
PFR_TYPES = ("pass", "rush", "rec", "def")
NGS_TYPES = ("passing", "rushing", "receiving")


def _put(frame: pd.DataFrame, *parts: str) -> None:
    write_parquet(frame, RAW_DIR.joinpath(*parts))
    log.info("player sources: %s %d rows", "/".join(parts), len(frame))


def ingest_player_sources(seasons: list[int], refresh_latest: bool = True) -> None:
    """Store every season's charting feeds that are missing, and refresh the
    latest season's so a weekly run sees the week just played."""
    seasons = sorted(set(seasons))
    latest = max(seasons)
    for season in seasons:
        fresh = refresh_latest and season == latest
        if season >= FTN_FIRST_SEASON:
            path = RAW_DIR / "ftn" / f"{season}.parquet"
            if fresh or not path.exists():
                _put(data.load_ftn_charting([season]), "ftn", f"{season}.parquet")
        if season >= PFR_FIRST_SEASON:
            for stat in PFR_TYPES:
                path = RAW_DIR / f"pfr_{stat}" / f"{season}.parquet"
                if fresh or not path.exists():
                    _put(
                        data.load_pfr_advstats([season], stat),
                        f"pfr_{stat}",
                        f"{season}.parquet",
                    )
        if season >= NGS_FIRST_SEASON:
            for stat in NGS_TYPES:
                path = RAW_DIR / "ngs" / f"{stat}_{season}.parquet"
                if fresh or not path.exists():
                    _put(
                        data.load_nextgen_stats([season], stat),
                        "ngs",
                        f"{stat}_{season}.parquet",
                    )
    _put(data.load_players(), "players.parquet")
