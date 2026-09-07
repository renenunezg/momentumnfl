"""QB observations for context diagnostics; never changes cached engine inputs."""

import pandas as pd

from backend.config import HISTORY_START_SEASON, RAW_DIR
from backend.etl import store
from backend.features.drives import competitive_plays

PBP_COLUMNS = [
    "game_id",
    "posteam",
    "defteam",
    "qb_dropback",
    "qb_scramble",
    "passer_player_id",
    "rusher_player_id",
    "epa",
    "qb_epa",
    "qb_hit",
    "sack",
    "qtr",
    "score_differential",
]


def build_context_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Competitive QB EPA, including scramble attribution and hit/sack exposure.

    Hit/sack exposure is a consistent historical proxy, not charted pressure
    or a causal offensive-line grade. QB EPA removes receiver fumble losses.
    """
    plays = pbp[
        pbp["qb_dropback"].eq(1) & pbp["epa"].notna() & competitive_plays(pbp)
    ].copy()
    scramble = plays["qb_scramble"].eq(1)
    plays.loc[scramble, "passer_player_id"] = plays.loc[scramble, "rusher_player_id"]
    plays["qb_value"] = plays["qb_epa"].fillna(plays["epa"])
    plays["hit_sack"] = plays["qb_hit"].eq(1) | plays["sack"].eq(1)
    return (
        plays.groupby(
            ["game_id", "posteam", "defteam", "passer_player_id"], as_index=False
        )
        .agg(
            dropbacks=("qb_value", "size"),
            epa=("qb_value", "sum"),
            hit_sacks=("hit_sack", "sum"),
        )
        .rename(columns={"posteam": "team", "defteam": "opponent"})
    )


def load_context_games(season: int) -> pd.DataFrame:
    """Read existing raw history without fetching data or overwriting artifacts."""
    seasons = [
        s
        for s in range(HISTORY_START_SEASON, season + 1)
        if str(s) in store.processed_names("qb_games")
    ]
    missing = [s for s in seasons if not (RAW_DIR / "pbp" / f"{s}.parquet").exists()]
    if missing:
        raise ValueError(f"QB context diagnostics need raw PBP for seasons {missing}")
    frames = [
        build_context_games(
            pd.read_parquet(RAW_DIR / "pbp" / f"{s}.parquet", columns=PBP_COLUMNS)
        )
        for s in seasons
    ]
    if not frames:
        raise ValueError("QB context diagnostics need historical QB games")
    return pd.concat(frames, ignore_index=True).merge(
        store.game_index(seasons), on="game_id", validate="many_to_one"
    )
