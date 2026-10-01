"""Pre-snap in-game state at every play about to be run.

Each row is what was knowable before the snap: the score and timeouts left by
earlier plays plus the situation of the play itself. nflverse records the
score before the play in posteam_score and defteam_score, and charges a
timeout on its own row after the play it follows, so no row carries its own
result. Rows that are not plays (timeouts, quarter ends) are dropped.
"""

import numpy as np
import pandas as pd

REGULATION_SECONDS = 3600.0
SCRIMMAGE, KICKOFF, EXTRA_POINT, TWO_POINT = range(4)

PBP_COLUMNS = [
    "season",
    "season_type",
    "game_id",
    "play_id",
    "home_team",
    "posteam",
    "defteam",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "play_type",
    "extra_point_attempt",
    "two_point_attempt",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_score",
    "defteam_score",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "home_wp",
    "vegas_home_wp",
]


def build_game_states(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per play, in nflverse play order."""
    # The team kicking off to open the game receives the second-half kickoff.
    opening_kicker = pbp.dropna(subset=["defteam"]).groupby("game_id")["defteam"]
    second_half_receiver = pbp["game_id"].map(opening_kicker.first())
    play_kind = pd.Series(
        np.select(
            [
                pbp["down"].notna(),
                pbp["play_type"].eq("kickoff"),
                pbp["two_point_attempt"].eq(1),
                pbp["extra_point_attempt"].eq(1),
            ],
            [SCRIMMAGE, KICKOFF, TWO_POINT, EXTRA_POINT],
            -1,
        ),
        index=pbp.index,
    )
    keep = (
        play_kind.ge(0) & pbp["posteam"].notna() & pbp["game_seconds_remaining"].notna()
    )
    plays, play_kind = pbp[keep], play_kind[keep]
    offense_is_home = plays["posteam"].eq(plays["home_team"])
    home_score = plays["posteam_score"].where(offense_is_home, plays["defteam_score"])
    away_score = plays["defteam_score"].where(offense_is_home, plays["posteam_score"])
    scrimmage = play_kind.eq(SCRIMMAGE)
    states = pd.DataFrame(
        {
            "season": plays["season"].astype(int),
            "season_type": plays["season_type"],
            "game_id": plays["game_id"],
            "play_id": plays["play_id"],
            "period": plays["qtr"].astype(int),
            "is_overtime": plays["qtr"].gt(4),
            # Both clocks restart in overtime.
            "seconds_remaining": plays["game_seconds_remaining"],
            "half_seconds_remaining": plays["half_seconds_remaining"],
            "score_margin": (home_score - away_score).astype(float),
            "offense_is_home": offense_is_home,
            "offense_receives_second_half": plays["qtr"].le(2)
            & plays["posteam"].eq(second_half_receiver[keep]),
            "play_kind": play_kind,
            "yards_to_goal": plays["yardline_100"].where(scrimmage),
            "down": plays["down"],
            "distance": plays["ydstogo"].where(scrimmage),
            "home_timeouts": plays["home_timeouts_remaining"].astype(float),
            "away_timeouts": plays["away_timeouts_remaining"].astype(float),
            "nflverse_home_wp": plays["home_wp"],
            "nflverse_vegas_home_wp": plays["vegas_home_wp"],
        }
    )
    if states[["score_margin", "home_timeouts", "away_timeouts"]].isna().any(axis=None):
        raise ValueError("Play-by-play is missing a pre-snap score or timeout count")
    return states.reset_index(drop=True)
