"""In-game states must never carry the result of the play about to run."""

import pandas as pd

from backend.features.ingame import (
    EXTRA_POINT,
    KICKOFF,
    SCRIMMAGE,
    build_game_states,
)


def test_states_are_pre_snap_at_scores_and_timeouts():
    # nflverse layout from 2024_01_ARI_BUF: the away touchdown, its try, the
    # kickoff Buffalo receives, then a home timeout charged on its own row.
    rows = [
        # posteam, play_type, down, posteam_score, defteam_score, home_timeouts
        ("ARI", "pass", 3.0, 0.0, 0.0, 3.0),
        ("ARI", "extra_point", None, 6.0, 0.0, 3.0),
        ("BUF", "kickoff", None, 0.0, 7.0, 3.0),
        ("BUF", "run", 1.0, 0.0, 7.0, 3.0),
        (None, "no_play", None, None, None, 2.0),
        ("BUF", "run", 2.0, 0.0, 7.0, 2.0),
    ]
    pbp = pd.DataFrame(
        rows,
        columns=[
            "posteam",
            "play_type",
            "down",
            "posteam_score",
            "defteam_score",
            "home_timeouts_remaining",
        ],
    ).assign(
        season=2024,
        season_type="REG",
        game_id="2024_01_ARI_BUF",
        play_id=range(len(rows)),
        home_team="BUF",
        defteam=lambda frame: frame.posteam.map({"ARI": "BUF", "BUF": "ARI"}),
        extra_point_attempt=lambda frame: frame.play_type.eq("extra_point") * 1.0,
        two_point_attempt=0.0,
        qtr=1.0,
        game_seconds_remaining=3000.0,
        half_seconds_remaining=1200.0,
        ydstogo=10.0,
        yardline_100=50.0,
        away_timeouts_remaining=3.0,
        home_wp=0.5,
        vegas_home_wp=0.5,
    )
    states = build_game_states(pbp)
    assert states["score_margin"].tolist() == [0, -6, -7, -7, -7]
    assert states["play_kind"].tolist() == [SCRIMMAGE, EXTRA_POINT, KICKOFF, 0, 0]
    # Arizona had the ball first, so Buffalo kicked off and receives after half.
    assert states["offense_receives_second_half"].tolist() == [0, 0, 1, 1, 1]
    assert states["home_timeouts"].tolist() == [3, 3, 3, 3, 2]
