"""Drive aggregation pinned against a known game (2023 opener, DET 21 @ KC 20)."""

from pathlib import Path

import pandas as pd

from backend.features.drives import build_team_games
from backend.features.qb import build_qb_games

FIXTURES = Path(__file__).parent / "fixtures"


def test_2023_opener_aggregates():
    pbp = pd.read_parquet(FIXTURES / "pbp_2023_01_DET_KC.parquet")
    schedules = pd.read_parquet(FIXTURES / "schedule_2023_01_DET_KC.parquet")
    tg = build_team_games(pbp, schedules)
    assert len(tg) == 1
    row = tg.iloc[0]
    assert row.home_team == "KC" and row.away_team == "DET"
    assert row.home_points == 20 and row.away_points == 21
    assert 8 <= row.home_drives <= 14 and 8 <= row.away_drives <= 14
    assert row.neutral_site == False  # noqa: E712
    assert row.home_competitive_points == 20
    assert row.away_competitive_points == 21
    # A dominant competitive start counts, but garbage-time scores cannot
    # inflate the competitive target or shorten the full-game pace input.
    blowout = pd.read_parquet(FIXTURES / "pbp_2023_07_DET_BAL.parquet")
    schedule = pd.read_parquet(FIXTURES / "schedule_2023_07_DET_BAL.parquet")
    built = build_team_games(blowout, schedule).iloc[0]
    assert built.game_drives == 20
    assert built.competitive_drives < built.game_drives
    assert built.home_competitive_points < built.home_points
    assert built.away_competitive_points < built.away_points
    shuffled = build_team_games(blowout.sample(frac=1, random_state=7), schedule)
    pd.testing.assert_series_equal(built, shuffled.iloc[0])
    # In-game injuries must not turn the replacement into the recorded starter.
    injury = pd.read_parquet(FIXTURES / "pbp_2023_01_BUF_NYJ.parquet")
    qbs = build_qb_games(injury)
    assert qbs.loc[qbs.team.eq("NYJ") & qbs.started, "passer"].item() == "A.Rodgers"
