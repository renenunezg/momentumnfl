"""Drive aggregation pinned against a known game (2023 opener, DET 21 @ KC 20)."""

from pathlib import Path

import pandas as pd

from backend.features.drives import build_team_games

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
