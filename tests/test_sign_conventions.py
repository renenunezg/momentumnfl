"""The one sign convention everything depends on, pinned against a real game:
nflverse spread_line is positive when the home team is favored, so the stored
sportsbook home line is its negation. 2023_01_DET_KC: KC favored by 4,
final score home 20 away 21."""

from pathlib import Path

import pandas as pd

from backend.features.drives import build_team_games
from backend.features.scoring import build_model_games

FIXTURES = Path(__file__).parent / "fixtures"


def test_spread_line_sign_pin():
    pbp = pd.read_parquet(FIXTURES / "pbp_2023_01_DET_KC.parquet")
    schedules = pd.read_parquet(FIXTURES / "schedule_2023_01_DET_KC.parquet")
    games = build_model_games(build_team_games(pbp, schedules), schedules)
    g = games.iloc[0]
    assert g.closing_spread == -4.0
    assert g.actual_margin == -1.0
    assert g.model_week == 1
