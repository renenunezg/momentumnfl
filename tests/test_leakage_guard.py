"""The fit must refuse to train on games at or after as_of."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.model.joint_scoring import fit_joint_scoring


def _toy_games() -> pd.DataFrame:
    rows = []
    for week in (1, 2, 3, 4, 5):
        rows.append(
            {
                "game_id": f"2023_{week:02d}_A_B",
                "season": 2023,
                "model_week": week,
                "home_team": "A",
                "away_team": "B",
                "neutral_site": False,
                "home_points": 24.0,
                "away_points": 20.0,
                "game_drives": 22.0,
                "home_epa_per_drive": 0.2,
                "away_epa_per_drive": -0.1,
                "start_date": datetime(2023, 9, week, tzinfo=UTC),
            }
        )
    return pd.DataFrame(rows)


def test_fit_refuses_future_games():
    games = _toy_games()
    with pytest.raises(ValueError):
        fit_joint_scoring(
            games,
            forecast_week=5,
            as_of=datetime(2023, 9, 1, tzinfo=UTC),
        )


def test_fit_accepts_clean_cut():
    fit = fit_joint_scoring(
        games=_toy_games(),
        forecast_week=5,
        as_of=datetime(2023, 9, 10, tzinfo=UTC),
    )
    assert fit.week == 5
    index = fit.team_index
    assert fit.offense_ppd[index["A"]] > fit.offense_ppd[index["B"]]
