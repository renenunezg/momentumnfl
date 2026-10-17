"""Position traits must never read the week being rated, and the frozen grade
model must refuse a season it was developed on."""

import pandas as pd
import pytest

from backend.players.adjust import adjust_rate
from backend.players.grade import WEIGHT_COLUMNS, GradeModel, validate_grade_model


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "model_week": [1, 2, 3],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "completed": [True, True, True],
        }
    )


def test_opponent_adjustment_uses_only_earlier_weeks():
    # Against B: a 1.0 rate in week 1, 0.0 in week 2, 1.0 in week 3. Against C
    # (the league's other opponent): 0.0 every week.
    events = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g1", "g2", "g3"],
            "opponent": ["B", "B", "B", "C", "C", "C"],
            "numerator": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "denominator": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    adjusted = adjust_rate(events, _games())
    # Week 1 has no prior evidence and is untouched.
    assert adjusted.loc[0, "numerator"] == 1.0
    # Week 2 against B: B allowed 1.0 through week 1, the league 0.5, so the
    # 0.0 numerator is marked up by 0.5. Week 3's result plays no part.
    assert adjusted.loc[1, "numerator"] == pytest.approx(-0.5)
    # Week 3 against B: B allowed 0.5 through week 2, the league 0.25.
    assert adjusted.loc[2, "numerator"] == pytest.approx(1.0 - 0.25)
    # Re-running with week 3 removed leaves weeks 1 and 2 identical.
    earlier = adjust_rate(events[events["game_id"].ne("g3")], _games())
    assert (
        earlier["numerator"].tolist()
        == adjusted.loc[[0, 1, 3, 4], "numerator"].tolist()
    )


def test_grade_model_refuses_a_season_it_was_developed_on():
    weights = pd.DataFrame(columns=WEIGHT_COLUMNS)
    model = GradeModel(weights=weights, dev_seasons=[2016, 2024])
    validate_grade_model(model, 2025)
    with pytest.raises(ValueError):
        validate_grade_model(model, 2024)
    with pytest.raises(ValueError):
        validate_grade_model(
            GradeModel(weights=weights, dev_seasons=[2016], model_version="other"), 2025
        )
