"""Season acceptance: a complete league, completed results, and QB substitutions."""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.model.joint_scoring import DEFAULT_CONFIG, JointScoringFit
from backend.model.season_wins import project_season


def test_season_forecast_preserves_schedule_and_completed_results():
    teams = [f"T{i:02d}" for i in range(32)]
    rotation = teams.copy()
    rows = []
    for week in range(1, 18):
        for i in range(16):
            rows.append(
                {
                    "season": 2026,
                    "week": week,
                    "game_type": "REG",
                    "game_id": f"2026_{week:02d}_{i:02d}",
                    "home_team": rotation[i],
                    "away_team": rotation[-i - 1],
                    "location": "Home",
                    "home_score": np.nan,
                    "away_score": np.nan,
                    "gameday": f"2026-09-{week:02d}",
                    "gametime": "13:00",
                    "spread_line": 1000.0,
                }
            )
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    schedule = pd.DataFrame(rows)
    as_of = datetime(2026, 9, 30, tzinfo=UTC)
    fit = JointScoringFit(
        season=2026,
        week=1,
        as_of=as_of,
        teams=teams,
        offense_ppd=np.zeros(32),
        defense_ppd=np.zeros(32),
        pace=np.zeros(32),
        base_ppd=2,
        base_drives=10,
        hfa_ppd=0,
        parameter_covariance=np.diag([0.1] * 64 + [0.0]),
        score_residual_covariance=np.eye(2) * 60,
        config=DEFAULT_CONFIG,
    )
    totals, games = project_season(fit, schedule, {}, {}, as_of, simulations=10_000)
    assert len(totals) == 32 and len(games) == 272
    assert totals.projected_wins.eq(8.5).all()
    assert np.allclose(games.home_win_probability + games.away_win_probability, 1)
    assert totals.games_remaining.eq(17).all()
    # Shared strength uncertainty produces wider season ranges than the
    # independent Binomial(17, .5) middle 80% interval of 6-11 wins.
    assert (totals.wins_p90 - totals.wins_p10).median() > 5
    repeated, _ = project_season(
        fit, schedule.sample(frac=1, random_state=4), {}, {}, as_of, simulations=10_000
    )
    pd.testing.assert_frame_equal(totals, repeated)
    # A starter change follows the team through its entire remaining schedule.
    shifts = {
        r.game_id: (
            5.0 if r.home_team == "T00" else 0.0,
            5.0 if r.away_team == "T00" else 0.0,
        )
        for r in schedule.itertuples()
    }
    swapped, _ = project_season(fit, schedule, shifts, {}, as_of, simulations=1000)
    assert swapped.set_index("team_abbr").loc["T00", "projected_wins"] > 10
    assert np.isclose(swapped.projected_wins.sum(), 272)
    # Completed wins cannot be re-simulated, and a tie is never half a win.
    schedule.loc[0, ["home_score", "away_score"]] = [24, 14]
    schedule.loc[1, ["home_score", "away_score"]] = [17, 17]
    partial, audit = project_season(fit, schedule, {}, {}, as_of, simulations=1000)
    partial = partial.set_index("team_abbr")
    assert len(audit) == 270
    assert partial.loc[schedule.iloc[0].home_team, "wins"] == 1
    assert partial.loc[schedule.iloc[0].home_team, "projected_wins"] == 9
    assert partial.loc[schedule.iloc[1].home_team, "ties"] == 1
    assert partial.loc[schedule.iloc[1].home_team, "projected_wins"] == 8
    assert np.isclose(partial.projected_wins.sum(), 271)
    assert np.allclose(
        partial.projected_wins, partial.wins + partial.remaining_expected_wins
    )
    schedule["home_score"], schedule["away_score"] = 20, 10
    final, audit = project_season(fit, schedule, {}, {}, as_of, simulations=1000)
    assert audit.empty and final.games_remaining.eq(0).all()
    assert final.projected_wins.eq(final.wins).all()
    assert final.wins_p10.eq(final.wins_p90).all()
    for invalid in (
        schedule.iloc[:-1],
        pd.concat([schedule.iloc[:-1], schedule.iloc[[0]]]),
    ):
        with pytest.raises(ValueError, match="272 unique"):
            project_season(fit, invalid, {}, {}, as_of, simulations=1000)
    with pytest.raises(ValueError, match="at or after as_of"):
        project_season(
            fit, schedule, {}, {}, datetime(2026, 8, 1, tzinfo=UTC), simulations=1000
        )
    # The copula uses exactly the engine's score design and covariance.
    game = SimpleNamespace(home_team="T00", away_team="T31", neutral_site=False)
    design = fit.score_design(game)
    variance = (
        np.array([1.0, -1.0])
        @ (design @ fit.parameter_covariance @ design.T + fit.score_residual_covariance)
        @ np.array([1.0, -1.0])
    )
    assert np.isclose(
        fit.engine_projection(game).margin_sd ** 2,
        variance * DEFAULT_CONFIG.score_covariance_scale**2,
    )
