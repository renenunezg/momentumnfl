"""Published performance aggregates grade each source against the same finals."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.performance import build_graded_games, compute_performance_metrics

KICKOFF = datetime(2026, 9, 13, 17, tzinfo=UTC)


def _frozen(game_id, pure, blended, closing, home, away, qb=0.0, captured=None):
    return {
        "game_id": game_id,
        "season": 2026,
        "week": 1,
        "season_type": "REG",
        "start_date": KICKOFF,
        "neutral_site": False,
        "home_team_abbr": "H",
        "away_team_abbr": "A",
        "home_team": "Home",
        "away_team": "Away",
        "home_points": home,
        "away_points": away,
        "closing_spread": closing,
        "captured_home_spread": captured,
        "captured_total": None if captured is None else 46.0,
        "captured_source": None if captured is None else "espn:draftkings",
        "score_source": "nflverse schedules",
        "source_fetched_at": KICKOFF + timedelta(days=1),
        "forecast_week": 1,
        "forecast_as_of": KICKOFF - timedelta(hours=2),
        "forecast_recorded_at": KICKOFF - timedelta(hours=1),
        "model_version": "test",
        "div_game": False,
        "home_qb_adjustment": qb,
        "away_qb_adjustment": 0.0,
        "rest_adjustment": 0.0,
        "pure_home_margin": pure,
        "market_informed_home_margin": blended,
        "market_weight": 0.5,
        "forecast_market_home_spread": -blended,
        "model_total": 44.0,
        "market_informed_total": None,
        "margin_sd": 12.0,
        "total_sd": 13.0,
        "distribution": "bivariate_student_t",
        "degrees_of_freedom": 7.0,
    }


def test_metrics_grade_every_source_on_the_same_finals():
    frozen = pd.DataFrame(
        [
            _frozen(
                "g1",
                pure=4.0,
                blended=3.0,
                closing=-2.0,
                home=27,
                away=20,
                captured=-2.5,
            ),
            _frozen("g2", pure=-6.0, blended=-5.0, closing=-4.0, home=17, away=24),
            _frozen(
                "g3", pure=1.0, blended=2.0, closing=None, home=10, away=30, qb=-2.5
            ),
        ]
    )
    totals = pd.Series({"g1": 45.5, "g2": 41.0})
    graded = build_graded_games(frozen, totals, graded_at=KICKOFF + timedelta(days=2))
    assert graded["actual_margin"].tolist() == [7, -7, -20]
    # The captured close is preferred and labelled; nflverse fills the rest.
    assert graded["closing_spread"].tolist()[:2] == [-2.5, -4.0]
    assert graded["closing_total"].tolist()[:2] == [46.0, 41.0]
    assert graded["closing_source"].tolist() == ["espn:draftkings", "nflverse", None]
    # A frozen probability stays on the pure model's own dispersion.
    assert 0.5 < graded.loc[0, "home_win_probability"] < 0.7

    metrics = compute_performance_metrics(graded, computed_at=KICKOFF)
    overall = metrics[metrics["segment_kind"].eq("overall")].set_index(
        "prediction_source"
    )
    assert overall.loc["pure_model", "games"] == 3
    assert overall.loc["pure_model", "margin_mae"] == (3 + 1 + 21) / 3
    assert overall.loc["closing_market", "games"] == 2
    assert overall.loc["closing_market", "margin_mae"] == (4.5 + 11) / 2
    # Market columns use only the games that have a closing line.
    assert overall.loc["pure_model", "games_with_market"] == 2
    assert overall.loc["pure_model", "model_minus_market_mae"] == (3 + 1) / 2 - 7.75
    assert overall.loc["market_informed", "closer_than_market_share"] == 1.0
    assert overall.loc["market_informed", "total_games"] == 0
    assert overall.loc["pure_model", "thin_sample"]
    # The close is scored as a probability on its own games, and the gap
    # carries a paired interval around it.
    assert overall.loc["closing_market", "probability_games"] == 2
    low, high = overall.loc[
        "pure_model",
        ["model_minus_market_mae_ci_low", "model_minus_market_mae_ci_high"],
    ]
    assert low <= overall.loc["pure_model", "model_minus_market_mae"] <= high

    qb = metrics[metrics["segment_kind"].eq("qb_adjustment")]
    assert qb[qb["prediction_source"].eq("pure_model")]["segment"].tolist() == [
        "none",
        "1 point or more",
    ]
    assert set(metrics["segment_kind"]) == {
        "overall",
        "week",
        "model_favorite_size",
        "qb_adjustment",
        "division_game",
    }
