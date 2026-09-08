"""Blend math: weighted average, cap, and missing-market fallback."""

import pandas as pd
import pytest

from backend.model.market_blend import (
    MARGINS,
    blend_margin,
    cover_push_probabilities,
    fit_margin_distribution,
    integer_margin_probabilities,
)


def test_blend_margin():
    assert blend_margin(6.0, 2.0, 0.25) == 5.0
    assert blend_margin(6.0, 2.0, 0.9) == 4.0  # capped at 0.5
    assert blend_margin(6.0, None, 0.25) == 6.0
    assert blend_margin(6.0, float("nan"), 0.25) == 6.0


def test_cover_push_probabilities_sum_and_push_mass():
    frame = pd.DataFrame(
        {
            "actual_margin": [-7, -3, -3, 0, 0, 3, 3, 3, 7, 10],
            "model_margin": 3.0,
            "margin_sd": 13.0,
            "season": 2016,
        }
    )
    dist = fit_margin_distribution(frame, 7)
    cover, push = cover_push_probabilities(3.0, -3.0, dist, 13, 7)
    assert push > 0
    assert 0 < cover < 1
    mass = integer_margin_probabilities(3, 13, 7, dist)
    assert mass.sum() == pytest.approx(1)
    assert push == pytest.approx(mass[MARGINS == 3].item())
    assert cover_push_probabilities(3, -3.5, dist, 13, 7)[1] == 0
    left = cover_push_probabilities(2.49, -3, dist, 13, 7)[0]
    right = cover_push_probabilities(2.51, -3, dist, 13, 7)[0]
    assert 0 < right - left < 0.001
    assert (
        cover_push_probabilities(7, -3, dist, 8, 7)[0]
        > cover_push_probabilities(7, -3, dist, 18, 7)[0]
    )
    with pytest.raises(ValueError, match="development seasons only"):
        fit_margin_distribution(frame.assign(season=2022), 7)
    # Exercise the production offer-pricing path with the same marginal law.
    from backend.odds.markets import compare_priced_offers

    projection = pd.DataFrame(
        [
            {
                "game_id": "game",
                "home_team": "Home",
                "away_team": "Away",
                "home_margin": 3.0,
                "home_spread": -3.0,
                "model_total": 44.0,
                "margin_sd": 13.0,
                "total_sd": 14.0,
                "degrees_of_freedom": 7.0,
                "as_of": "2026-09-01T00:00:00Z",
                "start_date": "2026-09-09T00:00:00Z",
            }
        ]
    )
    offers = pd.DataFrame(
        [
            {
                "game_id": "game",
                "point": -3.0,
                "price": -110.0,
                "market": "spreads",
                "selection": "home",
                "execution_eligibility_verified": False,
                "provider": "Book",
                "provider_key": "book",
                "provider_last_update": None,
                "event_link": None,
                "market_link": None,
                "bet_link": None,
            }
        ]
    )
    priced = compare_priced_offers(projection, offers, dist).iloc[0]
    assert priced.best_offer_model_cover_probability == pytest.approx(cover)
    assert priced.best_offer_expected_value_per_unit == pytest.approx(
        cover * (100 / 110) - (1 - cover - push)
    )
    assert priced.recommendation_status == "not_recommended"
