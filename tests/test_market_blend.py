"""Blend math: weighted average, cap, and missing-market fallback."""

import numpy as np

from backend.model.market_blend import (
    blend_margin,
    cover_push_probabilities,
    fit_margin_residual_distribution,
)


def test_blend_margin():
    assert blend_margin(6.0, 2.0, 0.25) == 5.0
    assert blend_margin(6.0, 2.0, 0.9) == 4.0  # capped at 0.5
    assert blend_margin(6.0, None, 0.25) == 6.0
    assert blend_margin(6.0, float("nan"), 0.25) == 6.0


def test_cover_push_probabilities_sum_and_push_mass():
    residuals = np.array([-7, -3, -3, 0, 0, 3, 3, 3, 7, 10])
    dist = fit_margin_residual_distribution(residuals)
    assert abs(dist.sum() - 1.0) < 1e-12
    # model margin 3, offer home spread -3: pushes exactly when residual is 0
    cover, push = cover_push_probabilities(3.0, -3.0, dist)
    assert push > 0
    assert 0 < cover < 1
