"""Output market blend and matchup-specific probabilities on actual NFL margins."""

import numpy as np
from scipy.stats import t as student_t

from backend.model.distributions import student_t_scale

MARKET_WEIGHT_CAP = 0.5
MARGIN_RANGE = 100
MARGINS = np.arange(-MARGIN_RANGE, MARGIN_RANGE + 1)


def capped_weight(weight: float) -> float:
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Market weight must be finite and nonnegative")
    return min(weight, MARKET_WEIGHT_CAP)


def blend_margin(pure_margin, market_margin, weight: float):
    """Missing market observations leave the pure forecast unchanged."""
    w = capped_weight(weight)
    pure = np.asarray(pure_margin, dtype=float)
    market = np.asarray(market_margin, dtype=float)
    blended = np.where(np.isnan(market), pure, (1.0 - w) * pure + w * market)
    return float(blended) if blended.ndim == 0 else blended


def integer_margin_probabilities(margin, sd, df, weights: np.ndarray) -> np.ndarray:
    """Smooth location/scale changes with fixed key numbers in outcome space.

    Tail mass is folded into the two endpoint bins. Prices outside the support
    are refused rather than silently treating truncated tails as impossible.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != MARGINS.shape or not np.isfinite(weights).all():
        raise ValueError("Expected calibrated actual-margin weights, schema v2")
    if (weights <= 0).any() or not np.isfinite(margin).all():
        raise ValueError("Margin weights must be positive and margins finite")
    scale = student_t_scale(np.asarray(sd), df)
    boundaries = MARGINS[:-1] + 0.5
    cdf = student_t.cdf(
        (boundaries - np.asarray(margin)[..., None]) / scale[..., None], df
    )
    probabilities = (
        np.diff(
            np.concatenate(
                [np.zeros(cdf.shape[:-1] + (1,)), cdf, np.ones(cdf.shape[:-1] + (1,))],
                axis=-1,
            ),
            axis=-1,
        )
        * weights
    )
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def fit_margin_distribution(development, df: float) -> np.ndarray:
    """Estimate key-number multipliers only from chronological development rows."""
    from backend.config import DEVELOPMENT_SEASONS

    if development.empty or not development["season"].isin(DEVELOPMENT_SEASONS).all():
        raise ValueError(
            "Margin distribution fitting requires development seasons only"
        )
    observed = development["actual_margin"].to_numpy(float)
    if not np.isfinite(observed).all() or (observed != np.round(observed)).any():
        raise ValueError("Observed margins must be finite integers")
    expected = integer_margin_probabilities(
        development["model_margin"].to_numpy(),
        development["margin_sd"].to_numpy(),
        df,
        np.ones(len(MARGINS)),
    ).sum(axis=0)
    counts = np.bincount(
        np.clip(observed, -MARGIN_RANGE, MARGIN_RANGE).astype(int) + MARGIN_RANGE,
        minlength=len(MARGINS),
    )
    # Pool home/away orientations and shrink sparse bins toward no adjustment.
    return (counts + counts[::-1] + 20.0) / (expected + expected[::-1] + 20.0)


def cover_push_probabilities(
    model_margin: float,
    offer_home_spread: float,
    distribution: np.ndarray,
    margin_sd: float,
    degrees_of_freedom: float,
) -> tuple[float, float]:
    if not np.isfinite(offer_home_spread) or abs(offer_home_spread) >= MARGIN_RANGE:
        raise ValueError("Spread lies outside the supported margin range")
    probabilities = integer_margin_probabilities(
        model_margin, margin_sd, degrees_of_freedom, distribution
    )
    edge = MARGINS + offer_home_spread
    return (
        float(probabilities[edge > 1e-9].sum()),
        float(probabilities[np.abs(edge) <= 1e-9].sum()),
    )
