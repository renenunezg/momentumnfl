"""Output-side market blend and key-number-aware cover/push pricing.

The blend never touches ratings: it moves the published margin toward the
market line by a capped weight. Cover and push probabilities come from a
discrete distribution of integer margin residuals (NFL margins mass on
3, 6, 7, 10), not a continuous CDF."""

import numpy as np

MARKET_WEIGHT_CAP = 0.5
RESIDUAL_RANGE = 60  # residual bins span [-60, 60]
LAPLACE_SMOOTHING = 0.5


def blend_margin(
    pure_margin: float, market_margin: float | None, weight: float
) -> float:
    """market_margin is -market home spread. Falls back to pure when the
    market is missing."""
    w = min(weight, MARKET_WEIGHT_CAP)
    if market_margin is None or (
        isinstance(market_margin, float) and np.isnan(market_margin)
    ):
        return pure_margin
    return (1.0 - w) * pure_margin + w * market_margin


def fit_margin_residual_distribution(residuals: np.ndarray) -> np.ndarray:
    """P(actual - model = k) for integer k in [-RESIDUAL_RANGE, RESIDUAL_RANGE],
    from integer-rounded residuals with Laplace smoothing."""
    bins = np.arange(-RESIDUAL_RANGE, RESIDUAL_RANGE + 1)
    rounded = np.clip(
        np.round(np.asarray(residuals, dtype=float)),
        -RESIDUAL_RANGE,
        RESIDUAL_RANGE,
    ).astype(int)
    counts = np.bincount(
        rounded + RESIDUAL_RANGE, minlength=len(bins)
    ).astype(float)
    counts += LAPLACE_SMOOTHING
    return counts / counts.sum()


def cover_push_probabilities(
    model_margin: float, offer_home_spread: float, distribution: np.ndarray
) -> tuple[float, float]:
    """P(home covers offer_home_spread), P(push), from the residual
    distribution shifted to the model margin. Home covers when
    actual_margin + offer_home_spread > 0."""
    margins = (
        np.arange(-RESIDUAL_RANGE, RESIDUAL_RANGE + 1)
        + round(model_margin)
    )
    edge = margins + offer_home_spread
    cover = float(distribution[edge > 1e-9].sum())
    push = float(distribution[np.abs(edge) <= 1e-9].sum())
    return cover, push
