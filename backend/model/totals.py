"""Dedicated league scoring and pace priors for game totals.

These settings affect the common score baseline, never team strengths or the
margin distribution. The fixed 64-game priors were selected on 2016-2021
regular-season total RMSE; 2022-2025 remains retrospective validation.
"""

from dataclasses import dataclass
from math import isfinite

TOTALS_LAYER_VERSION = "totals_v1"
# Total pricing owns its market blend independently of spread pricing.
DEFAULT_TOTAL_MARKET_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class TotalsConfig:
    scoring_prior_games: float = 64.0
    pace_prior_games: float = 64.0
    prior_half_life_weeks: float = 6.0

    def __post_init__(self):
        for value in (self.scoring_prior_games, self.pace_prior_games):
            if not isfinite(value) or value < 0:
                raise ValueError("Totals prior games must be finite and nonnegative")
        if not isfinite(self.prior_half_life_weeks) or self.prior_half_life_weeks <= 0:
            raise ValueError("Totals prior half-life must be finite and positive")


DEFAULT_TOTALS_CONFIG = TotalsConfig()


def pool_environment(
    current_ppd,
    current_drives,
    prior_ppd,
    prior_drives,
    observed_weight,
    forecast_week,
    config=DEFAULT_TOTALS_CONFIG,
):
    """Pool prior and observed environments, decaying prior evidence by week."""
    if not all(
        isfinite(value) and value > 0
        for value in (
            current_ppd,
            current_drives,
            prior_ppd,
            prior_drives,
            observed_weight,
        )
    ):
        raise ValueError("Totals environment requires positive finite inputs")
    decay = 0.5 ** (max(forecast_week - 1, 0) / config.prior_half_life_weeks)

    def pool(current, prior, prior_games):
        if prior_games == 0:
            return current
        prior_weight = prior_games * decay
        return (current * observed_weight + prior * prior_weight) / (
            observed_weight + prior_weight
        )

    return (
        pool(current_ppd, prior_ppd, config.scoring_prior_games),
        pool(current_drives, prior_drives, config.pace_prior_games),
    )
