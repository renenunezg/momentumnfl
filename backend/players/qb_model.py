"""The quarterback rating, version 3, on the shared rating engine.

The headline is the expected EPA per dropback over the rest of the season.
The measures that carry weight are the production core (EPA and success per
dropback, sacks, late-down conversion) and the processing measures from FTN
charting (success past the first read, sacks and success when blitzed), with
the prior season blended in. Every input publishes during the season.

Version 3 drops completion over expected (current and prior season), pressure
to sack, the rate of throws past the first read and the checkdown rate: on the
2016 to 2024 development seasons each carried a standardised coefficient near
zero, and leaving them out lowered the leave-one-season-out error and raised
the rating's rank correlation with the rest of the season. They stay on the
card as descriptive measures.

The rating is judged against the public composites to date (EPA per dropback,
Ben Baldwin's EPA + CPOE, nflfastR's DAKOTA, ESPN's Total QBR) and against
EPA per dropback with the prior season on this same engine, which holds the
information fixed and asks what the other measures add.
"""

import functools
import json

import numpy as np
import pandas as pd

from backend.config import STATIC_DIR
from backend.players.qb_study import (
    RATES,
    STUDY_DIR,
    build_qb_games,
    rate,
    with_espn_qbr,
)
from backend.players.rating import RatingSpec

DAKOTA_GAM = STATIC_DIR / "players" / "dakota_gam.json"

PRODUCTION_RATES = {
    **{
        name: RATES[name]
        for name in (
            "epa_db",
            "success_db",
            "cpoe",
            "anya",
            "sack_rate",
            "late_down_conv",
            "explosive_rate",
            "epa_in_pocket",
            "epa_no_play_action",
            "catchable_rate",
            "int_rate",
            "int_worthy_rate",
            "turnover_wpa_db",
            "beyond_first_read_rate",
            "success_beyond_first",
            "epa_beyond_first",
            "checkdown_rate",
            "blitz_sack_rate",
            "blitz_success",
            "int_worthy_beyond_first",
            "ngs_time_to_throw",
            "scramble_rate",
            "out_of_pocket_rate",
            "play_action_rate",
            "adot",
            "deep_rate",
            "ngs_aggressiveness",
            "throwaway_rate",
            "motion_rate",
            "rpo_rate",
            "screen_rate",
            "scramble_epa",
        )
    },
    "pfr_pressure_rate": ("pfr_pressured", "pfr_db"),
    "pfr_pressure_to_sack": ("pfr_sacked", "pfr_pressured"),
}


@functools.cache
def _dakota_gam() -> dict:
    payload = json.loads(DAKOTA_GAM.read_text())
    return {
        "intercept": float(payload["intercept"]),
        **{
            name: {
                "grid": np.asarray(payload[name]["grid"], dtype=float),
                "term": np.asarray(payload[name]["term"], dtype=float),
            }
            for name in ("epa_per_play", "cpoe")
        },
    }


def dakota(sums: pd.DataFrame) -> pd.Series:
    """nflfastR's DAKOTA to date: its GAM (``target ~ s(cpoe) + s(epa_per_play)``
    from nflverse's ``dakota_model.Rdata``, each term tabulated in
    ``DAKOTA_GAM`` and read linearly) on EPA per play and CPOE over every pass
    and rush play. An input outside the table is an error, not a clamp."""
    gam = _dakota_gam()
    inputs = {
        "epa_per_play": rate(sums, "dakota_epa_play").to_numpy(float),
        "cpoe": rate(sums, "dakota_cpoe").fillna(0.0).to_numpy(float),
    }
    total = np.full(len(sums), gam["intercept"])
    for name, values in inputs.items():
        grid, term = gam[name]["grid"], gam[name]["term"]
        known = values[np.isfinite(values)]
        if known.size and (known.min() < grid[0] or known.max() > grid[-1]):
            raise ValueError(f"DAKOTA {name} outside the tabulated range")
        total = total + np.interp(values, grid, term)
    return pd.Series(total, index=sums.index)


def qbr(sums: pd.DataFrame) -> pd.Series:
    """ESPN's Total QBR (opponent adjusted) to date: the weekly figures weighted
    by their plays, which is what publishes during the season. ESPN's own
    season figure is a different aggregate and tracks this at Spearman 0.98."""
    return rate(sums, "qbr")


def qbr_raw(sums: pd.DataFrame) -> pd.Series:
    """ESPN's raw (not opponent adjusted) Total QBR to date, weighted the same."""
    return rate(sums, "qbr_raw")


QB_SPEC = RatingSpec(
    position="QB",
    model_version="nfl_qb_rating_v3",
    rates=PRODUCTION_RATES,
    kinds={
        "quality": [
            "epa_db",
            "success_db",
            "cpoe",
            "anya",
            "sack_rate",
            "pfr_pressure_to_sack",
            "late_down_conv",
            "epa_in_pocket",
            "epa_no_play_action",
            "catchable_rate",
        ],
        "processing": [
            "beyond_first_read_rate",
            "success_beyond_first",
            "epa_beyond_first",
            "checkdown_rate",
            "blitz_sack_rate",
            "blitz_success",
            "int_worthy_beyond_first",
        ],
        "style": [
            "ngs_time_to_throw",
            "scramble_rate",
            "out_of_pocket_rate",
            "play_action_rate",
            "adot",
            "deep_rate",
            "ngs_aggressiveness",
            "throwaway_rate",
            "motion_rate",
            "rpo_rate",
            "screen_rate",
            "pfr_pressure_rate",
        ],
        "noise": [
            "int_rate",
            "int_worthy_rate",
            "turnover_wpa_db",
            "explosive_rate",
            "scramble_epa",
        ],
    },
    rating_features=[
        "epa_db",
        "success_db",
        "sack_rate",
        "late_down_conv",
        "success_beyond_first",
        "blitz_sack_rate",
        "blitz_success",
    ],
    prior_features=["epa_db", "success_db", "sack_rate"],
    target="epa_db",
    baselines={
        "epa_db": {"epa_db": 1.0},
        # The public EPA + CPOE composite (Ben Baldwin, guga31bb/nfl
        # code/season_qb_stats.R): 0.248070 x EPA per play + 0.006686 x CPOE in
        # percentage points; the intercept does not change a rank.
        "epa_cpoe": {"epa_db": 0.248070, "cpoe": 0.006686},
        "dakota": dakota,
        "qbr": qbr,
        "qbr_raw": qbr_raw,
    },
    volume="dropbacks",
    min_volume=60,
    min_rest_volume=100,
    prior_min_volume=100,
    # EPA per dropback with the prior season on the rating's own engine: the
    # equal-information control the other measures must improve on.
    fitted_baselines={"epa_prior": ["epa_db", "prior_epa_db", "prior_games"]},
    prior_baselines=("dakota", "qbr"),
    lower_is_better=frozenset(
        {
            "sack_rate",
            "pfr_pressure_to_sack",
            "int_rate",
            "int_worthy_rate",
            "checkdown_rate",
            "blitz_sack_rate",
            "int_worthy_beyond_first",
        }
    ),
)


def games_table(seasons: list[int], rebuild: bool = False) -> pd.DataFrame:
    path = STUDY_DIR / "qb_games.parquet"
    if path.exists() and not rebuild:
        games = pd.read_parquet(path)
        if set(seasons) <= set(games["season"].unique()):
            games = games[games["season"].isin(seasons)]
            # Tables cached before raw QBR was a baseline lack its sums.
            return games if "qbr_raw_weighted" in games else with_espn_qbr(games)
    return build_qb_games(seasons)
