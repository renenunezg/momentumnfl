"""The quarterback rating, version 2, on the shared rating engine.

The headline is the expected EPA per dropback over the rest of the season.
The measures that carry weight are the production core (EPA and success per
dropback, completion over expected, sacks, pressure to sack, late-down
conversion) and the processing measures from FTN charting (what happens past
the first read, the checkdown rate, sacks and success when blitzed), with the
prior season blended in. Every input publishes during the season.
"""

import pandas as pd

from backend.players.qb_study import RATES, STUDY_DIR, build_qb_games
from backend.players.rating import RatingSpec

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

QB_SPEC = RatingSpec(
    position="QB",
    model_version="nfl_qb_rating_v2",
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
        "cpoe",
        "sack_rate",
        "pfr_pressure_to_sack",
        "late_down_conv",
        "beyond_first_read_rate",
        "success_beyond_first",
        "checkdown_rate",
        "blitz_sack_rate",
        "blitz_success",
    ],
    prior_features=["epa_db", "success_db", "cpoe", "sack_rate"],
    target="epa_db",
    baseline="epa_db",
    volume="dropbacks",
    min_volume=60,
    min_rest_volume=100,
    prior_min_volume=100,
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
            return games[games["season"].isin(seasons)]
    return build_qb_games(seasons)
