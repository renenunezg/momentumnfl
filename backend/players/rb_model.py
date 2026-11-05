"""The running back rating, version 4, on the shared rating engine.

The headline is the expected EPA over the average back per game over the rest
of the season: a forecast of EPA over average per touch times the back's
shrunk touches per game to date. The per-touch forecast keeps what earned its
place leaving one season out of 2016 to 2024: EPA per touch, rushing success,
yards after catch and target rate. Carry share, touches per game and the
prior season added nothing once efficiency was in, and with them the ridge
flipped signs between collinear role measures; they stay on the card as
descriptive. Those efficiency measures repeat poorly, so the band is wide.

Version 3 forecast raw EPA per game from efficiency alone. With the average
back below zero per touch that rewarded playing less, and a part-time
receiving back on 50 touches led the board. On the development seasons the
workload product ranks the rest of the season as well as a linear fit on the
same target (Spearman 0.292 against 0.294), and touches per game as an added
linear feature did not help (0.292).
"""

import pandas as pd

from backend.players.qb_study import STUDY_DIR
from backend.players.rating import RatingSpec
from backend.players.rb_study import RB_RATES, build_rb_games

# EPA over the average back's rate per touch that season, per game: the
# target. Raw EPA per game sits below zero for the average back, so it
# rewarded playing less; this credits efficiency times workload.
RATES = {**RB_RATES, "epa_oe_game": ("epa_oe", "games_n")}

RB_SPEC = RatingSpec(
    position="RB",
    model_version="nfl_rb_rating_v4",
    rates=RATES,
    kinds={
        "role": ["carry_share", "touches_game", "target_rate"],
        "quality": [
            "epa_oe_game",
            "epa_game",
            "epa_touch",
            "first_down_rate",
            "success_carry",
            "stuff_rate",
            "yac_rec",
            "ngs_ryoe_att",
            "pfr_yac_carry",
            "pfr_broken_tackle_rate",
        ],
        "style": ["ngs_eight_box_rate", "pfr_ybc_carry", "ngs_expected_ypc"],
        "noise": [
            "epa_carry",
            "ypc",
            "yoe_carry",
            "explosive_rate",
            "rec_epa_target",
            "fumble_rate",
            "short_yardage_conv",
            "ngs_efficiency",
        ],
    },
    rating_features=["epa_touch", "success_carry", "yac_rec", "target_rate"],
    prior_features=[],
    target="epa_oe_game",
    baselines={"epa_oe_game": {"epa_oe_game": 1.0}},
    prior_baselines=("epa_oe_game",),
    volume="touches",
    workload="touches_game",
    min_volume=40,
    min_rest_volume=60,
    prior_min_volume=120,
    lower_is_better=frozenset({"stuff_rate", "fumble_rate"}),
)


def with_epa_over_average(games: pd.DataFrame) -> pd.DataFrame:
    """``epa_oe``: a back's EPA less his touches at the season's pooled EPA per
    touch over every back in ``games``."""
    totals = games.groupby("season")[["epa_total", "touches"]].transform("sum")
    average = totals["epa_total"] / totals["touches"]
    return games.assign(epa_oe=games["epa_total"] - games["touches"] * average)


def games_table(seasons: list[int], rebuild: bool = False) -> pd.DataFrame:
    path = STUDY_DIR / "rb_games.parquet"
    if path.exists() and not rebuild:
        games = pd.read_parquet(path)
        if set(seasons) <= set(games["season"].unique()):
            return with_epa_over_average(games[games["season"].isin(seasons)])
    return with_epa_over_average(build_rb_games(seasons))
