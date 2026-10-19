"""The running back rating, version 3, on the shared rating engine.

The headline is the expected EPA per game over the rest of the season. Role
(carry share, touches, target rate) is what repeats game to game for backs,
but on the 2016 to 2024 development seasons carry share, touches per game and
the prior season added nothing to rest-of-season EPA per game once efficiency
was in, and with them the ridge flipped signs between collinear role
measures. Version 3 keeps what earned its place leaving one season out: EPA
per touch, rushing success, yards after catch and target rate. Those
efficiency measures repeat poorly, so the rating is an efficiency forecast
with a wide band; the role measures stay on the card as descriptive.
"""

import pandas as pd

from backend.players.qb_study import STUDY_DIR
from backend.players.rating import RatingSpec
from backend.players.rb_study import RB_RATES, build_rb_games

RB_SPEC = RatingSpec(
    position="RB",
    model_version="nfl_rb_rating_v3",
    rates=RB_RATES,
    kinds={
        "role": ["carry_share", "touches_game", "target_rate"],
        "quality": [
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
    target="epa_game",
    baselines={"epa_game": {"epa_game": 1.0}},
    volume="touches",
    min_volume=40,
    min_rest_volume=60,
    prior_min_volume=120,
    lower_is_better=frozenset({"stuff_rate", "fumble_rate"}),
)


def games_table(seasons: list[int], rebuild: bool = False) -> pd.DataFrame:
    path = STUDY_DIR / "rb_games.parquet"
    if path.exists() and not rebuild:
        games = pd.read_parquet(path)
        if set(seasons) <= set(games["season"].unique()):
            return games[games["season"].isin(seasons)]
    return build_rb_games(seasons)
