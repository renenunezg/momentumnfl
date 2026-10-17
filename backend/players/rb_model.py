"""The running back rating, version 2, on the shared rating engine.

The headline is the expected EPA per game over the rest of the season, which
is role times efficiency. The study found no efficiency measure that repeats
game to game for backs, and only role (carry share, touches, target rate)
that does, so the rating is role first by evidence, the efficiency measures
are shown with their low reliabilities, and the band says so.
"""

import pandas as pd

from backend.players.qb_study import STUDY_DIR
from backend.players.rating import RatingSpec
from backend.players.rb_study import RB_RATES, build_rb_games

RB_SPEC = RatingSpec(
    position="RB",
    model_version="nfl_rb_rating_v2",
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
    rating_features=[
        "epa_touch",
        "success_carry",
        "first_down_rate",
        "stuff_rate",
        "yac_rec",
        "carry_share",
        "target_rate",
        "touches_game",
    ],
    prior_features=["epa_touch", "carry_share", "touches_game"],
    target="epa_game",
    baseline="epa_game",
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
