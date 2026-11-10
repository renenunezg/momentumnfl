"""The running back rating, version 5, on the shared rating engine.

The headline is the expected EPA over a replacement back per game over the
rest of the season: a forecast of EPA over replacement per touch times the
back's shrunk touches per game to date. The per-touch forecast keeps what
earned its place leaving one season out of 2016 to 2024: EPA per touch,
rushing success, yards after catch, target rate, receiving EPA per target and
rushing yards over expected. Carry share, touches per game and role as a
feature added nothing once efficiency was in; they stay on the card as
descriptive. Those efficiency measures repeat poorly, so the band is wide.

Version 5 pools each measure with the back's three previous seasons
(``HISTORY_DECAY``), adds the two measures above, and counts value over a
replacement back instead of the average one. Version 4 read this season
alone, so a proven back in a slow start sat below average and a hot half
season led the board. Leaving one season out of 2016 to 2024 the rating's
rank correlation with the rest of the season rises from 0.290 to 0.315
(paired gain 0.025, standard error 0.013) and its reliability at eight games
from 0.22 to 0.49; on held-out 2025, 0.154 to 0.188. Heavier history, role
measures, age, draft position and steadier targets (success, yards or first
downs over average) were each tried and did not help. The ceiling is low:
rest-of-season value per game is itself only 0.20 reliable at eight games.

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

# EPA per touch a replacement back gives up to the average one: every back's
# pooled EPA per touch less that of each team's backs outside its top two in
# touches, 2016 to 2024.
REPLACEMENT_GAP = 0.0289
# The share of its touches a season keeps per year back, from a continuous
# leave-one-season-out search on 2016 to 2024 (0.19 to 0.20 across feature
# sets); accuracy falls steadily past 0.25.
HISTORY_DECAY = 0.19
# EPA over a replacement back's rate per touch that season, per game: the
# target. Raw EPA per game sits below zero for the average back, so it
# rewarded playing less; this credits efficiency times workload, and a
# full-time average back above a part-time one.
RATES = {**RB_RATES, "epa_or_game": ("epa_or", "games_n")}

RB_SPEC = RatingSpec(
    position="RB",
    model_version="nfl_rb_rating_v5",
    rates=RATES,
    kinds={
        "role": ["carry_share", "touches_game", "target_rate"],
        "quality": [
            "epa_or_game",
            "epa_game",
            "epa_touch",
            "first_down_rate",
            "success_carry",
            "stuff_rate",
            "yac_rec",
            "rec_epa_target",
            "yoe_carry",
            "ngs_ryoe_att",
            "pfr_yac_carry",
            "pfr_broken_tackle_rate",
        ],
        "style": ["ngs_eight_box_rate", "pfr_ybc_carry", "ngs_expected_ypc"],
        "noise": [
            "epa_carry",
            "ypc",
            "explosive_rate",
            "fumble_rate",
            "short_yardage_conv",
            "ngs_efficiency",
        ],
    },
    rating_features=[
        "epa_touch",
        "success_carry",
        "yac_rec",
        "target_rate",
        "rec_epa_target",
        "yoe_carry",
    ],
    prior_features=[],
    target="epa_or_game",
    baselines={"epa_or_game": {"epa_or_game": 1.0}},
    prior_baselines=("epa_or_game",),
    volume="touches",
    workload="touches_game",
    min_volume=40,
    min_rest_volume=60,
    prior_min_volume=120,
    lower_is_better=frozenset({"stuff_rate", "fumble_rate"}),
    history_seasons=3,
    history_decay=HISTORY_DECAY,
)


def with_epa_over_replacement(games: pd.DataFrame) -> pd.DataFrame:
    """``epa_or``: a back's EPA less his touches at the season's pooled EPA per
    touch over every back in ``games``, lowered by ``REPLACEMENT_GAP``."""
    totals = games.groupby("season")[["epa_total", "touches"]].transform("sum")
    replacement = totals["epa_total"] / totals["touches"] - REPLACEMENT_GAP
    return games.assign(epa_or=games["epa_total"] - games["touches"] * replacement)


def games_table(seasons: list[int], rebuild: bool = False) -> pd.DataFrame:
    path = STUDY_DIR / "rb_games.parquet"
    if path.exists() and not rebuild:
        games = pd.read_parquet(path)
        if set(seasons) <= set(games["season"].unique()):
            return with_epa_over_replacement(games[games["season"].isin(seasons)])
    return with_epa_over_replacement(build_rb_games(seasons))
