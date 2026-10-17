"""Importing this module registers every trait, so ``TRAITS`` is complete.

The position modules register their traits as they are imported; anything that
reads the registry (snapshots, the grade, the backtest) imports this module
first so that no trait is missing from a vote, a percentile or a weight.
"""

from backend.players import EPA_TRAIT, TRAITS
from backend.players.credit import epa_credit_frames
from backend.players.defense import DEF_TRAITS, defense_traits
from backend.players.line import LINE_TRAITS, line_traits
from backend.players.offense import (
    QB_TRAITS,
    RB_TRAITS,
    REC_TRAITS,
    qb_traits,
    rb_traits,
    receiver_traits,
)
from backend.players.special import SPECIAL_TRAITS, special_traits

__all__ = [
    "DEF_TRAITS",
    "EPA_TRAIT",
    "LINE_TRAITS",
    "QB_TRAITS",
    "RB_TRAITS",
    "REC_TRAITS",
    "SPECIAL_TRAITS",
    "TRAITS",
    "defense_traits",
    "epa_credit_frames",
    "line_traits",
    "qb_traits",
    "rb_traits",
    "receiver_traits",
    "special_traits",
]
