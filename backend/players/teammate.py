"""Teammate-relative splits: a player's rate against the others in the same context."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def relative_rate(
    events: pd.DataFrame,
    player_key: str = "athlete_id",
    context_keys: Sequence[str] = ("game_id", "team"),
    numerator: str = "numerator",
    denominator: str = "denominator",
    min_context_denominator: float = 5.0,
) -> pd.DataFrame:
    """One row per (player, context) with the numerator net of the others' rate.

    ``numerator = player_rate - others_rate`` times the player's denominator,
    where the others are the remaining players sharing the context. It is NaN
    when the others' summed denominator is below ``min_context_denominator``.
    """
    keys = list(context_keys)
    players = events.groupby([player_key, *keys], as_index=False)[
        [numerator, denominator]
    ].sum()
    totals = players.groupby(keys)[[numerator, denominator]].transform("sum")
    others_numerator = totals[numerator] - players[numerator]
    others_denominator = totals[denominator] - players[denominator]
    enough = others_denominator >= min_context_denominator
    others_rate = others_numerator / others_denominator.where(enough)
    players["numerator"] = np.where(
        enough, players[numerator] - players[denominator] * others_rate, np.nan
    )
    players["denominator"] = players[denominator]
    return players[[player_key, *keys, "numerator", "denominator"]]
