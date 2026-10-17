"""Offensive line trait builders.

The offensive line is rated as one unit per team: ``athlete_id`` is
``unit:{team}`` and there are no individual lineman rows, because the feed
tags no lineman on any play. Team dropbacks and carries come from the team's
competitive offensive credit rows and the opponent is the defense faced.

``ol.protection`` expects the league rate of pressure events (sacks and
quarterback hits) per dropback through the previous week, so the season's
first week has no expectation: its rows keep their dropbacks with a NaN
numerator and are left out of the opponent adjustment.
"""

import pandas as pd

from backend.features.units import _adjusted_line_yards
from backend.players import TraitSpec, register
from backend.players.context import SeasonContext
from backend.players.credit import unit_rows
from backend.players.expectations import ExpectationTables
from backend.players.offense import (
    GAME_KEYS,
    attach_plays,
    conversion_over_expected,
    dropbacks,
    is_short_yardage,
    opponent_adjusted,
    per_game,
    prior_totals,
    trait_rows,
)

LINE_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("ol.protection", ("OL",), "inferred"),
        TraitSpec("ol.stuff_rate", ("OL",), "inferred"),
        TraitSpec("ol.short_yardage", ("OL",), "inferred"),
        TraitSpec("ol.line_yards", ("OL",), "inferred"),
    )
}


def _ol_protection(all_dropbacks, games):
    """Pressure events avoided against the league rate through the previous week."""
    pressured = (all_dropbacks["is_sack"] | all_dropbacks["qb_hit"].eq(1)).astype(float)
    team = per_game(all_dropbacks, pressured, 1.0).rename(
        columns={"numerator": "pressures", "denominator": "dropbacks"}
    )
    league = prior_totals(team, [], ["pressures", "dropbacks"])
    league["league_rate"] = league["pressures"] / league["dropbacks"].where(
        league["dropbacks"] > 0
    )
    merged = team.merge(
        league[["model_week", "league_rate"]], on="model_week", how="left"
    )
    merged["numerator"] = (
        merged["dropbacks"] * merged["league_rate"] - merged["pressures"]
    )
    merged["denominator"] = merged["dropbacks"]
    events = merged[[*GAME_KEYS, "numerator", "denominator"]]
    known = events["numerator"].notna()
    adjusted = opponent_adjusted(events[known], games)
    return trait_rows(
        pd.concat([adjusted, events[~known]], ignore_index=True),
        LINE_TRAITS["ol.protection"],
    )


def _ol_stuff_rate(carries, games):
    """The negative of stuffed carries (no gain or a loss) per carry."""
    stuffed = carries["yards_gained"].le(0).astype(float)
    events = per_game(carries, -stuffed, 1.0)
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.stuff_rate"])


def _ol_short_yardage(carries, tables, games):
    plays = carries[is_short_yardage(carries)]
    events = per_game(plays, conversion_over_expected(plays, tables), 1.0)
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.short_yardage"])


def _ol_line_yards(carries, games):
    line_yards = _adjusted_line_yards(carries["yards_gained"].astype(float))
    events = per_game(carries, line_yards, 1.0)
    return trait_rows(opponent_adjusted(events, games), LINE_TRAITS["ol.line_yards"])


def line_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """The four offensive line traits, one unit row per team and game."""
    credit = ctx.credit
    rows = attach_plays(unit_rows(credit[credit["is_competitive"]]), ctx.plays)
    carries = rows[rows["role"].eq("rusher") & rows["yards_gained"].notna()]
    frames = [
        _ol_protection(dropbacks(rows), ctx.games),
        _ol_stuff_rate(carries, ctx.games),
        _ol_short_yardage(carries, tables, ctx.games),
        _ol_line_yards(carries, ctx.games),
    ]
    return pd.concat(frames, ignore_index=True)
