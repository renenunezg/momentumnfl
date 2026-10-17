"""Kicker and punter traits.

A field goal is judged against what kicks of its distance usually make, from
the frozen expectation tables; a punt against the league's gross average
through the previous week. Neither is opponent adjusted.
"""

import pandas as pd

from backend.players import TraitSpec, register
from backend.players.context import SeasonContext
from backend.players.expectations import ExpectationTables, expected_field_goal
from backend.players.offense import GAME_KEYS, per_game, prior_totals, trait_rows

SPECIAL_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("k.field_goal", ("K",), "counted"),
        TraitSpec("p.punt_average", ("P",), "counted"),
    )
}
FIELD_GOAL_RESULTS = ("made", "missed", "blocked")


def _kick_rows(ctx: SeasonContext, play_type: str, player_column: str) -> pd.DataFrame:
    plays = ctx.plays[
        ctx.plays["play_type"].eq(play_type) & ctx.plays[player_column].notna()
    ]
    return plays.assign(
        athlete_id=plays[player_column].astype(str),
        team=plays["posteam"],
        opponent=plays["defteam"],
    )


def kicker_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """Field goals made above expectation per attempt."""
    kicks = _kick_rows(ctx, "field_goal", "kicker_player_id")
    kicks = kicks[
        kicks["field_goal_result"].isin(FIELD_GOAL_RESULTS)
        & kicks["kick_distance"].notna()
    ]
    over = kicks["field_goal_result"].eq("made").astype(float) - expected_field_goal(
        tables, kicks
    )
    return trait_rows(per_game(kicks, over, 1.0), SPECIAL_TRAITS["k.field_goal"])


def punter_traits(ctx: SeasonContext) -> pd.DataFrame:
    """Gross punt yards above the league average through the previous week, per punt."""
    punts = _kick_rows(ctx, "punt", "punter_player_id")
    punts = punts[punts["kick_distance"].notna()]
    games = per_game(punts, punts["kick_distance"].astype(float), 1.0).rename(
        columns={"numerator": "yards", "denominator": "punts"}
    )
    league = prior_totals(games, [], ["yards", "punts"])
    league["league_average"] = league["yards"] / league["punts"].where(
        league["punts"] > 0
    )
    merged = games.merge(
        league[["model_week", "league_average"]], on="model_week", how="left"
    )
    merged["numerator"] = merged["yards"] - merged["punts"] * merged["league_average"]
    merged["denominator"] = merged["punts"]
    return trait_rows(
        merged[[*GAME_KEYS, "numerator", "denominator"]],
        SPECIAL_TRAITS["p.punt_average"],
    )


def special_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    return pd.concat(
        [kicker_traits(ctx, tables), punter_traits(ctx)], ignore_index=True
    )
