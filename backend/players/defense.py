"""Defensive trait builders for DL, LB and DB.

Per-game traits have one row for every game a defender was present in, which
is every completed game of his team with a box row, a snap or a credited play
for him; a game without an event still scores zero. Counts come from the box
score, the win probability a sack swung and the stops come from the credited
plays, and the with-without impacts from the team's EPA allowed in games he
played against games he missed.

The opponent of a defender is the offense he faced, so ``adjust_rate`` nets
out how many such events that offense had allowed in earlier weeks.
"""

import pandas as pd

from backend.players import TraitSpec, register
from backend.players.absence import with_without
from backend.players.context import BOX_COLUMNS, SeasonContext, opponents
from backend.players.credit import scrimmage_plays
from backend.players.offense import (
    GAME_KEYS,
    attach_plays,
    opponent_adjusted,
    position_ids,
    trait_rows,
)

# Prior games of shrinkage for the with-without impact estimates: a sample
# floor on the product of games played and games missed, not a tuned value.
WITH_WITHOUT_SHRINK_GAMES = 6.0

DEFENSE_GROUPS = ("DL", "LB", "DB")
FRONT_GROUPS = ("DL", "LB")
STOP_ROLES = ("tackler", "tackler_for_loss", "sacker")

DEF_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("def.havoc", DEFENSE_GROUPS, "counted"),
        TraitSpec("def.pass_rush", FRONT_GROUPS, "counted"),
        TraitSpec("def.finishing", FRONT_GROUPS, "counted"),
        TraitSpec("def.run_stop", FRONT_GROUPS, "counted"),
        TraitSpec("def.stops", DEFENSE_GROUPS, "counted"),
        TraitSpec(
            "def.every_week", DEFENSE_GROUPS, "counted", aggregate="share_of_games"
        ),
        TraitSpec("def.tackle_share", ("LB",), "counted"),
        TraitSpec("def.coverage", ("LB",), "counted"),
        TraitSpec("def.ball_production", ("DB",), "counted"),
        TraitSpec("def.run_support", ("DB",), "counted", higher_is_better=None),
        TraitSpec("def.impact", DEFENSE_GROUPS, "inferred", aggregate="latest"),
        TraitSpec("def.pass_impact", ("DB",), "inferred", aggregate="latest"),
    )
}


def group_ids(positions: pd.DataFrame, groups: tuple[str, ...]) -> set[str]:
    return set().union(*(position_ids(positions, group) for group in groups))


def box_lines(stats: pd.DataFrame) -> pd.DataFrame:
    """The defensive box counts per player, team and game, one column per stat."""
    if stats.empty:
        return pd.DataFrame(
            columns=["athlete_id", "team", "game_id", *BOX_COLUMNS.values()]
        )
    present = [column for column in BOX_COLUMNS if column in stats]
    lines = stats[["athlete_id", "team", "game_id", *present]].rename(
        columns=BOX_COLUMNS
    )
    lines = lines.groupby(["athlete_id", "team", "game_id"], as_index=False).sum()
    for column in BOX_COLUMNS.values():
        if column not in lines:
            lines[column] = 0.0
        lines[column] = lines[column].fillna(0.0).astype(float)
    lines["tackles"] = lines["solo"] + lines["tackles_with_assist"] + lines["assists"]
    return lines


def defense_rows(ctx: SeasonContext) -> pd.DataFrame:
    """Competitive defensive credit rows with the play's situation attached."""
    credit = ctx.credit
    keep = credit["side"].eq("defense") & credit["is_competitive"].eq(True)
    return attach_plays(credit[keep], ctx.plays)


def defender_games(
    ctx: SeasonContext, rows: pd.DataFrame, defenders: set[str]
) -> pd.DataFrame:
    """One row per defender and game present, with box counts, sack WPA and stops."""
    played = ctx.presence[
        ctx.presence["present"] & ctx.presence["athlete_id"].isin(defenders)
    ]
    played = played.drop(columns="present").merge(
        opponents(ctx.games), on=["game_id", "team"], how="inner"
    )
    lines = box_lines(ctx.stats)
    out = played.merge(lines, on=["athlete_id", "team", "game_id"], how="left")
    sacks = rows[rows["role"].eq("sacker")]
    sack_wpa = (
        sacks.assign(sack_wpa=sacks["play_wpa"].abs().fillna(0.0) * sacks["share"])
        .groupby(["athlete_id", "team", "game_id"], as_index=False)["sack_wpa"]
        .sum()
    )
    stops = (
        rows[rows["role"].isin(STOP_ROLES)]
        .groupby(["athlete_id", "team", "game_id"], as_index=False)
        .size()
        .rename(columns={"size": "stops"})
    )
    out = out.merge(sack_wpa, on=["athlete_id", "team", "game_id"], how="left").merge(
        stops, on=["athlete_id", "team", "game_id"], how="left"
    )
    counts = [*BOX_COLUMNS.values(), "tackles", "sack_wpa", "stops"]
    out[counts] = out[counts].fillna(0.0)
    team_tackles = lines.groupby(["team", "game_id"], as_index=False)["tackles"].sum()
    return out.merge(
        team_tackles.rename(columns={"tackles": "team_tackles"}),
        on=["team", "game_id"],
        how="left",
    )


def events(frame: pd.DataFrame, numerator, denominator) -> pd.DataFrame:
    return frame[GAME_KEYS].assign(numerator=numerator, denominator=denominator)


def in_groups(frame, positions, groups):
    return frame[frame["athlete_id"].isin(group_ids(positions, groups))]


def _def_havoc(games, schedule):
    havoc = (
        games["tfl"]
        + games["forced_fumbles"]
        + games["passes_defended"]
        + games["interceptions"]
    )
    return trait_rows(
        opponent_adjusted(events(games, havoc, 1.0), schedule), DEF_TRAITS["def.havoc"]
    )


def sack_prices(sacks: pd.DataFrame, weeks: pd.Series) -> pd.Series:
    """Median absolute WPA of every tagged sack before each week, indexed by week."""
    plays = sacks.drop_duplicates("play_id")
    value = plays["play_wpa"].fillna(0.0).abs()
    return pd.Series(
        {
            week: value[plays["model_week"].lt(week)].median()
            for week in sorted(weeks.unique())
        },
        dtype=float,
    )


def _def_pass_rush(games, rows):
    """The WPA of the player's own sacks plus his hits priced at the median sack.

    A sack is worth its play's absolute WPA, split among its sackers. A hit is
    priced at the median absolute WPA of every tagged sack through the previous
    week; with no earlier sack (the season's first week) a game with hits has a
    NaN numerator.
    """
    price = games["model_week"].map(
        sack_prices(rows[rows["role"].eq("sacker")], games["model_week"])
    )
    hit_value = (games["qb_hits"] * price).where(games["qb_hits"] > 0, 0.0)
    numerator = games["sack_wpa"] + hit_value
    return trait_rows(events(games, numerator, 1.0), DEF_TRAITS["def.pass_rush"])


def _def_finishing(games):
    """Sacks per sack chance (sacks plus hits); a game with no chance gets no row."""
    chances = games["sacks"] + games["qb_hits"]
    out = events(games, games["sacks"], chances)
    return trait_rows(out[chances > 0], DEF_TRAITS["def.finishing"])


def _def_run_stop(games, schedule):
    run_stops = (games["tfl"] - games["sacks"]).clip(lower=0.0)
    return trait_rows(
        opponent_adjusted(events(games, run_stops, 1.0), schedule),
        DEF_TRAITS["def.run_stop"],
    )


def _def_stops(games, schedule):
    """Tackles, tackles for loss and sacks on plays the offense lost EPA on."""
    return trait_rows(
        opponent_adjusted(events(games, games["stops"], 1.0), schedule),
        DEF_TRAITS["def.stops"],
    )


def _def_every_week(games):
    disruptions = ["tfl", "sacks", "passes_defended", "interceptions", "forced_fumbles"]
    disrupted = games[disruptions].gt(0).any(axis=1).astype(float)
    return trait_rows(events(games, disrupted, 1.0), DEF_TRAITS["def.every_week"])


def _def_tackle_share(games):
    out = events(games, games["tackles"], games["team_tackles"])
    return trait_rows(out[games["team_tackles"] > 0], DEF_TRAITS["def.tackle_share"])


def _def_coverage(games, schedule):
    plays = games["passes_defended"] + games["interceptions"]
    return trait_rows(
        opponent_adjusted(events(games, plays, 1.0), schedule),
        DEF_TRAITS["def.coverage"],
    )


def _def_ball_production(games, schedule):
    plays = games["interceptions"] + games["passes_defended"]
    return trait_rows(
        opponent_adjusted(events(games, plays, 1.0), schedule),
        DEF_TRAITS["def.ball_production"],
    )


def _def_run_support(games, schedule):
    return trait_rows(
        opponent_adjusted(events(games, games["solo"], 1.0), schedule),
        DEF_TRAITS["def.run_support"],
    )


def epa_allowed(plays: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Each defense's competitive EPA allowed per play, all and passing, per game."""
    rows = scrimmage_plays(plays)
    rows = rows[rows["is_competitive"]]
    grouped = rows.groupby(["defteam", "game_id"])
    out = pd.DataFrame(
        {
            "epa_allowed": grouped["epa"].mean(),
            "pass_epa_allowed": rows[rows["play_type"].eq("pass")]
            .groupby(["defteam", "game_id"])["epa"]
            .mean(),
        }
    ).reset_index()
    out = out.rename(columns={"defteam": "team"})
    weeks = schedule[["game_id", "model_week"]].drop_duplicates("game_id")
    return out.merge(weeks, on="game_id", how="inner")


def weekly_impact(presence, unit_games, value_column, spec):
    """``with_without`` through each week over games of that week or earlier."""
    frames = []
    for week in sorted(presence["model_week"].unique()):
        impact = with_without(
            presence[presence["model_week"].le(week)],
            unit_games[unit_games["model_week"].le(week)],
            value_column,
            WITH_WITHOUT_SHRINK_GAMES,
        )
        frames.append(
            impact[["athlete_id"]].assign(
                model_week=week,
                game_id=None,
                numerator=impact["impact"],
                denominator=1.0,
            )
        )
    if not frames:
        return trait_rows(
            pd.DataFrame(
                columns=[
                    "athlete_id",
                    "model_week",
                    "game_id",
                    "numerator",
                    "denominator",
                ]
            ),
            spec,
        )
    return trait_rows(pd.concat(frames, ignore_index=True), spec)


def defense_traits(ctx: SeasonContext) -> pd.DataFrame:
    """The twelve DL, LB and DB traits in the trait frame contract.

    Per-game traits have one row per player and game; ``def.impact`` and
    ``def.pass_impact`` have one row per player and week with no ``game_id``.
    """
    rows = defense_rows(ctx)
    games = defender_games(ctx, rows, group_ids(ctx.positions, DEFENSE_GROUPS))
    front = in_groups(games, ctx.positions, FRONT_GROUPS)
    linebackers = in_groups(games, ctx.positions, ("LB",))
    backs = in_groups(games, ctx.positions, ("DB",))
    unit_games = epa_allowed(ctx.plays, ctx.games)
    presence = ctx.presence
    frames = [
        _def_havoc(games, ctx.games),
        _def_pass_rush(front, rows),
        _def_finishing(front),
        _def_run_stop(front, ctx.games),
        _def_stops(games, ctx.games),
        _def_every_week(games),
        _def_tackle_share(linebackers),
        _def_coverage(linebackers, ctx.games),
        _def_ball_production(backs, ctx.games),
        _def_run_support(backs, ctx.games),
        weekly_impact(
            in_groups(presence, ctx.positions, DEFENSE_GROUPS),
            unit_games,
            "epa_allowed",
            DEF_TRAITS["def.impact"],
        ),
        weekly_impact(
            in_groups(presence, ctx.positions, ("DB",)),
            unit_games,
            "pass_epa_allowed",
            DEF_TRAITS["def.pass_impact"],
        ),
    ]
    return pd.concat(frames, ignore_index=True)
