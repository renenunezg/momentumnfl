"""Offensive trait builders for quarterbacks, backs, receivers and tight ends.

Every builder reads the competitive credit rows with the play's situation
attached. A dropback is a passer credit row: completions, incompletions, sacks
taken and interceptions thrown. A target is a receiver credit row, which the
feed tags on completions and incompletions alike. A carry is a rusher row,
scrambles included. A lost fumble carries only the fumbler's row.
"""

import numpy as np
import pandas as pd

from backend.players import TraitSpec, register
from backend.players.adjust import adjust_rate
from backend.players.context import SeasonContext
from backend.players.credit import TRAIT_COLUMNS
from backend.players.expectations import (
    ExpectationTables,
    expected_conversion,
    expected_rush_yards,
)
from backend.players.teammate import relative_rate

GAME_KEYS = ["athlete_id", "game_id", "model_week", "team", "opponent"]
PLAY_COLUMNS = [
    "down",
    "ydstogo",
    "yardline_100",
    "yards_gained",
    "qtr",
    "quarter_seconds_remaining",
    "is_success",
    "play_type",
    "wpa",
    "epa",
    "complete_pass",
    "interception",
    "sack",
    "qb_hit",
    "cpoe",
    "yards_after_catch",
    "xyac_mean_yardage",
    "passer_player_id",
    "rush_attempt",
]

LATE_DOWNS = (3, 4)
TWO_MINUTE_PERIODS = (2, 4)
TWO_MINUTE_SECONDS = 120
EXPLOSIVE_PASS_YARDS = 20
FLOOR_MIN_DROPBACKS = 10
PRESSURE_MIN_OTHER_DROPBACKS = 30
RECEIVER_MIN_OTHER_TARGETS = 10
BREAKAWAY_RUSH_YARDS = 10
SHORT_YARDAGE_DISTANCE = 2
GOAL_LINE_YARDS = 3

QB_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("qb.drive_engine", ("QB",), "counted"),
        TraitSpec("qb.explosive", ("QB",), "counted"),
        TraitSpec("qb.accuracy", ("QB",), "counted"),
        TraitSpec("qb.ball_security", ("QB",), "counted"),
        TraitSpec("qb.floor", ("QB",), "counted", aggregate="quantile20"),
        TraitSpec("qb.legs", ("QB",), "counted"),
        TraitSpec("qb.pressure_handling", ("QB",), "inferred"),
        TraitSpec("qb.receiver_independence", ("QB",), "inferred"),
    )
}

RB_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("rb.yards_over_expected", ("RB",), "counted"),
        TraitSpec("rb.stuff_avoidance", ("RB",), "inferred"),
        TraitSpec("rb.breakaway", ("RB",), "counted"),
        TraitSpec("rb.short_yardage", ("RB",), "counted"),
        TraitSpec("rb.receiving", ("RB",), "counted"),
        TraitSpec("rb.ball_security", ("RB",), "counted"),
        TraitSpec("rb.line_independence", ("RB",), "inferred"),
    )
}

RECEIVER_GROUPS = ("WR", "TE")
REC_TRAITS = {
    spec.name: register(spec)
    for spec in (
        TraitSpec("rec.target_share", RECEIVER_GROUPS, "counted"),
        TraitSpec("rec.value_per_target", RECEIVER_GROUPS, "counted"),
        TraitSpec("rec.explosive", RECEIVER_GROUPS, "counted"),
        TraitSpec("rec.late_down", RECEIVER_GROUPS, "counted"),
        TraitSpec("rec.yac_over_expected", RECEIVER_GROUPS, "counted"),
        TraitSpec("rec.qb_independence", RECEIVER_GROUPS, "inferred"),
    )
}


def attach_plays(credit: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Credit rows with the play's situation; the play's own WPA is ``play_wpa``."""
    situation = plays[["play_id", *PLAY_COLUMNS]].rename(columns={"wpa": "play_wpa"})
    out = credit.merge(situation, on="play_id", how="left")
    out["is_success"] = out["is_success"].eq(True)
    return out


def offense_rows(ctx: SeasonContext) -> pd.DataFrame:
    """Competitive offensive credit rows with the play's situation attached."""
    credit = ctx.credit
    keep = credit["side"].eq("offense") & credit["is_competitive"].eq(True)
    return attach_plays(credit[keep], ctx.plays)


def position_ids(positions: pd.DataFrame, group: str) -> set[str]:
    return set(
        positions.loc[positions["position_group"].eq(group), "athlete_id"].astype(str)
    )


def dropbacks(rows: pd.DataFrame) -> pd.DataFrame:
    """Passer rows flagged as sack, interception, incompletion or completion."""
    out = rows[rows["role"].eq("passer")].copy()
    out["is_sack"] = out["sack"].eq(1)
    out["is_interception"] = out["interception"].eq(1)
    out["is_completion"] = out["complete_pass"].eq(1)
    out["is_incompletion"] = ~(
        out["is_sack"] | out["is_interception"] | out["is_completion"]
    )
    return out


def is_late_down(rows: pd.DataFrame) -> pd.Series:
    return rows["down"].isin(LATE_DOWNS)


def is_short_yardage(rows: pd.DataFrame) -> pd.Series:
    """Third or fourth and two or less, or inside the three."""
    short = rows["ydstogo"].le(SHORT_YARDAGE_DISTANCE) & is_late_down(rows)
    return short | rows["yardline_100"].le(GOAL_LINE_YARDS)


def is_two_minute(rows: pd.DataFrame) -> pd.Series:
    return rows["qtr"].isin(TWO_MINUTE_PERIODS) & rows["quarter_seconds_remaining"].le(
        TWO_MINUTE_SECONDS
    )


def conversion_over_expected(
    rows: pd.DataFrame, tables: ExpectationTables
) -> pd.Series:
    return rows["is_success"].astype(float) - expected_conversion(tables, rows)


def per_game(rows: pd.DataFrame, numerator, denominator) -> pd.DataFrame:
    """Sum a numerator and denominator per player and game."""
    frame = rows[GAME_KEYS].assign(numerator=numerator, denominator=denominator)
    return frame.groupby(GAME_KEYS, as_index=False)[["numerator", "denominator"]].sum()


def prior_totals(
    frame: pd.DataFrame, keys: list[str], columns: list[str]
) -> pd.DataFrame:
    """Sums of ``columns`` per ``keys`` over weeks strictly before each week."""
    weekly = (
        frame.groupby([*keys, "model_week"], as_index=False)[columns]
        .sum()
        .sort_values([*keys, "model_week"])
    )
    running = (
        weekly.groupby(keys)[columns].cumsum() if keys else weekly[columns].cumsum()
    )
    weekly[columns] = running - weekly[columns]
    return weekly


def opponent_adjusted(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """``adjust_rate`` by the opponent faced; the week comes from ``games``."""
    adjusted = adjust_rate(events.drop(columns="model_week"), games)
    return adjusted.assign(model_week=events["model_week"].to_numpy())


def trait_rows(frame: pd.DataFrame, spec: TraitSpec) -> pd.DataFrame:
    """Rows in the trait frame contract for one trait."""
    out = frame.assign(trait=spec.name, kind=spec.kind)
    out = out.astype({"numerator": float, "denominator": float})
    return out[TRAIT_COLUMNS]


def target_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Receiver credit rows with ``passer_id`` and ``is_reception``."""
    targets = rows[rows["role"].eq("receiver")].copy()
    targets["passer_id"] = targets["passer_player_id"].astype(str)
    targets["is_reception"] = targets["complete_pass"].eq(1)
    return targets


def teammate_relative(events: pd.DataFrame, context_keys: list[str]) -> pd.DataFrame:
    """``relative_rate`` over ``context_keys`` with the events' game keys kept."""
    relative = relative_rate(events, context_keys=context_keys)
    keys = events[list(dict.fromkeys([*GAME_KEYS, *context_keys]))].drop_duplicates()
    return relative.merge(keys, on=["athlete_id", *context_keys], how="left")


# Quarterbacks


def _qb_drive_engine(qb_rows, tables, games):
    plays = qb_rows[qb_rows["role"].isin(["passer", "rusher"])]
    plays = plays[is_late_down(plays) | is_two_minute(plays)]
    events = per_game(plays, conversion_over_expected(plays, tables), 1.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.drive_engine"])


def _qb_explosive(qb_dropbacks, games):
    explosive = qb_dropbacks["is_completion"] & qb_dropbacks["yards_gained"].ge(
        EXPLOSIVE_PASS_YARDS
    )
    events = per_game(qb_dropbacks, explosive.astype(float), 1.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.explosive"])


def _qb_accuracy(qb_dropbacks, games):
    """Completion percentage over expected per throw with a charted expectation."""
    throws = qb_dropbacks[qb_dropbacks["cpoe"].notna()]
    events = per_game(throws, throws["cpoe"].astype(float), 1.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.accuracy"])


def _qb_ball_security(qb_dropbacks, fumbles, games):
    """Offense-view WPA of the passer's interceptions and fumbles per dropback.

    Turnovers lose win probability, so the sum is usually negative and a clean
    game scores zero: higher is better.
    """
    interceptions = qb_dropbacks[qb_dropbacks["is_interception"]]
    turnovers = pd.concat([interceptions, fumbles], ignore_index=True)
    cost = per_game(turnovers, turnovers["play_wpa"].fillna(0.0), 0.0)
    exposure = per_game(qb_dropbacks, 0.0, 1.0)
    events = exposure.drop(columns="numerator").merge(
        cost.drop(columns="denominator"), on=GAME_KEYS, how="left"
    )
    events["numerator"] = events["numerator"].fillna(0.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.ball_security"])


def _qb_floor(qb_dropbacks):
    games = qb_dropbacks.groupby(GAME_KEYS, as_index=False).agg(
        epa=("adjusted_epa", "sum"), count=("play_id", "size")
    )
    games["numerator"] = (games["epa"] / games["count"]).where(
        games["count"] >= FLOOR_MIN_DROPBACKS
    )
    games["denominator"] = 1.0
    return trait_rows(games, QB_TRAITS["qb.floor"])


def _qb_legs(qb_rows, tables, games):
    rushes = qb_rows[qb_rows["role"].eq("rusher")]
    events = per_game(rushes, conversion_over_expected(rushes, tables), 1.0)
    return trait_rows(opponent_adjusted(events, games), QB_TRAITS["qb.legs"])


def _qb_pressure_handling(all_dropbacks, qbs, games):
    """Sacks avoided against the protection baseline the passer inherited.

    The baseline is the team's sack rate with its other passers through the
    previous week when they have at least 30 dropbacks, else the league rate
    through the previous week. A game with neither (the season's first week)
    gets no row.
    """
    counts = ["dropbacks", "sacks"]
    passers = per_game(
        all_dropbacks, all_dropbacks["is_sack"].astype(float), 1.0
    ).rename(columns={"numerator": "sacks", "denominator": "dropbacks"})

    def prior(keys, prefix):
        return prior_totals(passers, keys, counts).rename(
            columns={c: f"{prefix}_{c}" for c in counts}
        )

    merged = (
        passers.merge(prior([], "league"), on="model_week", how="left")
        .merge(prior(["team"], "team"), on=["team", "model_week"], how="left")
        .merge(
            prior(["team", "athlete_id"], "own"),
            on=["team", "athlete_id", "model_week"],
            how="left",
        )
    )
    other_dropbacks = merged["team_dropbacks"] - merged["own_dropbacks"]
    other_sacks = merged["team_sacks"] - merged["own_sacks"]
    league_rate = merged["league_sacks"] / merged["league_dropbacks"].where(
        merged["league_dropbacks"] > 0
    )
    baseline = np.where(
        other_dropbacks >= PRESSURE_MIN_OTHER_DROPBACKS,
        other_sacks / other_dropbacks,
        league_rate,
    )
    merged["numerator"] = merged["dropbacks"] * baseline - merged["sacks"]
    merged["denominator"] = merged["dropbacks"]
    events = merged[merged["athlete_id"].isin(qbs) & merged["numerator"].notna()]
    events = events[[*GAME_KEYS, "numerator", "denominator"]]
    return trait_rows(
        opponent_adjusted(events, games), QB_TRAITS["qb.pressure_handling"]
    )


def _qb_receiver_independence(all_dropbacks, targets, qbs):
    """Passer EPA on each target above the receiver's rate with other passers.

    The receiver's rate is the passer-row adjusted EPA per target his other
    passers produced through the previous week. Only targets whose receiver has
    at least ten such targets count; a game with none has a NaN numerator.
    """
    receivers = targets[["play_id", "athlete_id"]].rename(
        columns={"athlete_id": "receiver_id"}
    )
    throws = all_dropbacks[~all_dropbacks["is_sack"]].merge(
        receivers, on="play_id", how="inner"
    )
    throws = throws.assign(epa=throws["adjusted_epa"], targets=1.0)
    sums = ["epa", "targets"]
    receiver_prior = prior_totals(throws, ["receiver_id"], sums)
    pair_prior = prior_totals(throws, ["receiver_id", "athlete_id"], sums)
    qb_throws = throws[throws["athlete_id"].isin(qbs)]
    merged = qb_throws.merge(
        receiver_prior,
        on=["receiver_id", "model_week"],
        how="left",
        suffixes=("", "_receiver"),
    ).merge(
        pair_prior,
        on=["receiver_id", "athlete_id", "model_week"],
        how="left",
        suffixes=("", "_pair"),
    )
    other_targets = merged["targets_receiver"] - merged["targets_pair"]
    other_epa = merged["epa_receiver"] - merged["epa_pair"]
    qualifies = other_targets >= RECEIVER_MIN_OTHER_TARGETS
    rate = other_epa / other_targets.where(qualifies)
    value = (merged["epa"] - rate).where(qualifies)
    events = merged[GAME_KEYS].assign(value=value, denominator=qualifies.astype(float))
    events = events.groupby(GAME_KEYS, as_index=False).agg(
        numerator=("value", lambda v: v.sum(min_count=1)),
        denominator=("denominator", "sum"),
    )
    return trait_rows(events, QB_TRAITS["qb.receiver_independence"])


def qb_fumbles(ctx: SeasonContext, qbs: set[str]) -> pd.DataFrame:
    """Every competitive play a quarterback fumbled on, as rows under the game keys."""
    plays = ctx.plays[
        ctx.plays["is_competitive"]
        & ctx.plays["fumble"].eq(1)
        & ctx.plays["fumbled_1_player_id"].isin(qbs)
        & ctx.plays["fumbled_1_team"].eq(ctx.plays["posteam"])
    ]
    return pd.DataFrame(
        {
            "athlete_id": plays["fumbled_1_player_id"].astype(str),
            "game_id": plays["game_id"],
            "model_week": plays["model_week"],
            "team": plays["posteam"],
            "opponent": plays["defteam"],
            "play_wpa": plays["wpa"].astype(float),
        }
    )


def qb_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """The eight quarterback traits per player and game."""
    rows = offense_rows(ctx)
    qbs = position_ids(ctx.positions, "QB")
    qb_rows = rows[rows["athlete_id"].isin(qbs)]
    all_dropbacks = dropbacks(rows)
    qb_dropbacks = all_dropbacks[all_dropbacks["athlete_id"].isin(qbs)]
    frames = [
        _qb_drive_engine(qb_rows, tables, ctx.games),
        _qb_explosive(qb_dropbacks, ctx.games),
        _qb_accuracy(qb_dropbacks, ctx.games),
        _qb_ball_security(qb_dropbacks, qb_fumbles(ctx, qbs), ctx.games),
        _qb_floor(qb_dropbacks),
        _qb_legs(qb_rows, tables, ctx.games),
        _qb_pressure_handling(all_dropbacks, qbs, ctx.games),
        _qb_receiver_independence(all_dropbacks, target_rows(rows), qbs),
    ]
    return pd.concat(frames, ignore_index=True)


# Running backs


def _rb_yards_over_expected(carries, tables, games):
    over = carries["yards_gained"] - expected_rush_yards(tables, carries)
    events = per_game(carries, over, 1.0)
    return trait_rows(
        opponent_adjusted(events, games), RB_TRAITS["rb.yards_over_expected"]
    )


def _rb_stuff_avoidance(carries):
    """Stuffs (no gain or a loss) avoided against the other backs in the same game."""
    stuffed = carries["yards_gained"].le(0).astype(float)
    relative = teammate_relative(per_game(carries, stuffed, 1.0), ["game_id", "team"])
    relative["numerator"] = -relative["numerator"]
    return trait_rows(relative, RB_TRAITS["rb.stuff_avoidance"])


def _rb_breakaway(carries, games):
    breakaway = carries["yards_gained"].ge(BREAKAWAY_RUSH_YARDS).astype(float)
    events = per_game(carries, breakaway, 1.0)
    return trait_rows(opponent_adjusted(events, games), RB_TRAITS["rb.breakaway"])


def _rb_short_yardage(carries, tables, games):
    plays = carries[is_short_yardage(carries)]
    events = per_game(plays, conversion_over_expected(plays, tables), 1.0)
    return trait_rows(opponent_adjusted(events, games), RB_TRAITS["rb.short_yardage"])


def _rb_receiving(rb_targets):
    """The back's adjusted EPA per target; the credit is already opponent adjusted."""
    events = per_game(rb_targets, rb_targets["adjusted_epa"], 1.0)
    return trait_rows(events, RB_TRAITS["rb.receiving"])


def _rb_ball_security(touches, fumbles, games):
    """Offense-view WPA of the back's fumbles per touch; a clean game scores zero."""
    cost = per_game(fumbles, fumbles["play_wpa"].fillna(0.0), 0.0)
    exposure = per_game(touches, 0.0, 1.0)
    events = exposure.drop(columns="numerator").merge(
        cost.drop(columns="denominator"), on=GAME_KEYS, how="left"
    )
    events["numerator"] = events["numerator"].fillna(0.0)
    return trait_rows(opponent_adjusted(events, games), RB_TRAITS["rb.ball_security"])


def _rb_line_independence(carries):
    """Adjusted EPA per carry against the other backs behind the same line that game."""
    events = per_game(carries, carries["adjusted_epa"], 1.0)
    relative = teammate_relative(events, ["game_id", "team"])
    return trait_rows(relative, RB_TRAITS["rb.line_independence"])


def player_fumbles(ctx: SeasonContext, ids: set[str]) -> pd.DataFrame:
    return qb_fumbles(ctx, ids)


def rb_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """The seven running back traits per player and game."""
    rows = offense_rows(ctx)
    rbs = position_ids(ctx.positions, "RB")
    rb_rows = rows[rows["athlete_id"].isin(rbs)]
    carries = rb_rows[rb_rows["role"].eq("rusher") & rb_rows["yards_gained"].notna()]
    targets = target_rows(rows)
    rb_targets = targets[targets["athlete_id"].isin(rbs)]
    touches = pd.concat([carries, rb_targets], ignore_index=True)
    frames = [
        _rb_yards_over_expected(carries, tables, ctx.games),
        _rb_stuff_avoidance(carries),
        _rb_breakaway(carries, ctx.games),
        _rb_short_yardage(carries, tables, ctx.games),
        _rb_receiving(rb_targets),
        _rb_ball_security(touches, player_fumbles(ctx, rbs), ctx.games),
        _rb_line_independence(carries),
    ]
    return pd.concat(frames, ignore_index=True)


# Receivers


def _rec_target_share(rows, targets, receivers):
    """The player's targets over the team's targets in each game he appeared in."""
    appeared = rows[GAME_KEYS].drop_duplicates()
    appeared = appeared[appeared["athlete_id"].isin(receivers)]
    own = (
        targets.groupby(GAME_KEYS, as_index=False)
        .size()
        .rename(columns={"size": "numerator"})
    )
    team = (
        targets.groupby(["game_id", "team"], as_index=False)
        .size()
        .rename(columns={"size": "denominator"})
    )
    events = appeared.merge(own, on=GAME_KEYS, how="left").merge(
        team, on=["game_id", "team"], how="inner"
    )
    events["numerator"] = events["numerator"].fillna(0.0)
    return trait_rows(events, REC_TRAITS["rec.target_share"])


def _rec_value_per_target(player_targets):
    events = per_game(player_targets, player_targets["adjusted_epa"], 1.0)
    return trait_rows(events, REC_TRAITS["rec.value_per_target"])


def _rec_explosive(player_targets, games):
    explosive = player_targets["is_reception"] & player_targets["yards_gained"].ge(
        EXPLOSIVE_PASS_YARDS
    )
    events = per_game(player_targets, explosive.astype(float), 1.0)
    return trait_rows(opponent_adjusted(events, games), REC_TRAITS["rec.explosive"])


def _rec_late_down(player_targets, tables, games):
    late = player_targets[is_late_down(player_targets)]
    events = per_game(late, conversion_over_expected(late, tables), 1.0)
    return trait_rows(opponent_adjusted(events, games), REC_TRAITS["rec.late_down"])


def _rec_yac_over_expected(player_targets, games):
    """Yards after the catch above the feed's expectation, per reception."""
    caught = player_targets[
        player_targets["is_reception"]
        & player_targets["yards_after_catch"].notna()
        & player_targets["xyac_mean_yardage"].notna()
    ]
    over = caught["yards_after_catch"].astype(float) - caught[
        "xyac_mean_yardage"
    ].astype(float)
    events = per_game(caught, over, 1.0)
    return trait_rows(
        opponent_adjusted(events, games), REC_TRAITS["rec.yac_over_expected"]
    )


def _rec_qb_independence(targets, receivers):
    """Adjusted EPA per target against the same passer's other targets in the game."""
    keys = [*GAME_KEYS, "passer_id"]
    events = targets.groupby(keys, as_index=False).agg(
        numerator=("adjusted_epa", "sum"), denominator=("play_id", "size")
    )
    relative = teammate_relative(events, ["game_id", "passer_id"])
    relative = relative[relative["athlete_id"].isin(receivers)]
    relative = relative.assign(
        qualified=relative["denominator"].where(relative["numerator"].notna(), 0.0)
    )
    games = relative.groupby(GAME_KEYS, as_index=False).agg(
        numerator=("numerator", lambda values: values.sum(min_count=1)),
        qualified=("qualified", "sum"),
        targets=("denominator", "sum"),
    )
    games["denominator"] = games["qualified"].where(
        games["numerator"].notna(), games["targets"]
    )
    return trait_rows(games, REC_TRAITS["rec.qb_independence"])


def receiver_traits(ctx: SeasonContext, tables: ExpectationTables) -> pd.DataFrame:
    """The six wide receiver and tight end traits per player and game."""
    rows = offense_rows(ctx)
    receivers = set().union(
        *(position_ids(ctx.positions, group) for group in RECEIVER_GROUPS)
    )
    targets = target_rows(rows)
    player_targets = targets[targets["athlete_id"].isin(receivers)]
    frames = [
        _rec_target_share(rows, targets, receivers),
        _rec_value_per_target(player_targets),
        _rec_explosive(player_targets, ctx.games),
        _rec_late_down(player_targets, tables, ctx.games),
        _rec_yac_over_expected(player_targets, ctx.games),
        _rec_qb_independence(targets, receivers),
    ]
    return pd.concat(frames, ignore_index=True)
