"""Per-play credit: which players a play belongs to and what it was worth.

Offense: the passer takes a pass play, split in half with the receiver when one
is tagged; an interception or a sack is the passer's alone; a carry (a scramble
included) is the rusher's; a lost fumble is charged to the fumbler alone.
Defense: every defender the play tags for a sack, interception, pass defended,
forced fumble, tackle for loss or quarterback hit shares the play's negative
EPA equally, and so does a tackler on a play the offense lost EPA on (a stop).
A tackle after a long gain is not blamed on the tackler: coverage, not the
tackle, allowed it, and the feed does not say who was in coverage.

``adjusted_epa`` nets out the opponent through the previous week: a pass
play's EPA against a defense that had allowed more than the league through
last week is marked down, and a defender's credit against such an offense is
marked up. Nothing from the play's own week enters its adjustment.
"""

import numpy as np
import pandas as pd

from backend.players import CREDITED_GROUPS, EPA_TRAIT, TRAITS, UNIT_PREFIX
from backend.players.adjust import adjust_rate
from backend.players.context import SeasonContext, opponents

CREDIT_COLUMNS = [
    "game_id",
    "play_id",
    "model_week",
    "athlete_id",
    "team",
    "opponent",
    "side",
    "role",
    "share",
    "raw_epa",
    "adjusted_epa",
    "wpa",
    "is_competitive",
]
TRAIT_COLUMNS = [
    "athlete_id",
    "model_week",
    "game_id",
    "trait",
    "numerator",
    "denominator",
    "kind",
]
SCRIMMAGE = ("pass", "run")
# Defensive tag columns, in the priority a player's role takes when a play tags
# him more than once, with the team column that says which side tagged him.
DEFENSIVE_TAGS = (
    ("sacker", "sack_player_id", None),
    ("sacker", "half_sack_1_player_id", None),
    ("sacker", "half_sack_2_player_id", None),
    ("interceptor", "interception_player_id", None),
    (
        "fumble_forcer",
        "forced_fumble_player_1_player_id",
        "forced_fumble_player_1_team",
    ),
    (
        "fumble_forcer",
        "forced_fumble_player_2_player_id",
        "forced_fumble_player_2_team",
    ),
    ("pass_defender", "pass_defense_1_player_id", None),
    ("pass_defender", "pass_defense_2_player_id", None),
    ("tackler_for_loss", "tackle_for_loss_1_player_id", None),
    ("tackler_for_loss", "tackle_for_loss_2_player_id", None),
    ("qb_hitter", "qb_hit_1_player_id", None),
    ("qb_hitter", "qb_hit_2_player_id", None),
    ("tackler", "solo_tackle_1_player_id", "solo_tackle_1_team"),
    ("tackler", "solo_tackle_2_player_id", "solo_tackle_2_team"),
    ("tackler", "tackle_with_assist_1_player_id", "tackle_with_assist_1_team"),
    ("tackler", "tackle_with_assist_2_player_id", "tackle_with_assist_2_team"),
    ("tackler", "assist_tackle_1_player_id", "assist_tackle_1_team"),
    ("tackler", "assist_tackle_2_player_id", "assist_tackle_2_team"),
    ("tackler", "assist_tackle_3_player_id", "assist_tackle_3_team"),
    ("tackler", "assist_tackle_4_player_id", "assist_tackle_4_team"),
)
ROLE_PRIORITY = {
    role: rank
    for rank, role in enumerate(
        [
            "sacker",
            "interceptor",
            "fumble_forcer",
            "pass_defender",
            "tackler_for_loss",
            "qb_hitter",
            "tackler",
        ]
    )
}


def scrimmage_plays(plays: pd.DataFrame) -> pd.DataFrame:
    return plays[plays["play_type"].isin(SCRIMMAGE) & plays["epa"].notna()]


def play_adjustments(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """``play_id, epa_vs_defense, epa_vs_offense`` for every scrimmage play.

    ``epa_vs_defense`` is the play's EPA net of how the defense faced had
    compared with the league through the previous week on that play type; it is
    the offense's view. ``epa_vs_offense`` nets out the offense faced the same
    way; it is the defense's view. Competitive plays only set the baselines, and
    only competitive plays are adjusted.
    """
    rows = scrimmage_plays(plays)
    rows = rows[rows["is_competitive"]]
    parts = []
    for play_type, part in rows.groupby("play_type"):
        base = part[["game_id", "play_id"]].assign(
            numerator=part["epa"].astype(float), denominator=1.0
        )
        versus_defense = adjust_rate(
            base.assign(opponent=part["defteam"].to_numpy()), games
        )
        versus_offense = adjust_rate(
            base.assign(opponent=part["posteam"].to_numpy()), games
        )
        parts.append(
            pd.DataFrame(
                {
                    "play_id": part["play_id"].to_numpy(),
                    "epa_vs_defense": versus_defense["numerator"].to_numpy(),
                    "epa_vs_offense": versus_offense["numerator"].to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["play_id", "epa_vs_defense", "epa_vs_offense"])
    return pd.concat(parts, ignore_index=True)


def _rows(part: pd.DataFrame, athlete: pd.Series, role: str, share) -> pd.DataFrame:
    keep = athlete.notna()
    out = part.loc[keep, ["game_id", "play_id", "model_week", "is_competitive"]].copy()
    out["athlete_id"] = athlete[keep].astype(str).to_numpy()
    out["role"] = role
    out["share"] = share[keep] if isinstance(share, pd.Series) else share
    return out


def offense_credit(plays: pd.DataFrame) -> pd.DataFrame:
    """Offensive credit rows: passer, receiver, rusher and fumbler shares per play."""
    rows = scrimmage_plays(plays)
    passes = rows[rows["play_type"].eq("pass")]
    runs = rows[rows["play_type"].eq("run")]
    receiver_tagged = passes["receiver_player_id"].notna() & ~passes["interception"].eq(
        1
    )
    passer_share = pd.Series(np.where(receiver_tagged, 0.5, 1.0), index=passes.index)
    parts = [
        _rows(passes, passes["passer_player_id"], "passer", passer_share),
        _rows(
            passes[receiver_tagged],
            passes.loc[receiver_tagged, "receiver_player_id"],
            "receiver",
            0.5,
        ),
        _rows(runs, runs["rusher_player_id"], "rusher", 1.0),
    ]
    out = pd.concat(parts, ignore_index=True)
    # A lost fumble belongs to the fumbler alone.
    lost = rows[
        rows["fumble_lost"].eq(1)
        & rows["fumbled_1_player_id"].notna()
        & rows["fumbled_1_team"].eq(rows["posteam"])
    ]
    out = out[~out["play_id"].isin(set(lost["play_id"]))]
    fumblers = _rows(lost, lost["fumbled_1_player_id"], "fumbler", 1.0)
    out = pd.concat([out, fumblers], ignore_index=True)
    out["team"] = out["play_id"].map(rows.set_index("play_id")["posteam"])
    out["opponent"] = out["play_id"].map(rows.set_index("play_id")["defteam"])
    out["side"] = "offense"
    return out


def defense_credit(plays: pd.DataFrame) -> pd.DataFrame:
    """Defensive credit rows: every tagged disruptor, and tacklers on stops."""
    rows = scrimmage_plays(plays)
    by_play = rows.set_index("play_id")
    parts = []
    for role, column, team_column in DEFENSIVE_TAGS:
        if column not in rows:
            continue
        tagged = rows[rows[column].notna()]
        if team_column is not None and team_column in tagged:
            tagged = tagged[tagged[team_column].eq(tagged["defteam"])]
        if role == "tackler":
            tagged = tagged[tagged["epa"].lt(0)]
        parts.append(_rows(tagged, tagged[column], role, 1.0))
    out = pd.concat(parts, ignore_index=True)
    out["priority"] = out["role"].map(ROLE_PRIORITY)
    out = (
        out.sort_values(["play_id", "athlete_id", "priority"], kind="stable")
        .drop_duplicates(["play_id", "athlete_id"])
        .drop(columns="priority")
    )
    out["share"] = 1.0 / out.groupby("play_id")["athlete_id"].transform("size")
    out["team"] = out["play_id"].map(by_play["defteam"])
    out["opponent"] = out["play_id"].map(by_play["posteam"])
    out["side"] = "defense"
    return out


def build_credit(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Every credit row with raw, adjusted and win-probability value."""
    rows = pd.concat([offense_credit(plays), defense_credit(plays)], ignore_index=True)
    by_play = scrimmage_plays(plays).set_index("play_id")
    epa = rows["play_id"].map(by_play["epa"]).astype(float)
    wpa = rows["play_id"].map(by_play["wpa"]).astype(float)
    adjustments = play_adjustments(plays, games).set_index("play_id")
    versus_defense = rows["play_id"].map(adjustments["epa_vs_defense"]).astype(float)
    versus_offense = rows["play_id"].map(adjustments["epa_vs_offense"]).astype(float)
    sign = np.where(rows["side"].eq("offense"), 1.0, -1.0)
    adjusted = np.where(rows["side"].eq("offense"), versus_defense, versus_offense)
    rows["raw_epa"] = sign * epa * rows["share"]
    rows["adjusted_epa"] = (
        sign * np.where(np.isnan(adjusted), epa, adjusted) * rows["share"]
    )
    rows["wpa"] = sign * wpa
    return rows[CREDIT_COLUMNS].reset_index(drop=True)


def with_credit(ctx: SeasonContext) -> SeasonContext:
    """``ctx`` with its credit rows built, and its presence extended by them."""
    from backend.players.context import player_presence

    credit = build_credit(ctx.plays, ctx.games)
    seen = pd.concat(
        [
            ctx.presence.loc[
                ctx.presence["present"], ["athlete_id", "team", "game_id"]
            ],
            credit[["athlete_id", "team", "game_id"]],
        ],
        ignore_index=True,
    )
    ctx.presence = player_presence(seen, ctx.games)
    ctx.credit = credit
    return ctx


def epa_credit_frames(credit: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """One ``epa.credit`` row per credited player-game: adjusted competitive EPA."""
    spec = TRAITS[EPA_TRAIT]
    grouped = positions[positions["position_group"].isin(CREDITED_GROUPS)]
    rows = credit[
        credit["is_competitive"] & credit["athlete_id"].isin(set(grouped["athlete_id"]))
    ]
    out = rows.groupby(["athlete_id", "game_id", "model_week"], as_index=False)[
        "adjusted_epa"
    ].sum()
    out = out.rename(columns={"adjusted_epa": "numerator"})
    out["denominator"] = 1.0
    out["trait"] = spec.name
    out["kind"] = spec.kind
    return out[TRAIT_COLUMNS]


def credited_games(credit: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, game_id, model_week`` for every competitive credited game."""
    rows = credit[credit["is_competitive"]]
    return rows[["athlete_id", "game_id", "model_week"]].drop_duplicates(
        ignore_index=True
    )


def unit_rows(credit: pd.DataFrame) -> pd.DataFrame:
    """Offensive credit rows re-keyed to the team's line unit."""
    rows = credit[credit["side"].eq("offense")]
    return rows.assign(athlete_id=UNIT_PREFIX + rows["team"].astype(str))


__all__ = [
    "CREDIT_COLUMNS",
    "TRAIT_COLUMNS",
    "build_credit",
    "credited_games",
    "epa_credit_frames",
    "opponents",
    "unit_rows",
    "with_credit",
]
