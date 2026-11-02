"""Raw season stat lines for the position grade cards (``nfl.player_season_stats``).

One row per published athlete (every ``position_traits`` athlete but kickers)
with the season's numbers through the trait snapshot week, in the key order
and scale ``STAT_KEYS`` gives per position group; a key ending ``*10`` is
stored as the value times ten, rounded.

Every number is raw: no opponent adjustment, garbage time included, regular
season only. Box counts come from the nflverse weekly player stats; punting,
play counts and EPA from nflfastR play-by-play (two-point tries left out);
pressures and coverage from Pro Football Reference's advanced defense and
passing; rush yards over expected from Next Gen Stats' season row and Total QBR
from ESPN's season figure. Three values come from the trait code so they cannot
disagree with it: the quarterback's ``cpoe`` and ``adot`` are the rating's own
measures (competitive plays) and target share is the ``rec.target_share``
trait's own computation (competitive plays). ``games`` is the trait rows'
count, and ``epa_credit`` the defensive EPA credit unadjusted, on competitive
plays.

A value is null only where its source lacks the player: Next Gen Stats lists
rushers with enough carries only, ESPN rates passers with enough plays only,
and a player Pro Football Reference's ids do not reach has no pressures or
coverage.
"""

import numpy as np
import pandas as pd

from backend.config import RAW_DIR
from backend.features.units import _adjusted_line_yards
from backend.players import UNIT_PREFIX
from backend.players.context import SeasonContext
from backend.players.offense import (
    RECEIVER_GROUPS,
    offense_rows,
    position_ids,
    rec_target_share,
    target_rows,
)
from backend.players.qb_study import _sums_through, rate

SMALLINT_MAX = 32767
STAT_LINE_COLUMNS = ["season", "athlete_id", "position_group", "week", "stats"]
EXPLOSIVE_RUSH_YARDS = 10
TOUCHBACK_YARDS = 20
SKIPPED_GROUPS = frozenset({"K"})


def _line(*keys: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (key.removesuffix("*10"), 10 if key.endswith("*10") else 1) for key in keys
    )


DEFENSE_LINE = _line(
    "games", "tackles", "solo", "tfl*10", "sacks*10", "qb_hits", "pressures", "pbu",
    "int", "ff", "epa_credit*10", "cov_targets", "cov_cmp", "cov_yds", "cov_td",
    "cov_int",
)  # fmt: skip
RECEIVER_LINE = _line(
    "games", "targets", "rec", "rec_yds", "rec_td", "yac", "target_share*10",
    "adot*10", "epa_total*10",
)  # fmt: skip
STAT_KEYS: dict[str, tuple[tuple[str, int], ...]] = {
    "QB": _line(
        "games", "att", "cmp", "pass_yds", "pass_td", "int", "sacks", "rush_att",
        "rush_yds", "rush_td", "epa_plays", "epa_total*10", "cpoe*10", "adot*10",
        "espn_qbr*10",
    ),
    "RB": _line(
        "games", "rush_att", "rush_yds", "rush_td", "fumbles_lost", "targets", "rec",
        "rec_yds", "rec_td", "rush_success", "rush_explosive", "ryoe_total*10",
        "rush_epa*10", "epa_total*10",
    ),
    "WR": RECEIVER_LINE,
    "TE": RECEIVER_LINE,
    "DL": DEFENSE_LINE,
    "LB": DEFENSE_LINE,
    "DB": DEFENSE_LINE,
    "OL": _line(
        "games", "rush_att", "stuffs", "line_yds*10", "dropbacks", "sacks_allowed",
        "pressures_allowed",
    ),
    "P": _line(
        "games", "punts", "punt_yds", "net_yds", "inside_20", "touchbacks",
        "punt_epa*10",
    ),
}  # fmt: skip
# Keys whose source lists only some players, and ratios, which have no value
# without a denominator. Every other key is a count or a sum, zero for a player
# without the event.
SOURCED = frozenset(
    {
        "espn_qbr", "ryoe_total", "pressures", "cov_targets", "cov_cmp", "cov_yds",
        "cov_td", "cov_int",
    }
)  # fmt: skip
RATIOS = frozenset({"cpoe", "adot", "target_share"})
BOX_STATS = {
    "attempts": "att",
    "completions": "cmp",
    "passing_yards": "pass_yds",
    "passing_tds": "pass_td",
    "passing_interceptions": "int",
    "sacks_suffered": "sacks",
    "carries": "rush_att",
    "rushing_yards": "rush_yds",
    "rushing_tds": "rush_td",
    "rushing_fumbles_lost": "rushing_fumbles_lost",
    "receiving_fumbles_lost": "receiving_fumbles_lost",
    "targets": "targets",
    "receptions": "rec",
    "receiving_yards": "rec_yds",
    "receiving_tds": "rec_td",
    "receiving_yards_after_catch": "yac",
    "receiving_air_yards": "air_yards",
    "def_tackles_solo": "solo",
    "def_tackles_with_assist": "tackles_with_assist",
    "def_tackle_assists": "tackle_assists",
    "def_sacks": "sacks_made",
    "def_interceptions": "int_caught",
    "def_pass_defended": "pbu",
    "def_fumbles_forced": "ff",
}
PFR_DEFENSE = {
    "def_pressures": "pressures",
    "def_targets": "cov_targets",
    "def_completions_allowed": "cov_cmp",
    "def_yards_allowed": "cov_yds",
    "def_receiving_td_allowed": "cov_td",
    "def_ints": "cov_int",
}


def _per_athlete(ids: pd.Series, **sums) -> pd.DataFrame:
    """Sums per athlete; each value is a Series aligned with ``ids`` or a scalar."""
    frame = pd.DataFrame({"athlete_id": ids.astype(str).to_numpy()})
    for name, values in sums.items():
        frame[name] = (
            values.to_numpy(dtype=float) if isinstance(values, pd.Series) else values
        )
    return frame.groupby("athlete_id").sum()


def _tagged(scrimmage: pd.DataFrame, columns: list[str], shared: bool) -> pd.Series:
    """Plays per athlete tagged in any of ``columns``; ``shared`` splits a play
    between the athletes it tags, so a shared tackle for loss is half each."""
    tagged = scrimmage[columns].notna()
    weight = (
        1.0 / tagged.sum(axis=1).clip(lower=1) if shared else tagged.any(axis=1) * 1.0
    )
    ids = pd.concat([scrimmage.loc[tagged[c], c] for c in columns])
    weights = pd.concat([weight[tagged[c]] for c in columns])
    return _per_athlete(ids, n=weights)["n"]


def _box_totals(stats: pd.DataFrame) -> pd.DataFrame:
    """Season sums of ``BOX_STATS`` per athlete from the weekly player stats."""
    totals = stats.groupby("athlete_id")[list(BOX_STATS)].sum(min_count=1)
    return totals.rename(columns=BOX_STATS).fillna(0.0)


def _scrimmage(plays: pd.DataFrame) -> pd.DataFrame:
    """Valued passes and runs; two-point tries have no down and are left out."""
    return plays[
        plays["play_type"].isin(["pass", "run"])
        & plays["epa"].notna()
        & plays["down"].notna()
    ]


def _pfr_ids() -> dict[str, str]:
    players = pd.read_parquet(
        RAW_DIR / "players.parquet", columns=["gsis_id", "pfr_id"]
    )
    players = players.dropna().drop_duplicates("pfr_id")
    return dict(zip(players["pfr_id"], players["gsis_id"].astype(str)))


def _pfr(stat: str, season: int, week: int) -> pd.DataFrame:
    path = RAW_DIR / f"pfr_{stat}" / f"{season}.parquet"
    frame = pd.read_parquet(path)
    return frame[frame["game_type"].eq("REG") & frame["week"].le(week)]


def _espn_qbr(season: int) -> pd.Series:
    """ESPN's regular-season Total QBR per gsis id."""
    qbr = pd.read_parquet(RAW_DIR / "espn_qbr_season" / f"{season}.parquet")
    qbr = qbr[qbr["season_type"].eq("Regular")]
    players = pd.read_parquet(
        RAW_DIR / "players.parquet", columns=["gsis_id", "espn_id"]
    ).dropna()
    crosswalk = dict(
        zip(
            players["espn_id"].astype(float).astype("int64").astype(str),
            players["gsis_id"].astype(str),
        )
    )
    qbr = qbr.assign(athlete_id=qbr["player_id"].astype(str).map(crosswalk))
    qbr = qbr.dropna(subset=["athlete_id"])
    return pd.Series(qbr["qbr_total"].astype(float).to_numpy(), index=qbr["athlete_id"])


def _ryoe(season: int) -> pd.Series:
    """Next Gen Stats' regular-season rush yards over expected (its week 0 row)."""
    ngs = pd.read_parquet(RAW_DIR / "ngs" / f"rushing_{season}.parquet")
    rows = ngs[ngs["season_type"].eq("REG") & ngs["week"].eq(0)]
    return pd.Series(
        rows["rush_yards_over_expected"].astype(float).to_numpy(),
        index=rows["player_gsis_id"].astype(str),
    )


def _target_epa(scrimmage: pd.DataFrame) -> pd.Series:
    targets = scrimmage[
        scrimmage["play_type"].eq("pass") & scrimmage["receiver_player_id"].notna()
    ]
    return _per_athlete(targets["receiver_player_id"], epa=targets["epa"])["epa"]


def _quarterbacks(
    box: pd.DataFrame,
    scrimmage: pd.DataFrame,
    qb_games: pd.DataFrame,
    week: int,
    season: int,
) -> pd.DataFrame:
    """Box passing and rushing, EPA over dropbacks plus designed runs, the
    rating's completion over expected and depth of target, and ESPN's QBR.

    A pass play (a sack included) is the passer's dropback; a run, scramble or
    designed, is the rusher's.
    """
    athlete = scrimmage["passer_player_id"].where(
        scrimmage["play_type"].eq("pass"), scrimmage["rusher_player_id"]
    )
    plays = scrimmage[athlete.notna()]
    epa = _per_athlete(athlete[athlete.notna()], epa_plays=1.0, epa_total=plays["epa"])
    sums = _sums_through(qb_games[qb_games["season"].eq(season)], week)
    measures = pd.DataFrame(
        {"cpoe": rate(sums, "cpoe").to_numpy(), "adot": rate(sums, "adot").to_numpy()},
        index=sums["athlete_id"].astype(str),
    )
    out = box[
        [
            "att",
            "cmp",
            "pass_yds",
            "pass_td",
            "int",
            "sacks",
            "rush_att",
            "rush_yds",
            "rush_td",
        ]
    ]
    out = out.join(epa, how="outer").join(measures, how="outer")
    return out.join(_espn_qbr(season).rename("espn_qbr"), how="outer")


def _running_backs(
    box: pd.DataFrame, scrimmage: pd.DataFrame, season: int
) -> pd.DataFrame:
    carries = scrimmage[
        scrimmage["play_type"].eq("run") & scrimmage["rusher_player_id"].notna()
    ]
    rushing = _per_athlete(
        carries["rusher_player_id"],
        rush_success=carries["epa"].gt(0),
        rush_explosive=carries["yards_gained"].ge(EXPLOSIVE_RUSH_YARDS),
        rush_epa=carries["epa"],
    )
    out = box[
        ["rush_att", "rush_yds", "rush_td", "targets", "rec", "rec_yds", "rec_td"]
    ].assign(fumbles_lost=box["rushing_fumbles_lost"] + box["receiving_fumbles_lost"])
    out = out.join(rushing, how="outer")
    out = out.join(_target_epa(scrimmage).rename("target_epa"), how="outer")
    out[["rush_epa", "target_epa"]] = out[["rush_epa", "target_epa"]].fillna(0.0)
    out["epa_total"] = out["rush_epa"] + out.pop("target_epa")
    return out.join(_ryoe(season).rename("ryoe_total"), how="outer")


def _receivers(
    ctx: SeasonContext, box: pd.DataFrame, scrimmage: pd.DataFrame, week: int
) -> pd.DataFrame:
    """Box receiving, depth of target, target EPA and the trait target share (%)."""
    rows = offense_rows(ctx)
    rows = rows[rows["model_week"].le(week)]
    receivers = set().union(
        *(position_ids(ctx.positions, group) for group in RECEIVER_GROUPS)
    )
    share = rec_target_share(rows, target_rows(rows), receivers)
    sums = share.groupby("athlete_id")[["numerator", "denominator"]].sum()
    percent = (
        100 * sums["numerator"] / sums["denominator"].where(sums["denominator"] > 0)
    )
    out = box[["targets", "rec", "rec_yds", "rec_td", "yac"]].assign(
        adot=box["air_yards"] / box["targets"].where(box["targets"] > 0)
    )
    out = out.join(_target_epa(scrimmage).rename("epa_total"), how="outer")
    return out.join(percent.rename("target_share"), how="outer")


def _defenders(
    ctx: SeasonContext,
    box: pd.DataFrame,
    scrimmage: pd.DataFrame,
    week: int,
    season: int,
) -> pd.DataFrame:
    """Box defense, half-shared tackles for loss and QB hits from the
    play-by-play, the raw EPA credit and PFR's pressures and coverage."""
    events = pd.DataFrame(
        {
            "tfl": _tagged(
                scrimmage,
                ["tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id"],
                shared=True,
            ),
            "qb_hits": _tagged(
                scrimmage, ["qb_hit_1_player_id", "qb_hit_2_player_id"], shared=False
            ),
        }
    )
    credit = ctx.credit
    credit = credit[
        credit["side"].eq("defense")
        & credit["is_competitive"]
        & credit["model_week"].le(week)
    ]
    out = box[["solo", "pbu", "ff"]].assign(
        tackles=box["solo"] + box["tackles_with_assist"] + box["tackle_assists"],
        sacks=box["sacks_made"],
        int=box["int_caught"],
    )
    out = out.join(events, how="outer")
    out = out.join(
        _per_athlete(credit["athlete_id"], epa_credit=credit["raw_epa"]), how="outer"
    )
    pfr = _pfr("def", season, week)
    ids = _pfr_ids()
    pfr = pfr.assign(athlete_id=pfr["pfr_player_id"].map(ids)).dropna(
        subset=["athlete_id"]
    )
    coverage = (
        pfr.groupby("athlete_id")[list(PFR_DEFENSE)].sum().rename(columns=PFR_DEFENSE)
    )
    # A player PFR's ids reach but who has no advanced row was never charted
    # with an event, so his counts are zero; one they do not reach stays null.
    reached = pd.Index(sorted(set(ids.values()))).difference(coverage.index)
    coverage = pd.concat(
        [coverage, pd.DataFrame(0.0, index=reached, columns=coverage.columns)]
    )
    return out.join(coverage, how="left")


def _lines(scrimmage: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Designed runs, stuffs, line yards, dropbacks, sacks and pressures per unit.

    A dropback is nflfastR's ``qb_dropback`` (scrambles included), so a play
    is either one of the team's dropbacks or one of its designed runs.
    """
    dropback = scrimmage["qb_dropback"].eq(1)
    runs = scrimmage[scrimmage["play_type"].eq("run") & ~dropback]
    runs = runs[runs["yards_gained"].notna()]
    drops = scrimmage[dropback]
    pressured = _pfr("pass", season, week).groupby("team")["times_pressured"].sum()
    out = pd.concat(
        [
            runs.groupby("posteam").size().rename("rush_att"),
            runs["yards_gained"].le(0).groupby(runs["posteam"]).sum().rename("stuffs"),
            _adjusted_line_yards(runs["yards_gained"].astype(float))
            .groupby(runs["posteam"])
            .sum()
            .rename("line_yds"),
            drops.groupby("posteam").size().rename("dropbacks"),
            drops["sack"].eq(1).groupby(drops["posteam"]).sum().rename("sacks_allowed"),
            pressured.rename("pressures_allowed"),
        ],
        axis=1,
    )
    out.index = UNIT_PREFIX + out.index.astype(str)
    return out.astype(float)


def _punters(plays: pd.DataFrame) -> pd.DataFrame:
    """Punts, gross and net yards, inside the 20, touchbacks and EPA per punter.

    Net yards are gross less return yards less 20 per touchback. They come from
    the play-by-play: the weekly player stats' punt yards drop some punts.
    """
    punts = plays[plays["play_type"].eq("punt") & plays["punter_player_id"].notna()]
    gross = punts["kick_distance"].fillna(0.0)
    touchback = punts["touchback"].eq(1)
    return _per_athlete(
        punts["punter_player_id"],
        punts=1.0,
        punt_yds=gross,
        net_yds=gross - punts["return_yards"].fillna(0.0) - TOUCHBACK_YARDS * touchback,
        inside_20=punts["punt_inside_twenty"].eq(1),
        touchbacks=touchback,
        punt_epa=punts["epa"].fillna(0.0),
    )


def encode(line: pd.DataFrame, group: str) -> list[list[int | None]]:
    """Each row's ``STAT_KEYS[group]`` values, scaled and rounded; NaN becomes null.

    A value outside a smallint is a contract break and raises.
    """
    keys = STAT_KEYS[group]
    scaled = np.rint(
        np.column_stack(
            [line[name].to_numpy(dtype=float) * scale for name, scale in keys]
        )
    )
    too_big = np.abs(np.nan_to_num(scaled)) > SMALLINT_MAX
    if too_big.any():
        row, column = np.argwhere(too_big)[0]
        raise ValueError(
            f"{group} {keys[column][0]} of {line.index[row]} is {scaled[row, column]}, "
            "outside a smallint"
        )
    return [
        [None if np.isnan(value) else int(value) for value in row] for row in scaled
    ]


def season_stat_lines(
    ctx: SeasonContext, qb_games: pd.DataFrame, published: pd.DataFrame
) -> pd.DataFrame:
    """One ``STAT_LINE_COLUMNS`` row per athlete in ``published`` but kickers.

    ``ctx`` carries its credit; ``qb_games`` is the quarterback rating's game
    table; ``published`` is the snapshot week's published trait rows, which
    give the population, each athlete's group and games, and the week. Only
    model weeks through that week count.
    """
    people = published[~published["position_group"].isin(SKIPPED_GROUPS)]
    if people.empty:
        return pd.DataFrame(columns=STAT_LINE_COLUMNS)
    if people["week"].nunique() != 1:
        raise ValueError("Published trait rows span more than one week")
    week = int(people["week"].iloc[0])
    people = people.astype({"athlete_id": str}).drop_duplicates(
        ["athlete_id", "position_group"]
    )
    if people["athlete_id"].duplicated().any():
        raise ValueError("An athlete is published in more than one position group")
    unknown = set(people["position_group"]) - set(STAT_KEYS)
    if unknown:
        raise ValueError(f"no stat line for position groups {sorted(unknown)}")

    season = ctx.season
    box = _box_totals(ctx.stats[ctx.stats["model_week"].le(week)])
    plays = ctx.plays[ctx.plays["model_week"].le(week)]
    scrimmage = _scrimmage(plays)
    by_group = {
        "QB": _quarterbacks(box, scrimmage, qb_games, week, season),
        "RB": _running_backs(box, scrimmage, season),
        "WR": _receivers(ctx, box, scrimmage, week),
        "DL": _defenders(ctx, box, scrimmage, week, season),
        "OL": _lines(scrimmage, season, week),
        "P": _punters(plays),
    }
    for alias, source in (("TE", "WR"), ("LB", "DL"), ("DB", "DL")):
        by_group[alias] = by_group[source]

    parts = []
    for group, members in people.groupby("position_group"):
        names = [name for name, _ in STAT_KEYS[group]]
        line = by_group[group].reindex(members["athlete_id"])
        counted = [n for n in names if n in line and n not in SOURCED | RATIOS]
        line[counted] = line[counted].fillna(0.0)
        line["games"] = members["games"].to_numpy(dtype=float)
        line = line.reindex(columns=names)
        parts.append(
            pd.DataFrame(
                {
                    "season": season,
                    "athlete_id": members["athlete_id"].to_numpy(),
                    "position_group": group,
                    "week": week,
                    "stats": encode(line, group),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)[STAT_LINE_COLUMNS]
