"""Quarterback measure study: what the NFL data can say about a passer, and
which of it is stable, predictive and additive.

One row per quarterback and regular-season game with the raw sums behind every
candidate measure (so a measure can be read as a rate through any week), from
nflverse play-by-play, participation (pressure, pass rushers, time to throw),
FTN charting (interception-worthy throws, catchable balls, drops, play action,
blitzes, quarterback-fault sacks) and PFR game lines. Every measure is then
judged the same way: split-half reliability, prediction of the passer's
rest-of-season and next-season play, and what it adds beyond EPA alone.

This is research, not serving: nothing here is published or fitted into the
grade until the evidence says it should be.
"""

import logging

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.etl.store import write_parquet
from backend.players.context import load_plays, load_season_context
from backend.players.expectations import expected_conversion, fit_expectation_tables
from backend.players.offense import is_late_down, is_two_minute

log = logging.getLogger(__name__)

STUDY_DIR = PROCESSED_DIR / "players" / "research"
KEYS = ["season", "week", "game_id", "athlete_id", "team", "opponent"]
DEEP_AIR_YARDS = 20
EXPLOSIVE_YARDS = 20
# Numerator over denominator pairs; a rate is numerator / denominator.
RATES = {
    "epa_db": ("epa", "dropbacks"),
    "epa_db_adj": ("epa_adj", "dropbacks"),
    "epa_neutral": ("epa_neutral", "dropbacks"),
    "epa_drop_adj": ("epa_drop_adj", "ftn_db"),
    "success_db": ("success", "dropbacks"),
    "wpa_db": ("wpa", "dropbacks"),
    "cpoe": ("cpoe_sum", "cpoe_n"),
    "ypa": ("pass_yards", "attempts"),
    "anya": ("anya_num", "anya_den"),
    "adot": ("air_sum", "air_n"),
    "deep_rate": ("deep_att", "air_n"),
    "deep_cpoe": ("deep_cpoe_sum", "deep_cpoe_n"),
    "explosive_rate": ("explosive", "dropbacks"),
    "sack_rate": ("sacks", "dropbacks"),
    "int_rate": ("ints", "attempts"),
    "scramble_rate": ("scrambles", "dropbacks"),
    "scramble_epa": ("scramble_epa", "scrambles"),
    "late_down_conv": ("late_conv", "late_n"),
    "turnover_wpa_db": ("turnover_wpa", "dropbacks"),
    "pressure_rate": ("pressures", "pressure_n"),
    "pressure_to_sack": ("pressure_sacks", "pressures"),
    "epa_pressured": ("pressure_epa", "pressures"),
    "epa_clean": ("clean_epa", "clean_n"),
    "time_to_throw": ("ttt_sum", "ttt_n"),
    "pass_rushers": ("rushers_sum", "rushers_n"),
    "int_worthy_rate": ("int_worthy", "ftn_att"),
    "catchable_rate": ("catchable", "ftn_att"),
    "drop_rate": ("drops", "ftn_att"),
    "throwaway_rate": ("throwaways", "ftn_att"),
    "play_action_rate": ("pa_n", "ftn_db"),
    "epa_play_action": ("pa_epa", "pa_n"),
    "epa_no_play_action": ("nopa_epa", "nopa_n"),
    "out_of_pocket_rate": ("oop_n", "ftn_db"),
    "epa_out_of_pocket": ("oop_epa", "oop_n"),
    "epa_in_pocket": ("inpocket_epa", "inpocket_n"),
    "blitz_rate": ("blitz_n", "ftn_db"),
    "epa_blitzed": ("blitz_epa", "blitz_n"),
    "qb_fault_sack_share": ("qb_fault_sacks", "ftn_sacks"),
    "screen_rate": ("screen_n", "ftn_db"),
    "rpo_rate": ("rpo_n", "ftn_db"),
    "pfr_pressure_rate": ("pfr_pressured", "pfr_db"),
    "pfr_bad_throw_rate": ("pfr_bad_throws", "pfr_att"),
    "pfr_drop_rate": ("pfr_drops", "pfr_att"),
    "ngs_aggressiveness": ("ngs_aggr_sum", "ngs_n"),
    "ngs_time_to_throw": ("ngs_ttt_sum", "ngs_n"),
    "ngs_cpoe": ("ngs_cpoe_sum", "ngs_n"),
    "ngs_air_yards_to_sticks": ("ngs_ays_sum", "ngs_n"),
    # Processing: what happens past the first read, against the blitz, by
    # coverage, and when the ball is held.
    "beyond_first_read_rate": ("beyond_n", "read_n"),
    "epa_first_read": ("first_epa", "first_n"),
    "epa_beyond_first": ("beyond_epa", "beyond_n"),
    "success_beyond_first": ("beyond_success", "beyond_n"),
    "epa_second_read": ("second_epa", "second_n"),
    "epa_checkdown": ("check_epa", "check_n"),
    "epa_scramble_drill": ("sd_epa", "sd_n"),
    "checkdown_rate": ("check_n", "read_n"),
    "int_worthy_beyond_first": ("beyond_intw", "beyond_n"),
    "blitz_epa_delta": ("blitz_delta", "blitz_pairs"),
    "blitz_sack_rate": ("blitz_sacks", "blitz_n"),
    "blitz_success": ("blitz_success", "blitz_n"),
    "blitz_int_worthy": ("blitz_intw", "blitz_att"),
    "no_blitz_epa": ("noblitz_epa", "noblitz_n"),
    "pressure_scramble_rate": ("pressure_scrambles", "pressures"),
    "pressure_throwaway_rate": ("pressure_throwaways", "pressures"),
    "held_sack_rate": ("held_sacks", "ttt_n"),
    "late_throw_epa": ("late_throw_epa", "late_throw_n"),
    "quick_throw_epa": ("quick_throw_epa", "quick_throw_n"),
    "epa_vs_man": ("man_epa", "man_n"),
    "epa_vs_zone": ("zone_epa", "zone_n"),
    "man_zone_delta": ("man_zone_delta", "man_zone_pairs"),
    "epa_vs_cover0_1": ("c01_epa", "c01_n"),
    "epa_vs_cover2_4": ("c24_epa", "c24_n"),
    "int_worthy_vs_zone": ("zone_intw", "zone_att"),
    "motion_rate": ("motion_n", "ftn_db"),
    "epa_with_motion": ("motion_epa", "motion_n"),
    "epa_without_motion": ("nomotion_epa", "nomotion_n"),
}


def _play_key(frame: pd.DataFrame, game: str, play: str) -> pd.Series:
    return frame[game].astype(str) + ":" + frame[play].astype("int64").astype(str)


def _ftn(season: int) -> pd.DataFrame:
    path = RAW_DIR / "ftn" / f"{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    ftn = pd.read_parquet(path)
    ftn["play_id"] = _play_key(ftn, "nflverse_game_id", "nflverse_play_id")
    return ftn.drop_duplicates("play_id").set_index("play_id")


def _participation(season: int) -> pd.DataFrame:
    path = RAW_DIR / "participation" / f"{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    part = pd.read_parquet(
        path,
        columns=[
            "nflverse_game_id",
            "play_id",
            "number_of_pass_rushers",
            "time_to_throw",
            "was_pressure",
            "defense_man_zone_type",
            "defense_coverage_type",
        ],
    )
    part["play_id"] = _play_key(part, "nflverse_game_id", "play_id")
    return part.drop_duplicates("play_id").set_index("play_id")


def _pfr_pass(season: int, players: pd.DataFrame) -> pd.DataFrame:
    path = RAW_DIR / "pfr_pass" / f"{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    pfr = pd.read_parquet(path)
    pfr = pfr[pfr["game_type"].eq("REG")]
    ids = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    pfr = pfr.assign(
        athlete_id=pfr["pfr_player_id"].map(dict(zip(ids["pfr_id"], ids["gsis_id"])))
    )
    pfr = pfr.dropna(subset=["athlete_id"])
    out = pd.DataFrame(
        {
            "game_id": pfr["game_id"],
            "athlete_id": pfr["athlete_id"].astype(str),
            "pfr_pressured": pfr["times_pressured"].astype(float),
            "pfr_bad_throws": pfr["passing_bad_throws"].astype(float),
            "pfr_drops": pfr["passing_drops"].astype(float),
            "pfr_sacked": pfr["times_sacked"].astype(float),
        }
    )
    return out.groupby(["game_id", "athlete_id"], as_index=False).sum()


def _ngs_passing(season: int) -> pd.DataFrame:
    path = RAW_DIR / "ngs" / f"passing_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    ngs = pd.read_parquet(path)
    ngs = ngs[ngs["season_type"].eq("REG") & ngs["week"].gt(0)]
    out = pd.DataFrame(
        {
            "week": ngs["week"].astype(int),
            "athlete_id": ngs["player_gsis_id"].astype(str),
            "ngs_aggr_sum": ngs["aggressiveness"].astype(float),
            "ngs_ttt_sum": ngs["avg_time_to_throw"].astype(float),
            "ngs_cpoe_sum": ngs["completion_percentage_above_expectation"].astype(
                float
            ),
            "ngs_ays_sum": ngs["avg_air_yards_to_sticks"].astype(float),
            "ngs_n": 1.0,
        }
    )
    return out.drop_duplicates(["week", "athlete_id"])


def _read_sums(rows, joined, charted, att, epa) -> dict:
    """Sums for the progression, blitz and motion measures from FTN charting.

    ``read_thrown`` is 1 for the first read, 2 for the second, CHK a checkdown,
    SD a scramble drill and DES a designed throw; 2, CHK and SD together are
    what the passer did when the first read was not there.
    """
    read = pd.Series(joined["read_thrown"].to_numpy(), index=rows.index).astype(str)
    first = att & read.eq("1")
    second = att & read.eq("2")
    check = att & read.eq("CHK")
    drill = att & read.eq("SD")
    beyond = second | check | drill
    read_n = first | beyond
    intw = pd.Series(joined["is_interception_worthy"].to_numpy(), index=rows.index).eq(
        True
    )
    blitzers = pd.Series(joined["n_blitzers"].to_numpy(float), index=rows.index).fillna(
        0
    )
    blitzed = charted & blitzers.gt(0)
    unblitzed = charted & blitzers.eq(0)
    motion = charted & pd.Series(joined["is_motion"].to_numpy(), index=rows.index).eq(
        True
    )
    nomotion = charted & ~motion
    success = rows["is_success"].astype(float)
    return {
        "read_n": read_n.astype(float),
        "first_n": first.astype(float),
        "first_epa": epa.where(first, 0.0),
        "beyond_n": beyond.astype(float),
        "beyond_epa": epa.where(beyond, 0.0),
        "beyond_success": success.where(beyond, 0.0),
        "beyond_intw": (beyond & intw).astype(float),
        "second_n": second.astype(float),
        "second_epa": epa.where(second, 0.0),
        "check_n": check.astype(float),
        "check_epa": epa.where(check, 0.0),
        "sd_n": drill.astype(float),
        "sd_epa": epa.where(drill, 0.0),
        "blitz_n": blitzed.astype(float),
        "blitz_epa": epa.where(blitzed, 0.0),
        "blitz_success": success.where(blitzed, 0.0),
        "blitz_sacks": (blitzed & rows["is_sack"]).astype(float),
        "blitz_att": (blitzed & att).astype(float),
        "blitz_intw": (blitzed & att & intw).astype(float),
        "noblitz_n": unblitzed.astype(float),
        "noblitz_epa": epa.where(unblitzed, 0.0),
        "motion_n": motion.astype(float),
        "motion_epa": epa.where(motion, 0.0),
        "nomotion_n": nomotion.astype(float),
        "nomotion_epa": epa.where(nomotion, 0.0),
    }


def _deltas(out: pd.DataFrame) -> pd.DataFrame:
    """Game-level differences that only make sense once both sides have plays."""

    def delta(a_epa, a_n, b_epa, b_n, name, pairs, floor=3):
        ok = out[a_n].ge(floor) & out[b_n].ge(floor)
        out[name] = (out[a_epa] / out[a_n] - out[b_epa] / out[b_n]).where(ok, 0.0)
        out[pairs] = ok.astype(float)

    if "blitz_epa" in out:
        delta(
            "blitz_epa",
            "blitz_n",
            "noblitz_epa",
            "noblitz_n",
            "blitz_delta",
            "blitz_pairs",
        )
    if "man_epa" in out:
        delta(
            "man_epa", "man_n", "zone_epa", "zone_n", "man_zone_delta", "man_zone_pairs"
        )
    return out


def qb_game_table(season: int, tables) -> pd.DataFrame:
    """Raw sums per quarterback and game for every candidate measure."""
    ctx = load_season_context(season)
    plays = ctx.plays
    plays = plays[plays["is_competitive"]]
    qbs = set(
        ctx.positions.loc[
            ctx.positions["position_group"].eq("QB"), "athlete_id"
        ].astype(str)
    )
    passer = plays["passer_player_id"].astype(str)
    rusher = plays["rusher_player_id"].astype(str)
    is_scramble = plays["qb_scramble"].eq(1) & plays["play_type"].eq("run")
    qb = passer.where(plays["play_type"].eq("pass"), rusher.where(is_scramble))
    rows = plays[qb.isin(qbs)].copy()
    rows["athlete_id"] = qb[qb.isin(qbs)]
    rows["is_scramble"] = is_scramble[qb.isin(qbs)]
    rows["is_pass"] = rows["play_type"].eq("pass")
    rows["is_sack"] = rows["sack"].eq(1)
    rows["is_attempt"] = rows["is_pass"] & ~rows["is_sack"]
    rows["is_completion"] = rows["complete_pass"].eq(1)
    rows["is_int"] = rows["interception"].eq(1)
    epa = rows["epa"].astype(float)
    wpa = rows["wpa"].astype(float).fillna(0.0)
    air = rows["air_yards"].astype(float)
    has_air = rows["is_attempt"] & air.notna()
    deep = has_air & air.ge(DEEP_AIR_YARDS)
    cpoe = rows["cpoe"].astype(float)
    yards = rows["yards_gained"].astype(float).fillna(0.0)
    late = is_late_down(rows) | is_two_minute(rows)
    expected = expected_conversion(tables, rows)
    incompletion_epa = float(
        epa[rows["is_attempt"] & ~rows["is_completion"] & ~rows["is_int"]].mean()
    )
    completion_epa = float(epa[rows["is_completion"]].mean())
    turnover = rows["is_int"] | (
        rows["fumble_lost"].eq(1)
        & rows["fumbled_1_player_id"].astype(str).eq(rows["athlete_id"])
    )
    touchdown = (
        rows["yards_gained"].astype(float).ge(0)
        & rows["is_completion"]
        & (rows["yardline_100"].astype(float) <= yards)
    )

    sums = {
        "dropbacks": 1.0,
        "attempts": rows["is_attempt"].astype(float),
        "completions": rows["is_completion"].astype(float),
        "epa": epa,
        "success": rows["is_success"].astype(float),
        "wpa": wpa,
        "cpoe_sum": cpoe.where(rows["is_attempt"]).fillna(0.0),
        "cpoe_n": (rows["is_attempt"] & cpoe.notna()).astype(float),
        "pass_yards": yards.where(rows["is_pass"], 0.0),
        "anya_num": (yards + 20 * touchdown - 45 * rows["is_int"]).where(
            rows["is_pass"], 0.0
        ),
        "anya_den": rows["is_pass"].astype(float),
        "air_sum": air.where(has_air).fillna(0.0),
        "air_n": has_air.astype(float),
        "deep_att": deep.astype(float),
        "deep_cpoe_sum": cpoe.where(deep & cpoe.notna()).fillna(0.0),
        "deep_cpoe_n": (deep & cpoe.notna()).astype(float),
        "explosive": (rows["is_completion"] & yards.ge(EXPLOSIVE_YARDS)).astype(float),
        "sacks": rows["is_sack"].astype(float),
        "ints": rows["is_int"].astype(float),
        "scrambles": rows["is_scramble"].astype(float),
        "scramble_epa": epa.where(rows["is_scramble"], 0.0),
        "late_conv": (rows["is_success"].astype(float) - expected).where(late, 0.0),
        "late_n": late.astype(float),
        "turnover_wpa": wpa.where(turnover, 0.0),
        # Interceptions are noise game to game; a turnover play is valued as
        # the incompletion it would have been.
        "epa_neutral": epa.where(~turnover, incompletion_epa),
    }
    # FTN flags are read first because the pressure block uses them too.
    ftn = _ftn(season)
    ftn_flags = None
    if not ftn.empty:
        ftn_joined = ftn.reindex(rows["play_id"])
        ftn_flags = {
            name: pd.Series(ftn_joined[name].to_numpy(), index=rows.index)
            .eq(True)
            .to_numpy()
            for name in ("is_interception_worthy", "is_throw_away")
        }
    # Participation: pressure and pass rushers per dropback, time to throw.
    part = _participation(season)
    if not part.empty:
        joined = part.reindex(rows["play_id"])
        flagged = pd.Series(joined["was_pressure"].to_numpy(), index=rows.index)
        # The feed leaves was_pressure unset on sacks before 2023 and flags
        # essentially every sack from then on, so a sack always counts as a
        # pressure and a play is known when flagged or a sack.
        pressured = flagged.eq(True) | rows["is_sack"]
        known = flagged.notna() | rows["is_sack"]
        ttt = pd.Series(joined["time_to_throw"].to_numpy(float), index=rows.index)
        rushers = pd.Series(
            joined["number_of_pass_rushers"].to_numpy(float), index=rows.index
        )
        man_zone = pd.Series(
            joined["defense_man_zone_type"].to_numpy(), index=rows.index
        )
        coverage = pd.Series(
            joined["defense_coverage_type"].to_numpy(), index=rows.index
        )
        man = man_zone.eq("MAN_COVERAGE")
        zone = man_zone.eq("ZONE_COVERAGE")
        c01 = coverage.isin(["COVER_0", "COVER_1"])
        c24 = coverage.isin(["COVER_2", "COVER_3", "COVER_4", "COVER_6", "2_MAN"])
        held = ttt.gt(3.0)
        thrown = rows["is_attempt"] & ttt.notna()
        intw = (
            pd.Series(ftn_flags["is_interception_worthy"], index=rows.index)
            if ftn_flags is not None
            else pd.Series(False, index=rows.index)
        )
        sums.update(
            {
                "pressures": pressured.astype(float),
                "pressure_n": known.astype(float),
                "pressure_sacks": (pressured & rows["is_sack"]).astype(float),
                "pressure_epa": epa.where(pressured, 0.0),
                "clean_epa": epa.where(known & ~pressured, 0.0),
                "clean_n": (known & ~pressured).astype(float),
                "ttt_sum": ttt.fillna(0.0),
                "ttt_n": ttt.notna().astype(float),
                "rushers_sum": rushers.fillna(0.0),
                "rushers_n": rushers.notna().astype(float),
                "pressure_scrambles": (pressured & rows["is_scramble"]).astype(float),
                "pressure_throwaways": (
                    pressured
                    & (
                        pd.Series(ftn_flags["is_throw_away"], index=rows.index)
                        if ftn_flags is not None
                        else pd.Series(False, index=rows.index)
                    )
                ).astype(float),
                "held_sacks": (held & rows["is_sack"]).astype(float),
                "late_throw_epa": epa.where(thrown & held, 0.0),
                "late_throw_n": (thrown & held).astype(float),
                "quick_throw_epa": epa.where(thrown & ttt.le(2.5), 0.0),
                "quick_throw_n": (thrown & ttt.le(2.5)).astype(float),
                "man_epa": epa.where(man, 0.0),
                "man_n": man.astype(float),
                "zone_epa": epa.where(zone, 0.0),
                "zone_n": zone.astype(float),
                "c01_epa": epa.where(c01, 0.0),
                "c01_n": c01.astype(float),
                "c24_epa": epa.where(c24, 0.0),
                "c24_n": c24.astype(float),
                "zone_intw": (zone & intw).astype(float),
                "zone_att": (zone & rows["is_attempt"]).astype(float),
            }
        )
    # FTN charting.
    if not ftn.empty:
        joined = ftn_joined

        def flag(column):
            return pd.Series(joined[column].to_numpy(), index=rows.index).eq(True)

        charted = pd.Series(joined["ftn_play_id"].to_numpy(), index=rows.index).notna()
        att = charted & rows["is_attempt"]
        pa = flag("is_play_action")
        oop = flag("is_qb_out_of_pocket")
        blitz = (
            pd.Series(joined["n_blitzers"].to_numpy(float), index=rows.index)
            .fillna(0)
            .gt(0)
        )
        sums.update(
            {
                "ftn_db": charted.astype(float),
                "ftn_att": att.astype(float),
                "ftn_sacks": (charted & rows["is_sack"]).astype(float),
                "int_worthy": (att & flag("is_interception_worthy")).astype(float),
                "catchable": (att & flag("is_catchable_ball")).astype(float),
                "drops": (att & flag("is_drop")).astype(float),
                "throwaways": (att & flag("is_throw_away")).astype(float),
                "pa_n": (charted & pa).astype(float),
                "pa_epa": epa.where(charted & pa, 0.0),
                "nopa_n": (charted & ~pa).astype(float),
                "nopa_epa": epa.where(charted & ~pa, 0.0),
                "oop_n": (charted & oop).astype(float),
                "oop_epa": epa.where(charted & oop, 0.0),
                "inpocket_n": (charted & ~oop).astype(float),
                "inpocket_epa": epa.where(charted & ~oop, 0.0),
                "blitz_n": (charted & blitz).astype(float),
                "blitz_epa": epa.where(charted & blitz, 0.0),
                "qb_fault_sacks": (
                    charted & rows["is_sack"] & flag("is_qb_fault_sack")
                ).astype(float),
                "screen_n": (charted & flag("is_screen_pass")).astype(float),
                **_read_sums(rows, joined, charted, att, epa),
                # A catchable ball dropped is valued as the completion it was.
                "epa_drop_adj": epa.where(
                    ~(att & flag("is_drop")), completion_epa
                ).where(charted, 0.0),
                "rpo_n": (charted & flag("is_rpo")).astype(float),
            }
        )
    frame = rows[["game_id", "model_week", "athlete_id", "posteam", "defteam"]].rename(
        columns={"model_week": "week", "posteam": "team", "defteam": "opponent"}
    )
    for name, values in sums.items():
        frame[name] = values if isinstance(values, pd.Series) else float(values)
    out = frame.groupby(
        ["game_id", "week", "athlete_id", "team", "opponent"], as_index=False
    ).sum()
    out["season"] = season
    out = _deltas(out)
    # EPA net of the defense faced through the previous week.
    from backend.players.adjust import adjust_rate

    events = out[["game_id", "opponent"]].assign(
        numerator=out["epa"].astype(float), denominator=out["dropbacks"].astype(float)
    )
    out["epa_adj"] = adjust_rate(events, ctx.games)["numerator"].to_numpy()
    # Game-level PFR and weekly NGS lines.
    players = pd.read_parquet(
        RAW_DIR / "players.parquet", columns=["gsis_id", "pfr_id"]
    )
    pfr = _pfr_pass(season, players)
    if not pfr.empty:
        out = out.merge(pfr, on=["game_id", "athlete_id"], how="left")
        out["pfr_db"] = out["dropbacks"].where(out["pfr_pressured"].notna())
        out["pfr_att"] = out["attempts"].where(out["pfr_bad_throws"].notna())
    ngs = _ngs_passing(season)
    if not ngs.empty:
        out = out.merge(ngs, on=["week", "athlete_id"], how="left")
    identity = ctx.identity.set_index("athlete_id")["athlete_name"]
    out["athlete_name"] = out["athlete_id"].map(identity)
    return out[
        KEYS
        + ["athlete_name"]
        + [c for c in out.columns if c not in KEYS and c != "athlete_name"]
    ]


def build_qb_games(seasons: list[int]) -> pd.DataFrame:
    """Every season's quarterback-game table, written to the research directory."""
    plays = {
        season: load_plays(season)
        for season in range(min(seasons) - 1, max(seasons) + 1)
    }
    parts = []
    for season in seasons:
        tables = fit_expectation_tables(plays, season)
        parts.append(qb_game_table(season, tables))
        log.info("qb study %s: %d quarterback games", season, len(parts[-1]))
    out = pd.concat(parts, ignore_index=True)
    write_parquet(out, STUDY_DIR / "qb_games.parquet")
    return out


# Evaluation


def rate(frame: pd.DataFrame, name: str, rates: dict | None = None) -> pd.Series:
    numerator, denominator = (rates or RATES)[name]
    if numerator not in frame or denominator not in frame:
        return pd.Series(np.nan, index=frame.index)
    den = frame[denominator].astype(float)
    return frame[numerator].astype(float) / den.where(den.gt(0))


def _sums_through(
    games: pd.DataFrame, upto: int | None, after: bool = False
) -> pd.DataFrame:
    rows = (
        games[games["week"].gt(upto)]
        if after
        else games[games["week"].le(upto)]
        if upto
        else games
    )
    numeric = [c for c in rows.columns if c not in KEYS and c != "athlete_name"]
    out = rows.groupby(["season", "athlete_id"], as_index=False)[numeric].sum(
        min_count=1
    )
    out["games"] = rows.groupby(["season", "athlete_id"]).size().to_numpy()
    return out


def split_half(
    games: pd.DataFrame, min_games: int = 8, rates: dict | None = None
) -> pd.DataFrame:
    """Odd-game against even-game rates per measure, pooled over seasons."""
    rows = games.sort_values(["season", "athlete_id", "week"]).copy()
    rows["order"] = rows.groupby(["season", "athlete_id"]).cumcount()
    rows["played"] = rows.groupby(["season", "athlete_id"])["week"].transform("size")
    rows = rows[rows["played"].ge(min_games) & rows["order"].lt(min_games)]
    rows["half"] = rows["order"] % 2
    numeric = [
        c
        for c in rows.columns
        if c not in KEYS + ["athlete_name", "order", "played", "half"]
    ]
    halves = rows.groupby(["season", "athlete_id", "half"])[numeric].sum(min_count=1)
    results = []
    rates = rates or RATES
    for name in rates:
        values = rate(halves.reset_index(), name, rates)
        wide = pd.DataFrame(
            {"half": halves.index.get_level_values("half"), "value": values.to_numpy()},
            index=halves.index.droplevel("half"),
        )
        wide = wide.pivot_table(
            index=["season", "athlete_id"], columns="half", values="value"
        ).dropna()
        if len(wide) < 30:
            results.append(
                {"measure": name, "players": len(wide), "split_half": np.nan}
            )
            continue
        results.append(
            {
                "measure": name,
                "players": len(wide),
                "split_half": float(wide[0].corr(wide[1])),
            }
        )
    return pd.DataFrame(results)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    both = a.notna() & b.notna()
    if both.sum() < 20:
        return np.nan
    return float(a[both].rank().corr(b[both].rank()))


def checkpoint_prediction(
    games: pd.DataFrame,
    week: int,
    targets=("epa_db", "success_db"),
    rates: dict | None = None,
    volume: tuple[str, float, float] = ("dropbacks", 100, 0),
) -> pd.DataFrame:
    """Each measure through ``week`` against the rest-of-season targets, averaged."""
    before = _sums_through(games, week)
    after = _sums_through(games, week, after=True)
    after = after[after["games"].ge(3)]
    merged = before.merge(after, on=["season", "athlete_id"], suffixes=("", "_rest"))
    rates = rates or RATES
    column, floor, rest_floor = volume
    merged = merged[merged[column].ge(floor) & merged[f"{column}_rest"].ge(rest_floor)]
    results = []
    for name in rates:
        value = rate(merged, name, rates)
        row = {"measure": name, "week": week, "players": int(value.notna().sum())}
        for target in targets:
            numerator, denominator = rates[target]
            future = merged[f"{numerator}_rest"] / merged[f"{denominator}_rest"].where(
                merged[f"{denominator}_rest"].gt(0)
            )
            per_season = [
                _spearman(value[merged["season"].eq(s)], future[merged["season"].eq(s)])
                for s in sorted(merged["season"].unique())
            ]
            row[f"rho_{target}"] = (
                float(np.nanmean(per_season))
                if np.isfinite(per_season).any()
                else np.nan
            )
        results.append(row)
    return pd.DataFrame(results)


def next_season_prediction(
    games: pd.DataFrame,
    targets=("epa_db", "success_db"),
    rates: dict | None = None,
    volume: tuple[str, float] = ("dropbacks", 200),
) -> pd.DataFrame:
    """Each measure over a full season against the passer's next season."""
    rates = rates or RATES
    totals = _sums_through(games, None)
    totals = totals[totals[volume[0]].ge(volume[1])]
    nxt = totals.assign(season=totals["season"] - 1)
    merged = totals.merge(nxt, on=["season", "athlete_id"], suffixes=("", "_next"))
    results = []
    for name in rates:
        value = rate(merged, name, rates)
        row = {"measure": name, "players": int(value.notna().sum())}
        for target in targets:
            numerator, denominator = rates[target]
            future = merged[f"{numerator}_next"] / merged[f"{denominator}_next"].where(
                merged[f"{denominator}_next"].gt(0)
            )
            per_pair = [
                _spearman(value[merged["season"].eq(s)], future[merged["season"].eq(s)])
                for s in sorted(merged["season"].unique())
            ]
            row[f"rho_{target}_next"] = (
                float(np.nanmean(per_pair)) if np.isfinite(per_pair).any() else np.nan
            )
        results.append(row)
    return pd.DataFrame(results)


def run_study(seasons: list[int], rebuild: bool = False) -> dict[str, pd.DataFrame]:
    path = STUDY_DIR / "qb_games.parquet"
    games = (
        pd.read_parquet(path)
        if path.exists() and not rebuild
        else build_qb_games(seasons)
    )
    games = games[games["season"].isin(seasons)]
    out = {
        "split_half": split_half(games),
        "week6": checkpoint_prediction(games, 6),
        "week9": checkpoint_prediction(games, 9),
        "next_season": next_season_prediction(games),
    }
    summary = out["split_half"].merge(
        out["week6"].drop(columns="week"),
        on="measure",
        how="left",
        suffixes=("", "_w6"),
    )
    summary = summary.merge(
        out["week9"].drop(columns=["week", "players"]),
        on="measure",
        how="left",
        suffixes=("", "_w9"),
    )
    summary = summary.merge(
        out["next_season"].drop(columns="players"), on="measure", how="left"
    )
    out["summary"] = summary
    write_parquet(summary, STUDY_DIR / "qb_summary.parquet")
    return out


# Combinations


FEATURES = [
    "epa_db",
    "epa_db_adj",
    "epa_neutral",
    "success_db",
    "cpoe",
    "anya",
    "sack_rate",
    "pressure_to_sack",
    "pressure_rate",
    "epa_clean",
    "epa_pressured",
    "late_down_conv",
    "explosive_rate",
    "int_rate",
    "scramble_epa",
    "turnover_wpa_db",
]
PRIOR_FEATURES = ["epa_db", "success_db", "cpoe", "sack_rate", "pressure_to_sack"]
MODELS = {
    "epa_only": ["epa_db"],
    "epa_success": ["epa_db", "success_db"],
    "epa_prior": ["epa_db", "prior_epa_db", "prior_games"],
    "core": [
        "epa_db",
        "success_db",
        "cpoe",
        "sack_rate",
        "pressure_to_sack",
        "late_down_conv",
    ],
    "core_prior": [
        "epa_db",
        "success_db",
        "cpoe",
        "sack_rate",
        "pressure_to_sack",
        "late_down_conv",
        *[f"prior_{f}" for f in PRIOR_FEATURES],
        "prior_games",
    ],
    "all_prior": [*FEATURES, *[f"prior_{f}" for f in PRIOR_FEATURES], "prior_games"],
    "epa_adjusted": ["epa_db_adj"],
    "epa_neutral": ["epa_neutral"],
    "epa_decay": ["epa_db_decay"],
    "epa_bio": ["epa_db", "age", "experience", "draft_pick"],
    "core_prior_bio": [
        "epa_db",
        "success_db",
        "cpoe",
        "sack_rate",
        "pressure_to_sack",
        "late_down_conv",
        *[f"prior_{f}" for f in PRIOR_FEATURES],
        "prior_games",
        "age",
        "experience",
        "draft_pick",
    ],
}
BIO_FEATURES = ["age", "experience", "draft_pick"]
DECAY_HALF_LIFE_GAMES = 4.0
CHECKPOINTS = tuple(range(4, 14))
MIN_DROPBACKS = 60
MIN_REST_DROPBACKS = 100
PENALTY_GRID = np.logspace(-1, 4, 26)


def _z_within(frame: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        grouped = out.groupby(keys)[column]
        mean, sd = grouped.transform("mean"), grouped.transform("std")
        out[column] = (
            ((out[column] - mean) / sd.where(sd.gt(0))).fillna(0.0).clip(-3, 3)
        )
    return out


def checkpoint_rows(
    games: pd.DataFrame,
    rates: dict | None = None,
    features: list[str] | None = None,
    prior_features: list[str] | None = None,
    volume: tuple[str, float, float] = ("dropbacks", MIN_DROPBACKS, MIN_REST_DROPBACKS),
    prior_volume: float = 100,
    targets: dict[str, str] | None = None,
    decay: tuple[str, str] = ("epa", "dropbacks"),
) -> pd.DataFrame:
    """One row per player, season and checkpoint week: rates to date, prior-season
    rates, and the rest-of-season targets (``targets`` maps a target column to
    the rate it is built from)."""
    rates = rates or RATES
    features = features or FEATURES
    prior_features = prior_features or PRIOR_FEATURES
    targets = targets or {"target_epa": "epa_db", "target_success": "success_db"}
    column, floor, rest_floor = volume
    totals = _sums_through(games, None)
    prior = totals[totals[column].ge(prior_volume)].copy()
    for name in prior_features:
        prior[f"prior_{name}"] = rate(prior, name, rates)
    prior["prior_games"] = prior["games"].astype(float)
    prior = prior[
        ["season", "athlete_id", *[f"prior_{f}" for f in prior_features], "prior_games"]
    ]
    prior["season"] = prior["season"] + 1
    bio = _biography(games)
    parts = []
    for week in CHECKPOINTS:
        before = _sums_through(games, week)
        decayed = _decayed_rate(games, week, *decay)
        before = before[before[column].ge(floor)]
        after = _sums_through(games, week, after=True)
        after = after[after[column].ge(rest_floor)]
        merged = before.merge(
            after, on=["season", "athlete_id"], suffixes=("", "_rest")
        )
        row = merged[["season", "athlete_id"]].assign(week=week)
        for name in features:
            row[name] = rate(merged, name, rates).to_numpy()
        row["epa_db_decay"] = (
            merged[["season", "athlete_id"]]
            .merge(decayed, on=["season", "athlete_id"], how="left")["decayed"]
            .to_numpy()
        )
        for target, source in targets.items():
            numerator, denominator = rates[source]
            row[target] = (
                merged[f"{numerator}_rest"]
                / merged[f"{denominator}_rest"].where(
                    merged[f"{denominator}_rest"].gt(0)
                )
            ).to_numpy()
        parts.append(row)
    rows = pd.concat(parts, ignore_index=True).merge(
        prior, on=["season", "athlete_id"], how="left"
    )
    rows = rows.merge(bio, on=["season", "athlete_id"], how="left")
    rows[BIO_FEATURES] = rows[BIO_FEATURES].fillna(rows[BIO_FEATURES].mean())
    rows["epa_db_decay"] = rows["epa_db_decay"].fillna(rows[features[0]])
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    prior_columns = [f"prior_{f}" for f in prior_features]
    rows[prior_columns] = rows[prior_columns].fillna(rows[prior_columns].mean())
    rows[features] = rows[features].fillna(rows[features].mean())
    return _z_within(
        rows,
        [*features, *prior_columns, "prior_games", *BIO_FEATURES, "epa_db_decay"],
        ["season", "week"],
    )


def _biography(games: pd.DataFrame) -> pd.DataFrame:
    """Age at the season's start, seasons of experience and draft pick (undrafted
    counts as pick 262) per quarterback season."""
    players = pd.read_parquet(
        RAW_DIR / "players.parquet",
        columns=["gsis_id", "birth_date", "rookie_season", "draft_pick"],
    ).dropna(subset=["gsis_id"])
    players["athlete_id"] = players["gsis_id"].astype(str)
    seasons = games[["season", "athlete_id"]].drop_duplicates()
    out = seasons.merge(players, on="athlete_id", how="left")
    born = pd.to_datetime(out["birth_date"], errors="coerce")
    start = pd.to_datetime(out["season"].astype(str) + "-09-01")
    out["age"] = (start - born).dt.days / 365.25
    out["experience"] = (out["season"] - out["rookie_season"]).astype(float)
    out["draft_pick"] = out["draft_pick"].fillna(262.0).astype(float)
    return out[["season", "athlete_id", *BIO_FEATURES]]


def _decayed_rate(
    games: pd.DataFrame, week: int, numerator: str, denominator: str
) -> pd.DataFrame:
    """A rate through ``week`` with each game weighted by 0.5 ** (weeks ago /
    half-life), so the recent games count more."""
    rows = games[games["week"].le(week)]
    weight = 0.5 ** ((week - rows["week"]) / DECAY_HALF_LIFE_GAMES)
    frame = rows[["season", "athlete_id"]].assign(
        n=rows[numerator] * weight, d=rows[denominator] * weight
    )
    out = frame.groupby(["season", "athlete_id"], as_index=False).sum()
    out["decayed"] = out["n"] / out["d"].where(out["d"].gt(0))
    return out[["season", "athlete_id", "decayed"]]


def _ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    reg = penalty * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + x @ beta[1:]


def combination_study(
    rows: pd.DataFrame, target: str = "target_epa", models: dict | None = None
) -> pd.DataFrame:
    """Leave-one-season-out: each model's Spearman and R squared on the held-out
    season's rest-of-season target, with the penalty chosen by an inner
    leave-one-season-out on the training seasons."""
    seasons = sorted(rows["season"].unique())
    results = []
    for name, columns in (models or MODELS).items():
        rhos, r2s, coefs = [], [], []
        for held in seasons:
            train = rows[rows["season"].ne(held)]
            test = rows[rows["season"].eq(held)]
            x, y = train[columns].to_numpy(float), train[target].to_numpy(float)
            inner = []
            for penalty in PENALTY_GRID:
                sse = 0.0
                for inner_held in sorted(train["season"].unique()):
                    fit = train["season"].ne(inner_held).to_numpy()
                    beta = _ridge_fit(x[fit], y[fit], penalty)
                    resid = y[~fit] - _predict(beta, x[~fit])
                    sse += float(resid @ resid)
                inner.append(sse)
            penalty = float(PENALTY_GRID[int(np.argmin(inner))])
            beta = _ridge_fit(x, y, penalty)
            prediction = _predict(beta, test[columns].to_numpy(float))
            actual = test[target].to_numpy(float)
            rhos.append(_spearman(pd.Series(prediction), pd.Series(actual)))
            r2s.append(
                1
                - float(
                    ((actual - prediction) ** 2).sum()
                    / ((actual - actual.mean()) ** 2).sum()
                )
            )
            coefs.append(beta[1:])
        results.append(
            {
                "model": name,
                "target": target,
                "rho": float(np.nanmean(rhos)),
                "r2": float(np.mean(r2s)),
                "coefficients": " ".join(
                    f"{c}={v:+.3f}" for c, v in zip(columns, np.mean(coefs, axis=0))
                ),
            }
        )
    return pd.DataFrame(results)


# Lineup-adjusted quarterback effect (prototype)


def _dropback_rows(season: int) -> pd.DataFrame:
    """Competitive dropbacks with the quarterback, the defense and the ten other
    offensive players on the field (participation), one row per play."""
    ctx = load_season_context(season)
    plays = ctx.plays[ctx.plays["is_competitive"] & ctx.plays["qb_dropback"].eq(1)]
    qbs = set(ctx.positions.loc[ctx.positions["position_group"].eq("QB"), "athlete_id"])
    qb = (
        plays["passer_player_id"]
        .where(plays["play_type"].eq("pass"), plays["rusher_player_id"])
        .astype(str)
    )
    plays = plays[qb.isin(qbs)].assign(qb=qb[qb.isin(qbs)])
    part = pd.read_parquet(
        RAW_DIR / "participation" / f"{season}.parquet",
        columns=["nflverse_game_id", "play_id", "offense_players"],
    )
    part["play_id"] = _play_key(part, "nflverse_game_id", "play_id")
    part = part.drop_duplicates("play_id").set_index("play_id")["offense_players"]
    teammates = plays["play_id"].map(part).fillna("")
    out = plays[["game_id", "model_week", "qb", "posteam", "defteam", "epa"]].copy()
    out["teammates"] = [
        [p for p in str(line).split(";") if p and p != q]
        for line, q in zip(teammates, out["qb"])
    ]
    return out.rename(columns={"model_week": "week"})


def lineup_adjusted_qb(
    rows: pd.DataFrame, through_week: int, penalty_qb: float, penalty_other: float
) -> pd.Series:
    """Ridge of play EPA on quarterback, defense and teammate indicators over
    dropbacks through ``through_week``; the quarterback coefficients, by id."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    rows = rows[rows["week"].le(through_week)]
    qb_ids = sorted(rows["qb"].unique())
    def_ids = sorted(rows["defteam"].unique())
    mate_ids = sorted({p for line in rows["teammates"] for p in line})
    # Group prefixes keep a backup quarterback who lines up elsewhere apart
    # from himself as a passer.
    keys = (
        [f"qb:{k}" for k in qb_ids]
        + [f"def:{k}" for k in def_ids]
        + [f"mate:{k}" for k in mate_ids]
    )
    index = {key: i for i, key in enumerate(keys)}
    n_qb, n_def = len(qb_ids), len(def_ids)
    data, row_index, col_index = [], [], []
    for i, (qb, defense, mates) in enumerate(
        zip(rows["qb"], rows["defteam"], rows["teammates"])
    ):
        for key in (f"qb:{qb}", f"def:{defense}", *(f"mate:{m}" for m in mates)):
            row_index.append(i)
            col_index.append(index[key])
            data.append(1.0)
    n_rows, n_cols = len(rows), len(index)
    # Ridge as augmented rows: sqrt(penalty) on the diagonal, per group.
    scale = np.concatenate(
        [
            np.full(n_qb, np.sqrt(penalty_qb)),
            np.full(n_def, np.sqrt(penalty_other)),
            np.full(n_cols - n_qb - n_def, np.sqrt(penalty_other)),
        ]
    )
    aug_rows = list(range(n_rows, n_rows + n_cols))
    design = csr_matrix(
        (
            np.concatenate([data, scale]),
            (
                np.concatenate([row_index, aug_rows]),
                np.concatenate([col_index, range(n_cols)]),
            ),
        ),
        shape=(n_rows + n_cols, n_cols),
    )
    y = np.concatenate(
        [rows["epa"].to_numpy(float) - rows["epa"].mean(), np.zeros(n_cols)]
    )
    beta = lsqr(design, y, atol=1e-8, btol=1e-8)[0]
    return pd.Series(beta[:n_qb], index=qb_ids)


def lineup_study(
    seasons: list[int], weeks=(6, 9), penalties=((50, 200), (100, 400), (200, 800))
) -> pd.DataFrame:
    """Does a lineup-adjusted QB effect through week w predict the rest of the
    season better than raw EPA per dropback through week w?"""
    results = []
    for season in seasons:
        rows = _dropback_rows(season)
        for week in weeks:
            before = rows[rows["week"].le(week)]
            after = rows[rows["week"].gt(week)]
            raw = before.groupby("qb")["epa"].agg(["mean", "size"])
            rest = after.groupby("qb")["epa"].agg(["mean", "size"])
            keep = raw.index[(raw["size"] >= MIN_DROPBACKS)].intersection(
                rest.index[rest["size"] >= MIN_REST_DROPBACKS]
            )
            target = rest.loc[keep, "mean"]
            results.append(
                {
                    "season": season,
                    "week": week,
                    "model": "raw_epa",
                    "players": len(keep),
                    "rho": _spearman(raw.loc[keep, "mean"], target),
                }
            )
            for penalty_qb, penalty_other in penalties:
                adjusted = lineup_adjusted_qb(rows, week, penalty_qb, penalty_other)
                results.append(
                    {
                        "season": season,
                        "week": week,
                        "model": f"adjusted_{penalty_qb}_{penalty_other}",
                        "players": len(keep),
                        "rho": _spearman(adjusted.reindex(keep), target),
                    }
                )
    out = pd.DataFrame(results)
    return out.groupby(["week", "model"], as_index=False)["rho"].mean()


# Style against quality


def self_persistence(
    games: pd.DataFrame,
    rates: dict | None = None,
    volume: tuple[str, float] = ("dropbacks", 200),
) -> pd.DataFrame:
    """Each measure's Spearman with itself one season later: a measure that
    repeats season to season is a trait of the passer, whether or not it
    predicts his production."""
    rates = rates or RATES
    totals = _sums_through(games, None)
    totals = totals[totals[volume[0]].ge(volume[1])]
    nxt = totals.assign(season=totals["season"] - 1)
    merged = totals.merge(nxt, on=["season", "athlete_id"], suffixes=("", "_next"))
    rows = []
    for name in rates:
        now = rate(merged, name, rates)
        numerator, denominator = rates[name]
        later = merged[f"{numerator}_next"] / merged[f"{denominator}_next"].where(
            merged[f"{denominator}_next"].gt(0)
        )
        per_pair = [
            _spearman(now[merged["season"].eq(s)], later[merged["season"].eq(s)])
            for s in sorted(merged["season"].unique())
        ]
        rows.append(
            {
                "measure": name,
                "self_next": float(np.nanmean(per_pair))
                if np.isfinite(per_pair).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def style_table(summary: pd.DataFrame, persistence: pd.DataFrame) -> pd.DataFrame:
    """A measure is a *quality* when it is stable and predicts future production,
    a *style* when it is stable, repeats season to season, and does not predict
    production, and *noise* when it is not stable."""
    out = summary.merge(persistence, on="measure", how="left")
    predictive = out[[c for c in out.columns if c.startswith("rho_")]].abs().max(axis=1)
    stable = out["split_half"].ge(0.3) | out["self_next"].ge(0.5)
    out["kind"] = np.select(
        [~stable, predictive.ge(0.25), out["self_next"].ge(0.5)],
        ["noise", "quality", "style"],
        "weak",
    )
    out["predictive"] = predictive
    return out[
        ["measure", "split_half", "self_next", "predictive", "kind"]
    ].sort_values(["kind", "self_next"], ascending=[True, False])


# Multi-season lineup model


def _all_dropback_rows(seasons: list[int]) -> pd.DataFrame:
    parts = [_dropback_rows(season).assign(season=season) for season in seasons]
    return pd.concat(parts, ignore_index=True)


def dynamic_lineup_fit(
    rows: pd.DataFrame,
    penalty_qb: float,
    penalty_walk: float,
    penalty_other: float,
) -> pd.DataFrame:
    """Ridge of play EPA on quarterback, defense and teammate effects per season,
    with each entity's effect also pulled toward its own previous season
    (``penalty_walk``), so roster turnover across seasons identifies the split
    between a passer and the players around him. Returns ``entity, season,
    effect`` for the quarterbacks."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    rows = rows.reset_index(drop=True)
    keys: dict[tuple[str, str, int], int] = {}

    def column(group: str, entity: str, season: int) -> int:
        key = (group, entity, int(season))
        if key not in keys:
            keys[key] = len(keys)
        return keys[key]

    data, row_index, col_index = [], [], []
    for i, (season, qb, defense, mates) in enumerate(
        zip(rows["season"], rows["qb"], rows["defteam"], rows["teammates"])
    ):
        for group, entity in (
            ("qb", qb),
            ("def", defense),
            *(("mate", m) for m in mates),
        ):
            row_index.append(i)
            col_index.append(column(group, entity, season))
            data.append(1.0)
    n_rows, n_cols = len(rows), len(keys)
    penalty = {"qb": penalty_qb, "def": penalty_other, "mate": penalty_other}
    # Ridge toward zero, then the random walk between consecutive seasons.
    extra_rows, extra_cols, extra_data = [], [], []
    r = n_rows
    for (group, entity, season), j in keys.items():
        extra_rows.append(r)
        extra_cols.append(j)
        extra_data.append(np.sqrt(penalty[group]))
        r += 1
        previous = keys.get((group, entity, season - 1))
        if previous is not None:
            extra_rows += [r, r]
            extra_cols += [j, previous]
            extra_data += [np.sqrt(penalty_walk), -np.sqrt(penalty_walk)]
            r += 1
    design = csr_matrix(
        (
            np.concatenate([data, extra_data]),
            (
                np.concatenate([row_index, extra_rows]),
                np.concatenate([col_index, extra_cols]),
            ),
        ),
        shape=(r, n_cols),
    )
    y = np.concatenate(
        [rows["epa"].to_numpy(float) - rows["epa"].mean(), np.zeros(r - n_rows)]
    )
    beta = lsqr(design, y, atol=1e-8, btol=1e-8, iter_lim=20000)[0]
    out = pd.DataFrame(
        [
            (entity, season, beta[j])
            for (group, entity, season), j in keys.items()
            if group == "qb"
        ],
        columns=["athlete_id", "season", "effect"],
    )
    return out


def dynamic_lineup_study(
    seasons: list[int],
    weeks=(6, 9),
    penalties=((100, 200, 500), (200, 400, 800), (400, 800, 1600)),
) -> dict[str, pd.DataFrame]:
    """Walk-forward test of the multi-season lineup model.

    In-season: at week ``w`` of season ``t``, fit on every earlier season and
    season ``t`` through ``w``; the passer's season-``t`` effect against his
    rest-of-season EPA per dropback, beside raw EPA to date and a two-season
    blend of raw EPA. Transfer: for passers who changed teams between ``t - 1``
    and ``t``, the season ``t - 1`` effect (fit on seasons through ``t - 1``)
    against season-``t`` EPA per dropback, beside raw season ``t - 1`` EPA; the
    same for stayers.
    """
    rows = _all_dropback_rows(seasons)
    in_season, transfer = [], []
    for t in seasons[1:]:
        history = rows[rows["season"].lt(t)]
        current = rows[rows["season"].eq(t)]
        prior_raw = (
            history[history["season"].eq(t - 1)]
            .groupby("qb")["epa"]
            .agg(["mean", "size"])
        )
        season_raw = current.groupby("qb")["epa"].agg(["mean", "size"])
        # Transfer test uses fits through t - 1 only.
        for penalty_qb, penalty_walk, penalty_other in penalties:
            fitted = dynamic_lineup_fit(
                history, penalty_qb, penalty_walk, penalty_other
            )
            last = fitted[fitted["season"].eq(t - 1)].set_index("athlete_id")["effect"]
            eligible = prior_raw.index[prior_raw["size"].ge(200)].intersection(
                season_raw.index[season_raw["size"].ge(200)]
            )
            tb = (
                history[history["season"].eq(t - 1)]
                .groupby("qb")["posteam"]
                .agg(lambda s: s.mode().iloc[0])
            )
            ta = current.groupby("qb")["posteam"].agg(lambda s: s.mode().iloc[0])
            movers = [qb for qb in eligible if tb.get(qb) != ta.get(qb)]
            stayers = [qb for qb in eligible if qb not in movers]
            for label, group in (("movers", movers), ("stayers", stayers)):
                if len(group) < 5:
                    continue
                target = season_raw.loc[group, "mean"]
                transfer.append(
                    {
                        "season": t,
                        "group": label,
                        "penalties": f"{penalty_qb}/{penalty_walk}/{penalty_other}",
                        "players": len(group),
                        "rho_adjusted": _spearman(last.reindex(group), target),
                        "rho_raw_prior": _spearman(
                            prior_raw.loc[group, "mean"], target
                        ),
                    }
                )
        # In-season test.
        for week in weeks:
            before = current[current["week"].le(week)]
            after = current[current["week"].gt(week)]
            raw = before.groupby("qb")["epa"].agg(["mean", "size"])
            rest = after.groupby("qb")["epa"].agg(["mean", "size"])
            keep = raw.index[raw["size"].ge(MIN_DROPBACKS)].intersection(
                rest.index[rest["size"].ge(MIN_REST_DROPBACKS)]
            )
            target = rest.loc[keep, "mean"]
            blend_num = raw.loc[keep, "mean"] * raw.loc[keep, "size"] + 0.6 * prior_raw[
                "mean"
            ].reindex(keep).fillna(0) * prior_raw["size"].reindex(keep).fillna(0)
            blend_den = raw.loc[keep, "size"] + 0.6 * prior_raw["size"].reindex(
                keep
            ).fillna(0)
            base = [
                ("raw_epa", raw.loc[keep, "mean"]),
                ("blend_two_seasons", blend_num / blend_den),
            ]
            for name, values in base:
                in_season.append(
                    {
                        "season": t,
                        "week": week,
                        "model": name,
                        "players": len(keep),
                        "rho": _spearman(values, target),
                    }
                )
            fit_rows = pd.concat([history, before], ignore_index=True)
            for penalty_qb, penalty_walk, penalty_other in penalties:
                fitted = dynamic_lineup_fit(
                    fit_rows, penalty_qb, penalty_walk, penalty_other
                )
                now = fitted[fitted["season"].eq(t)].set_index("athlete_id")["effect"]
                in_season.append(
                    {
                        "season": t,
                        "week": week,
                        "model": (
                            f"adjusted_{penalty_qb}/{penalty_walk}/{penalty_other}"
                        ),
                        "players": len(keep),
                        "rho": _spearman(now.reindex(keep), target),
                    }
                )
        log.info("dynamic lineup study: season %s done", t)
    in_season = pd.DataFrame(in_season)
    transfer = pd.DataFrame(transfer)
    return {
        "in_season": in_season.groupby(["week", "model"], as_index=False)["rho"].mean(),
        "transfer": transfer.groupby(["group", "penalties"], as_index=False)[
            ["players", "rho_adjusted", "rho_raw_prior"]
        ].mean(),
        "transfer_by_season": transfer,
    }
