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
            }
        )
    # FTN charting.
    ftn = _ftn(season)
    if not ftn.empty:
        joined = ftn.reindex(rows["play_id"])

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


def rate(frame: pd.DataFrame, name: str) -> pd.Series:
    numerator, denominator = RATES[name]
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


def split_half(games: pd.DataFrame, min_games: int = 8) -> pd.DataFrame:
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
    for name in RATES:
        values = rate(halves.reset_index(), name)
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
    games: pd.DataFrame, week: int, targets=("epa_db", "success_db")
) -> pd.DataFrame:
    """Each measure through ``week`` against the rest-of-season targets, averaged."""
    before = _sums_through(games, week)
    after = _sums_through(games, week, after=True)
    after = after[after["games"].ge(3)]
    merged = before.merge(after, on=["season", "athlete_id"], suffixes=("", "_rest"))
    merged = merged[merged["dropbacks"].ge(100)]
    results = []
    for name in RATES:
        value = rate(merged, name)
        row = {"measure": name, "week": week, "players": int(value.notna().sum())}
        for target in targets:
            numerator, denominator = RATES[target]
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
    games: pd.DataFrame, targets=("epa_db", "success_db")
) -> pd.DataFrame:
    """Each measure over a full season against the passer's next season."""
    totals = _sums_through(games, None)
    totals = totals[totals["dropbacks"].ge(200)]
    nxt = totals.assign(season=totals["season"] - 1)
    merged = totals.merge(nxt, on=["season", "athlete_id"], suffixes=("", "_next"))
    results = []
    for name in RATES:
        value = rate(merged, name)
        row = {"measure": name, "players": int(value.notna().sum())}
        for target in targets:
            numerator, denominator = RATES[target]
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
}
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


def checkpoint_rows(games: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterback, season and checkpoint week: rates to date, prior-season
    rates, and the rest-of-season target."""
    totals = _sums_through(games, None)
    prior = totals[totals["dropbacks"].ge(100)].copy()
    for name in PRIOR_FEATURES:
        prior[f"prior_{name}"] = rate(prior, name)
    prior["prior_games"] = prior["games"].astype(float)
    prior = prior[
        ["season", "athlete_id", *[f"prior_{f}" for f in PRIOR_FEATURES], "prior_games"]
    ]
    prior["season"] = prior["season"] + 1
    parts = []
    for week in CHECKPOINTS:
        before = _sums_through(games, week)
        before = before[before["dropbacks"].ge(MIN_DROPBACKS)]
        after = _sums_through(games, week, after=True)
        after = after[after["dropbacks"].ge(MIN_REST_DROPBACKS)]
        merged = before.merge(
            after, on=["season", "athlete_id"], suffixes=("", "_rest")
        )
        row = merged[["season", "athlete_id"]].assign(week=week)
        for name in FEATURES:
            row[name] = rate(merged, name).to_numpy()
        row["target_epa"] = (merged["epa_rest"] / merged["dropbacks_rest"]).to_numpy()
        row["target_success"] = (
            merged["success_rest"] / merged["dropbacks_rest"]
        ).to_numpy()
        parts.append(row)
    rows = pd.concat(parts, ignore_index=True).merge(
        prior, on=["season", "athlete_id"], how="left"
    )
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    prior_columns = [f"prior_{f}" for f in PRIOR_FEATURES]
    rows[prior_columns] = rows[prior_columns].fillna(rows[prior_columns].mean())
    rows[FEATURES] = rows[FEATURES].fillna(rows[FEATURES].mean())
    return _z_within(
        rows, [*FEATURES, *prior_columns, "prior_games"], ["season", "week"]
    )


def _ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    reg = penalty * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + x @ beta[1:]


def combination_study(rows: pd.DataFrame, target: str = "target_epa") -> pd.DataFrame:
    """Leave-one-season-out: each model's Spearman and R squared on the held-out
    season's rest-of-season target, with the penalty chosen by an inner
    leave-one-season-out on the training seasons."""
    seasons = sorted(rows["season"].unique())
    results = []
    for name, columns in MODELS.items():
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
    out = plays[["game_id", "model_week", "qb", "defteam", "epa"]].copy()
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
