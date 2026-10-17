"""Running back measure study, on the quarterback study's machinery.

One row per back and regular-season game with the raw sums behind every
candidate measure: play-by-play rushing and receiving (EPA, success, yards
over the situation expectation, explosive and stuffed carries, fumbles), the
back's share of his team's carries, Next Gen tracking rushing (rush yards over
expected from the defenders' positions, defenders in the box faced) and PFR
charting (yards before and after contact, broken tackles).
"""

import logging

import numpy as np
import pandas as pd

from backend.config import RAW_DIR
from backend.etl.store import write_parquet
from backend.players.context import load_plays, load_season_context
from backend.players.expectations import expected_rush_yards, fit_expectation_tables
from backend.players.qb_study import (
    KEYS,
    STUDY_DIR,
    _spearman,
    checkpoint_prediction,
    checkpoint_rows,
    combination_study,
    next_season_prediction,
    self_persistence,
    split_half,
    style_table,
)

log = logging.getLogger(__name__)

RB_RATES = {
    "epa_touch": ("epa_total", "touches"),
    "epa_game": ("epa_total", "games_n"),
    "epa_carry": ("rush_epa", "carries"),
    "success_carry": ("rush_success", "carries"),
    "ypc": ("rush_yards", "carries"),
    "yoe_carry": ("yoe", "carries"),
    "explosive_rate": ("explosive", "carries"),
    "stuff_rate": ("stuffed", "carries"),
    "first_down_rate": ("rush_first_downs", "carries"),
    "short_yardage_conv": ("short_conv", "short_n"),
    "rec_epa_target": ("rec_epa", "targets"),
    "target_rate": ("targets", "touches"),
    "yac_rec": ("rec_yac", "receptions"),
    "fumble_rate": ("fumbles", "touches"),
    "carry_share": ("carries", "team_carries"),
    "touches_game": ("touches", "games_n"),
    "ngs_ryoe_att": ("ngs_ryoe", "ngs_att"),
    "ngs_expected_ypc": ("ngs_expected", "ngs_att"),
    "ngs_eight_box_rate": ("ngs_eight_box", "ngs_att"),
    "ngs_efficiency": ("ngs_efficiency", "ngs_n"),
    "pfr_yac_carry": ("pfr_yac", "pfr_carries"),
    "pfr_ybc_carry": ("pfr_ybc", "pfr_carries"),
    "pfr_broken_tackle_rate": ("pfr_broken", "pfr_touches"),
}
EXPLOSIVE_YARDS = 10
VOLUME = ("touches", 40, 60)
SEASON_VOLUME = ("touches", 120)


def _ngs_rushing(season: int) -> pd.DataFrame:
    path = RAW_DIR / "ngs" / f"rushing_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    ngs = pd.read_parquet(path)
    ngs = ngs[ngs["season_type"].eq("REG") & ngs["week"].gt(0)]
    att = ngs["rush_attempts"].astype(float)
    return pd.DataFrame(
        {
            "week": ngs["week"].astype(int),
            "athlete_id": ngs["player_gsis_id"].astype(str),
            "ngs_att": att,
            "ngs_ryoe": ngs["rush_yards_over_expected"].astype(float),
            "ngs_expected": ngs["expected_rush_yards"].astype(float),
            "ngs_eight_box": ngs["percent_attempts_gte_eight_defenders"].astype(float)
            / 100.0
            * att,
            "ngs_efficiency": ngs["efficiency"].astype(float),
            "ngs_n": 1.0,
        }
    ).drop_duplicates(["week", "athlete_id"])


def _pfr_rush(season: int, players: pd.DataFrame) -> pd.DataFrame:
    path = RAW_DIR / "pfr_rush" / f"{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    pfr = pd.read_parquet(path)
    pfr = pfr[pfr["game_type"].eq("REG")]
    ids = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    pfr = pfr.assign(
        athlete_id=pfr["pfr_player_id"].map(dict(zip(ids["pfr_id"], ids["gsis_id"])))
    ).dropna(subset=["athlete_id"])
    out = pd.DataFrame(
        {
            "game_id": pfr["game_id"],
            "athlete_id": pfr["athlete_id"].astype(str),
            "pfr_carries": pfr["carries"].astype(float),
            "pfr_yac": pfr["rushing_yards_after_contact"].astype(float),
            "pfr_ybc": pfr["rushing_yards_before_contact"].astype(float),
            "pfr_broken": pfr["rushing_broken_tackles"].astype(float)
            + pfr["receiving_broken_tackles"].astype(float).fillna(0.0),
        }
    )
    return out.groupby(["game_id", "athlete_id"], as_index=False).sum()


def rb_game_table(season: int, tables) -> pd.DataFrame:
    ctx = load_season_context(season)
    plays = ctx.plays[ctx.plays["is_competitive"]]
    backs = set(
        ctx.positions.loc[ctx.positions["position_group"].eq("RB"), "athlete_id"]
    )
    rushes = plays[plays["play_type"].eq("run") & plays["rusher_player_id"].isin(backs)]
    rushes = rushes[rushes["yards_gained"].notna()]
    targets = plays[
        plays["play_type"].eq("pass") & plays["receiver_player_id"].isin(backs)
    ]
    epa_r = rushes["epa"].astype(float)
    yards = rushes["yards_gained"].astype(float)
    short = rushes["ydstogo"].le(2) & rushes["down"].isin([3, 4])
    rush = pd.DataFrame(
        {
            "game_id": rushes["game_id"],
            "week": rushes["model_week"],
            "athlete_id": rushes["rusher_player_id"].astype(str),
            "team": rushes["posteam"],
            "opponent": rushes["defteam"],
            "carries": 1.0,
            "rush_epa": epa_r,
            "rush_success": rushes["is_success"].astype(float),
            "rush_yards": yards,
            "yoe": yards - expected_rush_yards(tables, rushes),
            "explosive": yards.ge(EXPLOSIVE_YARDS).astype(float),
            "stuffed": yards.le(0).astype(float),
            "rush_first_downs": rushes["success"].eq(1).astype(float) * 0.0
            + (
                rushes["yards_gained"].astype(float) >= rushes["ydstogo"].astype(float)
            ).astype(float),
            "short_n": short.astype(float),
            "short_conv": (short & (yards >= rushes["ydstogo"].astype(float))).astype(
                float
            ),
            "fumbles": (
                rushes["fumble"].eq(1)
                & rushes["fumbled_1_player_id"]
                .astype(str)
                .eq(rushes["rusher_player_id"].astype(str))
            ).astype(float),
            "targets": 0.0,
            "receptions": 0.0,
            "rec_epa": 0.0,
            "rec_yac": 0.0,
        }
    )
    rec = pd.DataFrame(
        {
            "game_id": targets["game_id"],
            "week": targets["model_week"],
            "athlete_id": targets["receiver_player_id"].astype(str),
            "team": targets["posteam"],
            "opponent": targets["defteam"],
            "carries": 0.0,
            "rush_epa": 0.0,
            "rush_success": 0.0,
            "rush_yards": 0.0,
            "yoe": 0.0,
            "explosive": 0.0,
            "stuffed": 0.0,
            "rush_first_downs": 0.0,
            "short_n": 0.0,
            "short_conv": 0.0,
            "fumbles": (
                targets["fumble"].eq(1)
                & targets["fumbled_1_player_id"]
                .astype(str)
                .eq(targets["receiver_player_id"].astype(str))
            ).astype(float),
            "targets": 1.0,
            "receptions": targets["complete_pass"].eq(1).astype(float),
            "rec_epa": targets["epa"].astype(float) * 0.5,
            "rec_yac": targets["yards_after_catch"].astype(float).fillna(0.0),
        }
    )
    frame = pd.concat([rush, rec], ignore_index=True)
    keys = ["game_id", "week", "athlete_id", "team", "opponent"]
    out = frame.groupby(keys, as_index=False).sum()
    out["touches"] = out["carries"] + out["targets"]
    out["epa_total"] = out["rush_epa"] + out["rec_epa"]
    out["games_n"] = 1.0
    team_carries = (
        rush.groupby(["game_id", "team"])["carries"].sum().rename("team_carries")
    )
    out = out.merge(team_carries.reset_index(), on=["game_id", "team"], how="left")
    out["season"] = season
    players = pd.read_parquet(
        RAW_DIR / "players.parquet", columns=["gsis_id", "pfr_id"]
    )
    pfr = _pfr_rush(season, players)
    if not pfr.empty:
        out = out.merge(pfr, on=["game_id", "athlete_id"], how="left")
        out["pfr_touches"] = out["touches"].where(out["pfr_broken"].notna())
    ngs = _ngs_rushing(season)
    if not ngs.empty:
        out = out.merge(ngs, on=["week", "athlete_id"], how="left")
    out["athlete_name"] = out["athlete_id"].map(
        ctx.identity.set_index("athlete_id")["athlete_name"]
    )
    rest = [c for c in out.columns if c not in KEYS and c != "athlete_name"]
    return out[KEYS + ["athlete_name"] + rest]


def build_rb_games(seasons: list[int]) -> pd.DataFrame:
    plays = {s: load_plays(s) for s in range(min(seasons) - 1, max(seasons) + 1)}
    parts = []
    for season in seasons:
        parts.append(rb_game_table(season, fit_expectation_tables(plays, season)))
        log.info("rb study %s: %d back games", season, len(parts[-1]))
    out = pd.concat(parts, ignore_index=True)
    write_parquet(out, STUDY_DIR / "rb_games.parquet")
    return out


def run_rb_study(seasons: list[int], rebuild: bool = False) -> dict[str, pd.DataFrame]:
    path = STUDY_DIR / "rb_games.parquet"
    games = (
        pd.read_parquet(path)
        if path.exists() and not rebuild
        else build_rb_games(seasons)
    )
    games = games[games["season"].isin(seasons)]
    targets = ("epa_touch", "success_carry", "epa_game")
    out = {
        "split_half": split_half(games, rates=RB_RATES),
        "week6": checkpoint_prediction(games, 6, targets, RB_RATES, VOLUME),
        "week9": checkpoint_prediction(games, 9, targets, RB_RATES, VOLUME),
        "next_season": next_season_prediction(games, targets, RB_RATES, SEASON_VOLUME),
        "persistence": self_persistence(games, RB_RATES, SEASON_VOLUME),
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
    out["style"] = style_table(summary, out["persistence"])
    write_parquet(summary, STUDY_DIR / "rb_summary.parquet")
    return out


__all__ = ["RB_RATES", "run_rb_study", "_spearman", "np"]


RB_FEATURES = [
    "epa_touch",
    "epa_carry",
    "success_carry",
    "first_down_rate",
    "yoe_carry",
    "stuff_rate",
    "explosive_rate",
    "rec_epa_target",
    "yac_rec",
    "target_rate",
    "carry_share",
    "touches_game",
    "fumble_rate",
]
RB_PRIOR = ["epa_touch", "success_carry", "first_down_rate", "carry_share", "yoe_carry"]
RB_MODELS = {
    "epa_only": ["epa_touch"],
    "epa_prior": ["epa_touch", "prior_epa_touch", "prior_games"],
    "core": ["epa_touch", "success_carry", "first_down_rate", "stuff_rate", "yac_rec"],
    "core_usage": [
        "epa_touch",
        "success_carry",
        "first_down_rate",
        "stuff_rate",
        "yac_rec",
        "carry_share",
        "target_rate",
    ],
    "core_usage_prior": [
        "epa_touch",
        "success_carry",
        "first_down_rate",
        "stuff_rate",
        "yac_rec",
        "carry_share",
        "target_rate",
        *[f"prior_{f}" for f in RB_PRIOR],
        "prior_games",
    ],
    "all_prior_bio": [
        *RB_FEATURES,
        *[f"prior_{f}" for f in RB_PRIOR],
        "prior_games",
        "age",
        "experience",
        "draft_pick",
    ],
}


def rb_combinations(games: pd.DataFrame) -> pd.DataFrame:
    rows = checkpoint_rows(
        games,
        RB_RATES,
        RB_FEATURES,
        RB_PRIOR,
        ("touches", 40, 60),
        120,
        {
            "target_epa_touch": "epa_touch",
            "target_success": "success_carry",
            "target_epa_game": "epa_game",
        },
        ("epa_total", "touches"),
    )
    parts = [
        combination_study(rows, target, RB_MODELS)
        for target in ("target_epa_touch", "target_success", "target_epa_game")
    ]
    return pd.concat(parts, ignore_index=True)
