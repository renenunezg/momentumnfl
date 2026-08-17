"""Simple v1 QB layer: rolling EPA-per-dropback value vs replacement, shrunk
by sample size; the game adjustment is the expected starter's value minus the
QB value already embedded in the team's training window."""

import numpy as np
import pandas as pd

REPLACEMENT_EPA_PER_DROPBACK = -0.08
LEAGUE_DROPBACKS_PER_GAME = 34.0
SHRINK_DROPBACKS = 100.0
DEFAULT_SPAN_DROPBACKS = 200.0


def qb_values(
    qb_games: pd.DataFrame,
    game_order: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
) -> pd.DataFrame:
    """Per (game, QB) rolling value in margin points per game, using only
    games strictly before each row. Columns: passer_player_id, game_id,
    value_points (value entering that game), dropbacks_before."""
    merged = qb_games.merge(
        game_order[["game_id", "start_date"]], on="game_id"
    ).sort_values(["passer_player_id", "start_date"])
    league_epa = float(merged["epa"].sum() / merged["dropbacks"].sum())

    rows = []
    for passer, group in merged.groupby("passer_player_id"):
        ew_value = 0.0
        ew_weight = 0.0
        seen_dropbacks = 0.0
        for row in group.itertuples():
            if ew_weight > 0:
                shrink = seen_dropbacks / (seen_dropbacks + SHRINK_DROPBACKS)
                raw = ew_value / ew_weight
                value = REPLACEMENT_EPA_PER_DROPBACK + shrink * (
                    raw - REPLACEMENT_EPA_PER_DROPBACK
                )
            else:
                value = REPLACEMENT_EPA_PER_DROPBACK
            rows.append(
                {
                    "passer_player_id": passer,
                    "game_id": row.game_id,
                    "value_points": (value - league_epa)
                    * LEAGUE_DROPBACKS_PER_GAME,
                    "dropbacks_before": seen_dropbacks,
                }
            )
            decay = 0.5 ** (row.dropbacks / span_dropbacks)
            ew_value = ew_value * decay + row.epa
            ew_weight = ew_weight * decay + row.dropbacks
            seen_dropbacks += row.dropbacks

    return pd.DataFrame(rows)


def latest_qb_value(
    values: pd.DataFrame,
    qb_games: pd.DataFrame,
    game_order: pd.DataFrame,
    passer_player_id: str,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
) -> float:
    """Value in margin points per game after all played games."""
    merged = qb_games[qb_games["passer_player_id"].eq(passer_player_id)].merge(
        game_order[["game_id", "start_date"]], on="game_id"
    )
    league_epa = float(qb_games["epa"].sum() / qb_games["dropbacks"].sum())
    if merged.empty:
        return (REPLACEMENT_EPA_PER_DROPBACK - league_epa) * LEAGUE_DROPBACKS_PER_GAME
    merged = merged.sort_values("start_date")
    ew_value = 0.0
    ew_weight = 0.0
    seen = 0.0
    for row in merged.itertuples():
        decay = 0.5 ** (row.dropbacks / span_dropbacks)
        ew_value = ew_value * decay + row.epa
        ew_weight = ew_weight * decay + row.dropbacks
        seen += row.dropbacks
    shrink = seen / (seen + SHRINK_DROPBACKS)
    raw = ew_value / ew_weight
    value = REPLACEMENT_EPA_PER_DROPBACK + shrink * (
        raw - REPLACEMENT_EPA_PER_DROPBACK
    )
    return (value - league_epa) * LEAGUE_DROPBACKS_PER_GAME


def team_baseline_values(
    values: pd.DataFrame,
    qb_games: pd.DataFrame,
    training_game_ids: pd.Series,
    recency_by_game: dict[str, float],
) -> pd.Series:
    """team -> dropback- and recency-weighted starter value embedded in the
    training window (what the team rating already contains)."""
    window = qb_games[qb_games["game_id"].isin(training_game_ids)]
    window = window[window["started"]]
    window = window.merge(values, on=["game_id", "passer_player_id"], how="left")
    window["weight"] = window["dropbacks"] * window["game_id"].map(
        recency_by_game
    ).fillna(0.0)
    window["value_points"] = window["value_points"].fillna(0.0)

    def weighted(group: pd.DataFrame) -> float:
        total = group["weight"].sum()
        if total <= 0:
            return 0.0
        return float((group["value_points"] * group["weight"]).sum() / total)

    return window.groupby("team").apply(weighted, include_groups=False)


def game_qb_adjustment(starter_value: float, team_baseline: float) -> float:
    return starter_value - team_baseline
