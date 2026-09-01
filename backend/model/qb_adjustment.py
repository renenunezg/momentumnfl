"""Simple v1 QB layer: rolling EPA-per-dropback value vs replacement, shrunk
by sample size; the game adjustment is the expected starter's value minus the
QB value already embedded in the team's training window."""

import pandas as pd

REPLACEMENT_EPA_PER_DROPBACK = -0.08
LEAGUE_DROPBACKS_PER_GAME = 34.0
SHRINK_DROPBACKS = 100.0
DEFAULT_SPAN_DROPBACKS = 200.0


def _shrunk_value(ew_value: float, ew_weight: float, seen: float) -> float:
    if ew_weight <= 0:
        return REPLACEMENT_EPA_PER_DROPBACK
    shrink = seen / (seen + SHRINK_DROPBACKS)
    raw = ew_value / ew_weight
    return REPLACEMENT_EPA_PER_DROPBACK + shrink * (raw - REPLACEMENT_EPA_PER_DROPBACK)


def _value_trajectory(
    games: pd.DataFrame, span_dropbacks: float
) -> tuple[list[tuple[float, float]], float]:
    """games are one passer's rows in chronological order. Returns the
    (value, dropbacks_before) pair entering each game and the value after the
    last one, as exponentially decayed EPA per dropback shrunk to replacement."""
    entering = []
    ew_value = ew_weight = seen = 0.0
    for row in games.itertuples():
        entering.append((_shrunk_value(ew_value, ew_weight, seen), seen))
        decay = 0.5 ** (row.dropbacks / span_dropbacks)
        ew_value = ew_value * decay + row.epa
        ew_weight = ew_weight * decay + row.dropbacks
        seen += row.dropbacks
    return entering, _shrunk_value(ew_value, ew_weight, seen)


def _ordered(qb_games: pd.DataFrame, game_order: pd.DataFrame) -> pd.DataFrame:
    return qb_games.merge(
        game_order[["game_id", "start_date"]], on="game_id"
    ).sort_values(["passer_player_id", "start_date"])


def _league_epa(qb_games: pd.DataFrame) -> float:
    return float(qb_games["epa"].sum() / qb_games["dropbacks"].sum())


def qb_values(
    qb_games: pd.DataFrame,
    game_order: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
) -> pd.DataFrame:
    """Per (game, QB) rolling value in margin points per game, using only
    games strictly before each row. Columns: passer_player_id, game_id,
    value_points (value entering that game), dropbacks_before."""
    merged = _ordered(qb_games, game_order)
    league_epa = _league_epa(merged)
    rows = []
    for passer, group in merged.groupby("passer_player_id"):
        entering, _ = _value_trajectory(group, span_dropbacks)
        for row, (value, dropbacks_before) in zip(group.itertuples(), entering):
            rows.append(
                {
                    "passer_player_id": passer,
                    "game_id": row.game_id,
                    "value_points": (value - league_epa) * LEAGUE_DROPBACKS_PER_GAME,
                    "dropbacks_before": dropbacks_before,
                }
            )
    return pd.DataFrame(rows)


def latest_qb_values(
    qb_games: pd.DataFrame,
    game_order: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
) -> tuple[dict[str, float], float]:
    """passer -> value in margin points per game after all played games, plus
    the replacement-level value for a passer with no history."""
    merged = _ordered(qb_games, game_order)
    league_epa = _league_epa(qb_games)
    values = {
        passer: (_value_trajectory(group, span_dropbacks)[1] - league_epa)
        * LEAGUE_DROPBACKS_PER_GAME
        for passer, group in merged.groupby("passer_player_id")
    }
    replacement = (
        REPLACEMENT_EPA_PER_DROPBACK - league_epa
    ) * LEAGUE_DROPBACKS_PER_GAME
    return values, replacement


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
