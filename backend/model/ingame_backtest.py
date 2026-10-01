"""Retrospective scoring of in-game win probabilities against final results."""

import numpy as np
import pandas as pd

from backend.config import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS

MARGIN_BUCKET_EDGES = (-np.inf, -16.5, -8.5, -3.5, 3.5, 8.5, 16.5, np.inf)
MARGIN_BUCKET_LABELS = ("<=-17", "-16:-9", "-8:-4", "-3:3", "4:8", "9:16", "17+")
PROBABILITY_BUCKET_EDGES = np.linspace(0.0, 1.0, 11)


def build_inputs(
    states: pd.DataFrame, anchors: pd.DataFrame, margin_column: str = "model_margin"
) -> pd.DataFrame:
    """Join play states to walk-forward pregame forecasts and final results.

    Ties have no winner and are dropped from scoring.
    """
    anchor = anchors[[margin_column, "game_id", "home_points", "away_points"]].rename(
        columns={margin_column: "pregame_margin"}
    )
    merged = states.merge(anchor, on="game_id", how="inner", validate="many_to_one")
    merged = merged[merged["home_points"].ne(merged["away_points"])].copy()
    merged["home_win"] = merged["home_points"].gt(merged["away_points"])
    return merged.reset_index(drop=True)


def without_situation(inputs: pd.DataFrame) -> pd.DataFrame:
    """Inputs as a score, clock, and possession feed would supply them."""
    degraded = inputs.copy()
    situation = ["yards_to_goal", "down", "distance", "home_timeouts", "away_timeouts"]
    degraded[situation] = np.nan
    return degraded


def state_log_loss(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))


def paired_delta_interval(
    delta: np.ndarray, game_ids: pd.Series, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for a mean per-state difference, resampling whole games.

    States within a game share one outcome, so games are the sampling unit.
    """
    per_game = (
        pd.DataFrame({"delta": delta, "game_id": game_ids.to_numpy()})
        .groupby("game_id")["delta"]
        .agg(["sum", "count"])
    )
    totals, counts = per_game["sum"].to_numpy(), per_game["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(totals), size=(draws, len(totals)))
    means = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _calibration_row(frame: pd.DataFrame, partition: str, scope: str, group) -> dict:
    outcome = frame["home_win"].to_numpy(float)
    predicted = frame["win_probability"].to_numpy(float)
    n_games = int(frame["game_id"].nunique())
    rate = float(outcome.mean())
    gap = float(predicted.mean() - rate)
    # States within a game share one outcome, so the binomial tolerance uses
    # the game count, not the play count.
    tolerance = max(
        0.02, 1.96 * np.sqrt(max(rate * (1.0 - rate), 1e-4) / max(n_games, 1))
    )
    return {
        "summary_type": "calibration",
        "partition": partition,
        "scope": scope,
        "group_value": str(group),
        "n_states": len(frame),
        "n_games": n_games,
        "mean_predicted": float(predicted.mean()),
        "empirical_rate": rate,
        "gap": gap,
        "tolerance": float(tolerance),
        "calibrated": bool(abs(gap) <= tolerance),
        "brier": float(np.mean(np.square(predicted - outcome))),
        "log_loss": float(state_log_loss(outcome, predicted).mean()),
    }


def evaluate(inputs: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    """Overall scores by partition and holdout calibration by segment."""
    frame = inputs.copy()
    frame["win_probability"] = probability
    frame["phase"] = np.where(
        frame["is_overtime"], "OT", "Q" + frame["period"].astype(str)
    )
    frame["margin_bucket"] = pd.cut(
        frame["score_margin"], bins=MARGIN_BUCKET_EDGES, labels=MARGIN_BUCKET_LABELS
    ).astype(str)
    frame["probability_bucket"] = pd.cut(
        frame["win_probability"], bins=PROBABILITY_BUCKET_EDGES, include_lowest=True
    ).astype(str)
    development = frame[frame["season"].isin(DEVELOPMENT_SEASONS)]
    holdout = frame[frame["season"].isin(HOLDOUT_SEASONS)]
    rows = [
        _calibration_row(development, "development", "overall", "all"),
        _calibration_row(holdout, "holdout", "overall", "all"),
    ]
    for scope, column in (
        ("phase", "phase"),
        ("margin", "margin_bucket"),
        ("probability", "probability_bucket"),
    ):
        for value, group in holdout.groupby(column, sort=True):
            rows.append(_calibration_row(group, "holdout", scope, value))
    return pd.DataFrame(rows)


def comparison_row(
    label: str, frame: pd.DataFrame, probability: np.ndarray, reference: np.ndarray
) -> dict:
    """Score one candidate against the reference on the same holdout states."""
    outcome = frame["home_win"].to_numpy(float)
    loss = state_log_loss(outcome, probability)
    delta = loss - state_log_loss(outcome, reference)
    low, high = paired_delta_interval(delta, frame["game_id"])
    return {
        "summary_type": "comparison",
        "partition": "holdout",
        "scope": "candidate",
        "group_value": label,
        "n_states": len(frame),
        "n_games": int(frame["game_id"].nunique()),
        "log_loss": float(loss.mean()),
        "brier": float(np.mean(np.square(probability - outcome))),
        "log_loss_delta": float(delta.mean()),
        "log_loss_delta_low": low,
        "log_loss_delta_high": high,
    }
