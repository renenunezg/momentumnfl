"""Live-season forecast grading: frozen pregame forecasts against finals.

Every graded row pairs the last forecast published before kickoff (an
immutable ``forecast_snapshots`` row) with the final score and the nflverse
closing line recorded in ``game_results``. Nothing is re-fit or re-priced
once a result is known; the closing line is the benchmark, never an input.
The aggregates feed the site's performance page directly, so the database
only ever serves these small precomputed rows.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from backend.model.distributions import student_t_scale
from backend.recommendations import PRICING_MARGIN_SD, marginal_cdf

PROBABILITY_METHOD = (
    "P(home margin > 0) from the frozen pregame margin marginal: "
    "t.cdf(pure_home_margin / (margin_sd * sqrt((df - 2) / df)), df)"
)
# Each model source is graded against the same final margin; the closing
# market is graded as a source so its own error sits next to the model's.
PREDICTION_SOURCES = ("pure_model", "market_informed", "closing_market")
COVERAGE_LEVELS = (0.5, 0.8, 0.9)
THIN_SAMPLE_GAMES = 30

GRADED_GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "forecast_week",
    "start_date",
    "neutral_site",
    "div_game",
    "home_team_abbr",
    "away_team_abbr",
    "home_team",
    "away_team",
    "model_version",
    "forecast_as_of",
    "forecast_recorded_at",
    "home_qb_adjustment",
    "away_qb_adjustment",
    "rest_adjustment",
    "pure_home_margin",
    "market_informed_home_margin",
    "market_weight",
    "forecast_market_home_spread",
    "model_total",
    "market_informed_total",
    "margin_sd",
    "total_sd",
    "distribution",
    "degrees_of_freedom",
    "home_win_probability",
    "probability_method",
    "closing_spread",
    "closing_total",
    "closing_source",
    "home_points",
    "away_points",
    "actual_margin",
    "actual_total",
    "score_source",
    "source_fetched_at",
    "graded_at",
]

PERFORMANCE_METRIC_COLUMNS = [
    "season",
    "prediction_source",
    "segment_kind",
    "segment",
    "segment_order",
    "games",
    "thin_sample",
    "margin_mae",
    "margin_rmse",
    "margin_bias",
    "total_games",
    "total_mae",
    "total_rmse",
    "total_bias",
    "coverage_50",
    "coverage_80",
    "coverage_90",
    "games_with_market",
    "market_mae",
    "model_minus_market_mae",
    "closer_than_market_share",
    "probability_games",
    "brier_score",
    "log_loss",
    "computed_at",
]

FAVORITE_SIZE_LABELS = ["under 3", "3 to 7", "7 to 10", "10 or more"]
QB_ADJUSTMENT_LABELS = ["none", "under 1 point", "1 point or more"]
DIVISION_LABELS = ["Divisional", "Non-divisional"]


def closing_totals(schedules: pd.DataFrame, season: int) -> pd.Series:
    """game_id -> nflverse closing total for the season's games."""
    games = schedules.loc[schedules["season"].eq(season)]
    totals = pd.to_numeric(games["total_line"], errors="coerce")
    if np.isinf(totals.to_numpy(dtype=float)).any():
        raise ValueError("Invalid closing total")
    return pd.Series(totals.to_numpy(), index=games["game_id"].astype(str))


def build_graded_games(
    frozen: pd.DataFrame,
    totals: pd.Series,
    graded_at: datetime | None = None,
) -> pd.DataFrame:
    """One graded row per completed game with a pregame forecast.

    ``frozen`` is the joined record of ``game_results`` and the last eligible
    ``forecast_snapshots`` row per game, as served by
    ``publish.fetch_frozen_forecasts``; ``totals`` maps game_id to the
    nflverse closing total. Rows are rebuilt from those frozen inputs on
    every run, so a corrected final flows through without editing history.
    """
    graded_at = graded_at or datetime.now(UTC)
    if frozen.empty:
        return pd.DataFrame(columns=GRADED_GAME_COLUMNS)
    out = frozen.copy()
    out["game_id"] = out["game_id"].astype(str)
    for column in ("start_date", "forecast_as_of", "forecast_recorded_at"):
        out[column] = pd.to_datetime(out[column], utc=True)
    if not (
        out["forecast_as_of"].lt(out["start_date"])
        & out["forecast_recorded_at"].lt(out["start_date"])
    ).all():
        raise ValueError("Only forecasts published before kickoff can be graded")
    required = ["pure_home_margin", "market_informed_home_margin", "margin_sd"]
    required += ["degrees_of_freedom", "home_points", "away_points"]
    if out[required].isna().any().any():
        raise ValueError("Frozen forecast is missing a graded field")
    out["closing_total"] = out["game_id"].map(totals)
    out["closing_source"] = np.where(
        out["closing_spread"].notna(), out["score_source"], None
    )
    out["actual_margin"] = (out["home_points"] - out["away_points"]).astype(int)
    out["actual_total"] = (out["home_points"] + out["away_points"]).astype(int)
    out["home_win_probability"] = _home_win_probability(out)
    out["probability_method"] = PROBABILITY_METHOD
    out["graded_at"] = graded_at
    return (
        out.reindex(columns=GRADED_GAME_COLUMNS)
        .sort_values(["start_date", "game_id"], kind="stable")
        .reset_index(drop=True)
    )


def _home_win_probability(frame: pd.DataFrame) -> pd.Series:
    df = _degrees_of_freedom(frame)
    return pd.Series(
        marginal_cdf(frame["pure_home_margin"].to_numpy(), frame["margin_sd"], df),
        index=frame.index,
    )


def _degrees_of_freedom(frame: pd.DataFrame) -> float:
    values = frame["degrees_of_freedom"].unique()
    if len(values) != 1:
        raise ValueError("Graded games must share one degrees-of-freedom value")
    return float(values[0])


def _favorite_size(margin: pd.Series) -> pd.Series:
    bins = [-np.inf, 3, 7, 10, np.inf]
    return pd.cut(
        margin.abs(), bins=bins, labels=FAVORITE_SIZE_LABELS, right=False
    ).astype(object)


def _qb_adjustment_size(frame: pd.DataFrame) -> pd.Series:
    size = np.maximum(
        frame["home_qb_adjustment"].fillna(0).abs(),
        frame["away_qb_adjustment"].fillna(0).abs(),
    )
    return pd.Series(
        np.select(
            [size.eq(0), size.lt(1)],
            QB_ADJUSTMENT_LABELS[:2],
            default=QB_ADJUSTMENT_LABELS[2],
        ),
        index=frame.index,
    )


def _segments(graded: pd.DataFrame) -> list[tuple[str, str, int, pd.Series]]:
    """(kind, label, order, mask) for every segment with any games."""
    segments: list[tuple[str, str, int, pd.Series]] = [
        ("overall", "All graded games", 0, pd.Series(True, index=graded.index))
    ]
    keyed = {
        "week": graded["week"].astype(int).astype(str),
        "model_favorite_size": _favorite_size(graded["pure_home_margin"]),
        "qb_adjustment": _qb_adjustment_size(graded),
        "division_game": pd.Series(
            np.where(graded["div_game"].fillna(False), *DIVISION_LABELS),
            index=graded.index,
        ),
    }
    orders = {
        "model_favorite_size": FAVORITE_SIZE_LABELS,
        "qb_adjustment": QB_ADJUSTMENT_LABELS,
        "division_game": DIVISION_LABELS,
    }
    for kind, labels in keyed.items():
        present = list(labels.dropna().unique())
        if kind == "week":
            ordered = sorted(present, key=int)
        else:
            ordered = [label for label in orders[kind] if label in present]
        for order, label in enumerate(ordered, start=1):
            segments.append((kind, str(label), order, labels.eq(label)))
    return segments


def _source_columns(graded: pd.DataFrame, source: str) -> tuple[pd.Series, pd.Series]:
    if source == "pure_model":
        return graded["pure_home_margin"], graded["model_total"]
    if source == "market_informed":
        # Games published before the blended total existed have none to grade.
        return graded["market_informed_home_margin"], graded["market_informed_total"]
    return -graded["closing_spread"], graded["closing_total"]


def _margin_sd(frame: pd.DataFrame, source: str) -> pd.Series:
    """The pure model's frozen predictive sd, or for the published line the
    dispersion its picks are priced with."""
    if source == "pure_model":
        return frame["margin_sd"]
    return pd.Series(PRICING_MARGIN_SD, index=frame.index)


def _coverage(error: pd.Series, sd: pd.Series, df: float, level: float) -> float:
    width = student_t.ppf((1 + level) / 2, df) * student_t_scale(sd, df)
    return float((error.abs() <= width).mean())


def _error_stats(prediction: pd.Series, actual: pd.Series) -> dict:
    error = (prediction - actual).dropna()
    if error.empty:
        return {"mae": None, "rmse": None, "bias": None, "n": 0}
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "n": int(len(error)),
    }


def compute_performance_metrics(
    graded: pd.DataFrame, computed_at: datetime | None = None
) -> pd.DataFrame:
    """Aggregate graded games per prediction source and segment."""
    computed_at = computed_at or datetime.now(UTC)
    if graded.empty:
        return pd.DataFrame(columns=PERFORMANCE_METRIC_COLUMNS)
    seasons = graded["season"].unique()
    if len(seasons) != 1:
        raise ValueError("Performance metrics are computed one season at a time")
    season = int(seasons[0])
    df = _degrees_of_freedom(graded)
    rows = []
    for source in PREDICTION_SOURCES:
        margin_prediction, total_prediction = _source_columns(graded, source)
        for kind, label, order, mask in _segments(graded):
            frame = graded[mask & margin_prediction.notna()]
            if frame.empty:
                continue
            prediction = margin_prediction[frame.index]
            margin = _error_stats(prediction, frame["actual_margin"])
            total = _error_stats(total_prediction[frame.index], frame["actual_total"])
            row: dict[str, object] = {
                "season": season,
                "prediction_source": source,
                "segment_kind": kind,
                "segment": label,
                "segment_order": order,
                "games": margin["n"],
                "thin_sample": margin["n"] < THIN_SAMPLE_GAMES,
                "margin_mae": margin["mae"],
                "margin_rmse": margin["rmse"],
                "margin_bias": margin["bias"],
                "total_games": total["n"],
                "total_mae": total["mae"],
                "total_rmse": total["rmse"],
                "total_bias": total["bias"],
                "computed_at": computed_at,
            }
            if source != "closing_market":
                error = prediction - frame["actual_margin"]
                sd = _margin_sd(frame, source)
                for level in COVERAGE_LEVELS:
                    row[f"coverage_{int(level * 100)}"] = _coverage(
                        error, sd, df, level
                    )
                probability = (
                    frame["home_win_probability"]
                    if source == "pure_model"
                    else pd.Series(
                        marginal_cdf(prediction.to_numpy(), sd, df), index=frame.index
                    )
                )
                won = (frame["actual_margin"] > 0).astype(float)
                clipped = probability.clip(1e-6, 1 - 1e-6)
                row["probability_games"] = int(len(probability))
                row["brier_score"] = float(((probability - won) ** 2).mean())
                row["log_loss"] = float(
                    -(won * np.log(clipped) + (1 - won) * np.log(1 - clipped)).mean()
                )
                benchmarked = frame[frame["closing_spread"].notna()]
                row["games_with_market"] = int(len(benchmarked))
                if not benchmarked.empty:
                    model_error = error[benchmarked.index].abs()
                    market_error = (
                        -benchmarked["closing_spread"] - benchmarked["actual_margin"]
                    ).abs()
                    row["market_mae"] = float(market_error.mean())
                    row["model_minus_market_mae"] = float(
                        model_error.mean() - market_error.mean()
                    )
                    row["closer_than_market_share"] = float(
                        (model_error < market_error).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=PERFORMANCE_METRIC_COLUMNS)
