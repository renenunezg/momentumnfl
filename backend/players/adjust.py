"""Opponent adjustment for rate traits.

An opponent's allowed rate and the league rate both use only games from
strictly earlier weeks of the season, so an event is never adjusted by
evidence that includes its own week.
"""

import numpy as np
import pandas as pd


def _with_week(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    weeks = games[["game_id", "model_week"]].drop_duplicates("game_id")
    return events.merge(weeks, on="game_id", how="left")


def _prior_rate(cumulative: pd.DataFrame, grouped_by: list[str]) -> pd.Series:
    """Cumulative numerator over denominator through the previous row."""
    columns = ["numerator", "denominator"]
    if grouped_by:
        shifted = cumulative.groupby(grouped_by)[columns].shift(1)
    else:
        shifted = cumulative[columns].shift(1)
    denominator = shifted["denominator"].where(shifted["denominator"] > 0)
    return shifted["numerator"] / denominator


def opponent_rates(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each opponent's allowed rate over games before every week it appears in."""
    weekly = (
        _with_week(events, games)
        .dropna(subset=["model_week"])
        .groupby(["opponent", "model_week"], as_index=False)[
            ["numerator", "denominator"]
        ]
        .sum()
        .sort_values(["opponent", "model_week"])
    )
    cumulative = weekly.assign(
        numerator=weekly.groupby("opponent")["numerator"].cumsum(),
        denominator=weekly.groupby("opponent")["denominator"].cumsum(),
    )
    weekly["allowed_rate"] = _prior_rate(cumulative, ["opponent"])
    return weekly[["model_week", "opponent", "allowed_rate"]].reset_index(drop=True)


def _league_rates(with_week: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        with_week.dropna(subset=["model_week"])
        .groupby("model_week", as_index=False)[["numerator", "denominator"]]
        .sum()
        .sort_values("model_week")
    )
    cumulative = weekly.assign(
        numerator=weekly["numerator"].cumsum(),
        denominator=weekly["denominator"].cumsum(),
    )
    weekly["league_rate"] = _prior_rate(cumulative, [])
    return weekly[["model_week", "league_rate"]]


def adjust_rate(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Replace ``numerator`` with its value net of the opponent's prior-week deviation.

    ``numerator - denominator * (allowed_rate - league_rate)``; the original is
    kept in ``raw_numerator``. Events with no prior evidence stay unchanged.
    """
    with_week = _with_week(events, games)
    merged = with_week.merge(
        opponent_rates(events, games), on=["model_week", "opponent"], how="left"
    ).merge(_league_rates(with_week), on="model_week", how="left")
    deviation = merged["allowed_rate"] - merged["league_rate"]
    adjusted = merged["numerator"] - merged["denominator"] * deviation
    out = events.copy()
    out["raw_numerator"] = events["numerator"].to_numpy()
    out["numerator"] = np.where(
        deviation.notna(), adjusted, events["numerator"]
    ).astype(float)
    return out
