"""Paper-tracked six-point teaser legs on small underdogs.

A prospective record, never staked and never written to nfl.recommendations:
every game whose underdog is +1.5 to +2.5 at a decision-time odds pull is
logged once, at its first qualifying observation, and graded at six extra
points. The line is the median dog line across the configured books, every
book counted alike, from quotes the pick pulls already fetched; tracking makes
no odds request of its own. The success criterion (a two-team
six-point teaser at -120 breaks even at 73.86% per leg) and the sample size
(200 graded legs) were fixed before the first leg was logged.
"""

import numpy as np
import pandas as pd

TEASER_POINTS = 6.0
DOG_POINTS = (1.5, 2.0, 2.5)
LEG_COLUMNS = [
    "game_id",
    "season",
    "week",
    "start_date",
    "decision_at",
    "side",
    "team",
    "books",
    "point",
    "teased_point",
]
GRADE_COLUMNS = ["game_id", "outcome", "home_points", "away_points", "graded_at"]


def _timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _dog_line(quotes):
    """The lower-middle home line across books, so the point is one a book
    quoted, and the number of books; None without a quote."""
    lines = (
        quotes.assign(
            home_point=np.where(
                quotes["selection"].eq("home"), quotes["point"], -quotes["point"]
            )
        )
        .drop_duplicates("provider_key")["home_point"]
        .sort_values()
        .to_numpy()
    )
    if not len(lines):
        return None
    return float(lines[(len(lines) - 1) // 2]), len(lines)


def qualifying_legs(projections, offers):
    """The legs one decision-time pull qualifies, before kickoff only."""
    if offers.empty:
        return pd.DataFrame(columns=LEG_COLUMNS)
    spreads = offers[offers["market"].eq("spreads")].dropna(subset=["point", "price"])
    decision_at = _timestamp(offers["market_fetched_at"]).max()
    rows = []
    for game in projections.itertuples():
        start = _timestamp(game.start_date)
        if pd.isna(start) or not decision_at < start:
            continue
        quote = _dog_line(spreads[spreads["game_id"].eq(game.game_id)])
        if quote is None or abs(quote[0]) not in DOG_POINTS:
            continue
        line, books = quote
        side, point = ("home" if line > 0 else "away"), abs(line)
        rows.append(
            dict(
                game_id=game.game_id,
                season=game.season,
                week=game.week,
                start_date=start,
                decision_at=decision_at,
                side=side,
                team=getattr(game, f"{side}_team"),
                books=books,
                point=point,
                teased_point=point + TEASER_POINTS,
            )
        )
    return pd.DataFrame(rows, columns=LEG_COLUMNS)


def grade_legs(legs, games, *, graded_at):
    """Settle pending legs from the recommendation schedule; a cancelled game
    or a moved kickoff voids the leg."""
    schedule = games.set_index("game_id")
    rows = []
    for leg in legs.itertuples():
        if leg.game_id not in schedule.index:
            continue
        game = schedule.loc[leg.game_id]
        state = game.get("game_status", "scheduled")
        if state in ("canceled", "cancelled", "postponed") or _timestamp(
            game["start_date"]
        ) != _timestamp(leg.start_date):
            outcome = "void"
        elif (
            pd.isna(game["completed"])
            or not bool(game["completed"])
            or pd.isna(game["home_points"])
            or pd.isna(game["away_points"])
        ):
            continue
        else:
            margin = game["home_points"] - game["away_points"]
            balance = (margin if leg.side == "home" else -margin) + leg.teased_point
            outcome = "win" if balance > 0 else "loss" if balance < 0 else "push"
        rows.append(
            dict(
                game_id=leg.game_id,
                outcome=outcome,
                home_points=game["home_points"],
                away_points=game["away_points"],
                graded_at=graded_at,
            )
        )
    return pd.DataFrame(rows, columns=GRADE_COLUMNS)
