"""Hold each game's closing line in memory and write it once at kickoff.

Two pregame feeds are watched. The Odds API is asked once per kickoff cluster
(games starting in the same ten-minute bucket) about three minutes before the
earliest kickoff, inside a credit reserve shared with the other sports. The
DraftKings quote on ESPN's scoreboard, which the live worker already polls,
is refreshed on every poll inside the last half hour before kickoff. At the
first poll where a game is live or past its scheduled start, the Odds API
offer is written when it exists and the ESPN quote otherwise; a game without
either gets no row and grades against the nflverse value instead.
"""

import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from backend.contracts import SCHEMA
from backend.odds.client import OddsAPIError
from backend.odds.closing import ClosingLine, from_odds_api_event, from_quote
from backend.odds.markets import match_event

log = logging.getLogger(__name__)

# An ESPN quote older than this before kickoff is not a close.
QUOTE_WINDOW = timedelta(minutes=30)
# The Odds API is asked this close to a cluster's earliest kickoff.
ODDS_API_LEAD = timedelta(minutes=3)
CLUSTER_SECONDS = 600
RESERVE_ENV = "ODDS_API_CLOSING_RESERVE"
DEFAULT_RESERVE = 60


class ClosingCapture:
    """One game's close is whatever was held when it started: never a later
    quote, never a second row."""

    def __init__(self, *, fetch_odds=None, write=None, reserve: int | None = None):
        self.fetch_odds = fetch_odds
        self.write = write
        self.reserve = (
            int(os.getenv(RESERVE_ENV, DEFAULT_RESERVE)) if reserve is None else reserve
        )
        self.remaining: int | None = None
        self.offers: dict[str, ClosingLine] = {}
        self.quotes: dict[str, ClosingLine] = {}
        self.requested: set[int] = set()
        self.closed: set[str] = set()

    def observe(self, games, states: dict, now: datetime) -> list[ClosingLine]:
        """One poll: refresh what is held for unstarted games and close the
        games that started. Returns the rows written (or that would be)."""
        self._request_offers(games, now)
        lines = []
        for game in games:
            if game.game_id in self.closed:
                continue
            state = states.get((game.home_abbr, game.away_abbr))
            status = None if state is None else state.status
            if now < game.start and status in (None, "scheduled"):
                quote = None if state is None else state.quote
                if quote is not None and game.start - now <= QUOTE_WINDOW:
                    self.quotes[game.game_id] = from_quote(
                        game.game_id, game.season, game.start, quote, now
                    )
                continue
            self.closed.add(game.game_id)
            offer = self.offers.pop(game.game_id, None)
            quote = self.quotes.pop(game.game_id, None)
            if status == "off":
                continue
            line = offer or quote
            if line is not None:
                lines.append(line)
        if lines:
            written = self.write(lines) if self.write else 0
            for line in lines:
                log.info(
                    "%s closing line %s: home %s, total %s%s",
                    line.game_id,
                    line.source,
                    line.home_spread,
                    line.total,
                    " (published)" if self.write else " (read only)",
                )
            if self.write and written < len(lines):
                log.warning(
                    "%d of %d closing lines were already present or refused",
                    len(lines) - written,
                    len(lines),
                )
        return lines

    def _request_offers(self, games, now: datetime) -> None:
        if self.fetch_odds is None:
            return
        clusters: dict[int, list] = {}
        for game in games:
            if game.game_id not in self.closed and now < game.start:
                bucket = int(game.start.timestamp()) // CLUSTER_SECONDS
                clusters.setdefault(bucket, []).append(game)
        for bucket, members in sorted(clusters.items()):
            earliest = min(game.start for game in members)
            if bucket in self.requested or earliest - now > ODDS_API_LEAD:
                continue
            # One attempt per cluster, whatever happens to it.
            self.requested.add(bucket)
            if self.remaining is not None and self.remaining < self.reserve:
                log.warning(
                    "Odds API closing request skipped: %d credits left, reserve %d",
                    self.remaining,
                    self.reserve,
                )
                continue
            latest = max(game.start for game in members)
            try:
                snapshot = self.fetch_odds(
                    earliest - timedelta(minutes=1), latest + timedelta(minutes=1)
                )
            except OddsAPIError as error:
                log.warning("Odds API closing request failed: %s", error)
                continue
            self.remaining = snapshot.requests_remaining
            schedule = pd.DataFrame(
                {
                    "game_id": [game.game_id for game in members],
                    "start_date": [game.start for game in members],
                    "home_team": [game.home_team for game in members],
                    "away_team": [game.away_team for game in members],
                }
            )
            by_id = {game.game_id: game for game in members}
            matched = 0
            for event in snapshot.events:
                try:
                    game_id, _ = match_event(
                        event["commence_time"],
                        event["home_team"],
                        event["away_team"],
                        schedule,
                    )
                    game = by_id.get(game_id)
                    if game is None:
                        continue
                    line = from_odds_api_event(
                        game.game_id,
                        game.season,
                        game.start,
                        event,
                        snapshot.fetched_at,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    log.warning("Skipping an unreadable Odds API event: %r", error)
                    continue
                if line is not None:
                    self.offers[game.game_id] = line
                    matched += 1
            log.info(
                "Odds API closing request for %d games: %d priced, cost %s, "
                "%s credits remaining",
                len(members),
                matched,
                snapshot.last_request_cost,
                snapshot.requests_remaining,
            )


def write_closing_lines(lines: list[ClosingLine]) -> int:
    """Insert once per game; the table's trigger drops anything late."""
    from sqlalchemy import text

    from backend.db import engine

    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.closing_lines "
                "(game_id, season, start_date, source, observed_at, fetched_at, "
                "home_spread, spread_home_price, spread_away_price, total, "
                "over_price, under_price, home_moneyline, away_moneyline, "
                "n_offers, price_book) "
                "SELECT game_id, season, start_date, source, observed_at, fetched_at, "
                "home_spread, spread_home_price, spread_away_price, total, "
                "over_price, under_price, home_moneyline, away_moneyline, "
                "n_offers, price_book "
                "FROM jsonb_to_recordset(CAST(:lines AS jsonb)) AS incoming("
                "game_id text, season int, start_date timestamptz, source text, "
                "observed_at timestamptz, fetched_at timestamptz, "
                "home_spread float8, spread_home_price float8, "
                "spread_away_price float8, total float8, over_price float8, "
                "under_price float8, home_moneyline float8, away_moneyline float8, "
                "n_offers int, price_book text) "
                "ON CONFLICT (game_id) DO NOTHING"
            ),
            {"lines": json.dumps([line.row() for line in lines], allow_nan=False)},
        )
        return result.rowcount
