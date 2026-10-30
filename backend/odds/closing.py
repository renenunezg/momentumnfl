"""Closing-line rows from the two pregame feeds the live worker sees.

The Odds API row is the median spread and total across bookmakers with the
prices of one named book; the ESPN row is DraftKings' quote as the scoreboard
carried it. Both produce the one ``nfl.closing_lines`` row a game gets.
"""

from dataclasses import asdict, dataclass
from datetime import datetime

from backend.odds.espn import Quote

ODDS_API_SOURCE = "odds_api:median"
ESPN_SOURCE = "espn:draftkings"
NFLVERSE_SOURCE = "nflverse"
# The book whose prices are recorded when it quotes the median point.
PREFERRED_BOOK = "draftkings"


@dataclass(frozen=True, slots=True)
class ClosingLine:
    game_id: str
    season: int
    start_date: datetime
    source: str
    observed_at: datetime
    fetched_at: datetime
    home_spread: float | None = None
    spread_home_price: float | None = None
    spread_away_price: float | None = None
    total: float | None = None
    over_price: float | None = None
    under_price: float | None = None
    home_moneyline: float | None = None
    away_moneyline: float | None = None
    n_offers: int | None = None
    price_book: str | None = None

    def row(self) -> dict:
        row = asdict(self)
        for column in ("start_date", "observed_at", "fetched_at"):
            row[column] = row[column].isoformat()
        return row


def from_quote(
    game_id: str, season: int, start_date: datetime, quote: Quote, fetched_at: datetime
) -> ClosingLine:
    """ESPN gives no provider stamp, so the fetch time stands for observed_at."""
    return ClosingLine(
        game_id=game_id,
        season=season,
        start_date=start_date,
        source=ESPN_SOURCE,
        observed_at=fetched_at,
        fetched_at=fetched_at,
        home_spread=quote.home_spread,
        spread_home_price=quote.spread_home_price,
        spread_away_price=quote.spread_away_price,
        total=quote.total,
        over_price=quote.over_price,
        under_price=quote.under_price,
        home_moneyline=quote.home_moneyline,
        away_moneyline=quote.away_moneyline,
        n_offers=1,
        price_book=PREFERRED_BOOK,
    )


@dataclass(frozen=True, slots=True)
class _Offer:
    book: str
    updated: datetime
    point: float | None
    first: float  # home or over price
    second: float  # away or under price


def _timestamp(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _offers(event: dict, market: str, first: str, second: str) -> list[_Offer]:
    """One priced offer per bookmaker for ``market``; ``first`` and ``second``
    name the two outcomes (home and away, or Over and Under)."""
    offers = []
    for book in event.get("bookmakers") or []:
        for priced in book.get("markets") or []:
            if priced.get("key") != market:
                continue
            sides = {o.get("name"): o for o in priced.get("outcomes") or []}
            if first not in sides or second not in sides:
                continue
            prices = (sides[first].get("price"), sides[second].get("price"))
            if any(p is None for p in prices):
                continue
            point = sides[first].get("point")
            offers.append(
                _Offer(
                    book=str(book["key"]),
                    updated=_timestamp(
                        priced.get("last_update") or book["last_update"]
                    ),
                    point=None if point is None else float(point),
                    first=float(prices[0]),
                    second=float(prices[1]),
                )
            )
    return offers


def _median_offer(offers: list[_Offer]) -> _Offer | None:
    """The offer at the median point: the lower middle of the sorted points,
    so the point is always one some book actually quoted. Among the books at
    that point DraftKings is recorded when present, otherwise the book updated
    most recently."""
    priced = sorted(
        (o for o in offers if o.point is not None), key=lambda o: (o.point, o.book)
    )
    if not priced:
        return None
    point = priced[(len(priced) - 1) // 2].point
    return _pick([o for o in priced if o.point == point])


def _pick(offers: list[_Offer]) -> _Offer | None:
    if not offers:
        return None
    preferred = [o for o in offers if o.book == PREFERRED_BOOK]
    return preferred[0] if preferred else max(offers, key=lambda o: o.updated)


def from_odds_api_event(
    game_id: str, season: int, start_date: datetime, event: dict, fetched_at: datetime
) -> ClosingLine | None:
    """The median close of one Odds API event, None when no book priced it.

    ``observed_at`` is the recorded book's own ``last_update`` for the spread
    (or the total, or the moneyline when nothing else is priced); ``n_offers``
    counts the bookmakers the spread median was taken from.
    """
    home, away = event["home_team"], event["away_team"]
    spreads = _offers(event, "spreads", home, away)
    totals = _offers(event, "totals", "Over", "Under")
    spread = _median_offer(spreads)
    total = _median_offer(totals)
    moneyline = _pick(_offers(event, "h2h", home, away))
    recorded = spread or total or moneyline
    if recorded is None:
        return None
    return ClosingLine(
        game_id=game_id,
        season=season,
        start_date=start_date,
        source=ODDS_API_SOURCE,
        observed_at=recorded.updated,
        fetched_at=fetched_at,
        home_spread=None if spread is None else spread.point,
        spread_home_price=None if spread is None else spread.first,
        spread_away_price=None if spread is None else spread.second,
        total=None if total is None else total.point,
        over_price=None if total is None else total.first,
        under_price=None if total is None else total.second,
        home_moneyline=None if moneyline is None else moneyline.first,
        away_moneyline=None if moneyline is None else moneyline.second,
        n_offers=len([o for o in spreads if o.point is not None]) or len(totals),
        price_book=recorded.book,
    )
