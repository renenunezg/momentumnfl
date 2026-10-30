"""DraftKings pregame odds from ESPN's public scoreboard feed.

Fallback for when The Odds API monthly quota is exhausted. Selected with
ODDS_SOURCE=espn; unset that variable to return to The Odds API. Events are
returned in The Odds API's shape, so offer flattening and grading are unchanged.
ESPN carries one book and no quote timestamp, so each price is stamped with
the fetch time. The live worker reads the same quote off the scoreboard it
already polls (``draftkings_quote``) to hold a game's closing line.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests

from backend.odds.client import REQUEST_TIMEOUT_SECONDS, OddsAPIError, OddsSnapshot

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
# ESPN's provider id is stable while its display name is not ("DraftKings" and
# "Draft Kings" both appear). Map the id to The Odds API's key and title.
ESPN_BOOKMAKERS = {"100": ("draftkings", "DraftKings")}


def _price(quote: dict) -> int | None:
    raw = str(quote.get("odds", "")).strip().upper()
    if raw == "EVEN":
        return 100
    try:
        return int(raw)
    except ValueError:
        return None


def _point(quote: dict) -> float | None:
    # Totals arrive as "o48.5" / "u48.5", spreads as "-9.5" / "+9.5".
    try:
        return float(str(quote.get("line", "")).strip().lstrip("ou"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Quote:
    """One bookmaker's pregame prices for a game in sportsbook notation: the
    home line (negative when the home team is favored), the total, and
    American prices. A moneyline counts only when both sides are priced."""

    home_spread: float | None = None
    spread_home_price: int | None = None
    spread_away_price: int | None = None
    total: float | None = None
    over_price: int | None = None
    under_price: int | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None


EMPTY_QUOTE = Quote()


def _quote(odds: dict) -> Quote:
    def close(market: str, side: str) -> dict:
        return (odds.get(market) or {}).get(side, {}).get("close") or {}

    home_moneyline = _price(close("moneyline", "home"))
    away_moneyline = _price(close("moneyline", "away"))
    if home_moneyline is None or away_moneyline is None:
        home_moneyline = away_moneyline = None
    home_spread = _point(close("pointSpread", "home"))
    if home_spread is None:
        away_line = _point(close("pointSpread", "away"))
        home_spread = None if away_line is None else -away_line
    total = _point(close("total", "over"))
    if total is None:
        total = _point(close("total", "under"))
    return Quote(
        home_spread=home_spread,
        spread_home_price=_price(close("pointSpread", "home")),
        spread_away_price=_price(close("pointSpread", "away")),
        total=total,
        over_price=_price(close("total", "over")),
        under_price=_price(close("total", "under")),
        home_moneyline=home_moneyline,
        away_moneyline=away_moneyline,
    )


def draftkings_quote(competition: dict) -> Quote | None:
    """DraftKings' quote from one scoreboard competition, None when absent."""
    for odds in competition.get("odds") or []:
        if str((odds.get("provider") or {}).get("id")) in ESPN_BOOKMAKERS:
            quote = _quote(odds)
            return None if quote == EMPTY_QUOTE else quote
    return None


def _markets(quote: Quote, home_team: str, away_team: str, updated: str) -> list[dict]:
    sides = ((home_team, quote.home_moneyline), (away_team, quote.away_moneyline))
    away_spread = None if quote.home_spread is None else 0.0 - quote.home_spread
    outcomes = {
        "h2h": [{"name": team, "price": price} for team, price in sides],
        "spreads": [
            {
                "name": home_team,
                "price": quote.spread_home_price,
                "point": quote.home_spread,
            },
            {"name": away_team, "price": quote.spread_away_price, "point": away_spread},
        ],
        "totals": [
            {"name": "Over", "price": quote.over_price, "point": quote.total},
            {"name": "Under", "price": quote.under_price, "point": quote.total},
        ],
    }
    return [
        {"key": key, "last_update": updated, "outcomes": priced}
        for key, priced in outcomes.items()
        if all(outcome["price"] is not None for outcome in priced)
    ]


class EspnOddsClient:
    def get_nfl_odds(
        self, commence_from: datetime, commence_to: datetime
    ) -> OddsSnapshot:
        if commence_from.tzinfo is None or commence_to.tzinfo is None:
            raise ValueError("odds query timestamps must be timezone-aware")
        # ESPN serves one US date per request and rejects date ranges; widen a
        # day and filter to the exact window below.
        day = commence_from.astimezone(UTC).date() - timedelta(days=1)
        scoreboard = {}
        while day <= commence_to.astimezone(UTC).date():
            try:
                response = requests.get(
                    ESPN_SCOREBOARD_URL,
                    params={"dates": day.strftime("%Y%m%d")},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as error:
                raise OddsAPIError(
                    f"The ESPN odds request failed: {type(error).__name__}"
                ) from None
            if response.status_code != 200:
                raise OddsAPIError(f"ESPN returned {response.status_code}")
            try:
                scoreboard.update(
                    {event["id"]: event for event in response.json()["events"]}
                )
            except (ValueError, KeyError, TypeError):
                raise OddsAPIError(
                    "ESPN returned an unexpected scoreboard body"
                ) from None
            day += timedelta(days=1)

        fetched_at = datetime.now(UTC)
        updated = fetched_at.isoformat()
        events = []
        for event in scoreboard.values():
            # Started games carry live or closing lines, never a pregame offer.
            if event["status"]["type"]["state"] != "pre":
                continue
            commence = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            if not commence_from <= commence <= commence_to:
                continue
            competition = event["competitions"][0]
            teams = {
                competitor["homeAway"]: competitor["team"]["displayName"]
                for competitor in competition["competitors"]
            }
            bookmakers = []
            for odds in competition.get("odds") or []:
                provider = str((odds.get("provider") or {}).get("id"))
                if provider not in ESPN_BOOKMAKERS:
                    continue
                key, title = ESPN_BOOKMAKERS[provider]
                bookmakers.append(
                    {
                        "key": key,
                        "title": title,
                        "last_update": updated,
                        "markets": _markets(
                            _quote(odds), teams["home"], teams["away"], updated
                        ),
                    }
                )
            events.append(
                {
                    "id": f"espn-{event['id']}",
                    "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "home_team": teams["home"],
                    "away_team": teams["away"],
                    "bookmakers": bookmakers,
                }
            )
        return OddsSnapshot(
            events=events,
            fetched_at=fetched_at,
            requests_remaining=None,
            configured_bookmakers=tuple(key for key, _ in ESPN_BOOKMAKERS.values()),
            source="espn",
        )
