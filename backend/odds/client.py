import os
from dataclasses import dataclass
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
REQUEST_TIMEOUT_SECONDS = 60
RETRY_POLICY = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
)


class OddsAPIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OddsSnapshot:
    events: list[dict]
    fetched_at: datetime
    requests_remaining: int | None
    configured_bookmakers: tuple[str, ...]


def _header_int(response: requests.Response, name: str) -> int | None:
    value = response.headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        regions: str | None = None,
        bookmakers: str | None = None,
    ):
        key = api_key or os.getenv("ODDS_API_KEY")
        if not key:
            raise OddsAPIError("ODDS_API_KEY is not set")
        self.api_key = key
        self.regions = regions or os.getenv("ODDS_API_REGIONS", "us")
        configured = bookmakers or os.getenv("ODDS_API_BOOKMAKERS", "")
        self.bookmakers = tuple(
            value.strip() for value in configured.split(",") if value.strip()
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=RETRY_POLICY))

    def get_nfl_odds(
        self, commence_from: datetime, commence_to: datetime
    ) -> OddsSnapshot:
        if commence_from.tzinfo is None or commence_to.tzinfo is None:
            raise ValueError("odds query timestamps must be timezone-aware")
        params = {
            "apiKey": self.api_key,
            "markets": "spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
            # The API rejects "+00:00" offsets; it requires the trailing Z form.
            "commenceTimeFrom": commence_from.astimezone(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "commenceTimeTo": commence_to.astimezone(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "includeLinks": "true",
        }
        if self.bookmakers:
            params["bookmakers"] = ",".join(self.bookmakers)
        else:
            params["regions"] = self.regions
        # The API only accepts the key as a query parameter, so transport
        # errors are re-raised without the request URL to keep it out of logs.
        try:
            response = self.session.get(
                f"{ODDS_API_BASE_URL}/sports/{NFL_SPORT_KEY}/odds",
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            raise OddsAPIError(
                f"The Odds API request failed: {type(error).__name__}"
            ) from None
        if response.status_code != 200:
            raise OddsAPIError(
                f"The Odds API returned {response.status_code}: {response.text[:300]}"
            )
        try:
            events = response.json()
        except ValueError:
            raise OddsAPIError("The Odds API returned a non-JSON body") from None
        if not isinstance(events, list):
            raise OddsAPIError("The Odds API returned a non-list payload")
        return OddsSnapshot(
            events=events,
            fetched_at=datetime.now(UTC),
            requests_remaining=_header_int(response, "x-requests-remaining"),
            configured_bookmakers=self.bookmakers,
        )
