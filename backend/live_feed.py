"""Live game states from ESPN's public scoreboard, one call for the slate."""

from dataclasses import dataclass

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
# ESPN abbreviations that differ from the nflverse codes projections use.
TEAM_CODES = {"WSH": "WAS", "LAR": "LA"}
CALLED_OFF = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FORFEIT"}
STATES = {"pre": "scheduled", "in": "live", "post": "final"}


@dataclass(frozen=True, slots=True)
class LiveState:
    """What a feed knows about one game; None means the feed did not say."""

    status: str  # scheduled, live, final, or off
    home_score: int = 0
    away_score: int = 0
    period: int | None = None
    clock_seconds: int | None = None
    possession: str | None = None  # home or away
    down: int | None = None
    distance: int | None = None
    yards_to_goal: int | None = None
    home_timeouts: int | None = None
    away_timeouts: int | None = None


def _whole(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    number = float(value)
    return int(number) if number == int(number) and number >= 0 else None


def _yards_to_goal(text, offense: str | None) -> int | None:
    """From a spot such as 'PIT 35': the side of the field and its yard line."""
    parts = str(text or "").split()
    if not parts or not parts[-1].isdigit() or not 1 <= int(parts[-1]) <= 50:
        return None
    line = int(parts[-1])
    if line == 50:
        return 50
    if len(parts) != 2 or offense is None:
        return None
    return 100 - line if TEAM_CODES.get(parts[0], parts[0]) == offense else line


def parse_event(event: dict) -> tuple[tuple[str, str], LiveState]:
    competition = event["competitions"][0]
    sides = {team["homeAway"]: team for team in competition["competitors"]}
    codes = {
        side: TEAM_CODES.get(code, code)
        for side, code in (
            (side, team["team"]["abbreviation"]) for side, team in sides.items()
        )
    }
    key = (codes["home"], codes["away"])
    status = competition["status"]
    if status["type"]["name"] in CALLED_OFF:
        return key, LiveState("off")
    state = STATES[status["type"]["state"]]
    if state == "scheduled":
        return key, LiveState(state)
    scores = {side: _whole(team.get("score")) for side, team in sides.items()}
    if None in scores.values():
        raise ValueError("Missing or invalid score")
    situation = competition.get("situation") or {}
    ids = {str(team["team"]["id"]): side for side, team in sides.items()}
    possession = ids.get(str(situation.get("possession")))
    down = _whole(situation.get("down"))
    return key, LiveState(
        status=state,
        home_score=scores["home"],
        away_score=scores["away"],
        period=_whole(status.get("period")) or None,
        clock_seconds=_whole(status.get("clock")),
        possession=possession,
        down=down if down in (1, 2, 3, 4) else None,
        distance=_whole(situation.get("distance")),
        yards_to_goal=_yards_to_goal(
            situation.get("possessionText"), codes.get(possession)
        ),
        home_timeouts=_whole(situation.get("homeTimeouts")),
        away_timeouts=_whole(situation.get("awayTimeouts")),
    )


def fetch_states() -> dict[tuple[str, str], LiveState]:
    """Every game on the current scoreboard, keyed by (home, away) code."""
    response = requests.get(SCOREBOARD_URL, timeout=15)
    response.raise_for_status()
    return dict(parse_event(event) for event in response.json()["events"])
