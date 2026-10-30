"""The live worker polls only while games are open and writes each final once."""

from datetime import UTC, datetime, timedelta

import numpy as np

from backend.live_feed import LiveState, parse_event
from backend.live_publish import Game, LivePublisher
from backend.model.ingame_nflfastr import FEATURES, TreeEnsemble

KICKOFF = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
# One split on the score: the side with the ball wins when it leads.
MODEL = TreeEnsemble(
    left=np.array([[1, -1, -1]], dtype=np.int32),
    right=np.array([[2, -1, -1]], dtype=np.int32),
    feature=np.array([[FEATURES.index("score_differential"), 0, 0]], dtype=np.int32),
    value=np.array([[0.5, -2.0, 2.0]], dtype=np.float32),
    default_left=np.zeros((1, 3), dtype=bool),
    base_margin=0.0,
)


def _game(home: str, away: str) -> Game:
    return Game(
        game_id=f"2026_05_{away}_{home}",
        season=2026,
        week=5,
        start=KICKOFF,
        home_abbr=home,
        away_abbr=away,
        home_team=home,
        away_team=away,
        home_margin=2.5,
        as_of=KICKOFF - timedelta(hours=6),
        model_version="test",
    )


def test_polling_starts_at_kickoff_and_each_game_closes_once():
    games = [_game("BUF", "NE"), _game("CHI", "NYJ"), _game("CIN", "JAX")]
    board, calls, written = {}, [], []

    def fetch_states():
        calls.append(1)
        return dict(board)

    publisher = LivePublisher(
        MODEL,
        load_games=lambda start, end: games,
        load_saved=lambda game_ids: {},
        fetch_states=fetch_states,
        write=written.extend,
    )

    # Inside the lead window the scoreboard is read for closing quotes only;
    # no state is scored before a game's own kickoff.
    assert publisher.poll(KICKOFF - timedelta(minutes=10))
    assert calls == [1] and {p["abstract_state"] for p in written} == {"Pre"}

    board[("BUF", "NE")] = LiveState(
        "live", 14, 0, period=2, clock_seconds=450, possession="home", down=1
    )
    board[("CHI", "NYJ")] = LiveState("off")
    board[("CIN", "JAX")] = LiveState("scheduled")
    assert publisher.poll(KICKOFF + timedelta(hours=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["2026_05_NE_BUF"]["home_win_probability"] > 0.8
    assert latest["2026_05_NYJ_CHI"]["abstract_state"] == "Off"

    del board[("BUF", "NE")]
    assert publisher.poll(KICKOFF + timedelta(hours=3, minutes=1))
    latest = {p["game_id"]: p for p in written}
    assert latest["2026_05_NE_BUF"]["abstract_state"] == "Live"
    publisher = LivePublisher(
        MODEL,
        load_games=lambda start, end: games,
        load_saved=lambda ids: latest,
        fetch_states=fetch_states,
        write=written.extend,
    )

    # A tie is a final with no winner, and a game that never starts is closed.
    board[("BUF", "NE")] = LiveState("final", 20, 20)
    assert not publisher.poll(KICKOFF + timedelta(hours=3, minutes=2))
    latest = {p["game_id"]: p for p in written}
    assert latest["2026_05_NE_BUF"]["home_win_probability"] == 0.5
    assert [p["s"] for p in latest["2026_05_NE_BUF"]["history"]] == [0, 1350, 3600]
    assert latest["2026_05_JAX_CIN"]["abstract_state"] == "Off"

    spent, rows = len(calls), len(written)
    assert not publisher.poll(KICKOFF + timedelta(hours=3, minutes=3))
    assert (len(calls), len(written)) == (spent, rows)


def test_espn_spot_is_measured_from_the_offense():
    event = {
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "7",
                        "team": {"id": "5", "abbreviation": "CLE"},
                    },
                    {
                        "homeAway": "away",
                        "score": "3",
                        "team": {"id": "23", "abbreviation": "PIT"},
                    },
                ],
                "status": {
                    "period": 2,
                    "clock": 312.0,
                    "type": {"name": "STATUS_IN_PROGRESS", "state": "in"},
                },
                "situation": {
                    "possession": "23",
                    "down": 3,
                    "distance": 4,
                    "possessionText": "PIT 35",
                    "homeTimeouts": 3,
                    "awayTimeouts": 2,
                },
            }
        ]
    }
    key, state = parse_event(event)
    assert key == ("CLE", "PIT")
    assert (state.possession, state.yards_to_goal, state.clock_seconds) == (
        "away",
        65,
        312,
    )


def _draftkings(line: str, total: str, home_ml: str, away_ml: str) -> dict:
    away_line = line.replace("-", "+") if line.startswith("-") else "-" + line[1:]
    return {
        "provider": {"id": "100", "name": "DraftKings"},
        "pointSpread": {
            "home": {"close": {"line": line, "odds": "-110"}},
            "away": {"close": {"line": away_line, "odds": "-110"}},
        },
        "total": {
            "over": {"close": {"line": f"o{total}", "odds": "-105"}},
            "under": {"close": {"line": f"u{total}", "odds": "-115"}},
        },
        "moneyline": {
            "home": {"close": {"odds": home_ml}},
            "away": {"close": {"odds": away_ml}},
        },
    }


def _book(key: str, updated: datetime, spread=None, total=None, h2h=None) -> dict:
    home, away = "Buffalo Bills", "New England Patriots"
    markets = []
    if spread is not None:
        point, first, second = spread
        markets.append(
            {
                "key": "spreads",
                "last_update": updated.isoformat(),
                "outcomes": [
                    {"name": home, "price": first, "point": point},
                    {"name": away, "price": second, "point": -point},
                ],
            }
        )
    if total is not None:
        point, over, under = total
        markets.append(
            {
                "key": "totals",
                "last_update": updated.isoformat(),
                "outcomes": [
                    {"name": "Over", "price": over, "point": point},
                    {"name": "Under", "price": under, "point": point},
                ],
            }
        )
    if h2h is not None:
        markets.append(
            {
                "key": "h2h",
                "last_update": updated.isoformat(),
                "outcomes": [
                    {"name": home, "price": h2h[0]},
                    {"name": away, "price": h2h[1]},
                ],
            }
        )
    return {"key": key, "last_update": updated.isoformat(), "markets": markets}


def test_closing_line_is_held_pregame_and_written_once():
    from dataclasses import replace

    from backend.closing_capture import ClosingCapture
    from backend.odds.client import OddsSnapshot
    from backend.odds.espn import Quote

    minute = timedelta(minutes=1)
    bills = replace(
        _game("BUF", "NE"), home_team="Buffalo Bills", away_team="New England Patriots"
    )
    bears = _game("CHI", "NYJ")
    late = replace(_game("CIN", "JAX"), start=KICKOFF + 30 * minute)
    games = [bills, bears, late]
    event = {
        "commence_time": KICKOFF.isoformat(),
        "home_team": "Buffalo Bills",
        "away_team": "New England Patriots",
        "bookmakers": [
            _book("fanduel", KICKOFF - 4 * minute, spread=(-3.5, -110, -110)),
            _book("betmgm", KICKOFF - 3 * minute, spread=(-3.0, -110, -110)),
            _book(
                "caesars",
                KICKOFF - 2 * minute,
                spread=(-3.0, -112, -108),
                total=(45.5, -110, -110),
                h2h=(-160, 140),
            ),
            _book(
                "draftkings",
                KICKOFF - 6 * minute,
                spread=(-2.5, -115, -105),
                total=(44.5, -105, -115),
                h2h=(-170, 150),
            ),
        ],
    }
    requests, written = [], []

    def fetch_odds(commence_from, commence_to):
        requests.append((commence_from, commence_to))
        return OddsSnapshot(
            events=[event],
            fetched_at=KICKOFF - 2 * minute,
            requests_remaining=50,
            configured_bookmakers=(),
            source="odds_api",
            last_request_cost=3,
        )

    def write(lines):
        written.extend(lines)
        return len(lines)

    board: dict = {}
    publisher = LivePublisher(
        MODEL,
        load_games=lambda start, end: games,
        load_saved=lambda ids: {},
        fetch_states=lambda: dict(board),
        write=lambda payloads: None,
        closing=ClosingCapture(fetch_odds=fetch_odds, write=write, reserve=60),
    )
    quote = Quote(-3.5, -110, -110, 44.5, -105, -115, -180, 155)
    for key in (("BUF", "NE"), ("CHI", "NYJ"), ("CIN", "JAX")):
        board[key] = LiveState("scheduled", quote=quote)

    # Outside the half-hour window the scoreboard is not even fetched.
    publisher.poll(KICKOFF - 40 * minute)
    assert publisher.closing.quotes == {}
    publisher.poll(KICKOFF - 10 * minute)
    assert set(publisher.closing.quotes) == {bills.game_id, bears.game_id}
    assert requests == []
    # The latest pregame quote replaces the earlier one.
    board[("CHI", "NYJ")] = LiveState("scheduled", quote=replace(quote, home_spread=-4))
    publisher.poll(KICKOFF - 5 * minute)
    assert publisher.closing.quotes[bears.game_id].home_spread == -4
    # Three minutes out the cluster's offers are fetched in one request.
    publisher.poll(KICKOFF - 2 * minute)
    assert requests == [(KICKOFF - minute, KICKOFF + minute)]
    assert set(publisher.closing.offers) == {bills.game_id}

    board[("BUF", "NE")] = LiveState("live", 0, 0, period=1, clock_seconds=880)
    publisher.poll(KICKOFF + minute)
    rows = {line.game_id: line for line in written}
    assert set(rows) == {bills.game_id, bears.game_id}
    # Lower-middle point of (-3.5, -3, -3, -2.5) is -3; DraftKings is not
    # there, so the most recently updated book at -3 is recorded.
    odds_api = rows[bills.game_id]
    assert odds_api.source == "odds_api:median"
    assert (odds_api.home_spread, odds_api.spread_home_price) == (-3.0, -112)
    assert (odds_api.price_book, odds_api.n_offers) == ("caesars", 4)
    assert odds_api.observed_at == KICKOFF - 2 * minute
    # DraftKings wins the total and the moneyline where it quotes the median.
    assert (odds_api.total, odds_api.over_price) == (44.5, -105)
    assert (odds_api.home_moneyline, odds_api.away_moneyline) == (-170, 150)
    espn = rows[bears.game_id]
    assert (espn.source, espn.home_spread, espn.total) == ("espn:draftkings", -4, 44.5)
    # ESPN has no provider stamp: the last pregame fetch is the observation.
    assert espn.observed_at == espn.fetched_at == KICKOFF - 2 * minute

    # Repeated polls never write a second row for a started game.
    publisher.poll(KICKOFF + 2 * minute)
    assert len(written) == 2
    # The later cluster is skipped under the credit reserve and falls back.
    publisher.poll(KICKOFF + 28 * minute)
    assert len(requests) == 1
    board[("CIN", "JAX")] = LiveState("live", 0, 0, period=1, clock_seconds=900)
    publisher.poll(KICKOFF + 31 * minute)
    assert [line.source for line in written][-1] == "espn:draftkings"
    assert written[-1].game_id == late.game_id


def test_scoreboard_quote_is_parsed_only_for_a_scheduled_game():
    from backend.odds.espn import Quote

    competition = {
        "competitors": [
            {"homeAway": "home", "team": {"id": "2", "abbreviation": "BUF"}},
            {"homeAway": "away", "team": {"id": "17", "abbreviation": "NE"}},
        ],
        "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre"}},
        "odds": [_draftkings("-3.5", "44.5", "-180", "+155")],
    }
    _, state = parse_event({"competitions": [competition]})
    assert state.quote == Quote(-3.5, -110, -110, 44.5, -105, -115, -180, 155)
    # A one-sided moneyline does not count.
    del competition["odds"][0]["moneyline"]["away"]
    _, state = parse_event({"competitions": [competition]})
    assert state.quote.home_moneyline is None and state.quote.home_spread == -3.5
