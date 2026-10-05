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

    # Inside the lead window the database alone answers; the feed is untouched.
    assert publisher.poll(KICKOFF - timedelta(minutes=10))
    assert calls == [] and {p["abstract_state"] for p in written} == {"Pre"}

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
