"""Production pipeline regressions for cached history and special game days."""

from datetime import UTC, datetime

import pandas as pd

from backend import pipeline


def test_history_cache_skips_complete_seasons(monkeypatch):
    present = {
        "team_games": ["2015", "2016"],
        "qb_games": ["2015"],
    }
    monkeypatch.setattr(
        pipeline.store,
        "processed_names",
        lambda directory: present[directory],
    )

    assert pipeline.missing_core_seasons(2016) == [2016]


def test_upcoming_games_does_not_assume_a_weekday():
    schedules = pd.DataFrame(
        [
            {
                "game_id": "2026_17_WED_GAME",
                "season": 2026,
                "home_score": None,
                "gameday": "2026-12-23",
                "gametime": "13:00",
            }
        ]
    )
    now = datetime(2026, 12, 23, 15, 0, tzinfo=UTC)

    games = pipeline.upcoming_games(schedules, 2026, hours=4, now=now)

    assert games["game_id"].tolist() == ["2026_17_WED_GAME"]


def test_incremental_games_include_recent_weeks_and_old_gaps():
    schedules = pd.DataFrame(
        [
            {"game_id": "W1", "week": 1, "home_score": 20},
            {"game_id": "W2", "week": 2, "home_score": 20},
            {"game_id": "W3", "week": 3, "home_score": 20},
            {"game_id": "W4", "week": 4, "home_score": None},
        ]
    )
    existing = [
        {"W1", "W2", "W3"},
        {"W1", "W2", "W3"},
        {"W2", "W3"},
    ]

    selected = pipeline.incremental_game_ids(schedules, existing, lookback_weeks=1)

    assert selected == {"W1", "W3"}


def test_incremental_write_replaces_game_and_keeps_prior_weeks(monkeypatch):
    existing = pd.DataFrame(
        [
            {"game_id": "W1", "value": 1, "start_date": "2026-09-01"},
            {"game_id": "W2", "value": 2, "start_date": "2026-09-08"},
        ]
    )
    rebuilt = pd.DataFrame([{"game_id": "W2", "value": 20, "start_date": "2026-09-08"}])
    written = []
    monkeypatch.setattr(
        pipeline.store,
        "read_processed",
        lambda *parts: existing,
    )
    monkeypatch.setattr(
        pipeline.store,
        "write_processed",
        lambda frame, *parts: written.append(frame),
    )

    pipeline.write_incremental_features(
        rebuilt, "team_games", 2026, ["start_date", "game_id"]
    )

    assert written[0][["game_id", "value"]].to_dict("records") == [
        {"game_id": "W1", "value": 1},
        {"game_id": "W2", "value": 20},
    ]
