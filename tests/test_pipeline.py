"""Production pipeline regressions for cached history and special game days."""

from datetime import UTC, datetime

import pandas as pd

from backend import pipeline
from backend.etl import ingest, store
from backend.features.qb import expected_starters


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


def test_preseason_runs_load_current_starters_before_pbp_opens(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        ingest.nflreadpy,
        "get_current_season",
        lambda roster=False: 2026 if roster else 2025,
    )
    monkeypatch.setattr(ingest.data, "load_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest.data, "load_teams", pd.DataFrame)
    charts = pd.DataFrame(
        [
            ("2026-09-06", "MIN", "former", 1),
            ("2026-09-07", "MIN", "transfer", 1),
            ("2026-09-06", "LV", "rookie", 1),
            ("2026-09-07", "NEW", "new_team_qb", 1),
        ],
        columns=["dt", "team", "gsis_id", "pos_rank"],
    ).assign(pos_abb="QB")
    monkeypatch.setattr(ingest.data, "load_depth_charts", lambda seasons: charts)

    def game_data_unavailable(*args, **kwargs):
        raise AssertionError("Preseason must not request unavailable game data")

    for loader in ("load_pbp", "load_injuries", "load_pfr_advstats"):
        monkeypatch.setattr(ingest.data, loader, game_data_unavailable)

    for run in (ingest.ingest_season, ingest.ingest_projection_inputs):
        assert run(2026) == []
        loaded = store.read_raw("depth_charts", "2026.parquet")
        pd.testing.assert_frame_equal(loaded, charts)
        (tmp_path / "depth_charts" / "2026.parquet").unlink()

    index = pd.DataFrame([{"game_id": "old", "season": 2025, "model_week": 18}])
    history = pd.DataFrame(
        [("old", "MIN", "former"), ("old", "LV", "incumbent")],
        columns=["game_id", "team", "passer_player_id"],
    ).assign(dropbacks=30, started=True)
    starters = expected_starters(history, index, loaded, 2026, 1)
    assert starters.to_dict() == {
        "MIN": "transfer",
        "LV": "rookie",
        "NEW": "new_team_qb",
    }
