"""Real PostgreSQL acceptance: publish, cross kickoff, refresh, and grade.

Set NFL_TEST_DATABASE_URL to a local disposable PostgreSQL admin database.
The fixture creates and removes its own database; production URLs are refused.
"""

import os
import uuid
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from backend.grading import result_frame
from backend.publish import grade_season, publish_week


@pytest.fixture
def database(request):
    value = os.getenv("NFL_TEST_DATABASE_URL")
    if not value:
        pytest.skip("NFL_TEST_DATABASE_URL required for PostgreSQL acceptance")
    url = make_url(value)
    assert url.host in {"127.0.0.1", "localhost", "::1"}
    name = "nfl_acceptance_" + uuid.uuid4().hex
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(url.set(database=name))
    try:
        with engine.begin() as conn:
            for path in sorted((Path(__file__).parents[1] / "sql").glob("*.sql")):
                if path.name.startswith("004") and getattr(request, "param", False):
                    conn.execute(
                        text("""
                        insert into nfl.game_projections
                          (game_id, season, week, as_of, model_version,
                           start_date, home_team, away_team)
                        values
                          ('future', 2026, 1, now(), 'test',
                           now() + interval '1 day', 'Home', 'Away'),
                          ('past', 2026, 1, now() - interval '2 days', 'test',
                           now() - interval '1 day', 'Home', 'Away')
                    """)
                    )
                conn.execute(text(path.read_text()))
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}"'))
        admin.dispose()


@pytest.mark.parametrize("database", [True], indirect=True)
def test_migration_seeds_existing_pregame_forecasts_in_one_transaction(database):
    with database.connect() as conn:
        assert conn.execute(
            text("select game_id from nfl.forecast_snapshots")
        ).scalars().all() == ["future"]
        assert (
            conn.execute(text("select count(*) from nfl.game_projections")).scalar_one()
            == 2
        )


def test_published_forecasts_freeze_and_grade_on_identical_cohort(database):
    now = pd.Timestamp.now(tz="UTC")
    kickoff = pd.Timestamp(now.to_pydatetime() + timedelta(seconds=10))
    projections = pd.DataFrame(
        [
            dict(
                game_id=game,
                season=2026,
                week=1,
                as_of=now,
                start_date=kickoff,
                model_version="acceptance",
                home_team="Home",
                away_team="Away",
                home_team_abbr="H",
                away_team_abbr="A",
                pure_home_margin=2.0,
                home_margin=3.0,
            )
            for game in ("frozen", "tie", "no-close")
        ]
    )

    def publish(frame):
        return publish_week(database, 2026, 1, None, None, frame, None, None)

    assert publish(projections)["forecast_snapshots"] == 3
    assert publish(projections)["forecast_snapshots"] == 3
    revised = projections.copy()
    revised.loc[0, "home_margin"] = 4.0
    revised["as_of"] = pd.Timestamp.now(tz="UTC")
    assert publish(revised)["forecast_snapshots"] == 6
    assert publish(projections)["forecast_snapshots"] == 6  # stale artifact
    with database.connect() as conn:
        conn.execute(
            text(
                "select pg_sleep(greatest(0, extract(epoch from "
                "(:kickoff - clock_timestamp())) + 0.02))"
            ),
            {"kickoff": kickoff.to_pydatetime()},
        )
    corrupt = revised.copy()
    corrupt["home_margin"] = 25.0
    assert publish(corrupt)["forecast_snapshots"] == 6
    # A late first publication cannot sneak in by backdating its as_of.
    late = corrupt.iloc[[0]].assign(game_id="missing")
    assert publish(late)["game_projections"] == 3
    # Moving a started game into the future cannot unfreeze it either.
    corrupt["start_date"] = kickoff.to_pydatetime() + timedelta(days=1)
    assert publish(corrupt)["forecast_snapshots"] == 6
    with database.connect() as conn:
        assert (
            conn.execute(
                text(
                    "select home_margin from nfl.game_projections "
                    "where game_id = 'frozen'"
                )
            ).scalar_one()
            == 4
        )
    for statement in (
        "update nfl.forecast_snapshots set home_margin = 99",
        "delete from nfl.forecast_snapshots",
        "truncate nfl.forecast_snapshots",
        "delete from nfl.game_projections",
        "truncate nfl.game_projections",
    ):
        with pytest.raises(DBAPIError), database.begin() as conn:
            conn.execute(text(statement))

    # Receipt alone is insufficient if the publication commits after kickoff.
    with pytest.raises(DBAPIError), database.begin() as conn:
        conn.execute(
            text("""
            insert into nfl.game_projections
              (game_id, season, week, as_of, model_version, start_date,
               home_team, away_team)
            values ('crossed-commit', 2026, 1, clock_timestamp(), 'test',
              clock_timestamp() + interval '0.1 seconds', 'Home', 'Away')
        """)
        )
        conn.execute(text("select pg_sleep(0.15)"))

    eastern = kickoff.tz_convert("America/New_York")
    schedules = pd.DataFrame(
        [
            dict(
                game_id=game,
                season=2026,
                week=1,
                game_type="REG",
                gameday=eastern.strftime("%Y-%m-%d"),
                gametime=eastern.strftime("%H:%M:%S.%f"),
                home_team="H",
                away_team="A",
                location="Home",
                home_score=home,
                away_score=away,
                result=home - away,
                spread_line=line,
            )
            for game, home, away, line in (
                ("frozen", 24, 20, 3.5),
                ("tie", 20, 20, 1.0),
                ("no-close", 24, 20, None),
                ("missing", 24, 20, 3.5),
            )
        ]
    )

    def results(frame):
        return result_frame(
            frame, 2026, {"H": "Home", "A": "Away"}, pd.Timestamp.now(tz="UTC")
        )

    original_results = results(schedules)
    metrics = grade_season(database, original_results, 2026)
    assert metrics == dict(
        completed_games=4,
        frozen_forecasts=3,
        benchmark_games=2,
        blended_mae=1.5,
        pure_mae=2.0,
        closing_mae=0.75,
    )
    assert grade_season(database, original_results, 2026) == metrics
    schedules.loc[0, ["home_score", "result", "spread_line"]] = [27, 7, 4]
    corrected = grade_season(database, results(schedules), 2026)
    assert corrected["blended_mae"] == 3.0
    assert corrected["closing_mae"] == 2.0
    assert grade_season(database, original_results, 2026) == corrected
    with database.begin() as conn:
        conn.execute(text("set local role anon"))
        assert (
            conn.execute(text("select count(*) from nfl.live_predictions")).scalar_one()
            == 4
        )
    # Incomplete and canceled games are not assigned a fabricated final result.
    schedules["away_score"] = None
    assert results(schedules).empty
