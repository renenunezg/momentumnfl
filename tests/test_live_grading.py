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


def test_forecast_input_bundles_are_immutable_and_deduplicated(
    database, tmp_path, monkeypatch
):
    import hashlib

    from backend import forecast_archive

    content = b"frozen source revision"
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / digest).write_bytes(content)
    monkeypatch.setattr(forecast_archive, "ARCHIVE", tmp_path)
    manifest = dict(
        season=2026,
        week=1,
        cutoff="2026-01-01T00:00:00Z",
        files={"source": dict(present=True, sha256=digest)},
    )
    with database.begin() as conn:
        forecast_archive.persist(conn, manifest)
        forecast_archive.persist(conn, manifest)
        assert (
            conn.execute(text("select count(*) from nfl.forecast_input_runs")).scalar()
            == 1
        )
        assert (
            conn.execute(
                text("select count(*) from nfl.forecast_input_objects")
            ).scalar()
            == 1
        )
    for table in ("forecast_input_runs", "forecast_input_objects"):
        for verb in ("delete from", "truncate"):
            with pytest.raises(DBAPIError), database.begin() as conn:
                conn.execute(text(f"{verb} nfl.{table}"))


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
    with database.connect() as conn:
        accuracy = conn.execute(
            text("select nfl.forecast_accuracy('live')")
        ).scalar_one()
        assert accuracy["completed"] == 4 and accuracy["missingForecast"] == 1
        assert accuracy["overall"]["games"] == 2
        assert accuracy["overall"]["modelMae"] == 1.5
        assert accuracy["overall"]["pureMae"] == 2.0
        assert accuracy["overall"]["marketMae"] == 0.75
        assert (
            conn.execute(text("select nfl.forecast_accuracy('backtest')")).scalar_one()[
                "completed"
            ]
            == 0
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


def test_recommendation_publication_settlement_and_filtered_history(database):
    """Exercise the actual builder, database freeze, grader and public read API."""
    import json

    import numpy as np

    from backend.model.market_blend import MARGINS
    from backend.publish import grade_picks, publish_recommendations
    from backend.recommendations import _probabilities, build_recommendations

    now = pd.Timestamp.now(tz="UTC")
    start = now + timedelta(seconds=10)
    source = {
        name: {
            "observed_at": (now - timedelta(minutes=5)).isoformat(),
            "sha256": "a" * 64,
        }
        for name in (
            "schedule",
            "depth_charts",
            "qb_overrides",
            "win_totals",
            "win_total_sources",
            "preseason_qbs",
        )
    }
    forecasts = pd.DataFrame(
        [
            dict(
                game_id=f"pick-{i:03}",
                season=2026 if i < 51 else 2027,
                week=1,
                start_date=start,
                as_of=now - timedelta(minutes=1),
                model_version="test",
                home_team="Home",
                away_team="Away",
                pure_home_margin=8.0,
                home_margin=-8.0,
                model_total=55.0,
                margin_sd=10.0,
                total_sd=10.0,
                degrees_of_freedom=7.0,
                home_missing_input_count=0,
                away_missing_input_count=0,
                source_timestamps=json.dumps(source),
                data_flags=json.dumps(
                    {
                        f"{side}_{key}": value
                        for side in ("home", "away")
                        for key, value in (
                            ("expected_qb", "test-qb"),
                            (
                                "qb_source_at",
                                (now - timedelta(minutes=5)).isoformat(),
                            ),
                        )
                    }
                ),
            )
            for i in range(53)
        ]
    )
    offers = pd.DataFrame(
        [
            dict(
                game_id=game,
                market=market,
                selection=side,
                price=price,
                point=point,
                provider="Test Book",
                provider_key="test",
                provider_last_update=now,
                market_fetched_at=now,
                odds_api_event_id=game,
                provider_start_date=start,
                match_score=1.0,
                execution_eligibility_verified=True,
            )
            for game in forecasts.game_id
            for market, side, point, price in (
                ("h2h", "home", None, -110),
                ("h2h", "away", None, 100),
                ("spreads", "home", -3.0, -110),
                ("spreads", "away", 3.0, -110),
                ("totals", "over", 45.0, 100),
                ("totals", "under", 45.0, -110),
            )
        ]
    )
    forecasts.loc[48, "pure_home_margin"] = -8.0
    forecasts.loc[50, "model_total"] = 35.0
    offers.loc[offers.game_id.eq("pick-049") & offers.market.eq("spreads"), "point"] = (
        0.0
    )
    offers.loc[
        offers.game_id.eq("pick-050")
        & offers.market.eq("h2h")
        & offers.selection.eq("home"),
        "price",
    ] = 120.0
    offers.loc[
        offers.game_id.eq("pick-051")
        & offers.market.eq("h2h")
        & offers.selection.eq("home"),
        "price",
    ] = -100.0
    weights = np.ones(len(MARGINS))
    decisions = build_recommendations(forecasts, offers, weights, decision_at=now)
    assert len(decisions) == 159 and decisions.status.eq("recommended").all()
    # A blended margin with the opposite sign must not price recommendations.
    assert (
        decisions[decisions.market.eq("spreads") & ~decisions.game_id.eq("pick-048")]
        .side.eq("home")
        .all()
    )
    for pick in decisions.itertuples():
        profit = pick.price / 100 if pick.price > 0 else 100 / abs(pick.price)
        loss = 1 - pick.win_probability - pick.push_probability
        assert pick.expected_value_per_unit == pytest.approx(
            pick.win_probability * profit - loss
        )
        assert pick.probability_edge == pytest.approx(
            pick.win_probability / (1 - pick.push_probability) - 1 / (profit + 1)
        )
    projection = next(forecasts.itertuples())
    home = _probabilities(projection, dict(market="h2h", side="home"), weights)
    away = _probabilities(projection, dict(market="h2h", side="away"), weights)
    assert home[1] > 0 and sum(home) == pytest.approx(1)
    assert home == pytest.approx((away[2], away[1], away[0]))
    for line in (45, 45.5):
        over = _probabilities(
            projection, dict(market="totals", side="over", point=line), weights
        )
        under = _probabilities(
            projection, dict(market="totals", side="under", point=line), weights
        )
        assert over == pytest.approx((under[2], under[1], under[0]))
        assert (over[1] > 0) == (line == 45)
    # Source and matching failures never become a wager, even with high EV.
    for column, bad in (
        ("provider_start_date", start + timedelta(hours=1)),
        ("provider_last_update", now - timedelta(hours=2)),
        ("match_score", 0.7),
        ("match_score", 1.1),
        ("execution_eligibility_verified", False),
    ):
        blocked = build_recommendations(
            forecasts, offers.assign(**{column: bad}), weights, decision_at=now
        )
        assert blocked.status.eq("no_play").all()
    assert (
        build_recommendations(
            forecasts.assign(home_missing_input_count=1),
            offers,
            weights,
            decision_at=now,
        )
        .status.eq("no_play")
        .all()
    )
    future_source = json.loads(json.dumps(source))
    future_source["depth_charts"]["observed_at"] = now.isoformat()
    assert (
        build_recommendations(
            forecasts.assign(source_timestamps=json.dumps(future_source)),
            offers,
            weights,
            decision_at=now,
        )
        .status.eq("no_play")
        .all()
    )
    assert build_recommendations(
        forecasts.assign(start_date=now), offers, weights, decision_at=now
    ).empty

    def publish(frame):
        with database.begin() as conn:
            publish_recommendations(conn, frame)

    # No Play is transparent, contributes no wager, and can qualify before kickoff.
    with database.connect() as conn:
        transaction = conn.begin()
        try:
            trial = decisions.iloc[:1].assign(
                game_id="no-play-trial",
                status="no_play",
                stake_units=0.0,
                reason="missing_model_inputs",
            )
            publish_recommendations(conn, trial)
            summary = (
                conn.execute(
                    text(
                        "select * from nfl.recommendation_summary() "
                        "where segment_kind='overall'"
                    )
                )
                .mappings()
                .one()
            )
            assert (
                summary["picks"] == 0
                and summary["no_plays"] == 1
                and summary["roi"] is None
            )
            publish_recommendations(
                conn,
                decisions.iloc[:1].assign(
                    game_id="no-play-trial", decision_at=pd.Timestamp.now(tz="UTC")
                ),
            )
            assert (
                conn.execute(
                    text(
                        "select status from nfl.recommendations "
                        "where game_id='no-play-trial'"
                    )
                ).scalar_one()
                == "recommended"
            )
        finally:
            transaction.rollback()
    publish(decisions)
    with database.connect() as conn:
        original = conn.execute(
            text(
                "select jsonb_agg(to_jsonb(r) order by "
                "game_id,market) from nfl.recommendations r"
            )
        ).scalar_one()
    publish(decisions.assign(price=200, decision_at=pd.Timestamp.now(tz="UTC")))
    with database.connect() as conn:
        assert (
            conn.execute(
                text(
                    "select jsonb_agg(to_jsonb(r) order by "
                    "game_id,market) from nfl.recommendations r"
                )
            ).scalar_one()
            == original
        )
    # Database checks independently reject rewriting, deletion and arithmetic lies.
    for sql in (
        "update nfl.recommendations set price=200",
        "delete from nfl.recommendations",
        "truncate nfl.recommendations",
    ):
        with pytest.raises(DBAPIError), database.begin() as conn:
            conn.execute(text(sql))
    with pytest.raises(DBAPIError):
        publish(
            decisions.iloc[:1].assign(game_id="bad-ev", expected_value_per_unit=999)
        )
    with pytest.raises(DBAPIError):
        publish(
            decisions.iloc[:1].assign(
                game_id="past", start_date=now - timedelta(days=1)
            )
        )

    # The source-date metadata is itself an input to preseason cutoff checks.
    for bad_sources in (
        {k: v for k, v in source.items() if k != "win_total_sources"},
        {
            **source,
            "win_total_sources": {"sha256": "a" * 64, "observed_at": now.isoformat()},
        },
    ):
        with pytest.raises(DBAPIError):
            publish(
                decisions.iloc[:1].assign(
                    game_id="bad-source-metadata",
                    source_timestamps=json.dumps(bad_sources),
                )
            )

    # Pending games are included; one game may contribute three recommendations.
    with database.begin() as conn:
        conn.execute(text("set local role anon"))
        h = conn.execute(
            text("select nfl.recommendation_history(p_page=>999)")
        ).scalar_one()
        assert (h["count"], h["page"], h["total_pages"], len(h["rows"])) == (
            159,
            4,
            4,
            9,
        )
        overall = next(m for m in h["metrics"] if m["segment_kind"] == "overall")
        assert overall["unique_games"] == 53 and overall["pending"] == 159
        all_ids = []
        for page in range(1, 5):
            part = conn.execute(
                text("select nfl.recommendation_history(p_page=>:page)"),
                dict(page=page),
            ).scalar_one()
            all_ids.extend((r["game_id"], r["market"]) for r in part["rows"])
        assert len(set(all_ids)) == 159
        for season in (None, 2026, 2027):
            for market in ("all", "h2h", "spreads", "totals"):
                h = conn.execute(
                    text(
                        "select nfl.recommendation_history(:season,:market,:cutoff,1)"
                    ),
                    dict(season=season, market=market, cutoff=now - timedelta(days=6)),
                ).scalar_one()
                overall = next(
                    m for m in h["metrics"] if m["segment_kind"] == "overall"
                )
                assert h["count"] == overall["picks"] + overall["no_plays"]
                for m in (m for m in h["metrics"] if m["segment_kind"] == "market"):
                    sides = [
                        s
                        for s in h["metrics"]
                        if s["segment_kind"] == "side"
                        and s["segment"].startswith(m["segment"] + ":")
                    ]
                    assert sum(s["picks"] for s in sides) == m["picks"]
        assert (
            conn.execute(
                text("select nfl.recommendation_history(p_from=>:cutoff)"),
                dict(cutoff=start),
            ).scalar_one()["count"]
            == 0
        )

    with database.connect() as conn:
        conn.execute(
            text(
                "select pg_sleep(greatest(0,extract(epoch from "
                "(:start-clock_timestamp()))+.01))"
            ),
            dict(start=start.to_pydatetime()),
        )
    fetched = pd.Timestamp.now(tz="UTC")
    schedule = pd.DataFrame(
        [
            dict(
                game_id=game,
                season=2026,
                start_date=start,
                home_team="Home",
                away_team="Away",
                game_status="scheduled",
                completed=True,
                home_points=24,
                away_points=21,
                observed_at=fetched,
            )
            for game in forecasts.game_id[:51]
        ]
    )
    # Spread -3 and total 45 both push; ML wins at the exact frozen -110.
    schedule.loc[0, ["home_points", "away_points"]] = [20, 20]
    schedule.loc[1, "start_date"] = start + timedelta(days=1)
    schedule.loc[1, "completed"] = False
    schedule.loc[1, ["home_points", "away_points"]] = None
    schedule.loc[2, "game_status"] = "canceled"
    schedule.loc[2, "completed"] = False
    schedule.loc[2, ["home_points", "away_points"]] = None
    assert grade_picks(database, schedule, 2026) == 153
    assert grade_picks(database, schedule, 2026) == 0
    with database.connect() as conn:
        summary = (
            conn.execute(text("select * from nfl.recommendation_summary()"))
            .mappings()
            .all()
        )
        labels = {m["segment"] for m in summary if m["segment_kind"] == "side"}
        assert {
            "spreads:favorite",
            "spreads:underdog",
            "spreads:pickem",
            "h2h:favorite",
            "h2h:underdog",
            "h2h:even",
            "totals:over",
            "totals:under",
        } <= labels
        for market in (m for m in summary if m["segment_kind"] == "market"):
            sides = [
                m
                for m in summary
                if m["segment_kind"] == "side"
                and m["segment"].startswith(market["segment"] + ":")
            ]
            for key in (
                "picks",
                "wins",
                "losses",
                "pushes",
                "pending",
                "voids",
                "profit_units",
                "staked_units",
            ):
                assert sum(m[key] for m in sides) == pytest.approx(market[key])
    with database.connect() as conn:
        outcomes = conn.execute(
            text(
                "select market,outcome,profit_units from "
                "nfl.recommendations where game_id='pick-003' order by market"
            )
        ).all()
        assert outcomes == [
            ("h2h", "win", pytest.approx(100 / 110)),
            ("spreads", "push", 0),
            ("totals", "push", 0),
        ]
        assert (
            conn.execute(
                text(
                    "select outcome from nfl.recommendations "
                    "where game_id='pick-000' and market='h2h'"
                )
            ).scalar_one()
            == "void"
        )
        assert (
            conn.execute(
                text(
                    "select count(*) from nfl.recommendations "
                    "where game_id in ('pick-001','pick-002') and outcome='void'"
                )
            ).scalar_one()
            == 6
        )
        settled = conn.execute(
            text(
                "select jsonb_agg(to_jsonb(r) order by "
                "game_id,market) from nfl.recommendations r"
            )
        ).scalar_one()
    revised = schedule.assign(
        home_points=35, away_points=3, observed_at=pd.Timestamp.now(tz="UTC")
    )
    assert grade_picks(database, revised, 2026) == 0
    with database.connect() as conn:
        assert (
            conn.execute(
                text(
                    "select jsonb_agg(to_jsonb(r) order by "
                    "game_id,market) from nfl.recommendations r"
                )
            ).scalar_one()
            == settled
        )
    with pytest.raises(DBAPIError):
        publish(decisions.iloc[:1].assign(game_id="late", decision_at=now))
