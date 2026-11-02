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
            conn.execute(
                text("""
                CREATE FUNCTION public.site_revalidate() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$
            """)
            )
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
    from sqlalchemy import inspect

    from backend import publish
    from backend.awards.pipeline import BOARD_COLUMNS, META_COLUMNS
    from backend.grading import RESULT_COLUMNS

    contracts = {
        "teams": publish.TEAMS_COLUMNS,
        "team_ratings": publish.TEAM_RATINGS_COLUMNS,
        "team_unit_ratings": publish.TEAM_UNIT_RATINGS_COLUMNS,
        "game_projections": publish.GAME_PROJECTIONS_COLUMNS,
        "market_comparisons": publish.MARKET_COMPARISONS_COLUMNS,
        "backtest_predictions": publish.BACKTEST_COLUMNS,
        "market_snapshots": publish.MARKET_SNAPSHOTS_COLUMNS,
        "season_win_totals": publish.SEASON_WIN_TOTALS_COLUMNS,
        "game_results": RESULT_COLUMNS,
        "award_boards": BOARD_COLUMNS,
        "award_model_meta": META_COLUMNS,
    }
    inspector = inspect(database)
    for table, columns in contracts.items():
        stored = {
            column["name"] for column in inspector.get_columns(table, schema="nfl")
        }
        assert set(columns) <= stored, table
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
                home_team_abbr="H" + game,
                away_team_abbr="A" + game,
                home_field_points=1.0,
                pure_home_margin=2.0,
                home_margin=3.0,
            )
            for game in ("frozen", "tie", "no-close")
        ]
    )

    ratings = pd.DataFrame(
        [
            dict(
                season=2026,
                week=1,
                as_of=now,
                team_abbr=team,
                team=team,
                offense_points=2.0,
                defense_points=1.0,
                power_rating=3.0,
                forecast_alignment_points=0.0,
                model_version="acceptance",
            )
            for game in ("frozen", "tie", "no-close")
            for team in ("H" + game, "A" + game)
        ]
    )

    def publish(frame):
        result = publish_week(database, 2026, 1, None, ratings, frame, None, None)
        with database.connect() as conn:
            differences = (
                conn.execute(
                    text("""
                SELECT h.power_rating - a.power_rating
                     + g.home_field_points - g.home_margin
                FROM nfl.game_projections g
                JOIN nfl.team_ratings h ON h.team_abbr = g.home_team_abbr
                JOIN nfl.team_ratings a ON a.team_abbr = g.away_team_abbr
            """)
                )
                .scalars()
                .all()
            )
        assert all(abs(delta) < 1e-8 for delta in differences)
        return result

    assert publish(projections)["forecast_snapshots"] == 3
    assert publish(projections)["forecast_snapshots"] == 3
    revised = projections.copy()
    revised.loc[0, "home_margin"] = 4.0
    revised["as_of"] = pd.Timestamp.now(tz="UTC")
    assert publish(revised)["forecast_snapshots"] == 6
    assert publish(projections)["forecast_snapshots"] == 6  # stale artifact

    # The live worker captures one close per game before kickoff; a quote
    # stamped at or after kickoff is dropped, and nothing captured ever moves.
    def capture(game_id, home_spread, fetched_at):
        with database.begin() as conn:
            return conn.execute(
                text("""
                insert into nfl.closing_lines
                  (game_id, season, start_date, source, observed_at, fetched_at,
                   home_spread, total, n_offers, price_book)
                values (:game_id, 2026, :start, 'espn:draftkings', :fetched,
                  :fetched, :spread, 47.5, 1, 'draftkings')
                on conflict (game_id) do nothing
            """),
                {
                    "game_id": game_id,
                    "start": kickoff.to_pydatetime(),
                    "fetched": fetched_at,
                    "spread": home_spread,
                },
            ).rowcount

    assert capture("frozen", -5.0, kickoff.to_pydatetime() - timedelta(minutes=1)) == 1
    assert capture("frozen", -9.0, kickoff.to_pydatetime() - timedelta(minutes=1)) == 0
    assert capture("tie", -1.5, kickoff.to_pydatetime()) == 0
    for statement in (
        "update nfl.closing_lines set home_spread = 0",
        "delete from nfl.closing_lines",
        "truncate nfl.closing_lines",
    ):
        with pytest.raises(DBAPIError), database.begin() as conn:
            conn.execute(text(statement))
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
                home_team="H" + game,
                away_team="A" + game,
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
            frame,
            2026,
            {
                team + game: name
                for game in ("frozen", "tie", "no-close", "missing")
                for team, name in (("H", "Home"), ("A", "Away"))
            },
            pd.Timestamp.now(tz="UTC"),
        )

    original_results = results(schedules)
    metrics = grade_season(database, original_results, 2026)
    # frozen grades against its captured -5 (error 1), tie against nflverse.
    assert metrics == dict(
        completed_games=4,
        frozen_forecasts=3,
        benchmark_games=2,
        blended_mae=1.5,
        pure_mae=2.0,
        closing_mae=1.0,
    )
    with database.connect() as conn:
        accuracy = conn.execute(
            text("select nfl.forecast_accuracy('live')")
        ).scalar_one()
        assert accuracy["completed"] == 4 and accuracy["missingForecast"] == 1
        assert accuracy["overall"]["games"] == 2
        assert accuracy["overall"]["modelMae"] == 1.5
        assert accuracy["overall"]["pureMae"] == 2.0
        assert accuracy["overall"]["marketMae"] == 1.0
        assert accuracy["closingSources"] == {"espn:draftkings": 1, "nflverse": 1}
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
    # A revised nflverse line does not reach a game with a captured close.
    assert corrected["closing_mae"] == 1.5
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
                pure_home_margin=-8.0,
                home_margin=8.0,
                market_home_spread=-20.0,
                model_total=55.0,
                market_informed_total=50.0,
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
                odds_source="odds_api",
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
    forecasts.loc[48, "home_margin"] = -8.0
    forecasts.loc[50, "model_total"] = 35.0
    forecasts.loc[50, "market_informed_total"] = 40.0
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
    # Two unbettable books set the market every pick is priced from: it
    # favours the home side by ten (the away side of pick-048) and the over
    # (the under of pick-050), so every bettable quote has positive value.
    consensus = offers.assign(
        execution_eligibility_verified=False,
        point=lambda o: np.select(
            [o.market.eq("h2h"), o.market.eq("totals"), o.selection.eq("home")],
            [np.nan, np.where(o.game_id.eq("pick-050"), 38.0, 52.0), -10.0],
            10.0,
        )
        * np.where(o.game_id.eq("pick-048") & o.market.eq("spreads"), -1, 1),
        price=lambda o: np.select(
            [o.market.eq("h2h") & o.selection.eq("home"), o.market.eq("h2h")],
            [-500, 380],
            o.price,
        ),
    )
    offers = pd.concat(
        [offers]
        + [consensus.assign(provider_key=key, provider=key) for key in ("c1", "c2")],
        ignore_index=True,
    )
    weights = np.ones(len(MARGINS))
    decisions = build_recommendations(forecasts, offers, weights, decision_at=now)
    assert len(decisions) == 159 and decisions.status.eq("recommended").all()
    assert decisions.reason.eq("positive_ev").all()
    assert decisions.provider_key.eq("test").all()
    # Totals price from the market total alone; the model total has no weight.
    totals = decisions[decisions.market.eq("totals")].set_index("game_id")
    assert totals.market_total.drop("pick-050").eq(52.0).all()
    assert totals.model_total.drop("pick-050").nunique() == 1
    assert totals.model_total.iloc[0] == pytest.approx(52.0, abs=0.5)
    assert totals.side.eq("over").sum() == 52 and totals.side["pick-050"] == "under"
    # Sides price from the devigged market margin, not the published margin.
    spreads = decisions[decisions.market.eq("spreads")].set_index("game_id")
    assert spreads.side.eq("home").sum() == 52 and spreads.side["pick-048"] == "away"
    assert spreads.model_home_margin.drop("pick-048").to_numpy() == pytest.approx(
        10.0 + 0.045 * (-8.0 - 10.0)
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
    # A lineup correction revises an open pick and preserves its original record.
    # Market-only refreshes, stale QB evidence and post-kickoff edits stay frozen.
    with database.connect() as conn:
        transaction = conn.begin()
        try:
            trial = decisions.iloc[:1].assign(
                game_id="qb-refresh-trial",
                start_date=now + timedelta(days=1),
                provider_start_date=now + timedelta(days=1),
            )
            publish_recommendations(conn, trial)
            original_pick = conn.execute(
                text(
                    "select to_jsonb(r) from nfl.recommendations r "
                    "where game_id='qb-refresh-trial'"
                )
            ).scalar_one()
            flags = json.loads(trial.iloc[0].data_flags)
            flags["home_expected_qb"] = "announced-backup"
            corrected = trial.assign(
                data_flags=json.dumps(flags),
                forecast_as_of=now - timedelta(seconds=15),
                decision_at=pd.Timestamp.now(tz="UTC"),
            )
            publish_recommendations(conn, corrected)
            assert (
                conn.execute(
                    text(
                        "select data_flags ->> 'home_expected_qb' "
                        "from nfl.recommendations "
                        "where game_id='qb-refresh-trial'"
                    )
                ).scalar_one()
                == "test-qb"
            )
            flags["home_qb_source_at"] = (now - timedelta(seconds=30)).isoformat()
            corrected["data_flags"] = json.dumps(flags)
            publish_recommendations(conn, corrected)
            saved = conn.execute(
                text(
                    "select to_jsonb(r) from nfl.recommendations r "
                    "where game_id='qb-refresh-trial'"
                )
            ).scalar_one()
            assert saved["data_flags"]["home_expected_qb"] == "announced-backup"
            assert saved["published_at"] > original_pick["published_at"]
            assert (
                conn.execute(
                    text(
                        "select previous_record from nfl.recommendation_revisions "
                        "where game_id='qb-refresh-trial'"
                    )
                ).scalar_one()
                == original_pick
            )
            # A changed QB can also remove a pick when its new edge is inadequate.
            flags["home_expected_qb"] = "third-string-starter"
            flags["home_qb_source_at"] = (now - timedelta(seconds=10)).isoformat()
            no_play = corrected.assign(
                data_flags=json.dumps(flags),
                forecast_as_of=now,
                decision_at=pd.Timestamp.now(tz="UTC"),
                status="no_play",
                reason="below_edge_threshold",
                stake_units=0.0,
            )
            publish_recommendations(conn, no_play)
            assert (
                conn.execute(
                    text(
                        "select status from nfl.recommendations "
                        "where game_id='qb-refresh-trial'"
                    )
                ).scalar_one()
                == "no_play"
            )
            assert (
                conn.execute(
                    text(
                        "select count(*) from nfl.recommendation_revisions "
                        "where game_id='qb-refresh-trial'"
                    )
                ).scalar_one()
                == 2
            )
            for mutation in ("delete from", "update", "truncate"):
                with pytest.raises(DBAPIError), conn.begin_nested():
                    sql = f"{mutation} nfl.recommendation_revisions"
                    if mutation == "update":
                        sql += " set reason='expected_qb_changed'"
                    conn.execute(text(sql))
        finally:
            transaction.rollback()

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


def test_slate_fills_with_model_side_leans_after_frozen_picks():
    """Without positive value, the five-game slate counts frozen picks and fills
    with the highest-value sides the model leans to, never totals."""
    import json

    import numpy as np

    from backend.model.market_blend import MARGINS
    from backend.recommendations import SLATE_SIZE, build_recommendations

    now = pd.Timestamp.now(tz="UTC")
    start = now + timedelta(hours=2)
    at = (now - timedelta(minutes=5)).isoformat()
    source = {
        name: {"observed_at": at, "sha256": "a" * 64}
        for name in (
            "schedule",
            "depth_charts",
            "qb_overrides",
            "win_totals",
            "win_total_sources",
            "preseason_qbs",
        )
    }
    flags = {
        f"{side}_{key}": value
        for side in ("home", "away")
        for key, value in (("expected_qb", "qb"), ("qb_source_at", at))
    }
    # The pure margin sits i points above the 3-point market line.
    forecasts = pd.DataFrame(
        dict(
            game_id=f"g{i}",
            season=2026,
            week=1,
            start_date=start,
            as_of=now - timedelta(minutes=1),
            model_version="test",
            home_team=f"H{i}",
            away_team=f"A{i}",
            pure_home_margin=3.0 + i,
            home_margin=3.0,
            model_total=60.0,
            margin_sd=13.0,
            total_sd=13.0,
            degrees_of_freedom=7.0,
            home_missing_input_count=0,
            away_missing_input_count=0,
            source_timestamps=json.dumps(source),
            data_flags=json.dumps(flags),
        )
        for i in range(1, 9)
    )
    offers = pd.DataFrame(
        dict(
            game_id=game,
            market=market,
            selection=side,
            price=price,
            point=point,
            provider="Book",
            provider_key="book",
            provider_last_update=now,
            market_fetched_at=now,
            odds_source="odds_api",
            odds_api_event_id=game,
            provider_start_date=start,
            match_score=1.0,
            execution_eligibility_verified=True,
        )
        for game in forecasts.game_id
        # The away side's cheaper price never outranks the model's home side.
        for market, side, point, price in (
            ("spreads", "home", -3.0, -110),
            ("spreads", "away", 3.0, -102),
            ("totals", "over", 44.0, -110),
            ("totals", "under", 44.0, -110),
        )
    )
    decisions = build_recommendations(
        forecasts,
        offers,
        np.ones(len(MARGINS)),
        decision_at=now,
        picked_games={"g1", "other-game"},
    )
    picks = decisions[decisions.status.eq("recommended")]
    assert picks.reason.eq("ranked_lean").all() and picks.market.eq("spreads").all()
    assert picks.side.eq("home").all() and picks.expected_value_per_unit.lt(0).all()
    assert sorted(picks.game_id) == ["g6", "g7", "g8"] and SLATE_SIZE == 5


def test_teaser_legs_use_the_consensus_line_freeze_and_grade(database):
    """A leg is the lower-middle dog line across books, Pinnacle counted like
    any other; the first qualifying pull is frozen, only grading may change
    it, and every pregame book quote is archived."""
    from backend.publish import _grade_teaser_legs, insert_book_quotes
    from backend.teaser_legs import qualifying_legs

    now = pd.Timestamp.now(tz="UTC")
    start = now + timedelta(days=2)
    games = pd.DataFrame(
        dict(game_id=["a", "b", "c"], season=2026, week=9, start_date=start)
    ).assign(home_team=["HA", "HB", "HC"], away_team=["AA", "AB", "AC"])

    def offers(fetched, lines):
        return pd.DataFrame(
            dict(
                game_id=game,
                market="spreads",
                selection=side,
                point=home if side == "home" else -home,
                price=-110,
                provider_key=book,
                market_fetched_at=fetched,
                provider_last_update=fetched,
                provider_start_date=start,
            )
            for game, book, home in lines
            for side in ("home", "away")
        )

    # a: home dog +2.5 at two of three books, Pinnacle +3.5: the middle is +2.5.
    # b: away dog +2.0 at both books.
    # c: Pinnacle has the away dog +2.5 but the middle of three is +3.0.
    first = offers(
        now,
        [
            ("a", "pinnacle", 3.5),
            ("a", "draftkings", 2.5),
            ("a", "fanduel", 2.5),
            ("b", "draftkings", -2.0),
            ("b", "fanduel", -2.0),
            ("c", "pinnacle", -2.5),
            ("c", "draftkings", -3.0),
            ("c", "fanduel", -3.5),
        ],
    )
    legs = qualifying_legs(games, first).set_index("game_id")
    assert list(legs.index) == ["a", "b"]
    assert legs.loc["a", ["side", "point", "books", "teased_point"]].tolist() == [
        "home",
        2.5,
        3,
        8.5,
    ]
    assert legs.loc["b", ["side", "team", "books"]].tolist() == ["away", "AB", 2]
    later = offers(now + timedelta(hours=1), [("a", "draftkings", 1.5)])
    with database.begin() as conn:
        from sqlalchemy import MetaData, Table
        from sqlalchemy.dialects.postgresql import insert

        table = Table("teaser_legs", MetaData(), schema="nfl", autoload_with=conn)
        for pull in (first, later):
            rows = qualifying_legs(games, pull).to_dict("records")
            conn.execute(
                insert(table)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["game_id"])
            )
            assert insert_book_quotes(conn, pull, "decision") == len(pull)
        assert insert_book_quotes(conn, first, "decision") == 0
        frozen = conn.execute(
            text("select point from nfl.teaser_legs where game_id = 'a'")
        ).scalar_one()
        assert frozen == 2.5
    with pytest.raises(DBAPIError, match="frozen"):
        with database.begin() as conn:
            conn.execute(text("update nfl.teaser_legs set point = 1.5"))
    schedule = games.assign(
        game_status="final",
        completed=True,
        home_points=[20, 10, 0],
        away_points=[27, 17, 0],
    )
    with database.begin() as conn:
        _grade_teaser_legs(conn, schedule)
        graded = dict(
            conn.execute(text("select game_id, outcome from nfl.teaser_legs")).all()
        )
    # a: home dog loses by 7 with +8.5 (win); b: away dog wins outright (win).
    assert graded == {"a": "win", "b": "win"}
