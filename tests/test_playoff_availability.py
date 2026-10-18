"""Production QB report revisions and freshness gates."""
import json

import pandas as pd
import pytest


def test_live_return_report_revisions(tmp_path):
    from backend.playoffs.availability import load_return_reports, return_share

    report = dict(
        season=2026,
        team="TB",
        player="Example QB",
        gsis_id="example",
        status="window",
        absence_start="2026-10-01",
        out_weeks_min=4,
        out_weeks_max=6,
        reported_at="2026-10-02T12:00:00Z",
        observed_at="2026-10-02T13:00:00Z",
        source_url="https://example.com/report",
    )
    later = dict(
        report,
        status="unknown",
        recovery_scenario_start="2026-10-18",
        reported_at="2026-10-10T12:00:00Z",
        observed_at="2026-10-10T13:00:00Z",
    )
    path = tmp_path / "windows.json"
    path.write_text(json.dumps([report, later]))
    cutoff = pd.Timestamp("2026-10-03T00:00:00Z")
    selected = load_return_reports(path, 2026, cutoff)[0]
    assert selected["return_earliest"] == "2026-10-29"
    assert selected["return_latest"] == "2026-11-12"
    assert return_share(selected, pd.Timestamp("2026-10-25T17:00:00Z")) == 0
    assert return_share(selected, pd.Timestamp("2026-11-05T18:00:00Z")) == 1
    assert return_share(selected, pd.Timestamp("2026-11-15T18:00:00Z")) == 1
    selected = load_return_reports(path, 2026, pd.Timestamp("2026-10-11T00:00:00Z"))[0]
    assert selected["status"] == "unknown"
    assert return_share(selected, pd.Timestamp("2026-11-15T18:00:00Z")) is None
    report["out_weeks_max"] = 3
    path.write_text(json.dumps([report]))
    with pytest.raises(ValueError, match="Invalid reported absence"):
        load_return_reports(path, 2026, cutoff)


def test_current_return_estimate_survives_undated_weekly_rows(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from backend.playoffs import availability

    cutoff = pd.Timestamp("2026-10-17T23:00:00Z")
    current = dict(
        team="TB",
        gsis_id="starter",
        position="QB",
        full_name="Starter",
        report_status="Out",
        estimated_return_date="2026-10-25",
        date_modified="2026-10-17T22:00:00Z",
        source_updated_at="2026-10-16T18:00:00Z",
        source_url="https://example.com/injuries",
    )
    old = dict(current, date_modified=None, estimated_return_date=None)
    injuries = pd.DataFrame([current, old])
    monkeypatch.setattr(availability, "load_injuries", lambda *args: injuries)
    monkeypatch.setattr(
        availability.store,
        "qb_games",
        lambda *args: pd.DataFrame(
            dict(passer_player_id=["starter", "backup"], passer=["Starter", "Backup"])
        ),
    )
    context = SimpleNamespace(
        mixtures={"TB": {"starter": 1}},
        strengths={"starter": 1},
        adjustment=lambda *args: 0,
    )
    monkeypatch.setattr(
        availability,
        "forecast_qb_context",
        lambda *args: (context, {"TB": "backup"}, {"TB": 1}),
    )
    path = tmp_path / "reports.json"
    path.write_text("[]")
    fit = SimpleNamespace(season=2026, teams=["TB"], config=None)
    schedule = pd.DataFrame(
        dict(
            home_score=[float("nan")] * 5,
            week=[6, 7, 8, 9, 11],
            home_team=["TB"] * 5,
            away_team=["PHI"] * 5,
            start_date=pd.to_datetime(
                [
                    "2026-10-18T17:00:00Z",
                    "2026-10-25T17:00:00Z",
                    "2026-11-01T18:00:00Z",
                    "2026-11-08T18:00:00Z",
                    "2026-11-22T18:00:00Z",
                ]
            ),
        )
    )
    reports = availability.prepare_availability(fit, schedule, None, cutoff, path)
    assert len(reports) == 1
    assert reports[0]["return_latest"] == "2026-10-25"
    assert "not confirmed medical clearance" in reports[0]["uncertainty"]
    # An authoritative unknown update must replace the provider's point estimate.
    update = dict(
        current,
        season=2026,
        player="Starter",
        status="unknown",
        reported_at="2026-10-17T20:00:00Z",
        observed_at="2026-10-17T22:30:00Z",
        source_summary="No firm return date.",
    )
    path.write_text(json.dumps([update]))
    reports = availability.prepare_availability(fit, schedule, None, cutoff, path)
    assert reports[0]["status"] == "unknown"
    assert "No firm return date." in reports[0]["uncertainty"]
    assert (
        availability.return_share(reports[0], pd.Timestamp("2026-11-01T18:00:00Z"))
        is None
    )

    assert (
        availability.return_share(reports[0], pd.Timestamp("2026-10-18T17:00:00Z")) == 0
    )
    assert (
        availability.return_share(reports[0], pd.Timestamp("2026-10-25T17:00:00Z"))
        is None
    )
    assert (
        availability.return_share(
            dict(reports[0], status="season_out"), pd.Timestamp("2027-02-01T17:00:00Z")
        )
        == 0
    )
    update.update(absence_start="2026-10-17", minimum_games_out=4)
    path.write_text(json.dumps([update]))
    ir = availability.prepare_availability(fit, schedule, None, cutoff, path)[0]
    assert ir["unavailable_through"] == "2026-11-08"
    assert "recovery_scenario_start" not in ir
    assert availability.return_share(ir, pd.Timestamp("2026-11-08T18:00:00Z")) == 0
    assert availability.return_share(ir, pd.Timestamp("2026-11-22T18:00:00Z")) is None

    # Weekly refresh: a newer provider ETA replaces the curated unknown report,
    # while retaining its independently sourced IR minimum.
    injuries.loc[0, "source_updated_at"] = "2026-10-17T22:45:00Z"
    injuries.loc[0, "date_modified"] = "2026-10-17T22:50:00Z"
    injuries.loc[0, "estimated_return_date"] = "2026-11-01"
    revised = availability.prepare_availability(fit, schedule, None, cutoff, path)[0]
    assert revised["expected_return_date"] == "2026-11-01"
    assert availability.return_share(revised, pd.Timestamp("2026-11-01T18:00:00Z")) == 0
    injuries.loc[0, "estimated_return_date"] = "2026-11-22"
    revised = availability.prepare_availability(fit, schedule, None, cutoff, path)[0]
    assert revised["expected_return_date"] == "2026-11-22"
    assert revised["source_url"] == current["source_url"]
    injuries.loc[0, "report_status"] = "Active"
    assert availability.prepare_availability(fit, schedule, None, cutoff, path) == []

    # A failed/stale refresh must fail the scheduled command before replacing
    # the previous snapshot or reaching publication.
    import sys

    from backend import availability as injury_feed
    from backend.model import fit_week
    from backend.playoffs import forecast as playoff_forecast

    output = tmp_path / "previous.json"
    output.write_text("previous snapshot")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecast",
            "--season",
            "2026",
            "--require-current-injuries",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        injury_feed, "current_reports", lambda *args: (pd.DataFrame(), set())
    )
    monkeypatch.setattr(fit_week, "load_depth_charts", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        availability.store, "current_teams", lambda: pd.DataFrame({"team_abbr": ["TB"]})
    )
    with pytest.raises(ValueError, match="fresh injury coverage"):
        playoff_forecast.main()
    assert output.read_text() == "previous snapshot"
