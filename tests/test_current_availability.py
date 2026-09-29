"""Week 4 regression: early injury news must survive lagging weekly reports."""

import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend import availability, source_inputs
from backend.etl import store
from backend.features import qb
from backend.model.fit_week import compute_qb_adjustments, load_injuries
from backend.model.joint_scoring import DEFAULT_CONFIG
from backend.model.market_blend import MARGINS
from backend.model.projections import LayerConfig
from backend.recommendations import build_recommendations


def test_current_feed_corrects_lagging_weekly_report_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    now = pd.Timestamp.now(tz="UTC")
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    for module in (availability, source_inputs, store):
        monkeypatch.setattr(module, "RAW_DIR", raw)
    monkeypatch.setattr(source_inputs, "PROCESSED_DIR", processed)
    monkeypatch.setattr(source_inputs, "ARCHIVE", processed / "source_archive")
    empty_overrides = pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"])
    monkeypatch.setattr(qb, "_overrides", lambda: empty_overrides)
    monkeypatch.setattr(source_inputs, "_overrides", lambda: empty_overrides)
    teams = pd.DataFrame(
        {
            "team_abbr": ["TB", "LA"],
            "team_name": ["Tampa Bay Buccaneers", "Los Angeles Rams"],
        }
    )
    depth = pd.DataFrame(
        [
            ("TB", "00-0034855", 3052587, "QB", 1),
            ("TB", "chart-backup", 3, "QB", 2),
            ("TB", "00-0041251", 4596472, "QB", 3),
            ("LA", "stafford", 1, "QB", 1),
            ("LA", "garrett", 2, "DE", 1),
        ],
        columns=["team", "gsis_id", "espn_id", "pos_abb", "pos_rank"],
    )
    depth["dt"] = (now - timedelta(hours=1)).isoformat()
    depth["player_name"] = [
        "Baker Mayfield",
        "Chart Backup",
        "Jalon Daniels",
        "Matthew Stafford",
        "Myles Garrett",
    ]
    schedules = pd.DataFrame(
        [
            dict(
                season=2026,
                week=3,
                game_id="old",
                home_team="TB",
                away_team="LA",
                start_date=now - timedelta(days=2),
            ),
            dict(
                season=2026,
                week=4,
                game_id="new",
                home_team="TB",
                away_team="LA",
                start_date=now + timedelta(days=5),
            ),
        ]
    )
    weekly = pd.DataFrame(
        [dict(season=2026, week=3, team="TB", gsis_id="00-0034855", report_status=None)]
    )
    for frame, path in [
        (teams, raw / "teams.parquet"),
        (depth, raw / "depth_charts/2026.parquet"),
        (weekly, raw / "injuries/2026.parquet"),
        (schedules, raw / "schedules.parquet"),
    ]:
        store.write_parquet(frame, path)
    index = pd.DataFrame(
        [
            dict(
                game_id="old",
                season=2026,
                model_week=3,
                start_date=now - timedelta(days=2),
            )
        ]
    )
    logs = pd.DataFrame(
        [
            dict(
                game_id="old",
                team="TB",
                passer_player_id="00-0034855",
                dropbacks=40,
                epa=10,
                started=True,
            )
        ]
    )
    monkeypatch.setattr(store, "game_index", lambda seasons: index)
    slate = pd.DataFrame([dict(game_id="new", home_team="TB", away_team="LA")])
    frame = pd.DataFrame(
        [
            dict(
                game_id="new",
                season=2026,
                week=4,
                as_of=now,
                home_team_abbr="TB",
                away_team_abbr="LA",
                home_team="Tampa",
                away_team="Rams",
                start_date=now + timedelta(days=5),
                model_version="test",
                home_margin=4.0,
                pure_home_margin=4.0,
                model_total=45.0,
                margin_sd=13.0,
                total_sd=14.0,
                degrees_of_freedom=7.0,
            )
        ]
    )
    before = source_inputs.attach_sources(frame)
    assert json.loads(before.data_flags.iloc[0])["home_expected_qb"] == "00-0034855"
    assert before.home_missing_input_count.iloc[0] == 1

    def entry(name, identity, position, status):
        return dict(
            status=status,
            date=now.isoformat(),
            athlete=dict(
                displayName=name,
                position={"abbreviation": position},
                links=[
                    {
                        "href": f"https://www.espn.com/nfl/player/_/id/{identity}/name",
                        "rel": ["playercard"],
                    }
                ],
            ),
            details={"returnDate": "2026-09-01"},
        )

    payload = dict(
        status="success",
        timestamp=now.isoformat(),
        season={"year": 2026},
        injuries=[
            dict(
                displayName="Tampa Bay Buccaneers",
                injuries=[
                    entry("Baker Mayfield", 3052587, "QB", "Out"),
                    entry("Jalon Daniels", 4596472, "QB", "Active"),
                ],
            ),
            dict(
                displayName="Los Angeles Rams",
                injuries=[entry("Myles Garrett", 2, "DE", "Injured Reserve")],
            ),
        ],
    )
    daniels = payload["injuries"][0]["injuries"][1]
    daniels["shortComment"] = (
        "Head coach Todd Bowles said Monday that Daniels will step in as the "
        "Buccaneers' new starting quarterback while Baker Mayfield is out."
    )
    daniels["longComment"] = "The Buccaneers will host the Rams this Sunday."
    monkeypatch.setattr(
        availability.requests,
        "get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )
    availability.refresh(2026, depth, teams)
    receipt = source_inputs.receipt_for(availability.snapshot_path(2026))
    frozen = json.loads((processed / receipt["archive"]).read_text())
    cutoff = pd.Timestamp.now(tz="UTC")
    injuries = load_injuries(2026, 4, cutoff)
    assert qb.ruled_out(injuries, 2026, 4, cutoff) == {"00-0034855", "garrett"}
    starter = qb.expected_starters(
        logs, index, depth, 2026, 4, as_of=cutoff, injuries=injuries
    )
    assert starter["TB"] == "00-0041251"
    adj = compute_qb_adjustments(
        2026,
        4,
        slate,
        logs,
        depth,
        DEFAULT_CONFIG,
        LayerConfig(),
        as_of=cutoff,
        injuries=injuries,
    )
    assert adj["new"][0] < 0
    after = source_inputs.attach_sources(frame.assign(as_of=cutoff))
    flags = json.loads(after.data_flags.iloc[0])
    assert flags["home_expected_qb"] == "00-0041251"
    assert flags["home_qb_selection"]["basis"] == "announced_starter"
    assert flags["home_qb_selection"]["game_id"] == "new"
    assert flags["away_reported_absences"] == [
        dict(player="Myles Garrett", position="DE", status="Injured Reserve")
    ]
    assert after.home_missing_input_count.iloc[0] == 0
    assert pd.Timestamp(flags["home_qb_source_at"]) > pd.Timestamp(depth.dt.iloc[0])

    # Speculation, negation, wrong-week news and old-game announcements must fall back.
    for headline, date in [
        ("Coach said Daniels could start against the Rams.", now),
        ("Coach said he was unsure whether Daniels will start against the Rams.", now),
        ("Coach said Daniels will start against the Rams in Week 3.", now),
        ("Coach said Daniels will start his rehab against the Rams.", now),
        (daniels["shortComment"], now - timedelta(days=3)),
    ]:
        revised = copy.deepcopy(frozen)
        revised["announcements"] = []
        note = revised["payload"]["injuries"][0]["injuries"][1]
        note.update(shortComment=headline, date=date.isoformat())
        parsed, _ = availability.parse_snapshot(
            revised, 2026, 4, cutoff, depth, teams, schedules
        )
        selected, evidence = qb.lineup_starters(
            depth, 2026, 4, cutoff, parsed, empty_overrides
        )
        assert selected["TB"] == "chart-backup"
        assert evidence["TB"]["basis"] == "depth_chart_fallback"

    # A later observation cannot enter the old forecast or expire by return estimate.
    old, coverage = availability.current_reports(2026, 4, now, depth, teams)
    assert old.empty and not coverage
    week5, _ = availability.parse_snapshot(frozen, 2026, 5, cutoff, depth, teams)
    assert "00-0034855" in qb.ruled_out(week5, 2026, 5, cutoff)
    # Unresolved QB IDs remove coverage, rather than silently ignoring the absence.
    unresolved = copy.deepcopy(frozen)
    unresolved["payload"]["injuries"][0]["injuries"][0]["athlete"]["links"] = []
    _, coverage = availability.parse_snapshot(unresolved, 2026, 4, cutoff, depth, teams)
    assert "TB" not in coverage

    # Routine later news must not erase a previously observed announcement.
    daniels["shortComment"] = "Daniels practiced fully Tuesday."
    availability.refresh(2026, depth, teams)
    later = pd.Timestamp.now(tz="UTC")
    retained = load_injuries(2026, 4, later)
    selected, evidence = qb.lineup_starters(
        depth, 2026, 4, later, retained, empty_overrides
    )
    assert selected["TB"] == "00-0041251"
    assert evidence["TB"]["basis"] == "announced_starter"

    revoked = copy.deepcopy(frozen)
    revoked["payload"]["injuries"][0]["injuries"][1]["shortComment"] = (
        "Coach said Daniels will not start against the Rams."
    )
    parsed, _ = availability.parse_snapshot(
        revoked, 2026, 4, later, depth, teams, schedules
    )
    selected, _ = qb.lineup_starters(depth, 2026, 4, later, parsed, empty_overrides)
    assert selected["TB"] == "chart-backup"

    payload["injuries"][0]["injuries"][0]["status"] = "Active"
    availability.refresh(2026, depth, teams)
    frozen_rows, _ = availability.parse_snapshot(frozen, 2026, 4, cutoff, depth, teams)
    assert "00-0034855" in qb.ruled_out(frozen_rows, 2026, 4, cutoff)
    fresh = load_injuries(2026, 4, pd.Timestamp.now(tz="UTC"))
    assert "00-0034855" not in qb.ruled_out(fresh, 2026, 4)
    selected, _ = qb.lineup_starters(
        depth, 2026, 4, pd.Timestamp.now(tz="UTC"), fresh, empty_overrides
    )
    assert selected["TB"] == "00-0034855"

    # An expired feed is missing evidence, even if its archived bytes still exist.
    stale_cutoff = cutoff + timedelta(hours=25)
    stale = source_inputs.attach_sources(frame.assign(as_of=stale_cutoff))
    assert stale.home_missing_input_count.iloc[0] == 1
    decisions = build_recommendations(
        stale, pd.DataFrame(), np.ones(len(MARGINS)), decision_at=stale_cutoff
    )
    assert decisions.status.eq("no_play").all()
    assert decisions.reason.eq("missing_model_inputs").all()
