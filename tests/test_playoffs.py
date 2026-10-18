"""Acceptance coverage for NFL qualification, reseeding, and forecast accounting."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.model.joint_scoring import DEFAULT_CONFIG, JointScoringFit
from backend.playoffs.forecast import forecast, play_bracket, validate
from backend.playoffs.seeding import Standings


def test_playoff_forecast_and_reseeding():
    teams = [f"T{i:02d}" for i in range(32)]
    conferences = ["AFC"] * 16 + ["NFC"] * 16
    divisions = [f"{conferences[i]} {i // 4}" for i in range(32)]
    rotation, rows = teams.copy(), []
    for week in range(1, 18):
        for i in range(16):
            rows.append(
                dict(
                    season=2026,
                    week=week,
                    game_type="REG",
                    game_id=f"2026_{week}_{i}",
                    home_team=rotation[i],
                    away_team=rotation[-i - 1],
                    location="Home",
                    home_score=np.nan,
                    away_score=np.nan,
                    gameday=(datetime(2026, 9, 1) + timedelta(weeks=week - 1))
                    .date()
                    .isoformat(),
                    gametime="13:00",
                )
            )
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    schedule = pd.DataFrame(rows)
    schedule.loc[0, ["home_score", "away_score"]] = [20, 20]
    schedule.loc[1, ["home_score", "away_score"]] = [24, 10]
    as_of = datetime(2026, 9, 30, tzinfo=UTC)
    fit = JointScoringFit(
        season=2026,
        week=1,
        as_of=as_of,
        teams=teams,
        offense_ppd=np.zeros(32),
        defense_ppd=np.zeros(32),
        pace=np.zeros(32),
        base_ppd=2,
        base_drives=10,
        hfa_ppd=0,
        parameter_covariance=np.diag([0.1] * 64 + [0.0]),
        score_residual_covariance=np.eye(2) * 60,
        config=DEFAULT_CONFIG,
    )
    metadata = pd.DataFrame(
        dict(
            team_abbr=teams,
            team_name=teams,
            team_conf=conferences,
            team_division=divisions,
        )
    )
    payload = forecast(fit, schedule, {}, metadata, as_of, draws=1000)
    validate(payload)
    by_id = {t["id"]: t for t in payload["teams"]}
    assert by_id["T00"]["ties"] == 1
    assert by_id["T01"]["wins"] == 1
    assert len(payload["games"]) == 270
    assert np.isclose(sum(t["expected_wins"] for t in payload["teams"]), 271)
    assert len(payload["path"]) == 13
    repeated = forecast(
        fit, schedule.sample(frac=1, random_state=7), {}, metadata, as_of, draws=1000
    )
    assert payload == repeated
    for conf in ("AFC", "NFC"):
        members = [t for t in payload["teams"] if t["conference"] == conf]
        for seed in range(7):
            assert np.isclose(sum(t["seed_probabilities"][seed] for t in members), 1)
        assert np.isclose(sum(t["super_bowl"] for t in members), 1)
    # All Wild Card road teams win. Seed 1 must host seed 7, not seed 5.
    probabilities = np.zeros((32, 32))
    fields = [list(range(7)), list(range(16, 23))]
    path = play_bracket(fields, probabilities)
    afc_divisional = [
        g for g in path if g["round"] == "Divisional" and g["conference"] == "AFC"
    ]
    assert [(g["home_seed"], g["away_seed"]) for g in afc_divisional] == [
        (1, 7),
        (5, 6),
    ]
    assert path[-1]["home"] < 16 <= path[-1]["away"]
    # Acceptance: a sourced absence keeps the backup early, mixes the window,
    # and restores the starter later, including postseason matchup generation.
    report = dict(
        team="T00",
        player="Example starter",
        backup="Example backup",
        gsis_id="example",
        status="window",
        return_earliest="2026-10-06",
        return_latest="2026-10-20",
        reported_at="2026-09-20T12:00:00Z",
        observed_at="2026-09-20T13:00:00Z",
        source_url="https://example.com/report",
        uncertainty="Reported estimate, not a guarantee.",
        backup_adjustment=0.0,
        starter_adjustment=4.0,
    )
    with_window = forecast(
        fit, schedule, {}, metadata, as_of, draws=1000, qb_availability=[report]
    )
    original_games = {g["id"]: g for g in payload["games"]}
    affected = [g for g in with_window["games"] if "T00" in (g["home"], g["away"])]
    for g in affected:
        baseline = original_games[g["id"]]["home_win_probability"]
        delta = g["home_win_probability"] - baseline
        if g["date"][:10] < "2026-10-13":
            assert delta == pytest.approx(0)
        elif g["home"] == "T00":
            assert delta > 0
        else:
            assert delta < 0
    windows = with_window["qb_availability"][0]
    assert {g["return_share"] for g in windows["games"]} == {0.0, 1.0}
    assert all(g["return_share"] == 1 for g in windows["postseason"])
    assert windows["source_url"] == report["source_url"]
    assert "starter_adjustment" not in windows
    assert (
        next(t for t in with_window["teams"] if t["id"] == "T00")["expected_wins"]
        > by_id["T00"]["expected_wins"]
    )
    late = dict(report, return_earliest="2027-01-01", return_latest="2027-01-20")
    late_result = forecast(
        fit, schedule, {}, metadata, as_of, draws=1000, qb_availability=[late]
    )
    stages = late_result["qb_availability"][0]["postseason"]
    assert stages[0]["return_share"] == 0
    assert stages[-1]["return_share"] == 1
    schedule.loc[2, "away_team"] = schedule.loc[2, "home_team"]
    with pytest.raises(ValueError):
        forecast(fit, schedule, {}, metadata, as_of, draws=1000)


def test_division_champions_and_tied_records():
    # Round robin: all clubs tie every game except T00 beats divisional T01.
    # Half-win credit and head-to-head must put T00 ahead of T01.
    teams = [f"T{i:02d}" for i in range(32)]
    conf = ["AFC"] * 16 + ["NFC"] * 16
    divisions = [f"{conf[i]} {i // 4}" for i in range(32)]
    home, away = np.triu_indices(32, 1)
    results = np.full(len(home), 0.5)
    results[(home == 0) & (away == 1)] = 1
    standings = Standings(teams, conf, divisions, home, away)
    fields, unresolved = standings.seed(results, np.random.default_rng(1))
    assert fields[0][0] == 0
    assert 1 not in fields[0][:4]
    for field in fields:
        assert len(set(field)) == 7
        assert len({divisions[t] for t in field[:4]}) == 4
    assert unresolved  # Deep ties are visible, never disguised as official seeds.
