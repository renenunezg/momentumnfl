"""Prospective receipt boundaries and restoration, without historical proxies."""

import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

import pandas as pd
import pytest

from backend import forecast_archive as archive


def test_json_source_receipt_preserves_sportsbook_payload(tmp_path, monkeypatch):
    from backend import source_inputs

    processed = tmp_path / "processed"
    monkeypatch.setattr(source_inputs, "PROCESSED_DIR", processed)
    monkeypatch.setattr(source_inputs, "ARCHIVE", processed / "source_archive")
    source = tmp_path / "win_total_sources.json"
    payload = '{"2026":{"source":"BetMGM","date":"2026-09-01"}}'
    source.write_text(payload)
    receipt = source_inputs.archive_source(source)
    assert (processed / receipt["archive"]).read_text() == payload
    assert source_inputs.receipt_for(source) == receipt
    assert source_inputs.archive_source(source) == receipt


def test_replay_restores_frozen_inputs_and_refuses_missing_late_or_corrupt_bytes(
    tmp_path, monkeypatch
):
    root = tmp_path / "working"
    processed = root / "backend/data/processed"
    raw = root / "backend/data/raw"
    raw.mkdir(parents=True)
    (raw / "depth_charts").mkdir()
    pd.DataFrame({"revision": [1]}).to_parquet(raw / "schedules.parquet")
    pd.DataFrame(
        {
            "team": ["H", "A"],
            "gsis_id": ["qh", "qa"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1, 1],
            "dt": ["2026-09-01T00:00:00Z"] * 2,
        }
    ).to_parquet(raw / "depth_charts/2026.parquet")
    monkeypatch.setattr(archive, "REPO_ROOT", root)
    monkeypatch.setattr(archive, "PROCESSED_DIR", processed)
    monkeypatch.setattr(archive, "ARCHIVE", processed / "forecast_archive")
    monkeypatch.setattr(
        archive,
        "receipt_for",
        lambda p: {
            "observed_at": "2026-09-01T00:00:00Z",
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        },
    )
    args = Namespace(season=2026, week=1)
    archive.prepare(args)
    expected = pd.DataFrame(
        [
            dict(
                game_id="game",
                season=2026,
                week=1,
                model_version="test",
                home_team_abbr="H",
                away_team_abbr="A",
                home_margin=3.25,
                as_of=args.forecast_cutoff.isoformat(),
            )
        ]
    )
    archive.finish(args, expected)
    path = next((archive.ARCHIVE / "runs").glob("*.json"))
    frozen = json.loads(path.read_text())
    pd.DataFrame({"revision": [99]}).to_parquet(raw / "schedules.parquet")
    (root / "overrides").mkdir()
    (root / "overrides/qb_starters.csv").write_text("later override")

    def run(*args, cwd, **kwargs):
        assert (
            pd.read_parquet(cwd / "backend/data/raw/schedules.parquet").revision[0] == 1
        )
        assert not (cwd / "overrides/qb_starters.csv").exists()
        target = cwd / "backend/data/processed/projections/2026_01.parquet"
        target.parent.mkdir(parents=True)
        expected.to_parquet(target, index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(archive.subprocess, "run", run)
    assert archive.replay(path)["games_reproduced"] == 1
    source = "backend/data/raw/schedules.parquet"
    frozen["files"][source]["source_receipt"]["observed_at"] = "2200-01-01T00:00:00Z"
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="pre-cutoff source coverage"):
        archive.replay(path)
    frozen["files"][source]["source_receipt"] = None
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="pre-cutoff source coverage"):
        archive.replay(path)
    frozen["files"][source]["source_receipt"] = {
        "observed_at": "2026-09-01T00:00:00Z",
        "sha256": frozen["files"][source]["sha256"],
    }
    path.write_text(json.dumps(frozen))
    (archive.ARCHIVE / "objects" / frozen["files"][source]["sha256"]).write_bytes(
        b"bad"
    )
    with pytest.raises(ValueError, match="Corrupt archived input"):
        archive.replay(path)


def test_prospective_horizons_keep_versions_separate_and_expose_missing_coverage():
    from backend.prospective import summarize

    rows = []
    for number, version, margin in ((1, "early", 1.0), (2, "closing", 7.0)):
        rows.append(
            dict(
                game_id="g",
                snapshot_id=number,
                model_version=version,
                as_of=f"2026-09-0{number}T00:00:00Z",
                recorded_at=f"2026-09-0{number}T00:01:00Z",
                start_date="2026-09-10T00:00:00Z",
                pure_home_margin=margin,
                home_margin=margin,
                margin_sd=12.0,
                degrees_of_freedom=7.0,
            )
        )
    results = pd.DataFrame(
        [dict(game_id="g", home_points=24, away_points=17, closing_spread=-3.0)]
    )
    report = summarize(pd.DataFrame(rows), results)
    metrics = {r["horizon"]: r for r in report["metrics"]}
    assert metrics["early_week"]["model_version"] == "early"
    assert metrics["closing"]["model_version"] == "closing"
    assert metrics["early_week"]["pure_mae"] == 6
    assert metrics["closing"]["pure_mae"] == 0
    assert metrics["closing"]["missing_source_archives"] == 1
    assert metrics["closing"]["pure_brier"] < metrics["early_week"]["pure_brier"]
    assert sum(r["games"] for r in report["calibration"]) == 4
