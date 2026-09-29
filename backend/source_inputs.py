"""Receipt-time source archives attached to prospective recommendation inputs.

Source dates and local file modification times cannot establish when we knew
something. New receipts retain the bytes and a timestamp; old files without
receipts remain missing coverage rather than receiving invented timestamps.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from backend import availability
from backend.config import PROCESSED_DIR, RAW_DIR, STATIC_DIR
from backend.features.qb import (
    OVERRIDES_PATH,
    _overrides,
    lineup_starters,
    ruled_out,
)

ARCHIVE = PROCESSED_DIR / "source_archive"


def archive_source(path: Path, *, refresh=False) -> dict:
    content = path.read_bytes() if path.exists() else b""
    digest = hashlib.sha256(content).hexdigest()
    key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    directory = ARCHIVE / key
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.json"
    if not refresh:
        previous = receipt_for(path)
        if previous is not None:
            return previous
    now = datetime.now(UTC)
    name = now.strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = directory / f"{name}.source{path.suffix or '.bin'}"
    snapshot.write_bytes(content)
    receipt = dict(
        observed_at=now.isoformat(),
        sha256=digest,
        archive=str(snapshot.relative_to(PROCESSED_DIR)),
        present=path.exists(),
    )
    (directory / f"{name}.json").write_text(json.dumps(receipt))
    temporary = latest.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt))
    temporary.replace(latest)
    return receipt


def receipt_for(path: Path) -> dict | None:
    key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    latest = ARCHIVE / key / "latest.json"
    if not latest.exists():
        return None
    try:
        receipt = json.loads(latest.read_text())
        content = path.read_bytes() if path.exists() else b""
        snapshot = (PROCESSED_DIR / receipt["archive"]).resolve()
        if not snapshot.is_relative_to(ARCHIVE.resolve()):
            return None
        digest = hashlib.sha256(content).hexdigest()
        if (
            receipt["sha256"] != digest
            or receipt.get("present") is not path.exists()
            or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return receipt


def capture_static_inputs():
    for path in (
        OVERRIDES_PATH,
        STATIC_DIR / "win_totals.csv",
        STATIC_DIR / "win_total_sources.json",
        STATIC_DIR / "preseason_qbs.csv",
    ):
        archive_source(path)


def source_reason(value, forecast, now):
    try:
        sources = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return "missing_source_archive"
    if not isinstance(sources, dict):
        return "missing_source_archive"
    current = sources.get("current_injuries")
    if isinstance(current, dict) and current.get("present"):
        observed = pd.to_datetime(current.get("observed_at"), utc=True, errors="coerce")
        if pd.isna(observed) or observed > forecast:
            return "source_after_forecast"
        if now - observed > availability.MAX_AGE or not current.get("sha256"):
            return "stale_availability_inputs"
    for name in (
        "schedule",
        "depth_charts",
        "qb_overrides",
        "win_totals",
        "win_total_sources",
        "preseason_qbs",
    ):
        receipt = sources.get(name)
        if not isinstance(receipt, dict) or not receipt.get("sha256"):
            return "missing_source_archive"
        observed = pd.to_datetime(receipt.get("observed_at"), utc=True, errors="coerce")
        if pd.isna(observed) or observed > forecast:
            return "source_after_forecast"
        age = now - observed
        if (name == "schedule" and age > timedelta(hours=24)) or (
            name == "depth_charts" and age > timedelta(hours=48)
        ):
            return "stale_availability_inputs"
    return None


def attach_sources(frame):
    """Only receipts for the exact files read by this run qualify."""
    rows = frame.copy()
    if rows.empty:
        return rows
    season, week = int(rows.iloc[0].season), int(rows.iloc[0].week)
    sources = {
        name: receipt_for(path)
        for name, path in {
            "schedule": RAW_DIR / "schedules.parquet",
            "depth_charts": RAW_DIR / "depth_charts" / f"{season}.parquet",
            "injuries": RAW_DIR / "injuries" / f"{season}.parquet",
            "current_injuries": availability.snapshot_path(season),
            "qb_overrides": OVERRIDES_PATH,
            "win_totals": STATIC_DIR / "win_totals.csv",
            "win_total_sources": STATIC_DIR / "win_total_sources.json",
            "preseason_qbs": STATIC_DIR / "preseason_qbs.csv",
        }.items()
    }
    depth_path = RAW_DIR / "depth_charts" / f"{season}.parquet"
    depth = pd.read_parquet(depth_path) if depth_path.exists() else pd.DataFrame()
    injury_path = RAW_DIR / "injuries" / f"{season}.parquet"
    weekly = pd.read_parquet(injury_path) if injury_path.exists() else pd.DataFrame()
    teams_path = RAW_DIR / "teams.parquet"
    teams = (
        pd.read_parquet(teams_path)
        if teams_path.exists()
        else pd.DataFrame(columns=["team_name", "team_abbr"])
    )
    overrides = _overrides()
    overrides = overrides[overrides.season.eq(season) & overrides.week.eq(week)]
    flags = []
    missing = {"home": [], "away": []}
    availability_by_cutoff = {}
    for row in rows.itertuples():
        forecast = pd.to_datetime(row.as_of, utc=True)
        if forecast not in availability_by_cutoff:
            current, covered = availability.current_reports(
                season, week, forecast, depth, teams
            )
            injuries = availability.merge_reports(weekly, current)
            unavailable = ruled_out(injuries, season, week, forecast)
            selected, evidence = lineup_starters(
                depth, season, week, forecast, injuries, overrides
            )
            availability_by_cutoff[forecast] = (
                current,
                covered,
                unavailable,
                selected.to_dict(),
                evidence,
            )
        current, covered, unavailable, expected, evidence = availability_by_cutoff[
            forecast
        ]
        current_receipt = sources["current_injuries"] or {}
        if not current_receipt.get("present"):
            covered = set()
        data = {
            "availability_basis": (
                "announcements, injuries, overrides, depth-chart fallback"
            ),
            "non_qb_injuries": "not modeled; no comprehensive injury clearance",
        }
        for side in ("home", "away"):
            team = getattr(row, f"{side}_team_abbr")
            qb = expected.get(team)
            selection = evidence.get(team, dict(basis="missing"))
            data[f"{side}_qb_selection"] = selection
            data[f"{side}_injury_coverage"] = (
                "current_snapshot" if team in covered else "missing_current_snapshot"
            )
            absent = (
                current[current.team.eq(team) & current.report_status.eq("Out")]
                if not current.empty
                else current
            )
            data[f"{side}_reported_absences"] = [
                dict(player=r.full_name, position=r.position, status=r.current_status)
                for r in absent.itertuples()
            ]
            data[f"{side}_expected_qb"] = qb if pd.notna(qb) else None
            if selection["basis"] == "announced_starter":
                qb_at = pd.Timestamp(selection["observed_at"])
            elif team in set(overrides.team_abbr):
                qb_at = pd.to_datetime(
                    (sources["qb_overrides"] or {}).get("observed_at"),
                    utc=True,
                    errors="coerce",
                )
            elif {"team", "dt", "pos_abb"}.issubset(depth.columns):
                dates = pd.to_datetime(depth.dt, utc=True, errors="coerce")
                qb_at = dates[
                    depth.team.eq(team)
                    & depth.pos_abb.eq("QB")
                    & depth.gsis_id.eq(qb)
                    & dates.le(forecast)
                ].max()
            else:
                qb_at = pd.NaT
            stale_qb = pd.isna(qb_at) or not (
                forecast - timedelta(hours=48) <= qb_at <= forecast
            )
            # A newly observed absence can select QB2 from the same chart.
            # Preserve the chart freshness gate while dating that new evidence.
            if team in covered and not stale_qb:
                qb_at = max(qb_at, pd.Timestamp(current_receipt["observed_at"]))
            data[f"{side}_qb_source_at"] = None if pd.isna(qb_at) else qb_at.isoformat()
            missing[side].append(
                int(
                    qb is None
                    or pd.isna(qb)
                    or stale_qb
                    or qb in unavailable
                    or team not in covered
                )
            )
        flags.append(json.dumps(data))
    rows["source_timestamps"] = json.dumps(sources)
    rows["data_flags"] = flags
    for side in missing:
        rows[f"{side}_missing_input_count"] = missing[side]
    return rows
