"""Observed current injury reports, independent of the weekly game-status feed.

Snapshots are prospective evidence only. Their receipt time, not the date of
an older injury article, determines the earliest forecast allowed to use them.
"""

import json
import re
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from backend.config import RAW_DIR

INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
UNAVAILABLE = {"Out", "Doubtful", "Injured Reserve", "Physically Unable to Perform"}
STATUSES = UNAVAILABLE | {"Active", "Questionable", "Probable"}
MAX_AGE = timedelta(hours=24)


def snapshot_path(season):
    return RAW_DIR / "current_injuries" / f"{season}.json"


def parse_snapshot(snapshot, season, week, as_of, depth, teams, schedules=None):
    """Return report rows and teams with complete, timely identity coverage.

    An absent team or unresolved QB identity never counts as injury clearance.
    Return dates are estimates, so they never automatically clear an absence.
    """
    cutoff = pd.Timestamp(as_of)
    observed = pd.to_datetime(snapshot.get("observed_at"), utc=True, errors="coerce")
    payload = snapshot["payload"]
    published = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
    if (
        payload.get("status") != "success"
        or payload.get("season", {}).get("year") != season
        or pd.isna(observed)
        or pd.isna(published)
        or not cutoff - MAX_AGE <= observed <= cutoff
        or not cutoff - MAX_AGE <= published <= observed
    ):
        raise ValueError("Missing, stale, future, or wrong-season current injury feed")
    groups = payload.get("injuries")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Empty current injury coverage")
    names = dict(zip(teams.team_name, teams.team_abbr))
    mapping = {}
    if {"espn_id", "gsis_id", "dt"}.issubset(depth.columns):
        eligible = depth[pd.to_datetime(depth.dt, utc=True).le(cutoff)]
        for espn, group in eligible.dropna(subset=["espn_id", "gsis_id"]).groupby(
            "espn_id"
        ):
            ids = group.gsis_id.unique()
            if len(ids) == 1:
                mapping[str(int(espn))] = ids[0]
    rows, covered, blocked = [], set(), set()
    retracted = {}
    for group in groups:
        team = names.get(group.get("displayName"))
        entries = group.get("injuries")
        if team is None or not isinstance(entries, list) or team in covered:
            raise ValueError("Unrecognized or duplicate injury team coverage")
        covered.add(team)
        for entry in entries:
            athlete = entry["athlete"]
            position = athlete.get("position", {}).get("abbreviation")
            status = entry["status"]
            links = athlete.get("links", [])
            ids = {
                match.group(1)
                for link in links
                if (
                    match := re.search(
                        r"espn\.com/nfl/player/(?:[^ ]*/)?id/(\d+)",
                        link.get("href", ""),
                    )
                )
            }
            if athlete.get("id"):
                ids.add(str(athlete["id"]))
            gsis = mapping.get(next(iter(ids))) if len(ids) == 1 else None
            date = pd.to_datetime(entry.get("date"), utc=True, errors="coerce")
            if (
                not position
                or pd.isna(date)
                or date > observed
                or status not in STATUSES
            ):
                blocked.add(team)
                continue
            if position == "QB" and gsis is None:
                blocked.add(team)
            starter = None
            if position == "QB" and gsis is not None:
                from backend.starter_announcements import announcement, retraction

                starter = announcement(
                    entry, team, gsis, season, week, observed, depth, teams, schedules
                )
                if retraction(entry, week):
                    retracted[team, gsis] = date
            rows.append(
                dict(
                    season=season,
                    week=week,
                    team=team,
                    gsis_id=gsis,
                    position=position,
                    full_name=athlete.get("displayName"),
                    report_status="Out" if status in UNAVAILABLE else status,
                    current_status=status,
                    date_modified=observed.isoformat(),
                    source_updated_at=date.isoformat(),
                    source_url=INJURY_URL,
                    starter_announcement=starter,
                )
            )
    columns = [
        "season",
        "week",
        "team",
        "gsis_id",
        "position",
        "full_name",
        "report_status",
        "current_status",
        "date_modified",
        "source_updated_at",
        "source_url",
        "starter_announcement",
    ]
    # Later practice notes may replace a player's headline. Retain a previously
    # observed announcement only for its original game and forecast cutoff.
    if schedules is not None and {"season", "week", "game_id"}.issubset(schedules):
        game_ids = set(
            schedules.loc[
                schedules.season.eq(season) & schedules.week.eq(week), "game_id"
            ]
        )
        for saved in snapshot.get("announcements", []):
            evidence = saved["evidence"]
            if (
                saved["season"] == season
                and saved["week"] == week
                and evidence["game_id"] in game_ids
                and pd.Timestamp(evidence["observed_at"]) <= observed
                and pd.Timestamp(evidence["published_at"]) <= observed
                and retracted.get(
                    (saved["team"], evidence["gsis_id"]),
                    pd.Timestamp.min.tz_localize("UTC"),
                )
                < pd.Timestamp(evidence["published_at"])
            ):
                rows.append(
                    dict(
                        season=season,
                        week=week,
                        team=saved["team"],
                        gsis_id=evidence["gsis_id"],
                        position="QB",
                        full_name=evidence["player_name"],
                        report_status=None,
                        current_status=None,
                        date_modified=evidence["observed_at"],
                        source_updated_at=evidence["published_at"],
                        source_url=evidence["source_url"],
                        starter_announcement=evidence,
                    )
                )
    return pd.DataFrame(rows, columns=columns), covered - blocked


def refresh(season, depth, teams):
    """Fetch, validate and atomically retain a complete raw provider response."""
    response = requests.get(INJURY_URL, timeout=30)
    response.raise_for_status()
    snapshot = dict(observed_at=datetime.now(UTC).isoformat(), payload=response.json())
    parse_snapshot(snapshot, season, 0, snapshot["observed_at"], depth, teams)
    path = snapshot_path(season)
    schedule_path = RAW_DIR / "schedules.parquet"
    if schedule_path.exists():
        from backend.features.drives import kickoff_utc

        schedules = pd.read_parquet(schedule_path)
        required = {"season", "week", "game_id", "home_team", "away_team"}
        if required.issubset(schedules):
            dates = (
                pd.to_datetime(schedules.start_date, utc=True)
                if "start_date" in schedules
                else kickoff_utc(schedules)
            )
            upcoming = schedules[
                schedules.season.eq(season)
                & dates.gt(snapshot["observed_at"])
                & dates.le(pd.Timestamp(snapshot["observed_at"]) + timedelta(days=7))
            ]
            if path.exists():
                snapshot["announcements"] = json.loads(path.read_text()).get(
                    "announcements", []
                )
            retained = {}
            for week in upcoming.week.unique():
                reports, _ = parse_snapshot(
                    snapshot,
                    season,
                    int(week),
                    snapshot["observed_at"],
                    depth,
                    teams,
                    schedules,
                )
                for row in reports.itertuples():
                    evidence = row.starter_announcement
                    if not isinstance(evidence, dict):
                        continue
                    key = (
                        evidence["game_id"],
                        evidence["gsis_id"],
                        evidence["published_at"],
                    )
                    retained[key] = dict(
                        season=season, week=int(week), team=row.team, evidence=evidence
                    )
            snapshot["announcements"] = list(retained.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot))
    temporary.replace(path)
    from backend.source_inputs import archive_source

    archive_source(path, refresh=True)


def current_reports(season, week, as_of, depth, teams):
    path = snapshot_path(season)
    if not path.exists():
        return pd.DataFrame(), set()
    try:
        schedule_path = RAW_DIR / "schedules.parquet"
        schedules = pd.read_parquet(schedule_path) if schedule_path.exists() else None
        return parse_snapshot(
            json.loads(path.read_text()), season, week, as_of, depth, teams, schedules
        )
    except (ValueError, KeyError, TypeError):
        # Publication sees missing coverage; malformed data cannot clear a QB.
        return pd.DataFrame(), set()


def merge_reports(weekly, current):
    """A current snapshot supplements absences; conflicting reports stay Out.

    Weekly statuses are scoped to their actual week. Current reports are scoped
    to this forecast and reobserved on each refresh, including persistent IR.
    """
    if current.empty:
        return weekly
    if weekly is not None and "date_modified" not in weekly.columns:
        weekly = weekly.assign(untimestamped_weekly_report=True)
    return pd.concat([weekly, current], ignore_index=True)
