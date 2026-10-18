"""Reported QB return windows for forward-looking playoff projections."""

import json
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from backend.config import HISTORY_START_SEASON, REPO_ROOT
from backend.etl import store
from backend.model.fit_week import forecast_qb_context, load_injuries
from backend.model.projections import LayerConfig

RETURN_WINDOWS = REPO_ROOT / "overrides" / "qb_return_windows.json"


def load_return_reports(path, season, as_of):
    """Keep immutable revisions; a future observation cannot affect this run."""
    records = json.loads(Path(path).read_text())
    selected = {}
    for row in records:
        if row["season"] != season:
            continue
        reported = pd.Timestamp(row["reported_at"])
        observed = pd.Timestamp(row["observed_at"])
        if reported.tzinfo is None or observed.tzinfo is None or reported > observed:
            raise ValueError(
                "Return reports require ordered, timezone-aware timestamps"
            )
        if observed > as_of:
            continue
        source_url = urlparse(row["source_url"])
        if source_url.scheme not in {"http", "https"} or not source_url.netloc:
            raise ValueError("Return report requires an HTTP source URL")
        if row["status"] not in {"window", "unknown", "season_out", "cleared"}:
            raise ValueError("Invalid return report status")
        row = dict(row)
        if row["status"] == "window" and "out_weeks_min" in row:
            low, high = row["out_weeks_min"], row["out_weeks_max"]
            if not (
                isinstance(low, int)
                and isinstance(high, int)
                and 0 <= low <= high <= 52
            ):
                raise ValueError("Invalid reported absence in weeks")
            anchor = pd.Timestamp(row["absence_start"]).date()
            row["return_earliest"] = (anchor + timedelta(weeks=low)).isoformat()
            row["return_latest"] = (anchor + timedelta(weeks=high)).isoformat()
        if row["status"] in {"window", "cleared"}:
            lo, hi = (
                pd.Timestamp(row["return_earliest"]),
                pd.Timestamp(row["return_latest"]),
            )
            if pd.isna(lo) or pd.isna(hi) or lo > hi:
                raise ValueError("Invalid reported return window")
        key = row["team"]
        order = (reported, observed)
        if key not in selected or order > selected[key][0]:
            selected[key] = (order, row)
    return [v[1] for _, v in sorted(selected.items())]


def return_share(report, kickoff):
    """Deterministic reported midpoint; None means no reported return ETA."""
    if report["status"] == "season_out":
        return 0.0
    date = pd.Timestamp(kickoff).tz_convert("America/New_York").date()
    if (
        report.get("unavailable_through")
        and date <= pd.Timestamp(report["unavailable_through"]).date()
    ):
        return 0.0
    if report["status"] in {"unknown", "overdue"}:
        through = report.get("unavailable_through")
        return 0.0 if through and date <= pd.Timestamp(through).date() else None
    lo = pd.Timestamp(report["return_earliest"]).date()
    hi = pd.Timestamp(report["return_latest"]).date()
    midpoint = lo + timedelta(days=((hi - lo).days + 1) // 2)
    return float(date >= midpoint)


def prepare_availability(fit, schedule, charts, as_of, reports_path=RETURN_WINDOWS):
    """Value both QBs from the existing cutoff model; no new coefficients."""
    week = (
        int(schedule.loc[schedule.home_score.isna(), "week"].min())
        if schedule.home_score.isna().any()
        else 19
    )
    injuries = load_injuries(fit.season, week, as_of)
    qb_games = store.qb_games(list(range(HISTORY_START_SEASON, fit.season + 1)))
    context, expected, weights = forecast_qb_context(
        fit.season,
        week,
        qb_games,
        charts,
        fit.config,
        LayerConfig(),
        as_of,
        injuries,
    )
    reports = load_return_reports(reports_path, fit.season, as_of)
    manual = {r["team"]: r for r in reports}
    # Provider point estimates remain explicitly labelled as point estimates.
    if "estimated_return_date" in injuries:
        latest = (
            injuries.dropna(subset=["date_modified"])
            .sort_values("date_modified")
            .drop_duplicates(["team", "gsis_id"], keep="last")
        )
        for row in latest.itertuples():
            if row.position != "QB":
                continue
            previous = manual.get(row.team)
            if previous:
                if row.gsis_id != previous["gsis_id"]:
                    continue
                if pd.Timestamp(row.source_updated_at) <= pd.Timestamp(
                    previous["reported_at"]
                ):
                    continue
                # Re-observing old news is not an update. A genuinely newer
                # provider report can revise the ETA or clear the absence.
                reports.remove(previous)
            if row.report_status not in {"Out", "Doubtful"}:
                continue
            date = row.estimated_return_date
            has_date = isinstance(date, str) and bool(date)
            mixture = context.mixtures.get(row.team, {})
            if not previous and (
                not mixture or max(mixture, key=mixture.get) != row.gsis_id
            ):
                continue
            if expected.get(row.team) == row.gsis_id:
                continue
            reports.append(
                dict(
                    **{
                        k: previous[k]
                        for k in (
                            "absence_start",
                            "minimum_games_out",
                            "unavailable_through",
                        )
                        if previous and k in previous
                    },
                    season=fit.season,
                    team=row.team,
                    gsis_id=row.gsis_id,
                    player=row.full_name,
                    status="window" if has_date else "unknown",
                    return_earliest=date if has_date else None,
                    return_latest=date if has_date else None,
                    reported_at=row.source_updated_at,
                    observed_at=row.date_modified,
                    source_url=row.source_url,
                    estimate="Provider estimated return date"
                    if has_date
                    else "Return date unknown",
                )
            )
    names = (
        qb_games.drop_duplicates("passer_player_id", keep="last")
        .set_index("passer_player_id")
        .passer.to_dict()
    )
    prepared = []
    for source in reports:
        r = dict(source)
        team, player = r["team"], r["gsis_id"]
        if team not in fit.teams or player not in context.strengths:
            raise ValueError(f"Return report has unknown team or QB: {team} {player}")
        backup = r.get("backup_gsis_id") or expected.get(team)
        if backup is None or (backup == player and r["status"] != "cleared"):
            raise ValueError(f"Return report needs an identified backup for {team}")
        r["backup"] = r.get("backup_name") or names.get(backup, backup)
        r["backup_adjustment"] = context.adjustment(team, backup, weights[team])
        r["starter_adjustment"] = context.adjustment(team, player, weights[team])
        if (
            r["status"] == "window"
            and pd.Timestamp(r["return_latest"]).date()
            < as_of.astimezone(ZoneInfo("America/New_York")).date()
        ):
            r["status"] = "overdue"
        if r["status"] in {"unknown", "overdue"} or "minimum_games_out" in r:
            games = schedule[
                schedule.home_team.eq(team) | schedule.away_team.eq(team)
            ].sort_values("start_date")
            dates = (
                pd.to_datetime(games.start_date, utc=True)
                .dt.tz_convert("America/New_York")
                .dt.date
            )
            if "minimum_games_out" in r:
                count = r["minimum_games_out"]
                if not isinstance(count, int) or count < 1:
                    raise ValueError("Invalid minimum games out")
                missed = dates[dates >= pd.Timestamp(r["absence_start"]).date()]
                if len(missed) < count:
                    raise ValueError("Insufficient schedule to enforce minimum absence")
                r["unavailable_through"] = missed.iloc[count - 1].isoformat()
            elif not r.get("unavailable_through"):
                future = dates[
                    dates >= as_of.astimezone(ZoneInfo("America/New_York")).date()
                ]
                if future.empty:
                    raise ValueError(
                        "Unknown return needs a confirmed absence boundary"
                    )
                r["unavailable_through"] = future.iloc[0].isoformat()
        if r["status"] in {"window", "cleared"}:
            lo = pd.Timestamp(r["return_earliest"]).date()
            hi = pd.Timestamp(r["return_latest"]).date()
            r["expected_return_date"] = (
                lo + timedelta(days=((hi - lo).days + 1) // 2)
            ).isoformat()
        r["uncertainty"] = (
            f"Working ETA: {r['expected_return_date']}, the midpoint of the reported "
            "range. Backup before that date; starter from that date. This is an "
            "estimate, not clearance. The next forecast uses the latest report."
            if r["status"] in {"window", "cleared"}
            else f"Backup through {r['unavailable_through']}. Later availability is "
            "unknown: no recovery odds or return date invented. Later projections "
            "use the baseline QB assumptions and remain provisional pending an ETA."
            if r["status"] in {"unknown", "overdue"}
            else "Reported out for the season; backup used for remaining games."
        )
        if r.get("source_summary"):
            r["uncertainty"] = r["source_summary"] + " " + r["uncertainty"]
        if r.get("estimate") == "Provider estimated return date":
            r["uncertainty"] = (
                "Provider point estimate, not confirmed medical clearance. "
                + r["uncertainty"]
            )
        prepared.append(r)
    if len({r["team"] for r in prepared}) != len(prepared):
        raise ValueError("Multiple injured QBs require an explicit team return report")
    return prepared


def project_with_availability(
    fit, slate, as_of, names, adjustments, reports, market=None
):
    """Apply reported ETAs without inventing recovery probabilities."""
    from backend.model.projections import assemble_projections
    from backend.model.season_wins import win_probabilities

    by_team = {r["team"]: r for r in reports}
    scenario = dict(adjustments)
    for game in slate.itertuples():
        pair = list(scenario.get(game.game_id, (0.0, 0.0)))
        for side, team in enumerate((game.home_team, game.away_team)):
            report = by_team.get(team)
            if report is not None:
                returned = return_share(report, game.start_date)
                if returned is not None:
                    pair[side] = report[
                        "starter_adjustment" if returned else "backup_adjustment"
                    ]
        scenario[game.game_id] = tuple(pair)
    projections = assemble_projections(fit, slate, as_of, names, scenario, market)
    return projections, win_probabilities(fit, projections)
