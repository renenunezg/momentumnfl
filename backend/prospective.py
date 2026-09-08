"""Read-only prospective accuracy using immutable database forecast receipts."""

import json

import numpy as np
import pandas as pd
from scipy.stats import t

from backend.forecast_archive import ARCHIVE
from backend.model.distributions import student_t_scale


def summarize(snapshots, results, archive_keys=frozenset()):
    """First available and last pregame forecasts are separate cohorts.

    A closing forecast is the latest recorded pregame forecast, not a claim
    that a job executed at kickoff. Lead time is reported explicitly.
    """
    if snapshots.empty:
        return {"status": "no_frozen_forecasts", "metrics": [], "calibration": []}
    frames = []
    snapshots = snapshots.copy()
    for column in ("as_of", "recorded_at", "start_date"):
        snapshots[column] = pd.to_datetime(snapshots[column], utc=True)
    if "start_date" in results:
        actual_kickoffs = pd.to_datetime(
            results.set_index("game_id").start_date, utc=True
        )
        actual = snapshots.game_id.map(actual_kickoffs)
        snapshots = snapshots[actual.isna() | snapshots.recorded_at.lt(actual)]
    snapshots = snapshots[
        snapshots.recorded_at.lt(snapshots.start_date)
        & snapshots.as_of.le(snapshots.recorded_at)
    ].sort_values(["recorded_at", "snapshot_id"])
    for horizon, keep in (("early_week", "first"), ("closing", "last")):
        selected = snapshots.drop_duplicates("game_id", keep=keep).copy()
        selected["horizon"] = horizon
        selected["source_archive_available"] = [
            (r.game_id, r.as_of.isoformat(), r.model_version) in archive_keys
            for r in selected.itertuples()
        ]
        frames.append(selected)
    frame = pd.concat(frames).merge(
        results[["game_id", "home_points", "away_points", "closing_spread"]],
        on="game_id",
        how="left",
    )
    frame["actual_margin"] = frame.home_points - frame.away_points
    frame["lead_hours"] = (
        frame.start_date - frame.recorded_at
    ).dt.total_seconds() / 3600
    metrics, calibration = [], []
    for (version, horizon), group in frame.groupby(["model_version", "horizon"]):
        complete = group[group.actual_margin.notna()]
        paired = complete.dropna(
            subset=["pure_home_margin", "home_margin", "closing_spread"]
        )
        record = dict(
            model_version=version,
            horizon=horizon,
            games=len(group),
            completed=len(complete),
            paired_games=len(paired),
            missing_source_archives=int((~group.source_archive_available).sum()),
            minimum_lead_hours=float(group.lead_hours.min()),
        )
        for label, column in (("pure", "pure_home_margin"), ("blended", "home_margin")):
            record[f"{label}_mae"] = (
                float((paired[column] - paired.actual_margin).abs().mean())
                if len(paired)
                else None
            )
            eligible = complete.dropna(
                subset=[column, "margin_sd", "degrees_of_freedom"]
            )
            eligible = eligible[eligible.actual_margin.ne(0)]
            probabilities, outcomes = [], []
            for row in eligible.itertuples():
                probability = float(
                    t.cdf(
                        getattr(row, column)
                        / student_t_scale(row.margin_sd, row.degrees_of_freedom),
                        row.degrees_of_freedom,
                    )
                )
                probabilities.append(probability)
                outcomes.append(float(row.actual_margin > 0))
            p, y = np.asarray(probabilities), np.asarray(outcomes)
            record[f"{label}_probability_games"] = len(p)
            record[f"{label}_brier"] = float(np.mean((p - y) ** 2)) if len(p) else None
            clipped = np.clip(p, 1e-12, 1 - 1e-12)
            record[f"{label}_log_loss"] = (
                float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
                if len(p)
                else None
            )
            for bucket in range(10):
                selected = np.minimum((p * 10).astype(int), 9) == bucket
                if selected.any():
                    calibration.append(
                        dict(
                            model_version=version,
                            horizon=horizon,
                            forecast=label,
                            bin_lower=bucket / 10,
                            games=int(selected.sum()),
                            mean_probability=float(p[selected].mean()),
                            observed_home_win_rate=float(y[selected].mean()),
                        )
                    )
        record["closing_line_mae"] = (
            float((-paired.closing_spread - paired.actual_margin).abs().mean())
            if len(paired)
            else None
        )
        metrics.append(record)
    return {
        "status": "ok",
        "completed_games_missing_forecasts": len(
            set(results.game_id) - set(snapshots.game_id)
        ),
        "metrics": metrics,
        "calibration": calibration,
        "definitions": {
            "early_week": "first eligible publication per game",
            "closing": "last eligible pregame publication; see lead hours",
            "probabilities": "Student-t home-win probabilities; ties excluded",
            "archive_coverage": "matching archived bundle; not proof of replay",
        },
    }


def report(season):
    from sqlalchemy import text

    from backend.db import engine

    keys = set()
    for path in (ARCHIVE / "runs").glob(f"{season}_*.json"):
        manifest = json.loads(path.read_text())
        for row in manifest["forecasts"]:
            keys.add(
                (
                    row["game_id"],
                    pd.Timestamp(row["as_of"]).isoformat(),
                    row["model_version"],
                )
            )
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        manifests = connection.execute(
            text("select manifest from nfl.forecast_input_runs where season=:season"),
            {"season": season},
        ).scalars()
        for manifest in manifests:
            for row in manifest["forecasts"]:
                keys.add(
                    (
                        row["game_id"],
                        pd.Timestamp(row["as_of"]).isoformat(),
                        row["model_version"],
                    )
                )
        snapshots = pd.read_sql_query(
            text("select * from nfl.forecast_snapshots where season=:season"),
            connection,
            params={"season": season},
        )
        results = pd.read_sql_query(
            text(
                "select game_id,start_date,home_points,away_points,closing_spread "
                "from nfl.game_results where season=:season"
            ),
            connection,
            params={"season": season},
        )
    return summarize(snapshots, results, keys)
