"""Weekly local artifacts and a reproducible read-only award evaluation command."""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from backend.awards import AWARDS, MODEL_VERSION, features, ingest, model
from backend.config import PROCESSED_DIR, STATIC_DIR
from backend.etl import store

BOARD_COLUMNS = [
    "season",
    "week",
    "award",
    "as_of",
    "model_version",
    "candidate_id",
    "candidate_name",
    "headshot_url",
    "team",
    "position",
    "predicted_rank",
    "performance_rank",
    "performance_score",
    "win_probability",
    "probability_status",
    "rank_change",
    "games",
    "projected_stats",
    "drivers",
    "context_source",
    "context_reason",
]
META_COLUMNS = [
    "season",
    "week",
    "award",
    "as_of",
    "model_version",
    "status",
    "candidate_count",
    "training_seasons",
    "validation",
    "provenance",
]


def history_for(season: int, week: int, awards: list[str]) -> tuple[dict, dict]:
    winners = pd.read_csv(STATIC_DIR / "award_winners.csv")
    if winners.duplicated(["season", "award", "candidate_name"]).any():
        raise ValueError("Duplicate historical award labels")
    races = {award: [] for award in awards}
    for year in sorted(winners.loc[winners.season.lt(season), "season"].unique()):
        historical_week = min(week, 18 if year >= 2021 else 17)
        players, coaches, _ = features.snapshot(int(year), historical_week)
        for award in awards:
            # The revised comeback criteria have only two completed seasons.
            # Earlier rebound awards cannot train current-rule probabilities.
            if award == "CPOY" and year < 2024:
                continue
            pool = features.candidates(players, coaches, award)
            if not pool.empty:
                races[award].append(model.label(pool, winners, award, int(year)))
    evaluations = {award: model.evaluate(rows, award) for award, rows in races.items()}
    return races, evaluations


def build(season: int, week: int, as_of=None, awards=None, write=True) -> tuple:
    awards = list(awards or AWARDS)
    if set(awards) - AWARDS.keys():
        raise ValueError("Unknown AP award")
    players, coaches, provenance = features.snapshot(
        season, week, as_of or datetime.now(UTC).isoformat()
    )
    stamp = provenance["cutoff"]
    races, evaluations = history_for(season, week, awards) if week else ({}, {})
    boards, metadata = [], []
    previous_path = (
        PROCESSED_DIR / "awards" / "boards" / f"{season}_{week - 1:02d}.parquet"
    )
    previous = (
        pd.read_parquet(previous_path) if previous_path.exists() else pd.DataFrame()
    )
    for award in awards:
        pool = features.candidates(players, coaches, award)
        report = model.calibration_report(evaluations.get(award, pd.DataFrame()))
        training = [race for race in races.get(award, []) if race.winner.sum() > 0]
        status = "ready"
        watchlist = (
            features.comeback_context(season, stamp)
            if award == "CPOY" and len(training) < 4
            else pd.DataFrame()
        )
        if not watchlist.empty:
            pool = watchlist.rename(
                columns={"source_url": "context_source", "reason": "context_reason"}
            )
            pool = pool.assign(
                week=week,
                as_of=stamp,
                award=award,
                model_version=MODEL_VERSION,
                games=0,
                projected_stats="{}",
                drivers="[]",
                probability_status="insufficient_history",
            )
            boards.append(
                pool.reindex(columns=BOARD_COLUMNS).sort_values("candidate_name")
            )
            status = "watchlist"
        elif pool.empty:
            status = (
                "missing_comeback_context"
                if award == "CPOY" and not players.empty
                else "awaiting_games"
            )
        elif len(training) < 4:
            status = "insufficient_history"
        else:
            fitted = model.fit(training, award)
            if write:
                model_path = (
                    PROCESSED_DIR
                    / "awards"
                    / "models"
                    / f"{season}_{week:02d}_{award}.json"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_text(
                    json.dumps(
                        {
                            "model_version": MODEL_VERSION,
                            "as_of": stamp,
                            "features": fitted.features,
                            "mean": fitted.mean.tolist(),
                            "scale": fitted.scale.tolist(),
                            "coefficients": fitted.coefficients.tolist(),
                            "training_seasons": [
                                int(y) for y in fitted.training_seasons
                            ],
                            "winner_seed_sha256": hashlib.sha256(
                                (STATIC_DIR / "award_winners.csv").read_bytes()
                            ).hexdigest(),
                            "validation": report,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            probabilities, contributions = fitted.predict(pool)
            pool["win_probability"] = (
                probabilities if report["probabilities_publishable"] else np.nan
            )
            pool["probability_status"] = report["status"]
            pool["predicted_rank"] = (
                pd.Series(probabilities, index=pool.index)
                .rank(ascending=False, method="first")
                .astype(int)
            )
            pool["performance_rank"] = pool.performance_score.rank(
                ascending=False, method="min"
            )
            pool["rank_change"] = np.nan
            if not previous.empty:
                old = previous[previous.award.eq(award)].set_index("candidate_id")
                pool["rank_change"] = (
                    pool.candidate_id.map(old.predicted_rank) - pool.predicted_rank
                )
            pool["award"], pool["model_version"] = award, MODEL_VERSION
            stat_columns = [c for c in pool if c.startswith("projected_")]
            pool["projected_stats"] = [
                json.dumps({k: round(float(row[k]), 2) for k in stat_columns})
                for _, row in pool.iterrows()
            ]
            pool["drivers"] = [
                json.dumps(
                    [
                        {
                            "feature": fitted.features[j],
                            "contribution": round(float(row[j]), 3),
                        }
                        for j in np.argsort(-np.abs(row))[:3]
                    ]
                )
                for row in contributions
            ]
            pool["context_source"] = pool.get("source_url", "")
            pool["context_reason"] = pool.get("reason", "")
            boards.append(
                pool.reindex(columns=BOARD_COLUMNS).sort_values("predicted_rank")
            )
        metadata.append(
            {
                "season": season,
                "week": week,
                "award": award,
                "as_of": stamp,
                "model_version": MODEL_VERSION,
                "status": status,
                "candidate_count": len(pool),
                "training_seasons": json.dumps(
                    [int(r.season.iloc[0]) for r in training]
                ),
                "validation": json.dumps(report),
                "provenance": json.dumps(provenance),
            }
        )
    board = (
        pd.concat(boards, ignore_index=True)
        if boards
        else pd.DataFrame(columns=BOARD_COLUMNS)
    )
    meta = pd.DataFrame(metadata, columns=META_COLUMNS)
    # Presentation metadata joins by player ID after every model calculation.
    # Coach IDs cannot match player IDs and retain the UI's fallback avatar.
    if not board.empty:
        roster = ingest.read("rosters", season)
        photos = roster.dropna(subset=["gsis_id", "headshot_url"])
        photos = photos[photos.headshot_url.str.startswith("https://")]
        photos = photos.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
        board["headshot_url"] = board.candidate_id.map(photos.headshot_url)
    if write:
        store.write_processed(board, "awards", "boards", f"{season}_{week:02d}.parquet")
        store.write_processed(meta, "awards", "meta", f"{season}_{week:02d}.parquet")
        for award, history in evaluations.items():
            if not history.empty:
                store.write_processed(
                    history, "awards", "evaluation", f"{award}_{week:02d}.parquet"
                )
    return board, meta, evaluations
