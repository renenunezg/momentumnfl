"""Fit and score the fitted ratings (quarterbacks, running backs) into the
position trait row shape the serving table and the page read.

A rated position's rows per player and week are: ``grade`` (20 to 80 among
the week's qualified players, ``uncertainty`` the rating's own standard
error), ``rating`` (the forecast in its own units: EPA per dropback for a
passer, EPA per game for a back), and one row per profile measure with its
value, merit percentile, kind, reliability and whether it carries weight.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.players import qb_model, rb_model
from backend.players.rating import (
    MODELS_DIR,
    RatingSpec,
    fit_rating,
    load_model,
    model_path,
    profile_rows,
    save_model,
    score_week,
    validate_model,
)

log = logging.getLogger(__name__)

SPECS: dict[str, tuple[RatingSpec, object]] = {
    "QB": (qb_model.QB_SPEC, qb_model.games_table),
    "RB": (rb_model.RB_SPEC, rb_model.games_table),
}
RATED_POSITIONS = tuple(SPECS)
GRADE_ROW, RATING_ROW = "grade", "rating"
RATING_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "athlete_id",
    "athlete_name",
    "headshot_url",
    "team",
    "position_group",
    "games",
    "trait",
    "value",
    "percentile",
    "kind",
    "in_grade",
    "uncertainty",
    "reliability",
    "points_above_replacement",
    "qualified",
]


def fit_ratings(dev_seasons: list[int], validation_season: int) -> dict[str, Path]:
    """Fit every rated position on ``dev_seasons`` and freeze the models."""
    seasons = list(range(min(dev_seasons), validation_season + 1))
    out = {}
    for position, (spec, games_table) in SPECS.items():
        games = games_table(seasons, rebuild=True)
        model = fit_rating(games, spec, dev_seasons, validation_season)
        out[position] = save_model(model, model_path(spec, max(dev_seasons)))
        held = model.validation[model.validation["set"].eq("validation")]
        rho = held.filter(like="rho_").mean().round(3).to_dict()
        log.info("%s rating validation %s: mean %s", position, validation_season, rho)
    return out


def latest_model_path(spec: RatingSpec, season: int) -> Path:
    stem = f"{spec.position.lower()}_rating_"
    cutoffs = {
        int(path.stem[len(stem) :]): path
        for path in MODELS_DIR.glob(f"{stem}*.json")
        if path.stem[len(stem) :].isdigit()
    }
    earlier = [cutoff for cutoff in cutoffs if cutoff < season]
    if not earlier:
        raise FileNotFoundError(
            f"no {spec.position} rating model before {season}; run position-traits-fit"
        )
    return cutoffs[max(earlier)]


def _identity(games: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = games[games["season"].eq(season)].sort_values("week")
    return rows.drop_duplicates("athlete_id", keep="last").set_index("athlete_id")[
        ["athlete_name", "team"]
    ]


def score_ratings(
    season: int, weeks: list[int], as_of: datetime, headshots: pd.Series
) -> pd.DataFrame:
    """Rows for every rated position at each of ``weeks`` of ``season``."""
    parts = []
    for position, (spec, games_table) in SPECS.items():
        path = latest_model_path(spec, season)
        model = load_model(path)
        validate_model(model, spec, season)
        games = games_table(list(range(min(model.dev_seasons), season + 1)))
        if games[games["season"].eq(season)].empty:
            log.info("%s rating %s: no games to score", position, season)
            continue
        identity = _identity(games, season)
        for week in weeks:
            scored = score_week(games, model, spec, season, week)
            if scored.empty:
                continue
            base = pd.DataFrame(
                {
                    "athlete_id": scored["athlete_id"],
                    "athlete_name": scored["athlete_id"].map(identity["athlete_name"]),
                    "headshot_url": scored["athlete_id"].map(headshots),
                    "team": scored["athlete_id"].map(identity["team"]),
                    "position_group": position,
                    "games": scored["games"].astype(float),
                }
            )
            grade = base.assign(
                trait=GRADE_ROW,
                value=scored["grade"].to_numpy(),
                percentile=scored["grade_percentile"].to_numpy(),
                kind="quality",
                in_grade=True,
                uncertainty=scored["grade_sd"].to_numpy(),
                reliability=np.nan,
            )
            rating = base.assign(
                trait=RATING_ROW,
                value=scored["rating"].to_numpy(),
                percentile=scored["grade_percentile"].to_numpy(),
                kind="quality",
                in_grade=True,
                uncertainty=scored["outcome_sd"].to_numpy(),
                reliability=np.nan,
            )
            profile = profile_rows(scored, model, spec).merge(
                base, on="athlete_id", how="left"
            )
            profile = profile.rename(
                columns={"measure": "trait", "in_rating": "in_grade"}
            )
            profile["uncertainty"] = np.nan
            rows = pd.concat([grade, rating, profile], ignore_index=True)
            rows["week"] = week
            rows["model_version"] = model.model_version
            parts.append(rows)
    if not parts:
        return pd.DataFrame(columns=RATING_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["season"] = season
    out["as_of"] = as_of.isoformat()
    out["points_above_replacement"] = np.nan
    out["qualified"] = True
    return out[RATING_COLUMNS]
