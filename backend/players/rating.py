"""The fitted player rating engine shared by the positions with their own
production (quarterbacks, running backs).

A rating is a forecast of a player's rest-of-season production, fitted on
player-weeks: every qualified player at every checkpoint week of the
development seasons, a ridge of the rest-of-season outcome on his measures to
date and his prior season, standardised among the week's qualified players so
a week-4 number and a week-12 number mean the same thing. The penalty is
chosen leave-one-season-out, the validation season is scored once after the
fit, and the band is the rating's own standard error of measurement from odd
against even games, not a formula inherited from elsewhere.

A ``RatingSpec`` says what a position's version of this is: its measures
(all from feeds that publish during the season), which carry weight, the
kind of each (quality, processing, role, style, noise), the target and the
fixed-weight baselines the rating must beat.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.players.grade import reliability_at
from backend.players.qb_study import (
    _ridge_fit,
    _spearman,
    _sums_through,
    _z_within,
    rate,
    split_half,
)
from backend.players.snapshots import spearman_brown

log = logging.getLogger(__name__)

MODELS_DIR = PROCESSED_DIR / "players" / "models"
PENALTY_GRID = np.logspace(-1, 4, 26)
GRADE_MEAN, GRADE_SD, GRADE_MIN, GRADE_MAX = 50.0, 10.0, 20.0, 80.0
RELIABILITY_GAMES = (4, 6, 8, 12)
TARGET = "target"
BASELINE_PREFIX = "baseline_"
MIN_RELIABILITY_PLAYERS = 30


@dataclass(frozen=True)
class RatingSpec:
    position: str
    model_version: str
    # Measure name to (numerator, denominator) columns of the game table.
    rates: dict[str, tuple[str, str]]
    kinds: dict[str, list[str]]
    rating_features: list[str]
    prior_features: list[str]
    # The measure the rating forecasts over the rest of the season.
    target: str
    # The baselines the rating must beat: name to fixed weights on raw measures
    # to date. Nothing in a baseline is fitted here.
    baselines: dict[str, dict[str, float]]
    # Volume column with its floors: to date, rest of season, prior season.
    volume: str
    min_volume: float
    min_rest_volume: float
    prior_min_volume: float
    checkpoints: tuple[int, ...] = tuple(range(3, 17))
    lower_is_better: frozenset[str] = frozenset()

    def feature_columns(self) -> list[str]:
        """The rating features, then the prior season's (with its games) when
        the position uses one."""
        prior = [f"prior_{f}" for f in self.prior_features]
        return [*self.rating_features, *prior, *(["prior_games"] if prior else [])]

    def kind_of(self, measure: str) -> str:
        for kind, names in self.kinds.items():
            if measure in names:
                return kind
        return "noise"


@dataclass
class RatingModel:
    spec_position: str
    model_version: str
    features: list[str]
    coefficients: list[float]
    intercept: float
    penalty: float
    dev_seasons: list[int]
    # Out-of-sample residual sd per checkpoint week: the range of outcomes,
    # which includes the rest of the season's own noise.
    residual_sd: dict[int, float]
    # Per measure: game-to-game reliability, kind and rating membership.
    measures: pd.DataFrame
    # Reliability of the rating itself at each game count (``rating_reliability``).
    reliability: dict[int, float] = field(default_factory=dict)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.intercept + rows[self.features].to_numpy(float) @ np.asarray(
            self.coefficients
        )


# Rows


def _prior_table(games: pd.DataFrame, spec: RatingSpec) -> pd.DataFrame:
    totals = _sums_through(games, None)
    prior = totals[totals[spec.volume].ge(spec.prior_min_volume)].copy()
    for name in spec.prior_features:
        prior[f"prior_{name}"] = rate(prior, name, spec.rates)
    prior["prior_games"] = prior["games"].astype(float)
    columns = ["season", "athlete_id", *[f"prior_{f}" for f in spec.prior_features]]
    return prior[[*columns, "prior_games"]]


def _standardise(rows: pd.DataFrame, spec: RatingSpec) -> pd.DataFrame:
    columns = spec.feature_columns()
    rows = rows.copy()
    rows[columns] = rows[columns].fillna(
        rows.groupby(["season", "week"])[columns].transform("mean")
    )
    rows[columns] = rows[columns].fillna(0.0)
    return _z_within(rows, columns, ["season", "week"])


def checkpoint_rows(games: pd.DataFrame, spec: RatingSpec) -> pd.DataFrame:
    """One row per player, season and checkpoint week with standardised features
    and the rest-of-season target; a measure a season does not have sits at the
    week's average."""
    prior = _prior_table(games, spec)
    prior["season"] = prior["season"] + 1
    numerator, denominator = spec.rates[spec.target]
    parts = []
    for week in spec.checkpoints:
        before = _sums_through(games, week)
        before = before[before[spec.volume].ge(spec.min_volume)]
        after = _sums_through(games, week, after=True)
        after = after[after[spec.volume].ge(spec.min_rest_volume)]
        merged = before.merge(
            after, on=["season", "athlete_id"], suffixes=("", "_rest")
        )
        row = merged[["season", "athlete_id"]].assign(week=week)
        for name in spec.rating_features:
            row[name] = rate(merged, name, spec.rates).to_numpy()
        for name, weights in spec.baselines.items():
            row[f"{BASELINE_PREFIX}{name}"] = sum(
                weight * rate(merged, measure, spec.rates).to_numpy()
                for measure, weight in weights.items()
            )
        row[TARGET] = (
            merged[f"{numerator}_rest"]
            / merged[f"{denominator}_rest"].where(merged[f"{denominator}_rest"].gt(0))
        ).to_numpy()
        parts.append(row)
    rows = pd.concat(parts, ignore_index=True).merge(
        prior, on=["season", "athlete_id"], how="left"
    )
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    rows = rows.dropna(subset=[TARGET])
    return _standardise(rows, spec)


# Fitting


def _choose_penalty(rows: pd.DataFrame, columns: list[str]) -> float:
    seasons = sorted(rows["season"].unique())
    x, y = rows[columns].to_numpy(float), rows[TARGET].to_numpy(float)
    errors = []
    for penalty in PENALTY_GRID:
        sse = 0.0
        for held in seasons:
            fit = rows["season"].ne(held).to_numpy()
            beta = _ridge_fit(x[fit], y[fit], penalty)
            resid = y[~fit] - (beta[0] + x[~fit] @ beta[1:])
            sse += float(resid @ resid)
        errors.append(sse)
    return float(PENALTY_GRID[int(np.argmin(errors))])


def _out_of_sample(
    rows: pd.DataFrame, columns: list[str], penalty: float
) -> np.ndarray:
    x, y = rows[columns].to_numpy(float), rows[TARGET].to_numpy(float)
    prediction = np.full(len(rows), np.nan)
    for held in sorted(rows["season"].unique()):
        fit = rows["season"].ne(held).to_numpy()
        beta = _ridge_fit(x[fit], y[fit], penalty)
        prediction[~fit] = beta[0] + x[~fit] @ beta[1:]
    return prediction


def evaluate(rows: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Per week: Spearman of the forecast (``rho_rating``) and of each baseline
    (``rho_<name>``) with the target, averaged over the seasons that have one."""
    frame = rows.assign(rating=prediction)
    scores = {
        "rating": "rating",
        **{
            column[len(BASELINE_PREFIX) :]: column
            for column in frame.columns
            if column.startswith(BASELINE_PREFIX)
        },
    }
    out = []
    for week, part in frame.groupby("week"):
        row = {"week": int(week), "rows": len(part)}
        for name, column in scores.items():
            rho = [_spearman(p[column], p[TARGET]) for _, p in part.groupby("season")]
            finite = [r for r in rho if np.isfinite(r)]
            row[f"rho_{name}"] = float(np.mean(finite)) if finite else np.nan
        out.append(row)
    return pd.DataFrame(out)


def measure_table(games: pd.DataFrame, spec: RatingSpec) -> pd.DataFrame:
    """Per measure: its kind, rating membership and reliability at 8 games (the
    split-half correlation of 4-game halves, stepped up by Spearman-Brown)."""
    reliability = (
        split_half(games, rates=spec.rates)
        .set_index("measure")["split_half"]
        .map(spearman_brown)
    )
    rows = []
    for kind, names in spec.kinds.items():
        for name in names:
            rows.append(
                {
                    "measure": name,
                    "kind": kind,
                    "reliability": float(reliability.get(name, np.nan)),
                    "in_rating": name in spec.rating_features,
                }
            )
    return pd.DataFrame(rows)


def _half_sums(games: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = games.sort_values(["season", "athlete_id", "week"]).copy()
    rows["order"] = rows.groupby(["season", "athlete_id"]).cumcount()
    rows["played"] = rows.groupby(["season", "athlete_id"])["week"].transform("size")
    rows = rows[rows["played"].ge(n) & rows["order"].lt(n)]
    rows["half"] = rows["order"] % 2
    skip = {"season", "week", "order", "played", "half"}
    numeric = [c for c in rows.columns if rows[c].dtype.kind in "fi" and c not in skip]
    return rows.groupby(["season", "athlete_id", "half"], as_index=False)[numeric].sum(
        min_count=1
    )


def rating_reliability(
    games: pd.DataFrame, model: RatingModel, spec: RatingSpec
) -> dict:
    """The rating's reliability at each game count ``n``: the correlation of the
    rating on a player's odd games with the rating on his even games (``n / 2``
    each), stepped up to ``n`` games by Spearman-Brown. Both halves carry the
    player's own prior season, as the published rating does, so the band is the
    grade's error from which games were sampled this season, not the prior's."""
    prior = _prior_table(games, spec)
    prior["season"] = prior["season"] + 1
    out = {}
    for n in RELIABILITY_GAMES:
        halves = _half_sums(games, n)
        halves = halves[halves[spec.volume].ge(spec.min_volume / 2)]
        rows = halves[["season", "athlete_id", "half"]].copy()
        for name in spec.rating_features:
            rows[name] = rate(halves, name, spec.rates).to_numpy()
        rows = rows.merge(prior, on=["season", "athlete_id"], how="left")
        rows["prior_games"] = rows["prior_games"].fillna(0.0)
        rows["week"] = rows["half"]
        rows = _standardise(rows, spec)
        rows["rating"] = model.predict(rows)
        wide = rows.pivot_table(
            index=["season", "athlete_id"], columns="half", values="rating"
        ).dropna()
        out[n] = (
            spearman_brown(float(wide[0].corr(wide[1])))
            if len(wide) >= MIN_RELIABILITY_PLAYERS
            else np.nan
        )
    return out


def band_at(model: RatingModel, games: pd.Series) -> np.ndarray:
    """GRADE_SD x sqrt(1 - r), r the rating's reliability at the player's games
    on the position grade's curve (``grade.reliability_at``)."""
    table = pd.DataFrame(
        [(g, r) for g, r in model.reliability.items() if np.isfinite(r)],
        columns=["games", "correlation"],
    )
    if table.empty:
        return np.full(len(games), np.nan)
    return GRADE_SD * np.sqrt(1.0 - reliability_at(table, games))


def fit_rating(
    games: pd.DataFrame,
    spec: RatingSpec,
    dev_seasons: list[int],
    validation_season: int,
) -> RatingModel:
    """Fit on ``dev_seasons``; the validation season is scored once, after."""
    rows = checkpoint_rows(games, spec)
    dev = rows[rows["season"].isin(dev_seasons)]
    columns = spec.feature_columns()
    penalty = _choose_penalty(dev, columns)
    beta = _ridge_fit(
        dev[columns].to_numpy(float), dev[TARGET].to_numpy(float), penalty
    )
    held = _out_of_sample(dev, columns, penalty)
    residual = (dev[TARGET] - held).groupby(dev["week"]).std()
    dev_games = games[games["season"].isin(dev_seasons)]
    model = RatingModel(
        spec_position=spec.position,
        model_version=spec.model_version,
        features=columns,
        coefficients=beta[1:].tolist(),
        intercept=float(beta[0]),
        penalty=penalty,
        dev_seasons=sorted(int(s) for s in dev_seasons),
        residual_sd={int(w): float(v) for w, v in residual.items()},
        measures=measure_table(dev_games, spec),
    )
    model.reliability = rating_reliability(dev_games, model, spec)
    valid = rows[rows["season"].eq(validation_season)]
    model.validation = pd.concat(
        [
            evaluate(dev, held).assign(set="development"),
            evaluate(valid, model.predict(valid)).assign(set="validation"),
        ],
        ignore_index=True,
    )
    log.info(
        "%s rating: penalty %.1f, reliability %s",
        spec.position,
        penalty,
        {k: round(v, 3) for k, v in model.reliability.items()},
    )
    return model


# Artifact


def model_path(spec: RatingSpec, cutoff_season: int) -> Path:
    return MODELS_DIR / f"{spec.position.lower()}_rating_{cutoff_season}.json"


def save_model(model: RatingModel, path: Path) -> Path:
    payload = {
        "spec_position": model.spec_position,
        "model_version": model.model_version,
        "features": model.features,
        "coefficients": model.coefficients,
        "intercept": model.intercept,
        "penalty": model.penalty,
        "dev_seasons": model.dev_seasons,
        "residual_sd": {str(k): v for k, v in model.residual_sd.items()},
        "reliability": {str(k): v for k, v in model.reliability.items()},
        "measures": json.loads(model.measures.to_json(orient="records")),
        "validation": json.loads(model.validation.to_json(orient="records")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_model(path: Path) -> RatingModel:
    payload = json.loads(Path(path).read_text())
    return RatingModel(
        spec_position=payload["spec_position"],
        model_version=payload["model_version"],
        features=payload["features"],
        coefficients=payload["coefficients"],
        intercept=payload["intercept"],
        penalty=payload["penalty"],
        dev_seasons=payload["dev_seasons"],
        residual_sd={int(k): float(v) for k, v in payload["residual_sd"].items()},
        measures=pd.DataFrame(payload["measures"]),
        reliability={
            int(k): float(v) for k, v in payload.get("reliability", {}).items()
        },
        validation=pd.DataFrame(payload["validation"]),
    )


def validate_model(model: RatingModel, spec: RatingSpec, season: int) -> None:
    if (
        model.model_version != spec.model_version
        or model.spec_position != spec.position
    ):
        raise ValueError(f"{model.model_version} is not the {spec.position} rating")
    if not model.dev_seasons or max(model.dev_seasons) >= season:
        raise ValueError(f"rating model on {model.dev_seasons} cannot score {season}")


# Scoring


def week_rows(
    games: pd.DataFrame, spec: RatingSpec, season: int, week: int
) -> pd.DataFrame:
    """Every player with the floor of volume through ``week`` of ``season``, with
    the standardised rating features and every production measure (raw)."""
    before = _sums_through(games[games["season"].eq(season)], week)
    before = before[before[spec.volume].ge(spec.min_volume)]
    if before.empty:
        return pd.DataFrame()
    prior = _prior_table(games[games["season"].eq(season - 1)], spec).drop(
        columns="season"
    )
    rows = before[["season", "athlete_id", spec.volume, "games"]].assign(week=week)
    for name in spec.rates:
        rows[name] = rate(before, name, spec.rates).to_numpy()
    rows = rows.merge(prior, on="athlete_id", how="left")
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    raw = rows[list(spec.rates)].copy()
    rows = _standardise(rows, spec)
    for name in spec.rates:
        rows[f"raw_{name}"] = raw[name].to_numpy()
    return rows


def score_week(
    games: pd.DataFrame, model: RatingModel, spec: RatingSpec, season: int, week: int
) -> pd.DataFrame:
    """Rating, grade, band and raw measures for every player at ``week``."""
    rows = week_rows(games, spec, season, week)
    if rows.empty:
        return rows
    rating = model.predict(rows)
    rows["rating"] = rating
    sd = float(np.std(rating, ddof=1)) if len(rating) > 1 else np.nan
    z = (rating - rating.mean()) / sd if sd and sd > 0 else np.zeros(len(rating))
    rows["grade"] = np.clip(GRADE_MEAN + GRADE_SD * z, GRADE_MIN, GRADE_MAX)
    rows["grade_sd"] = band_at(model, rows["games"])
    nearest = min(model.residual_sd, key=lambda w: abs(w - week))
    rows["outcome_sd"] = (
        GRADE_SD * model.residual_sd[nearest] / sd if sd and sd > 0 else np.nan
    )
    rows["grade_percentile"] = (
        (rows["rating"].rank(method="average") - 1) / max(len(rows) - 1, 1) * 100
    )
    names = (
        games[["athlete_id", "athlete_name"]]
        .drop_duplicates("athlete_id", keep="last")
        .set_index("athlete_id")["athlete_name"]
    )
    rows["athlete_name"] = rows["athlete_id"].map(names)
    return rows


def profile_rows(
    scored: pd.DataFrame, model: RatingModel, spec: RatingSpec
) -> pd.DataFrame:
    """Long form: one row per player and measure with value, percentile, kind,
    reliability and rating membership. Style and role measures rank ascending
    (a percentile of the trait, not of merit); the rest rank by merit."""
    measures = model.measures.set_index("measure")
    parts = []
    for name in spec.rates:
        value = scored[f"raw_{name}"]
        kind = spec.kind_of(name)
        merit = kind in ("quality", "processing", "noise")
        ascending = not (merit and name in spec.lower_is_better)
        ranked = (
            (value.rank(method="average", ascending=ascending) - 1)
            / max(value.notna().sum() - 1, 1)
            * 100
        )
        parts.append(
            pd.DataFrame(
                {
                    "athlete_id": scored["athlete_id"],
                    "measure": name,
                    "value": value,
                    "percentile": ranked.where(value.notna()),
                    "kind": kind,
                    "reliability": float(measures["reliability"].get(name, np.nan)),
                    "in_rating": bool(measures["in_rating"].get(name, False)),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
