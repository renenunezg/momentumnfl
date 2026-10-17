"""The quarterback rating, version 2: a fitted forecast of the passer's play.

The headline is the expected EPA per dropback over the rest of the season,
fitted on quarterback-weeks (every passer at every checkpoint week of the
development seasons) by a ridge of the rest-of-season outcome on the passer's
measures to date and his prior season, standardised among the week's
qualified passers. The measures that carry weight are the production core (EPA
and success per dropback, completion over expected, sacks, pressure to sack,
late-down conversion) and the processing measures from FTN charting (what
happens past the first read, the checkdown rate, sacks and success when
blitzed). Every input publishes during the season, so the live page and the
backtest read the same numbers.

The band is the out-of-sample error of the forecast at that week, measured on
held-out seasons, not inherited from a formula. The profile beside the rating
reports every measure with its game-to-game reliability and its kind: quality,
processing, style or noise.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.players.qb_study import (
    RATES,
    STUDY_DIR,
    _ridge_fit,
    _spearman,
    _sums_through,
    _z_within,
    build_qb_games,
    rate,
    split_half,
)

log = logging.getLogger(__name__)

MODEL_VERSION = "nfl_qb_rating_v2"
MODELS_DIR = PROCESSED_DIR / "players" / "models"
# Measures the page shows, built from feeds that publish during the season.
PRODUCTION_RATES = {
    **{
        name: RATES[name]
        for name in (
            "epa_db",
            "success_db",
            "cpoe",
            "anya",
            "sack_rate",
            "late_down_conv",
            "explosive_rate",
            "epa_in_pocket",
            "epa_no_play_action",
            "catchable_rate",
            "int_rate",
            "int_worthy_rate",
            "turnover_wpa_db",
            "beyond_first_read_rate",
            "success_beyond_first",
            "epa_beyond_first",
            "checkdown_rate",
            "blitz_sack_rate",
            "blitz_success",
            "int_worthy_beyond_first",
            "ngs_time_to_throw",
            "scramble_rate",
            "out_of_pocket_rate",
            "play_action_rate",
            "adot",
            "deep_rate",
            "ngs_aggressiveness",
            "throwaway_rate",
            "motion_rate",
            "rpo_rate",
            "screen_rate",
            "scramble_epa",
        )
    },
    "pfr_pressure_rate": ("pfr_pressured", "pfr_db"),
    "pfr_pressure_to_sack": ("pfr_sacked", "pfr_pressured"),
}
MEASURE_KINDS = {
    "quality": [
        "epa_db",
        "success_db",
        "cpoe",
        "anya",
        "sack_rate",
        "pfr_pressure_to_sack",
        "late_down_conv",
        "epa_in_pocket",
        "epa_no_play_action",
        "catchable_rate",
    ],
    "processing": [
        "beyond_first_read_rate",
        "success_beyond_first",
        "epa_beyond_first",
        "checkdown_rate",
        "blitz_sack_rate",
        "blitz_success",
        "int_worthy_beyond_first",
    ],
    "style": [
        "ngs_time_to_throw",
        "scramble_rate",
        "out_of_pocket_rate",
        "play_action_rate",
        "adot",
        "deep_rate",
        "ngs_aggressiveness",
        "throwaway_rate",
        "motion_rate",
        "rpo_rate",
        "screen_rate",
        "pfr_pressure_rate",
    ],
    "noise": [
        "int_rate",
        "int_worthy_rate",
        "turnover_wpa_db",
        "explosive_rate",
        "scramble_epa",
    ],
}
RATING_FEATURES = [
    "epa_db",
    "success_db",
    "cpoe",
    "sack_rate",
    "pfr_pressure_to_sack",
    "late_down_conv",
    "beyond_first_read_rate",
    "success_beyond_first",
    "checkdown_rate",
    "blitz_sack_rate",
    "blitz_success",
]
PRIOR_FEATURES = ["epa_db", "success_db", "cpoe", "sack_rate"]
CHECKPOINTS = tuple(range(3, 17))
MIN_DROPBACKS = 60
MIN_REST_DROPBACKS = 100
PRIOR_MIN_DROPBACKS = 100
PENALTY_GRID = np.logspace(-1, 4, 26)
GRADE_MEAN, GRADE_SD, GRADE_MIN, GRADE_MAX = 50.0, 10.0, 20.0, 80.0
TARGET = "target_epa"
LOW_RELIABILITY = 0.10


def feature_columns() -> list[str]:
    return [*RATING_FEATURES, *[f"prior_{f}" for f in PRIOR_FEATURES], "prior_games"]


def checkpoint_rows(games: pd.DataFrame) -> pd.DataFrame:
    """One row per passer, season and checkpoint week with the standardised
    features and the rest-of-season target; unavailable measures (seasons before
    charting) sit at the week's average."""
    totals = _sums_through(games, None)
    prior = totals[totals["dropbacks"].ge(PRIOR_MIN_DROPBACKS)].copy()
    for name in PRIOR_FEATURES:
        prior[f"prior_{name}"] = rate(prior, name, PRODUCTION_RATES)
    prior["prior_games"] = prior["games"].astype(float)
    prior = prior[
        ["season", "athlete_id", *[f"prior_{f}" for f in PRIOR_FEATURES], "prior_games"]
    ]
    prior["season"] = prior["season"] + 1
    parts = []
    for week in CHECKPOINTS:
        before = _sums_through(games, week)
        before = before[before["dropbacks"].ge(MIN_DROPBACKS)]
        after = _sums_through(games, week, after=True)
        after = after[after["dropbacks"].ge(MIN_REST_DROPBACKS)]
        merged = before.merge(
            after, on=["season", "athlete_id"], suffixes=("", "_rest")
        )
        row = merged[["season", "athlete_id"]].assign(week=week)
        for name in RATING_FEATURES:
            row[name] = rate(merged, name, PRODUCTION_RATES).to_numpy()
        row[TARGET] = (merged["epa_rest"] / merged["dropbacks_rest"]).to_numpy()
        parts.append(row)
    rows = pd.concat(parts, ignore_index=True).merge(
        prior, on=["season", "athlete_id"], how="left"
    )
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    return _standardise(rows)


def _standardise(rows: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns()
    rows = rows.copy()
    rows[columns] = rows[columns].fillna(
        rows.groupby(["season", "week"])[columns].transform("mean")
    )
    rows[columns] = rows[columns].fillna(0.0)
    return _z_within(rows, columns, ["season", "week"])


@dataclass
class QbModel:
    features: list[str]
    coefficients: list[float]
    intercept: float
    penalty: float
    dev_seasons: list[int]
    # Out-of-sample residual sd of the forecast per checkpoint week: the range
    # of outcomes, which includes the rest of the season's own noise.
    residual_sd: dict[int, float]
    # Per measure: game-to-game reliability, kind and whether it is in the rating.
    measures: pd.DataFrame
    # Split-half reliability of the rating itself at a passer's game count, from
    # odd against even games pooled over the development seasons. The band is the
    # standard error of measurement, GRADE_SD x sqrt(1 - r).
    reliability: dict[int, float] = field(default_factory=dict)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_version: str = MODEL_VERSION

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.intercept + rows[self.features].to_numpy(float) @ np.asarray(
            self.coefficients
        )


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
) -> pd.DataFrame:
    """Leave-one-season-out forecasts for every development row."""
    x, y = rows[columns].to_numpy(float), rows[TARGET].to_numpy(float)
    prediction = np.full(len(rows), np.nan)
    for held in sorted(rows["season"].unique()):
        fit = rows["season"].ne(held).to_numpy()
        beta = _ridge_fit(x[fit], y[fit], penalty)
        prediction[~fit] = beta[0] + x[~fit] @ beta[1:]
    return rows.assign(prediction=prediction)


def evaluate(rows: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Per week: Spearman of the forecast and of EPA to date with the target."""
    frame = rows.assign(prediction=prediction)
    out = []
    for week, part in frame.groupby("week"):
        per_season = [
            (
                _spearman(p["prediction"], p[TARGET]),
                _spearman(p["epa_db"], p[TARGET]),
            )
            for _, p in part.groupby("season")
        ]
        rho = np.array(per_season, dtype=float)
        out.append(
            {
                "week": int(week),
                "rows": len(part),
                "rho_rating": float(np.nanmean(rho[:, 0])),
                "rho_epa": float(np.nanmean(rho[:, 1])),
            }
        )
    return pd.DataFrame(out)


def fit_qb_model(
    games: pd.DataFrame, dev_seasons: list[int], validation_season: int
) -> QbModel:
    """Fit on ``dev_seasons``; the validation season is scored once, after."""
    rows = checkpoint_rows(games)
    dev = rows[rows["season"].isin(dev_seasons)]
    columns = feature_columns()
    penalty = _choose_penalty(dev, columns)
    beta = _ridge_fit(
        dev[columns].to_numpy(float), dev[TARGET].to_numpy(float), penalty
    )
    held = _out_of_sample(dev, columns, penalty)
    residual = (held[TARGET] - held["prediction"]).groupby(held["week"]).std()
    dev_eval = evaluate(dev, held["prediction"].to_numpy()).assign(set="development")
    model = QbModel(
        features=columns,
        coefficients=beta[1:].tolist(),
        intercept=float(beta[0]),
        penalty=penalty,
        dev_seasons=sorted(int(s) for s in dev_seasons),
        residual_sd={int(w): float(v) for w, v in residual.items()},
        measures=measure_table(games[games["season"].isin(dev_seasons)]),
    )
    model.reliability = rating_reliability(
        games[games["season"].isin(dev_seasons)], model
    )
    log.info(
        "qb rating reliability by games: %s",
        {k: round(v, 3) for k, v in model.reliability.items()},
    )
    valid = rows[rows["season"].eq(validation_season)]
    valid_eval = evaluate(valid, model.predict(valid)).assign(set="validation")
    model.validation = pd.concat([dev_eval, valid_eval], ignore_index=True)
    for row in valid_eval.itertuples():
        log.info(
            "qb rating week %d held out: rating %.3f, EPA alone %.3f",
            row.week,
            row.rho_rating,
            row.rho_epa,
        )
    return model


RELIABILITY_GAMES = (4, 6, 8, 12)


def _half_sums(games: pd.DataFrame, n: int) -> pd.DataFrame:
    """Each passer-season's odd and even games among his first ``n``, summed."""
    rows = games.sort_values(["season", "athlete_id", "week"]).copy()
    rows["order"] = rows.groupby(["season", "athlete_id"]).cumcount()
    rows["played"] = rows.groupby(["season", "athlete_id"])["week"].transform("size")
    rows = rows[rows["played"].ge(n) & rows["order"].lt(n)]
    rows["half"] = rows["order"] % 2
    numeric = [
        c
        for c in rows.columns
        if rows[c].dtype.kind in "fi"
        and c not in ("season", "week", "order", "played", "half")
    ]
    return rows.groupby(["season", "athlete_id", "half"], as_index=False)[numeric].sum(
        min_count=1
    )


def rating_reliability(games: pd.DataFrame, model: QbModel) -> dict[int, float]:
    """Correlation of the rating computed on a passer's odd games with the rating
    on his even games, at each game count, with prior-season features held
    out so only the current games are tested."""
    out = {}
    for n in RELIABILITY_GAMES:
        halves = _half_sums(games, n)
        halves = halves[halves["dropbacks"].ge(MIN_DROPBACKS / 2)]
        rows = halves[["season", "athlete_id", "half"]].copy()
        for name in RATING_FEATURES:
            rows[name] = rate(halves, name, PRODUCTION_RATES).to_numpy()
        for name in PRIOR_FEATURES:
            rows[f"prior_{name}"] = 0.0
        rows["prior_games"] = 0.0
        rows["week"] = rows["half"]
        rows = _standardise(rows)
        rows["rating"] = model.predict(rows)
        wide = rows.pivot_table(
            index=["season", "athlete_id"], columns="half", values="rating"
        ).dropna()
        out[n] = float(wide[0].corr(wide[1])) if len(wide) >= 30 else np.nan
    return out


def band_at(model: QbModel, games: pd.Series) -> np.ndarray:
    """GRADE_SD x sqrt(1 - r) with r the rating's reliability interpolated at the
    passer's games (r = 0 at zero games, flat past the last point)."""
    table = sorted((g, r) for g, r in model.reliability.items() if np.isfinite(r))
    if not table:
        return np.full(len(games), np.nan)
    x = np.array([0.0, *[g for g, _ in table]])
    r = np.array([0.0, *[v for _, v in table]])
    rel = np.clip(np.interp(games.to_numpy(float), x, r), 0.0, 1.0)
    return GRADE_SD * np.sqrt(1.0 - rel)


def measure_table(games: pd.DataFrame) -> pd.DataFrame:
    """Reliability, kind and rating membership per production measure."""
    reliability = split_half(games, rates=PRODUCTION_RATES).set_index("measure")[
        "split_half"
    ]
    rows = []
    for kind, names in MEASURE_KINDS.items():
        for name in names:
            rows.append(
                {
                    "measure": name,
                    "kind": kind,
                    "reliability": float(reliability.get(name, np.nan)),
                    "in_rating": name in RATING_FEATURES,
                }
            )
    return pd.DataFrame(rows)


def qb_model_path(cutoff_season: int) -> Path:
    return MODELS_DIR / f"qb_rating_{cutoff_season}.json"


def save_qb_model(model: QbModel, path: Path) -> Path:
    payload = {
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


def load_qb_model(path: Path) -> QbModel:
    payload = json.loads(Path(path).read_text())
    return QbModel(
        features=payload["features"],
        coefficients=payload["coefficients"],
        intercept=payload["intercept"],
        penalty=payload["penalty"],
        dev_seasons=payload["dev_seasons"],
        residual_sd={int(k): float(v) for k, v in payload["residual_sd"].items()},
        reliability={
            int(k): float(v) for k, v in payload.get("reliability", {}).items()
        },
        measures=pd.DataFrame(payload["measures"]),
        validation=pd.DataFrame(payload["validation"]),
        model_version=payload["model_version"],
    )


# Scoring


def _week_rows(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Every passer with the floor of dropbacks through ``week`` of ``season``,
    with the standardised rating features and every production measure."""
    season_games = games[games["season"].eq(season)]
    before = _sums_through(season_games, week)
    before = before[before["dropbacks"].ge(MIN_DROPBACKS)]
    if before.empty:
        return pd.DataFrame()
    totals = _sums_through(games[games["season"].eq(season - 1)], None)
    prior = totals[totals["dropbacks"].ge(PRIOR_MIN_DROPBACKS)].copy()
    for name in PRIOR_FEATURES:
        prior[f"prior_{name}"] = rate(prior, name, PRODUCTION_RATES)
    prior["prior_games"] = prior["games"].astype(float)
    prior = prior[
        ["athlete_id", *[f"prior_{f}" for f in PRIOR_FEATURES], "prior_games"]
    ]
    rows = before[["season", "athlete_id", "dropbacks", "games"]].assign(week=week)
    for name in PRODUCTION_RATES:
        rows[name] = rate(before, name, PRODUCTION_RATES).to_numpy()
    rows = rows.merge(prior, on="athlete_id", how="left")
    rows["prior_games"] = rows["prior_games"].fillna(0.0)
    raw = rows[list(PRODUCTION_RATES)].copy()
    rows = _standardise(rows)
    for name in PRODUCTION_RATES:
        rows[f"raw_{name}"] = raw[name].to_numpy()
    return rows


def score_qb_week(
    games: pd.DataFrame, model: QbModel, season: int, week: int
) -> pd.DataFrame:
    """Rating, grade, band and the profile for every passer at ``week``."""
    rows = _week_rows(games, season, week)
    if rows.empty:
        return pd.DataFrame()
    rating = model.predict(rows)
    rows["rating"] = rating
    sd = float(np.std(rating, ddof=1)) if len(rating) > 1 else np.nan
    z = (rating - rating.mean()) / sd if sd and sd > 0 else np.zeros(len(rating))
    rows["grade"] = np.clip(GRADE_MEAN + GRADE_SD * z, GRADE_MIN, GRADE_MAX)
    nearest = min(model.residual_sd, key=lambda w: abs(w - week))
    rows["rating_sd"] = model.residual_sd[nearest]
    rows["outcome_sd"] = GRADE_SD * rows["rating_sd"] / sd if sd and sd > 0 else np.nan
    rows["grade_sd"] = band_at(model, rows["games"])
    rows["grade_percentile"] = rows["rating"].rank(pct=True) * 100
    names = (
        games[["athlete_id", "athlete_name"]]
        .drop_duplicates("athlete_id")
        .set_index("athlete_id")["athlete_name"]
    )
    rows["athlete_name"] = rows["athlete_id"].map(names)
    return rows


def profile_rows(scored: pd.DataFrame, model: QbModel) -> pd.DataFrame:
    """Long form: one row per passer and measure with value, percentile, kind,
    reliability and rating membership, for the serving table."""
    measures = model.measures.set_index("measure")
    parts = []
    for name in PRODUCTION_RATES:
        value = scored[f"raw_{name}"]
        higher = name not in (
            "sack_rate",
            "pfr_pressure_to_sack",
            "int_rate",
            "int_worthy_rate",
            "checkdown_rate",
            "blitz_sack_rate",
            "int_worthy_beyond_first",
            "throwaway_rate",
            "pfr_pressure_rate",
        )
        kind = measures["kind"].get(name, "noise")
        ranked = (
            value.rank(pct=True, ascending=higher) * 100
            if kind != "style"
            else value.rank(pct=True) * 100
        )
        parts.append(
            pd.DataFrame(
                {
                    "athlete_id": scored["athlete_id"],
                    "measure": name,
                    "value": value,
                    "percentile": ranked.where(value.notna()),
                    "kind": kind,
                    "reliability": measures["reliability"].get(name, np.nan),
                    "in_rating": bool(measures["in_rating"].get(name, False)),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def games_table(seasons: list[int], rebuild: bool = False) -> pd.DataFrame:
    path = STUDY_DIR / "qb_games.parquet"
    if path.exists() and not rebuild:
        games = pd.read_parquet(path)
        if set(seasons) <= set(games["season"].unique()):
            return games[games["season"].isin(seasons)]
    return build_qb_games(seasons)
