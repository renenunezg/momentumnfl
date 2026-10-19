"""The position grade: a 20 to 80 scouting number built from a player's traits.

Each trait's weight at a position comes from player-level evidence on the
development seasons only: its reliability at 8 games (his odd games against
his even games, stepped up to 8 games by Spearman-Brown) times its portability
(the share of its year-over-year persistence that survives a change of team),
or, where it predicts the player's own next season clearly better, a bounded
ridge fit of next season's EPA credit on the traits. Unreliable traits
(reliability under ``NOISE_RELIABILITY``), traits whose reliability cannot be
measured, and traits that do not travel (portability at or below zero) get no
weight and stay on the card as descriptive.

A player's ``composite`` is the weighted mean of his oriented, capped trait
z-scores among qualified players at the group that week, over the traits
available at the group that week, with an available trait he has no value for
counted at the position average (z = 0). He is graded only when he has his
position's primary exposure trait and his traits cover at least
``MIN_WEIGHT_COVERAGE`` of the position's weight. The ``grade`` puts the
composite on the 20 to 80 scale (mean 50, sd 10 among graded players). Nothing
is fitted to team margins or to any betting outcome.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from backend.config import PROCESSED_DIR
from backend.players import MODEL_VERSION, TRAITS

NOISE_RELIABILITY = 0.10
# The fewest players a pooled split-half reliability may rest on before it
# counts. A sample floor, not a tuned value.
MIN_RELIABILITY_PLAYERS = 30
WEIGHT_GAMES = 8
# The trait whose denominator measures a player's usage at the position. Without
# it the player has no measured role there, so he is not graded however extreme
# his other values are.
PRIMARY_TRAITS = {
    "QB": "qb.explosive",
    "RB": "rb.yards_over_expected",
    "WR": "rec.value_per_target",
    "TE": "rec.value_per_target",
    "DL": "def.every_week",
    "LB": "def.every_week",
    "DB": "def.every_week",
    "OL": "ol.line_yards",
    "K": "k.field_goal",
    "P": "p.punt_average",
}
MIN_WEIGHT_COVERAGE = 0.6
# A trait enters the week's composite only where at least this share of the
# week's qualified players at the group have a value for it, so structural gaps
# (a floor before four games, prior-week baselines in week one, a teammate
# comparison most one-quarterback teams cannot make) drop out for everyone while
# one thin profile still fails the coverage bar. A coverage floor, not tuned.
MIN_TRAIT_AVAILABILITY = 0.75
# Trait z-scores are capped at this many sd before weighting: a guard against
# single-game extremes and thin pools, not a tuned value.
Z_CAP = 3.0
SHRINK_RELIABILITY = 0.5
SHRINK_CAP_GAMES = 12.0
# The ridge penalty grid for fitted weights, chosen by leave-one-pair-out error:
# fine and log-spaced so the choice is not a coarse round number.
PENALTIES = np.logspace(-2, 4, 61)
ZERO_COEFFICIENT = 1e-9
GRADE_MEAN, GRADE_SD, GRADE_MIN, GRADE_MAX = 50.0, 10.0, 20.0, 80.0
WEIGHT_COLUMNS = [
    "position_group",
    "trait",
    "sign",
    "sign_source",
    "reliability",
    "portability",
    "portability_imputed",
    "coefficient",
    "weight",
    "weight_method",
    "status",
]
ELIGIBILITY_COLUMNS = [
    "athlete_id",
    "position_group",
    "games",
    "has_primary",
    "coverage",
    "graded",
    "composite",
]
ORIENTATION_COLUMNS = [
    "position_group",
    "trait",
    "sign",
    "sign_source",
    "spearman_epa",
    "players",
]
RELIABILITY_COLUMNS = ["position_group", "games", "players", "correlation"]
OUTPUT_COLUMNS = [
    "athlete_id",
    "position_group",
    "games",
    "composite",
    "grade",
    "grade_percentile",
    "uncertainty",
]
GRADE_NAME = re.compile(r"position_grade_(\d{4})\.json")
MODELS_DIR = PROCESSED_DIR / "players" / "models"


# Weights


def group_traits() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (group, name)
            for name, spec in TRAITS.items()
            for group in spec.position_groups
        ],
        columns=["position_group", "trait"],
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    both = left.notna() & right.notna()
    left, right = left[both].rank(), right[both].rank()
    if len(left) < 3 or not (left.std() > 0 and right.std() > 0):
        return np.nan
    return float(left.corr(right))


def orientations(final_weeks: pd.DataFrame) -> pd.DataFrame:
    """The sign each trait enters the grade with, per group.

    ``higher_is_better`` True is +1 and False is -1 (``sign_source`` ``registered``).
    ``None`` takes the sign of the trait's Spearman correlation with EPA credit
    among the rows of ``final_weeks`` at the group (``sign_source``
    ``epa_spearman``; +1 when the correlation is missing). ``final_weeks`` holds
    ``athlete_id, position_group, trait, value, epa`` for qualified players at the
    final week of each development season.
    """
    rows = []
    for group, trait in group_traits().itertuples(index=False):
        spec = TRAITS[trait]
        part = final_weeks[
            final_weeks["position_group"].eq(group) & final_weeks["trait"].eq(trait)
        ]
        part = part.dropna(subset=["value", "epa"])
        rho = _spearman(part["value"], part["epa"]) if len(part) else np.nan
        if spec.higher_is_better is None:
            sign, source = (-1.0 if rho < 0 else 1.0), "epa_spearman"
        else:
            sign, source = (1.0 if spec.higher_is_better else -1.0), "registered"
        rows.append(
            {
                "position_group": group,
                "trait": trait,
                "sign": sign,
                "sign_source": source,
                "spearman_epa": rho,
                "players": len(part),
            }
        )
    return pd.DataFrame(rows, columns=ORIENTATION_COLUMNS)


def trait_weights(
    reliability: pd.DataFrame, portability: pd.DataFrame, orientation: pd.DataFrame
) -> pd.DataFrame:
    """One row per group and trait with its weight in the grade and why.

    ``weight = max(reliability, 0) x portability``, normalized within the group
    over weighted traits. ``status`` is ``unmeasured_reliability`` when no
    reliability could be measured, ``noise`` when it is under
    ``NOISE_RELIABILITY``, ``team-driven`` when portability is measured and at or
    below zero, ``unmeasured`` when weighted on an imputed portability (the mean
    measured share of the group's traits, or of every group's), else
    ``weighted``. Only ``weighted`` and ``unmeasured`` traits carry weight.
    """
    rel = reliability
    if "games" in rel.columns:
        rel = rel[rel["games"].eq(WEIGHT_GAMES)]
    if "players" in rel.columns:
        rel = rel.assign(
            correlation=rel["correlation"].where(
                rel["players"].ge(MIN_RELIABILITY_PLAYERS)
            )
        )
    rel = rel.drop_duplicates(["position_group", "trait"]).set_index(
        ["position_group", "trait"]
    )
    port = portability.drop_duplicates(["position_group", "trait"]).set_index(
        ["position_group", "trait"]
    )
    out = orientation[["position_group", "trait", "sign", "sign_source"]].copy()
    keys = pd.MultiIndex.from_frame(out[["position_group", "trait"]])
    out["reliability"] = rel["correlation"].reindex(keys).to_numpy(float)
    raw_share = port["portable_share"].reindex(keys).to_numpy(float)
    measured_share = pd.Series(np.clip(raw_share, 0.0, 1.0), index=out.index)

    unmeasured_reliability = out["reliability"].isna()
    noise = ~unmeasured_reliability & out["reliability"].lt(NOISE_RELIABILITY)
    measured = measured_share.notna()
    usable = ~unmeasured_reliability & ~noise
    team_driven = usable & measured & (raw_share <= 0)
    out["status"] = np.select(
        [unmeasured_reliability, noise, team_driven, ~measured],
        ["unmeasured_reliability", "noise", "team-driven", "unmeasured"],
        "weighted",
    )
    group_mean = measured_share.groupby(out["position_group"]).transform("mean")
    out["portability"] = measured_share.fillna(group_mean).fillna(measured_share.mean())
    out["portability_imputed"] = ~measured
    raw = out["reliability"].clip(lower=0.0) * out["portability"]
    raw = raw.where(out["status"].isin(["weighted", "unmeasured"]), 0.0).fillna(0.0)
    total = raw.groupby(out["position_group"]).transform("sum")
    out["weight"] = (raw / total.where(total.gt(0))).fillna(0.0)
    out["coefficient"] = np.nan
    out["weight_method"] = "reliability"
    return out[WEIGHT_COLUMNS].reset_index(drop=True)


# Fitted weights


def _ridge(
    x: np.ndarray, y: np.ndarray, lower: np.ndarray, penalty: float
) -> np.ndarray:
    """Bounded ridge: a free intercept, then coefficients bounded below by ``lower``."""
    width = x.shape[1]
    design = np.vstack(
        [
            np.column_stack([np.ones(len(x)), x]),
            np.column_stack([np.zeros(width), np.sqrt(penalty) * np.eye(width)]),
        ]
    )
    target = np.concatenate([y, np.zeros(width)])
    bounds = (np.concatenate([[-np.inf], lower]), np.full(width + 1, np.inf))
    return lsq_linear(design, target, bounds=bounds).x


def fit_ridge_weights(
    pairs: dict[int, tuple[pd.DataFrame, pd.Series]], lower: pd.Series
) -> tuple[pd.Series, float]:
    """Coefficients of next-season EPA credit on season-t trait z-scores, and penalty.

    The penalty minimizes the pooled squared error of leave-one-pair-out refits
    over ``PENALTIES``; the coefficients are the refit on every pair at that
    penalty. With one pair the smallest penalty is used.
    """
    columns = list(lower.index)
    bound = lower.to_numpy(float)
    seasons = sorted(pairs)
    stacked = {
        season: (
            pairs[season][0][columns].to_numpy(float),
            pairs[season][1].to_numpy(float),
        )
        for season in seasons
    }
    if len(seasons) > 1:
        errors = []
        for penalty in PENALTIES:
            sse, n = 0.0, 0
            for held in seasons:
                x = np.vstack([stacked[s][0] for s in seasons if s != held])
                y = np.concatenate([stacked[s][1] for s in seasons if s != held])
                beta = _ridge(x, y, bound, penalty)
                x_held, y_held = stacked[held]
                residual = y_held - beta[0] - x_held @ beta[1:]
                sse += float(residual @ residual)
                n += len(residual)
            errors.append(sse / n)
        penalty = float(PENALTIES[int(np.argmin(errors))])
    else:
        penalty = float(PENALTIES[0])
    x = np.vstack([stacked[s][0] for s in seasons])
    y = np.concatenate([stacked[s][1] for s in seasons])
    beta = _ridge(x, y, bound, penalty)
    return pd.Series(beta[1:], index=columns), penalty


def fitted_group_weights(
    base: pd.DataFrame, group: str, coefficients: pd.Series
) -> pd.DataFrame:
    """``base``'s rows for ``group`` with the fitted coefficients as weights."""
    rows = base[base["position_group"].eq(group)].copy()
    coefficients = coefficients.where(coefficients.abs().gt(ZERO_COEFFICIENT), 0.0)
    coefficient = rows["trait"].map(coefficients)
    total = coefficients.abs().sum()
    rows["coefficient"] = coefficient
    rows["weight"] = (coefficient / total if total > 0 else coefficient * 0).fillna(0.0)
    rows["weight_method"] = "fitted"
    entered = coefficient.notna()
    weighted = entered & coefficient.ne(0)
    team_driven = weighted & ~rows["portability_imputed"] & rows["portability"].le(0)
    rows["status"] = np.select(
        [~entered, team_driven, weighted],
        [rows["status"], "team-driven (weighted)", "weighted"],
        "unweighted",
    )
    return rows


# The model artifact


@dataclass
class GradeModel:
    """``weights`` is ``trait_weights`` output; ``reliability`` the composite's pooled
    split-half correlation per group at each game count."""

    weights: pd.DataFrame
    dev_seasons: list[int]
    reliability: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=RELIABILITY_COLUMNS)
    )
    model_version: str = MODEL_VERSION
    # Per group, the k in games / (games + k) the composite is shrunk by; absent
    # or 0 means no shrinkage at that group.
    shrink_games: dict[str, float] = field(default_factory=dict)
    # Per group, the k the EPA credit is shrunk by for wins above replacement.
    epa_shrink_games: dict[str, float] = field(default_factory=dict)


def grade_model_path(cutoff_season: int) -> Path:
    return MODELS_DIR / f"position_grade_{cutoff_season}.json"


def grade_cutoff(path: Path) -> int:
    match = GRADE_NAME.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"{path} is not a position grade model")
    return int(match.group(1))


def latest_grade_model_path(season: int, models_dir: Path = MODELS_DIR) -> Path:
    """The grade model in ``models_dir`` with the greatest cutoff before ``season``."""
    cutoffs = {
        int(match.group(1)): path
        for path in Path(models_dir).glob("position_grade_*.json")
        if (match := GRADE_NAME.fullmatch(path.name))
    }
    earlier = [cutoff for cutoff in cutoffs if cutoff < season]
    if not earlier:
        raise FileNotFoundError(
            f"no position grade model with a cutoff before {season} in {models_dir}; "
            "run position-traits-fit first"
        )
    return cutoffs[max(earlier)]


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def save_grade_model(model: GradeModel, path: Path) -> Path:
    payload = {
        "model_version": model.model_version,
        "dev_seasons": [int(season) for season in model.dev_seasons],
        "weights": _records(model.weights[WEIGHT_COLUMNS]),
        "reliability": _records(model.reliability[RELIABILITY_COLUMNS]),
        "shrink_games": {group: float(k) for group, k in model.shrink_games.items()},
        "epa_shrink_games": {
            group: float(k) for group, k in model.epa_shrink_games.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)
    return path


def grade_model_from_payload(payload: dict) -> GradeModel:
    weights = pd.DataFrame(payload["weights"], columns=WEIGHT_COLUMNS)
    for column in ("sign", "reliability", "portability", "coefficient", "weight"):
        weights[column] = weights[column].astype(float)
    weights["portability_imputed"] = weights["portability_imputed"].astype(bool)
    reliability = pd.DataFrame(payload["reliability"], columns=RELIABILITY_COLUMNS)
    return GradeModel(
        weights=weights,
        dev_seasons=[int(season) for season in payload["dev_seasons"]],
        reliability=reliability.astype({"games": int, "correlation": float}),
        model_version=payload["model_version"],
        shrink_games={g: float(k) for g, k in payload.get("shrink_games", {}).items()},
        epa_shrink_games={
            g: float(k) for g, k in payload.get("epa_shrink_games", {}).items()
        },
    )


def load_grade_model(path: Path) -> GradeModel:
    return grade_model_from_payload(json.loads(Path(path).read_text()))


def validate_grade_model(model: GradeModel, season: int) -> None:
    """Refuse a model that saw ``season`` or later, or that another version made."""
    if model.model_version != MODEL_VERSION:
        raise ValueError(
            f"grade model is {model.model_version!r}, expected {MODEL_VERSION!r}"
        )
    if not model.dev_seasons or max(model.dev_seasons) >= season:
        raise ValueError(f"grade model on {model.dev_seasons} cannot score {season}")


# Scoring


def _scaled(
    values: pd.Series, reference: pd.Series, keys: list[pd.Series]
) -> pd.Series:
    """(value - mean) / sd with the mean and sd of ``reference`` within ``keys``;
    0 where sd is 0 or missing, NaN where the value is."""
    mean = reference.groupby(keys).transform("mean")
    sd = reference.groupby(keys).transform("std")
    z = (values - mean) / sd.where(sd.gt(0))
    return z.where(values.isna() | z.notna(), 0.0)


def _prophecy(games: np.ndarray, at: float, r: float) -> np.ndarray:
    """Spearman-Brown: the reliability at ``games`` from ``r`` measured at ``at``."""
    scale = games / at
    return scale * r / (1.0 + (scale - 1.0) * r)


def reliability_at(table: pd.DataFrame, games: pd.Series) -> np.ndarray:
    """Spearman-Brown from the first measured point below it, linear between
    measured points, flat past the last, clipped to [0, 1]."""
    table = table.dropna(subset=["correlation"])
    if table.empty:
        return np.zeros(len(games))
    table = table.sort_values("games")
    x, r = table["games"].to_numpy(float), table["correlation"].to_numpy(float)
    played = games.to_numpy(float)
    out = np.interp(played, x, r)
    below = played < x[0]
    out[below] = _prophecy(played[below], x[0], r[0])
    return np.clip(out, 0.0, 1.0)


def graded_groups(model: GradeModel) -> set[str]:
    return set(model.weights.loc[model.weights["weight"].ne(0), "position_group"])


def shrink_games_from(reliability: pd.DataFrame) -> dict[str, float]:
    """Per group, the games at which the ``reliability_at`` curve first reaches
    ``SHRINK_RELIABILITY``, or ``SHRINK_CAP_GAMES``."""
    out = {}
    for group, table in reliability.dropna(subset=["correlation"]).groupby(
        "position_group"
    ):
        table = table.sort_values("games")
        games = table["games"].to_numpy(float)
        r = table["correlation"].to_numpy(float)
        target, k = SHRINK_RELIABILITY, SHRINK_CAP_GAMES
        if r[0] >= target:
            # Below the first point the curve is Spearman-Brown, which reaches
            # the target at games x target (1 - r) / (r (1 - target)).
            k = games[0] * target * (1.0 - r[0]) / (r[0] * (1.0 - target))
        else:
            for i in range(1, len(r)):
                if r[i] >= target:
                    k = games[i - 1] + (target - r[i - 1]) / (r[i] - r[i - 1]) * (
                        games[i] - games[i - 1]
                    )
                    break
        out[group] = float(min(k, SHRINK_CAP_GAMES))
    return out


def _week_pool(snapshot_week: pd.DataFrame, traits: pd.DataFrame, groups: set[str]):
    """The week's qualified players at ``groups``, their rows of ``traits`` with
    capped oriented z-scores, and the (group, trait) pairs available that week."""
    keys = ["athlete_id", "position_group"]
    rows = snapshot_week.astype({"athlete_id": str})
    rows = rows.assign(qualified=rows["qualified"].eq(True))
    players = rows.groupby(keys, as_index=False).agg(
        games=("games", "max"), qualified=("qualified", "any")
    )
    players = players[players["qualified"] & players["position_group"].isin(groups)][
        [*keys, "games"]
    ].reset_index(drop=True)
    pool = rows.merge(players[keys], on=keys).merge(
        traits, on=["position_group", "trait"]
    )
    value = pool["value"].astype(float)
    z = _scaled(value, value, [pool["position_group"], pool["trait"]]) * pool["sign"]
    pool = pool.assign(z=z.clip(-Z_CAP, Z_CAP))[z.notna()]
    pool_size = players.groupby("position_group").size()
    share = (
        pool.groupby(["position_group", "trait"])["athlete_id"]
        .nunique()
        .div(pool_size, level="position_group")
    )
    available = share[share.ge(MIN_TRAIT_AVAILABILITY)].index
    return players, pool, available


def trait_z_frame(
    snapshot_week: pd.DataFrame, traits: pd.DataFrame, group: str
) -> pd.DataFrame:
    """One row per qualified player at ``group``, one column per trait of ``traits``
    at the group: the capped oriented z, 0 where missing or unavailable that week."""
    traits = traits[traits["position_group"].eq(group)]
    players, pool, available = _week_pool(snapshot_week, traits, {group})
    pool = pool[
        pd.MultiIndex.from_frame(pool[["position_group", "trait"]]).isin(available)
    ]
    wide = pool.pivot_table(
        index="athlete_id", columns="trait", values="z", aggfunc="first"
    )
    return wide.reindex(index=players["athlete_id"], columns=traits["trait"]).fillna(
        0.0
    )


def grade_eligibility(snapshot_week: pd.DataFrame, model: GradeModel) -> pd.DataFrame:
    """Every qualified player at a graded group in one snapshot week, graded or not.

    ``composite = sum(w_t z_t) / sum(|w_t|)`` over the traits available at the
    group that week (``MIN_TRAIT_AVAILABILITY``), with ``z = 0`` for an available
    trait the player has no value for, then multiplied by ``games / (games + k_g)``
    where the model shrinks the group. ``coverage`` is the absolute weight of the
    available traits he has a value for over that of every available trait;
    ``graded`` needs the primary trait and coverage of at least
    ``MIN_WEIGHT_COVERAGE``. ``composite`` is NaN where he is not graded.
    """
    keys = ["athlete_id", "position_group"]
    weights = model.weights[model.weights["weight"].ne(0)].assign(
        magnitude=lambda frame: frame["weight"].abs()
    )
    players, pool, available = _week_pool(
        snapshot_week,
        weights[["position_group", "trait", "sign", "weight", "magnitude"]],
        graded_groups(model),
    )
    available_weight = (
        weights.set_index(["position_group", "trait"])["magnitude"]
        .reindex(available)
        .groupby(level="position_group")
        .sum()
    )
    in_available = pd.MultiIndex.from_frame(pool[["position_group", "trait"]]).isin(
        available
    )
    sums = (
        pool.assign(
            weighted=(pool["weight"] * pool["z"]).where(in_available, 0.0),
            covered=pool["magnitude"].where(in_available, 0.0),
        )
        .groupby(keys)[["weighted", "covered"]]
        .sum()
        .reindex(pd.MultiIndex.from_frame(players[keys]))
        .fillna(0.0)
    )
    rows = snapshot_week.astype({"athlete_id": str})
    primary = rows[
        rows["trait"].eq(rows["position_group"].map(PRIMARY_TRAITS))
        & rows["value"].notna()
    ]
    primary_keys = set(zip(primary["athlete_id"], primary["position_group"]))
    players["has_primary"] = [
        key in primary_keys
        for key in zip(players["athlete_id"], players["position_group"])
    ]
    denominator = players["position_group"].map(available_weight)
    denominator = denominator.where(denominator.gt(0))
    players["coverage"] = sums["covered"].to_numpy() / denominator
    players["graded"] = players["has_primary"] & players["coverage"].ge(
        MIN_WEIGHT_COVERAGE
    )
    k = players["position_group"].map(model.shrink_games).fillna(0.0)
    games = players["games"].astype(float)
    shrink = (games / (games + k)).where(k.gt(0), 1.0).to_numpy()
    composite = sums["weighted"].to_numpy() / denominator.to_numpy() * shrink
    players["composite"] = np.where(players["graded"], composite, np.nan)
    return players[ELIGIBILITY_COLUMNS]


COVERAGE_LOSS_COLUMNS = [
    "position_group",
    "qualified",
    "graded",
    "no_primary",
    "low_coverage",
]


def coverage_losses(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Per group: qualified players, how many are graded, and how many lose the
    grade for lacking the primary trait or, with it, for low coverage."""
    frame = eligibility.assign(
        no_primary=~eligibility["has_primary"],
        low_coverage=eligibility["has_primary"] & ~eligibility["graded"],
    )
    out = frame.groupby("position_group").agg(
        qualified=("athlete_id", "size"),
        graded=("graded", "sum"),
        no_primary=("no_primary", "sum"),
        low_coverage=("low_coverage", "sum"),
    )
    return out.reset_index()[COVERAGE_LOSS_COLUMNS].astype(
        {name: int for name in COVERAGE_LOSS_COLUMNS[1:]}
    )


def score_grades(snapshot_week: pd.DataFrame, model: GradeModel) -> pd.DataFrame:
    """Each graded player's grade from one snapshot week (see ``grade_eligibility``).

    ``grade = 50 + 10 x (composite - mean) / sd`` over the graded players at the
    group, clipped to [20, 80]; ``grade_percentile`` ranks the composite (0 to 100,
    100 for a lone player); ``uncertainty = 10 x sqrt(1 - r)`` with ``r`` the
    model's composite reliability at the player's games.
    """
    out = grade_eligibility(snapshot_week, model)
    out = out[out["graded"]].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    group = out["position_group"]
    z = _scaled(out["composite"], out["composite"], [group])
    out["grade"] = (GRADE_MEAN + GRADE_SD * z).clip(GRADE_MIN, GRADE_MAX)
    rank = out["composite"].groupby(group).rank(method="average")
    n = out["composite"].groupby(group).transform("size")
    out["grade_percentile"] = (100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0)
    reliability = np.concatenate(
        [
            reliability_at(
                model.reliability[model.reliability["position_group"].eq(g)],
                out.loc[group.eq(g), "games"],
            )
            for g in group.unique()
        ]
    )
    order = np.concatenate(
        [np.flatnonzero(group.eq(g).to_numpy()) for g in group.unique()]
    )
    r = np.empty(len(out))
    r[order] = reliability
    out["uncertainty"] = GRADE_SD * np.sqrt(np.clip(1.0 - r, 0.0, 1.0))
    return out[OUTPUT_COLUMNS]
