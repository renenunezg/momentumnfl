"""The player grade head-to-heads published in nfl.position_grade_evaluation.

Every row is regenerated from the frozen models and the cached seasons: the
grade's Spearman rank correlation with its target against a baseline's on the
same players, their difference, a player bootstrap 95% interval, and the
holdout status of the season it was measured on. Nothing here fits or changes
a served model; development numbers for ratings use the frozen penalty
leave-one-season-out, as ``fit_rating`` does.

- Trait grades (``next_season``): the final-week grade in season t against
  per-game EPA credit in t + 1. Development is the mean over the 2016 to 2024
  pairs; validation is the 2024 to 2025 pair. A group whose weights were
  fitted or shrunk leave-one-pair-out takes its development numbers from the
  backtest's cross-fitted pairs and has no interval, since those per-pair
  models are not kept.
- QB and RB ratings (``rest_of_season``): the rating at each checkpoint week
  against the rest of that season, Spearman per week and season, averaged
  over seasons and then weeks 3 to 16.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from backend.players import MODEL_VERSION as TRAITS_VERSION
from backend.players import backtest, grade
from backend.players.rating import (
    BASELINE_PREFIX,
    TARGET,
    RatingModel,
    RatingSpec,
    _out_of_sample,
    checkpoint_rows,
    evaluate,
    with_fitted_baselines,
)

# The first published ratings used 1,000 and 250 resamples. At 2,000 the
# rating interval ends still move by up to 0.01 between seeds (measured on the
# 2016 to 2024 and 2025 rows), so a verdict whose end lies within 0.01 of
# zero is not settled by the interval; the trait ends move by about 0.002.
TRAIT_RESAMPLES = 4000
RATING_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261029
INTERVAL = 0.95
THIN_PLAYER_SEASONS = 100
# How each model's validation season was used before the model froze. This is
# history, not something the data can compute: the 2024 to 2025 trait pair was
# printed before the last model choices; 2025 also screened the ratings'
# candidate measures. A new model version must add its own entry.
HOLDOUT_STATUS = {
    TRAITS_VERSION: "viewed",
    "nfl_qb_rating_v3": "screened",
    "nfl_rb_rating_v3": "screened",
}
TRAIT_TARGET = "epa_credit_per_game_next_season"
RATING_TARGETS = {
    "QB": "epa_per_dropback_rest_of_season",
    "RB": "epa_per_game_rest_of_season",
}
CROSS_FITTED = {"fitted_kept": "fitted", "shrink_kept": "shrinkage"}


def _interval(differences: np.ndarray) -> tuple[float | None, float | None]:
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return None, None
    tail = (1 - INTERVAL) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    return float(low), float(high)


def _row_spearman(x: np.ndarray, y: np.ndarray, minimum: int) -> np.ndarray:
    """Spearman of each row's pairs, ignoring missing values; NaN below
    ``minimum`` complete pairs, as the backtest's ``_spearman``."""
    both = np.isfinite(x) & np.isfinite(y)
    count = both.sum(axis=1)
    x, y = np.where(both, x, np.nan), np.where(both, y, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rx = rankdata(x, axis=1, nan_policy="omit")
        ry = rankdata(y, axis=1, nan_policy="omit")
        rx = rx - np.nansum(rx, axis=1, keepdims=True) / count[:, None]
        ry = ry - np.nansum(ry, axis=1, keepdims=True) / count[:, None]
        rho = np.nansum(rx * ry, axis=1) / np.sqrt(
            np.nansum(rx**2, axis=1) * np.nansum(ry**2, axis=1)
        )
    return np.where(count >= minimum, rho, np.nan)


def _pair_bootstrap(pairs: pd.DataFrame, baseline: str, rng) -> np.ndarray:
    """Resampled grade-minus-baseline differences: players drawn with
    replacement within each season pair, the pair Spearmans averaged."""
    total = np.zeros(TRAIT_RESAMPLES)
    count = 0
    for _, pair in pairs.groupby("season"):
        draws = rng.integers(0, len(pair), (TRAIT_RESAMPLES, len(pair)))
        target = pair["epa_next"].to_numpy(float)[draws]
        grade_rho = _row_spearman(
            pair["grade"].to_numpy(float)[draws], target, backtest.MIN_PAIR_PLAYERS
        )
        base_rho = _row_spearman(
            pair[baseline].to_numpy(float)[draws], target, backtest.MIN_PAIR_PLAYERS
        )
        total += grade_rho - base_rho
        count += 1
    return total / count


def _trait_row(group, sample_set, seasons_label, frame, **values) -> dict:
    player_seasons = int(values.pop("player_seasons"))
    return {
        "model_version": TRAITS_VERSION,
        "position_group": group,
        "evaluation": "next_season",
        "sample_set": sample_set,
        "seasons": seasons_label,
        "checkpoint_week": None,
        "target": TRAIT_TARGET,
        "players": int(frame["athlete_id"].nunique()),
        "player_seasons": player_seasons,
        "thin_sample": player_seasons < THIN_PLAYER_SEASONS,
        **values,
    }


def trait_rows(
    model: grade.GradeModel,
    seasons: dict,
    dev_seasons: list[int],
    validation_season: int,
    evaluation: pd.DataFrame,
) -> list[dict]:
    """``evaluation`` is the backtest's long evaluation frame (its selection
    and development pair tables), written by ``write_backtest``."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairs = backtest.next_season_pairs(seasons, model, validation_season)
    selection = evaluation[evaluation["table"].eq("selection")].set_index(
        "position_group"
    )
    dev_pairs = evaluation[evaluation["table"].eq("dev_pairs")]
    baselines = {"epa_credit": "epa", "standard": "standard"}
    dev_label = f"{min(dev_seasons)}-{max(dev_seasons)}"
    rows = []
    for group, frame in pairs.groupby("position_group", sort=False):
        dev = frame[frame["next_season"].le(max(dev_seasons))]
        valid = frame[frame["next_season"].eq(validation_season)]
        names = [
            "epa_credit",
            *(["standard"] if group in backtest.STANDARD_BASELINES else []),
        ]
        variant = next(
            (
                name
                for flag, name in CROSS_FITTED.items()
                if str(selection.loc[group, flag]) == "True"
            ),
            None,
        )
        for baseline in names:
            column = baselines[baseline]
            if variant is None:
                means = (
                    dev.groupby("season")
                    .apply(
                        lambda p, c=column: pd.Series(
                            {
                                "grade": backtest._spearman(p["grade"], p["epa_next"]),
                                "base": backtest._spearman(p[c], p["epa_next"]),
                            }
                        ),
                        include_groups=False,
                    )
                    .mean()
                )
                low, high = _interval(_pair_bootstrap(dev, column, rng))
                dev_players = len(dev)
            else:
                table = dev_pairs[
                    dev_pairs["variant"].eq(variant)
                    & dev_pairs["position_group"].eq(group)
                ]
                means = pd.Series(
                    {
                        "grade": pd.to_numeric(table["spearman_grade_next_epa"]).mean(),
                        "base": pd.to_numeric(
                            table[f"spearman_{column}_next_epa"]
                        ).mean(),
                    }
                )
                low = high = None
                dev_players = int(pd.to_numeric(table["players"]).sum())
            if not np.isfinite(means["grade"]) or not np.isfinite(means["base"]):
                continue
            rows.append(
                _trait_row(
                    group,
                    "development_cv",
                    dev_label,
                    dev,
                    player_seasons=dev_players,
                    validation_season=validation_season,
                    grade_spearman=float(means["grade"]),
                    baseline=baseline,
                    baseline_spearman=float(means["base"]),
                    difference=float(means["grade"] - means["base"]),
                    difference_ci_low=low,
                    difference_ci_high=high,
                    holdout_status="development",
                )
            )
            grade_rho = backtest._spearman(valid["grade"], valid["epa_next"])
            base_rho = backtest._spearman(valid[column], valid["epa_next"])
            if not np.isfinite(grade_rho) or not np.isfinite(base_rho):
                continue
            low, high = _interval(_pair_bootstrap(valid, column, rng))
            rows.append(
                _trait_row(
                    group,
                    "validation",
                    f"{validation_season - 1}-{validation_season}",
                    valid,
                    player_seasons=len(valid),
                    validation_season=validation_season,
                    grade_spearman=float(grade_rho),
                    baseline=baseline,
                    baseline_spearman=float(base_rho),
                    difference=float(grade_rho - base_rho),
                    difference_ci_low=low,
                    difference_ci_high=high,
                    holdout_status=HOLDOUT_STATUS[TRAITS_VERSION],
                )
            )
    return rows


def _week_season_means(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``evaluate``'s summary for each column: Spearman with the target per
    week and season, averaged over seasons per week, then over weeks."""
    by_week = []
    for _, week in frame.groupby("week"):
        per_season = []
        for _, part in week.groupby("season"):
            target = part[TARGET].to_numpy(float)[None, :]
            values = part[columns].to_numpy(float).T
            per_season.append(
                _row_spearman(values, np.repeat(target, len(columns), axis=0), 20)
            )
        by_week.append(_nanmean(np.vstack(per_season)))
    return _nanmean(np.vstack(by_week))


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Column means over finite entries; NaN where a column has none."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _rating_bootstrap(
    frame: pd.DataFrame, columns: list[str], resamples: int, rng
) -> np.ndarray:
    """``(resamples, len(columns))`` summaries, resampling players with
    replacement within each season and keeping all their checkpoint weeks."""
    groups = {
        season: [part.index.to_numpy() for _, part in rows.groupby("athlete_id")]
        for season, rows in frame.groupby("season")
    }
    out = np.zeros((resamples, len(columns)))
    for b in range(resamples):
        chosen = [
            np.concatenate(
                [players[i] for i in rng.integers(0, len(players), len(players))]
            )
            for players in groups.values()
        ]
        out[b] = _week_season_means(frame.loc[np.concatenate(chosen)], columns)
    return out


def rating_rows(
    spec: RatingSpec,
    model: RatingModel,
    games: pd.DataFrame,
    dev_seasons: list[int],
    validation_season: int,
) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = with_fitted_baselines(checkpoint_rows(games, spec), spec, dev_seasons)
    dev = rows[rows["season"].isin(dev_seasons)].copy()
    dev["rating"] = _out_of_sample(dev, model.features, model.penalty)
    valid = rows[rows["season"].eq(validation_season)].copy()
    valid["rating"] = model.predict(valid)
    names = [c[len(BASELINE_PREFIX) :] for c in rows if c.startswith(BASELINE_PREFIX)]
    columns = ["rating", *[f"{BASELINE_PREFIX}{n}" for n in names]]
    out = []
    for sample_set, frame, resamples, label, status in (
        (
            "development_cv",
            dev,
            RATING_RESAMPLES,
            f"{min(dev_seasons)}-{max(dev_seasons)}",
            "development",
        ),
        (
            "validation",
            valid,
            RATING_RESAMPLES,
            str(validation_season),
            HOLDOUT_STATUS[model.model_version],
        ),
    ):
        frame = frame.reset_index(drop=True)
        summary = evaluate(frame, frame["rating"].to_numpy())
        point = {
            name: float(summary[f"rho_{name}"].mean()) for name in ["rating", *names]
        }
        draws = _rating_bootstrap(frame, columns, resamples, rng)
        player_seasons = int(frame[["season", "athlete_id"]].drop_duplicates().shape[0])
        for i, name in enumerate(names, start=1):
            low, high = _interval(draws[:, 0] - draws[:, i])
            out.append(
                {
                    "model_version": model.model_version,
                    "position_group": spec.position,
                    "evaluation": "rest_of_season",
                    "sample_set": sample_set,
                    "seasons": label,
                    "validation_season": validation_season,
                    "checkpoint_week": None,
                    "target": RATING_TARGETS[spec.position],
                    "players": int(frame["athlete_id"].nunique()),
                    "player_seasons": player_seasons,
                    "grade_spearman": point["rating"],
                    "baseline": name,
                    "baseline_spearman": point[name],
                    "difference": point["rating"] - point[name],
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "thin_sample": player_seasons < THIN_PLAYER_SEASONS,
                    "holdout_status": status,
                }
            )
    return out


def evaluation_rows(dev_seasons: list[int], validation_season: int) -> list[dict]:
    """Every published row, from the frozen grade and rating models fit on
    ``dev_seasons`` and the cached seasons through ``validation_season``."""
    from backend.etl import store
    from backend.players.rating import load_model, model_path
    from backend.players.ratings import SPECS

    cutoff = max(dev_seasons)
    seasons = {
        season: backtest.load_season(season)
        for season in range(min(dev_seasons), validation_season + 1)
    }
    rows = trait_rows(
        grade.load_grade_model(grade.grade_model_path(cutoff)),
        seasons,
        dev_seasons,
        validation_season,
        store.read_processed(*backtest.traits_artifact(f"evaluation_{cutoff}")),
    )
    years = list(range(min(dev_seasons), validation_season + 1))
    for spec, games_table in SPECS.values():
        model = load_model(model_path(spec, cutoff))
        rows += rating_rows(
            spec, model, games_table(years), dev_seasons, validation_season
        )
    return rows
