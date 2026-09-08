"""Regularized season-choice models with strictly chronological evaluation.

Targets are award winners, never invented ballot shares. Weekly models are
fit separately so each season contributes one race per forecast horizon.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from backend.awards.features import features_for, name_key


@dataclass
class WinnerModel:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    training_seasons: list[int]

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame[self.features].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite award model inputs")
        contributions = ((matrix - self.mean) / self.scale) * self.coefficients
        return softmax(contributions.sum(axis=1)), contributions


def fit(races: list[pd.DataFrame], award: str) -> WinnerModel:
    if len(races) < 4:
        raise ValueError("At least four earlier award seasons are required")
    features = features_for(award)
    combined = pd.concat(races, ignore_index=True)
    matrix = combined[features].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite historical award inputs")
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-8] = 1
    groups = []
    for race in races:
        target = race.winner.to_numpy(float)
        if target.sum() <= 0:
            raise ValueError("Training race has no observed winner in candidate pool")
        groups.append(
            ((race[features].to_numpy(float) - mean) / scale, target / target.sum())
        )

    def objective(beta):
        loss, gradient = 0.5 * np.dot(beta, beta), beta.copy()
        for x, target in groups:
            logits = x @ beta
            loss += logsumexp(logits) - target @ logits
            gradient += x.T @ (softmax(logits) - target)
        return loss, gradient

    # Sparse winner labels and correlated fields can otherwise learn that
    # more production hurts an otherwise identical candidate.
    bounds = [
        (
            (None, None)
            if name.startswith("is_") or name == "remaining_games"
            else (None, 0)
            if "passing_interceptions" in name
            else (0, None)
        )
        for name in features
    ]
    result = minimize(
        objective, np.zeros(len(features)), jac=True, method="L-BFGS-B", bounds=bounds
    )
    if not result.success:
        raise ValueError(f"Awards optimizer failed: {result.message}")
    return WinnerModel(
        features, mean, scale, result.x, sorted(combined.season.unique().astype(int))
    )


def label(
    pool: pd.DataFrame, winners: pd.DataFrame, award: str, season: int
) -> pd.DataFrame:
    race = pool.copy()
    target = winners[winners.award.eq(award) & winners.season.eq(season)]
    if target.empty:
        raise ValueError(f"Missing official award result: {award} {season}")
    winner_keys = set(target.candidate_name.map(name_key))
    race["winner"] = race.candidate_name.map(name_key).isin(winner_keys).astype(float)
    if race.loc[race.winner.gt(0), "candidate_id"].nunique() > len(winner_keys):
        raise ValueError(f"Ambiguous winner identity: {award} {season}")
    return race


def evaluate(races: list[pd.DataFrame], award: str) -> pd.DataFrame:
    history, training = [], []
    for race in sorted(races, key=lambda r: int(r.season.iloc[0])):
        if len(training) >= 4:
            model = fit(training, award)
            probabilities, _ = model.predict(race)
            order = np.argsort(-probabilities, kind="stable")
            target = race.winner.to_numpy(float)
            in_pool = target.sum() > 0
            target /= max(target.sum(), 1)
            winner_mass = float(probabilities @ (target > 0))
            # An omitted winner is a full miss, including a missing outcome
            # component in the Brier score; never silently drop that season.
            history.append(
                {
                    "season": int(race.season.iloc[0]),
                    "award": award,
                    "week": int(race.week.iloc[0]),
                    "as_of": race.as_of.iloc[0],
                    "training_through": max(model.training_seasons),
                    "candidate_count": len(race),
                    "winner_in_pool": bool(in_pool),
                    "winner_hit": bool(target[order[0]] > 0),
                    "top_three_hit": bool(target[order[:3]].sum() > 0),
                    "winner_probability": winner_mass,
                    "leader_probability": float(probabilities[order[0]]),
                    "log_loss": float(-np.log(max(winner_mass, 1e-15))),
                    "brier": float(
                        np.square(probabilities - target).sum() + (not in_pool)
                    ),
                    "uniform_log_loss": float(np.log(len(race)))
                    if in_pool
                    else -np.log(1e-15),
                    "split": "development"
                    if int(race.season.iloc[0]) <= 2019
                    else "retrospective_validation",
                }
            )
        if race.winner.sum() > 0:
            training.append(race)
    return pd.DataFrame(history)


def calibration_report(history: pd.DataFrame) -> dict:
    if history.empty:
        return {
            "status": "insufficient_history",
            "seasons": 0,
            "probabilities_publishable": False,
        }
    validation = history[history.split.eq("retrospective_validation")]
    if validation.empty:
        return {
            "status": "insufficient_validation",
            "seasons": 0,
            "probabilities_publishable": False,
        }
    # Confidence is assessed over entire independent seasons, not thousands
    # of candidate rows that would create a misleading effective sample size.
    gap = abs(
        float(validation.leader_probability.mean() - validation.winner_hit.mean())
    )
    passed = (
        len(validation) >= 8
        and validation.winner_in_pool.all()
        and validation.log_loss.mean() < validation.uniform_log_loss.mean()
        and gap <= 0.10
    )
    return {
        "status": "validated" if passed else "experimental",
        "seasons": len(validation),
        "probabilities_publishable": bool(passed),
        "winner_hit_rate": float(validation.winner_hit.mean()),
        "top_three_rate": float(validation.top_three_hit.mean()),
        "winner_pool_coverage": float(validation.winner_in_pool.mean()),
        "log_loss": float(validation.log_loss.mean()),
        "brier": float(validation.brier.mean()),
        "leader_calibration_gap": gap,
        "basis": "expanding-window retrospective validation; no fresh holdout",
    }
