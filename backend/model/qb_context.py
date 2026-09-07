"""Experimental decomposition of QB production, supporting cast, and opponents.

The historical promotion gate does not currently admit its point estimates
or additional spread variance. These estimates are inspection diagnostics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, hstack
from scipy.sparse.linalg import splu

from backend.model.qb_adjustment import LEAGUE_DROPBACKS_PER_GAME


@dataclass
class ContextFit:
    strengths: dict[str, float]
    effective_dropbacks: dict[str, float]
    hit_sack_rates: dict[str, float]
    supporting_cast: dict[str, float]
    opponents: dict[str, float]
    pressure_coefficient: float
    qb_index: dict[str, int]
    execution_solver: object
    pressure_solver: object
    execution_noise: float
    pressure_noise: float
    parameter_count: int
    qb_penalty: float

    def contrast_sd(self, contrasts: list[dict[str, float]]) -> np.ndarray:
        """Conditional parameter uncertainty, in points, for arbitrary QB swaps.

        Marginalizing the ridge includes uncertainty in team/opponent effects.
        Sum component SDs conservatively because their covariance is unknown.
        This is not a calibrated prediction interval for game outcomes.
        """
        design = np.zeros((self.parameter_count, len(contrasts)))
        unseen = np.zeros(len(contrasts))
        for j, contrast in enumerate(contrasts):
            for passer, weight in contrast.items():
                if passer in self.qb_index:
                    design[self.qb_index[passer], j] = weight
                else:
                    unseen[j] += weight**2 / self.qb_penalty
        execution = self.execution_noise * (
            np.einsum("ij,ij->j", design, self.execution_solver.solve(design)) + unseen
        )
        pressure = self.pressure_noise * (
            np.einsum("ij,ij->j", design[:-1], self.pressure_solver.solve(design[:-1]))
            + unseen
        )
        return LEAGUE_DROPBACKS_PER_GAME * (
            np.sqrt(np.maximum(execution, 0))
            + abs(self.pressure_coefficient) * np.sqrt(np.maximum(pressure, 0))
        )


def fit_context(
    observations: pd.DataFrame,
    season: int,
    week: int,
    qb_penalty: float = 100.0,
    team_penalty: float = 100.0,
) -> ContextFit:
    """Fit only prior weeks, using a 52-week calendar half-life.

    Separate regularized effects represent QBs, team-seasons, and opposing
    defense-seasons. A second fit estimates each QB's own hit/sack tendency,
    which is restored to his value rather than attributed entirely to the line.
    Penalties and error floors define conditional diagnostic uncertainty.
    """
    if not all(np.isfinite(p) and p > 0 for p in (qb_penalty, team_penalty)):
        raise ValueError("Context penalties must be finite and positive")
    d = observations[
        (observations["season"] < season)
        | (observations["season"].eq(season) & observations["model_week"].lt(week))
    ].copy()
    if d.empty:
        raise ValueError("QB context needs observations before the forecast week")
    if (
        d["dropbacks"].le(0).any()
        or d["hit_sacks"].lt(0).any()
        or d["hit_sacks"].gt(d["dropbacks"]).any()
        or not np.isfinite(d[["dropbacks", "epa", "hit_sacks"]]).all().all()
    ):
        raise ValueError("Invalid QB context observations")
    age = (season - d["season"]) * 52 + week - d["model_week"]
    weights = (d["dropbacks"] * 0.5 ** (age / 52)).to_numpy()
    qb, qb_names = pd.factorize(d["passer_player_id"], sort=True)
    team, team_names = pd.factorize(d["team"] + "_" + d["season"].astype(str))
    opponent, opponent_names = pd.factorize(
        d["opponent"] + "_" + d["season"].astype(str)
    )
    n, nq, nt, no = len(d), len(qb_names), len(team_names), len(opponent_names)
    columns = np.concatenate(
        [np.zeros(n, dtype=int), 1 + qb, 1 + nq + team, 1 + nq + nt + opponent]
    )
    base = coo_matrix(
        (np.ones(4 * n), (np.tile(np.arange(n), 4), columns)),
        shape=(n, 1 + nq + nt + no),
    ).tocsr()
    penalty = np.r_[
        1e-6, np.full(nq, qb_penalty), np.full(nt, team_penalty), np.full(no, 300.0)
    ]
    pressure = (d["hit_sacks"] / d["dropbacks"]).to_numpy()
    centered_pressure = pressure - np.average(pressure, weights=weights)
    design = hstack([base, coo_matrix(centered_pressure[:, None])], format="csr")
    weighted = design.multiply(weights[:, None])
    execution_solver = splu((design.T @ weighted + diags(np.r_[penalty, 1])).tocsc())
    target = (d["epa"] / d["dropbacks"]).to_numpy()
    execution = execution_solver.solve(weighted.T @ target)
    base_weighted = base.multiply(weights[:, None])
    pressure_solver = splu((base.T @ base_weighted + diags(penalty)).tocsc())
    pressure_effects = pressure_solver.solve(base_weighted.T @ pressure)
    beta = float(execution[-1])
    points = LEAGUE_DROPBACKS_PER_GAME
    combined = (execution[:-1] + beta * pressure_effects) * points
    # Conservative floors prevent small or constant samples implying certainty.
    noise = max(
        1.0,
        float(
            np.sum(weights * (target - design @ execution) ** 2)
            / max(n - design.shape[1], 1)
        ),
    )
    pressure_noise = max(
        0.05,
        float(
            np.sum(weights * (pressure - base @ pressure_effects) ** 2)
            / max(n - base.shape[1], 1)
        ),
    )
    d["weight"] = weights
    d["weighted_pressure"] = weights * pressure
    d["weighted_support"] = weights * combined[1 + nq + team]
    d["weighted_opponents"] = weights * combined[1 + nq + nt + opponent]
    counts = d.groupby("passer_player_id")[
        ["weight", "weighted_pressure", "weighted_support", "weighted_opponents"]
    ].sum()
    return ContextFit(
        strengths=dict(zip(qb_names, combined[1 : 1 + nq])),
        effective_dropbacks=counts["weight"].to_dict(),
        hit_sack_rates=(counts["weighted_pressure"] / counts["weight"]).to_dict(),
        supporting_cast=(counts["weighted_support"] / counts["weight"]).to_dict(),
        opponents=(counts["weighted_opponents"] / counts["weight"]).to_dict(),
        pressure_coefficient=beta,
        qb_index={q: i + 1 for i, q in enumerate(qb_names)},
        execution_solver=execution_solver,
        pressure_solver=pressure_solver,
        execution_noise=noise,
        pressure_noise=pressure_noise,
        parameter_count=design.shape[1],
        qb_penalty=qb_penalty,
    )
