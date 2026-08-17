"""Walk-forward calibration. Every week of every evaluation season is refit
and projected strictly out-of-sample; engine hyperparameters are selected on
development-season bivariate log loss, layer parameters on development spread
log loss and MAE; holdout seasons are reported untouched.

The expensive pass (generate_walk_forward) produces per-game engine numbers;
layer parameters rescore those vectorially without refitting."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import itertools

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t, t as student_t

from backend.config import DEVELOPMENT_SEASONS, HOLDOUT_SEASONS
from backend.etl import store
from backend.model import qb_adjustment as qb_layer
from backend.model.fit_week import (
    load_game_index,
    load_qb_games,
    load_season_games,
    recency_by_game,
)
from backend.model.joint_scoring import (
    JointScoringConfig,
    fit_joint_scoring,
)
from backend.model.preseason import (
    PreseasonConfig,
    build_preseason_prior,
)

EVAL_SEASONS = tuple(DEVELOPMENT_SEASONS) + tuple(HOLDOUT_SEASONS)
QB_SPANS = (150.0, 250.0)


@dataclass(frozen=True, slots=True)
class LayerParams:
    market_weight: float = 0.25
    rest_points_per_day: float = 0.06
    playoff_multiplier: float = 1.0
    qb_span: float = 250.0
    carryover: float = 0.60
    win_total_blend: float = 0.35


class WalkForwardData:
    """Config-independent inputs prepared once: game index, QB games,
    starters, per-span entering QB values, and per-season win-total slopes."""

    def __init__(self, seasons: tuple[int, ...], qb_spans=QB_SPANS):
        history = list(range(2015, max(seasons) + 1))
        self.game_index = load_game_index(history)
        self.qb_games = load_qb_games(history)
        starters = self.qb_games[self.qb_games["started"]]
        self.starter_of = {
            (row.game_id, row.team): row.passer_player_id
            for row in starters.itertuples()
        }
        self.entering: dict[float, dict[tuple[str, str], float]] = {}
        for span in qb_spans:
            values = qb_layer.qb_values(self.qb_games, self.game_index, span)
            self.entering[span] = {
                (row.game_id, row.passer_player_id): row.value_points
                for row in values.itertuples()
            }
        # One slope per season from a long-memory reference config; the map
        # from win totals to points is not an engine-selection question.
        reference = JointScoringConfig(
            rating_half_life_weeks=float("inf"),
            strength_prior_sd_ppd=0.35,
            student_t_degrees_of_freedom=500.0,
        )
        from backend.model.preseason import points_per_win

        self.slopes = {}
        for season in seasons:
            self.slopes[season] = points_per_win(
                list(range(2015, season)), reference
            )
        self.season_games = {
            season: load_season_games(season) for season in seasons
        }
        self.previous_games = {
            season: load_season_games(season - 1) for season in seasons
        }


def generate_walk_forward(
    engine_config: JointScoringConfig,
    seasons: tuple[int, ...],
    use_prior_means: bool = True,
    preseason_config: PreseasonConfig = PreseasonConfig(),
    qb_spans: tuple[float, ...] = QB_SPANS,
    data: WalkForwardData | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per played game per forecast week: engine margin/total and
    everything the layers need to rescore."""
    data = data or WalkForwardData(seasons, qb_spans)
    qb_games = data.qb_games
    starter_of = data.starter_of
    entering = data.entering

    rows = []
    for season in seasons:
        games = data.season_games[season]
        previous_games = data.previous_games[season]
        final_week = int(previous_games["model_week"].max()) + 1
        previous_fit = fit_joint_scoring(
            previous_games,
            final_week,
            datetime.now(timezone.utc),
            engine_config,
        )
        prior = build_preseason_prior(
            season,
            as_of=datetime.now(timezone.utc),
            config=preseason_config,
            engine_config=engine_config,
            previous_fit=previous_fit,
            slope=data.slopes[season],
        )
        prior_means = prior.strength_prior_means() if use_prior_means else None
        weeks = sorted(games["model_week"].unique())
        for week in weeks:
            slate = games[games["model_week"].eq(week)]
            if week == weeks[0]:
                fit = prior.week1_fit()
            else:
                as_of = pd.to_datetime(
                    slate["start_date"], utc=True
                ).min().to_pydatetime() - timedelta(seconds=1)
                try:
                    fit = fit_joint_scoring(
                        games, week, as_of, engine_config, prior_means
                    )
                except ValueError:
                    continue
            weights = recency_by_game(games, week, engine_config)
            window = qb_games[
                qb_games["game_id"].isin(
                    set(games.loc[games["model_week"] < week, "game_id"])
                )
            ]
            baselines: dict[float, pd.Series] = {}
            for span in qb_spans:
                values_frame = pd.DataFrame(
                    [
                        {
                            "game_id": row.game_id,
                            "passer_player_id": row.passer_player_id,
                            "value_points": entering[span].get(
                                (row.game_id, row.passer_player_id), 0.0
                            ),
                        }
                        for row in window[window["started"]].itertuples()
                    ]
                )
                baselines[span] = qb_layer.team_baseline_values(
                    values_frame, window, window["game_id"], weights
                ) if not values_frame.empty else pd.Series(dtype=float)
            for game in slate.itertuples():
                game_id = str(game.game_id)
                (
                    expected_home,
                    expected_away,
                    _home_field,
                    margin_sd,
                    total_sd,
                    correlation,
                ) = fit.engine_projection(game)
                record = {
                    "season": season,
                    "model_week": int(week),
                    "week": int(game.week),
                    "season_type": game.season_type,
                    "game_id": game_id,
                    "neutral_site": bool(game.neutral_site),
                    "engine_margin": expected_home - expected_away,
                    "engine_total": expected_home + expected_away,
                    "margin_sd": margin_sd,
                    "total_sd": total_sd,
                    "correlation": correlation,
                    "rest_diff": float(
                        np.clip(game.home_rest - game.away_rest, -7, 7)
                    )
                    if pd.notna(game.home_rest) and pd.notna(game.away_rest)
                    else 0.0,
                    "market_margin": (
                        float(game.spread_line)
                        if pd.notna(game.spread_line)
                        else np.nan
                    ),
                    "closing_spread": (
                        -float(game.spread_line)
                        if pd.notna(game.spread_line)
                        else np.nan
                    ),
                    "actual_margin": float(game.home_points - game.away_points),
                    "actual_total": float(game.home_points + game.away_points),
                    "is_playoff": game.season_type != "REG",
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_points": game.home_points,
                    "away_points": game.away_points,
                }
                for span in qb_spans:
                    adj = 0.0
                    for sign, team in (
                        (1.0, str(game.home_team)),
                        (-1.0, str(game.away_team)),
                    ):
                        passer = starter_of.get((game_id, team))
                        if passer is None:
                            continue
                        value = entering[span].get((game_id, passer))
                        if value is None:
                            continue
                        baseline = float(baselines[span].get(team, 0.0))
                        adj += sign * (value - baseline)
                    record[f"qb_adj_{int(span)}"] = adj
                rows.append(record)
        if verbose:
            print(f"walk-forward {season}: {len(rows)} cumulative rows")
    return pd.DataFrame(rows)


def apply_layers(
    predictions: pd.DataFrame, params: LayerParams
) -> pd.DataFrame:
    out = predictions.copy()
    qb_column = f"qb_adj_{int(params.qb_span)}"
    if qb_column not in out.columns:
        raise KeyError(f"{qb_column} not precomputed")
    pure = (
        out["engine_margin"]
        + out[qb_column]
        + params.rest_points_per_day * out["rest_diff"]
    )
    pure = np.where(
        out["is_playoff"], pure * params.playoff_multiplier, pure
    )
    market = out["market_margin"]
    weight = min(params.market_weight, 0.5)
    blended = np.where(
        market.notna(), (1 - weight) * pure + weight * market, pure
    )
    out["pure_model_margin"] = pure
    out["model_margin"] = blended
    return out


def margin_log_loss(frame: pd.DataFrame, scale: float, df: float) -> float:
    z = (frame["actual_margin"] - frame["model_margin"]) / (
        frame["margin_sd"] * scale
    )
    density = student_t.pdf(z, df) / (frame["margin_sd"] * scale)
    return float(-np.mean(np.log(np.maximum(density, 1e-300))))


def bivariate_log_loss(frame: pd.DataFrame, scale: float, df: float) -> float:
    losses = []
    for row in frame.itertuples():
        margin_sd = row.margin_sd * scale
        total_sd = row.total_sd * scale
        covariance = np.array(
            [
                [margin_sd**2, row.correlation * margin_sd * total_sd],
                [row.correlation * margin_sd * total_sd, total_sd**2],
            ]
        )
        density = multivariate_t.pdf(
            [row.actual_margin, row.actual_total],
            loc=[row.model_margin, row.engine_total],
            shape=covariance * (df - 2) / df,
            df=df,
        )
        losses.append(-np.log(max(density, 1e-300)))
    return float(np.mean(losses))


def honesty_report(frame: pd.DataFrame) -> pd.DataFrame:
    graded = frame[frame["closing_spread"].notna()].copy()
    graded["model_error"] = (
        graded["actual_margin"] - graded["model_margin"]
    ).abs()
    graded["market_error"] = (
        graded["actual_margin"] + graded["closing_spread"]
    ).abs()
    report = graded.groupby("season").agg(
        games=("game_id", "size"),
        model_mae=("model_error", "mean"),
        market_mae=("market_error", "mean"),
    )
    report["gap"] = report["model_mae"] - report["market_mae"]
    return report.round(3)


def coverage_report(
    frame: pd.DataFrame, scale: float, df: float
) -> dict[str, float]:
    z = (frame["actual_margin"] - frame["model_margin"]) / (
        frame["margin_sd"] * scale
    )
    out = {}
    for level in (0.5, 0.8, 0.95):
        bound = student_t.ppf(0.5 + level / 2, df)
        out[f"coverage_{int(level * 100)}"] = float(np.mean(np.abs(z) <= bound))
    return out


ENGINE_GRID = [
    JointScoringConfig(
        rating_half_life_weeks=half_life,
        strength_prior_sd_ppd=prior_sd,
        covariance_shrinkage=shrinkage,
        student_t_degrees_of_freedom=500.0,
    )
    for half_life, prior_sd, shrinkage in itertools.product(
        (float("inf"), 12.0, 6.0), (0.25, 0.35, 0.45), (0.1, 0.5, 0.8)
    )
]

SD_GRID = list(itertools.product((0.85, 0.925, 1.0, 1.075), (7.0, 50.0, 500.0)))

LAYER_GRID = [
    LayerParams(market_weight=w, rest_points_per_day=r,
                playoff_multiplier=p, qb_span=s)
    for w, r, p, s in itertools.product(
        (0.0, 0.15, 0.25, 0.35, 0.5),
        (0.0, 0.04, 0.08),
        (1.0, 1.05, 1.1),
        QB_SPANS,
    )
]

PRESEASON_GRID = [
    PreseasonConfig(carryover=c, win_total_blend=q)
    for c, q in itertools.product((0.5, 0.6, 0.7), (0.2, 0.35, 0.5))
]


def run_calibration(verbose: bool = True) -> dict:
    development = tuple(DEVELOPMENT_SEASONS)
    results = []

    def log(message: str) -> None:
        if verbose:
            print(message)

    data = WalkForwardData(EVAL_SEASONS)
    log("prepared walk-forward inputs")

    best = None
    # Stage 1: engine core + prior-means flag, scored on dev margin log loss
    # with layers at defaults.
    for use_prior in (True, False):
        for config in ENGINE_GRID:
            predictions = generate_walk_forward(
                config, development, use_prior, data=data
            )
            layered = apply_layers(predictions, LayerParams())
            loss = margin_log_loss(layered, 1.0, 500.0)
            results.append(
                {
                    "stage": "engine",
                    "half_life": config.rating_half_life_weeks,
                    "prior_sd": config.strength_prior_sd_ppd,
                    "shrinkage": config.covariance_shrinkage,
                    "use_prior_means": use_prior,
                    "margin_log_loss": loss,
                }
            )
            if best is None or loss < best[0]:
                best = (loss, config, use_prior)
            log(
                f"engine hl={config.rating_half_life_weeks} "
                f"sd={config.strength_prior_sd_ppd} "
                f"shr={config.covariance_shrinkage} prior={use_prior} "
                f"-> {loss:.4f}"
            )
    _, engine_config, use_prior_means = best

    # Stage 2: layers, on the winning engine's dev predictions.
    dev_predictions = generate_walk_forward(
        engine_config, development, use_prior_means, data=data
    )
    best_layer = None
    for layer in LAYER_GRID:
        layered = apply_layers(dev_predictions, layer)
        loss = margin_log_loss(layered, 1.0, 500.0)
        mae = float(
            (layered["actual_margin"] - layered["model_margin"]).abs().mean()
        )
        results.append(
            {"stage": "layers", **asdict(layer), "margin_log_loss": loss,
             "mae": mae}
        )
        if best_layer is None or loss < best_layer[0]:
            best_layer = (loss, layer)
    _, layer_params = best_layer
    log(f"selected layers: {layer_params}")

    # Stage 3: preseason carryover/blend, scored on weeks 1-4 only.
    best_preseason = None
    for preseason_config in PRESEASON_GRID:
        predictions = generate_walk_forward(
            engine_config, development, use_prior_means,
            preseason_config, data=data,
        )
        early = apply_layers(
            predictions[predictions["model_week"] <= 4], layer_params
        )
        loss = margin_log_loss(early, 1.0, 500.0)
        results.append(
            {"stage": "preseason", **asdict(preseason_config),
             "early_margin_log_loss": loss}
        )
        if best_preseason is None or loss < best_preseason[0]:
            best_preseason = (loss, preseason_config)
        log(
            f"preseason c={preseason_config.carryover} "
            f"q={preseason_config.win_total_blend} -> {loss:.4f}"
        )
    _, preseason_config = best_preseason

    # Stage 4: sd scale and df on dev, then untouched holdout report.
    dev_predictions = generate_walk_forward(
        engine_config, development, use_prior_means, preseason_config,
        data=data,
    )
    dev_layered = apply_layers(dev_predictions, layer_params)
    best_sd = None
    for scale, df in SD_GRID:
        loss = margin_log_loss(dev_layered, scale, df)
        results.append(
            {"stage": "sd", "scale": scale, "df": df, "margin_log_loss": loss}
        )
        if best_sd is None or loss < best_sd[0]:
            best_sd = (loss, scale, df)
    _, scale, df = best_sd
    engine_config = replace(
        engine_config,
        student_t_degrees_of_freedom=df,
        score_covariance_scale=scale,
    )
    log(f"selected engine: {engine_config}")
    log(f"selected preseason: {preseason_config}")

    holdout_predictions = generate_walk_forward(
        engine_config, tuple(HOLDOUT_SEASONS), use_prior_means,
        preseason_config, data=data,
    )
    holdout = apply_layers(holdout_predictions, layer_params)
    dev_final = apply_layers(dev_predictions, layer_params)
    summary = {
        "engine_config": engine_config,
        "layer_params": layer_params,
        "preseason_config": preseason_config,
        "use_prior_means": use_prior_means,
        "dev_margin_log_loss": margin_log_loss(dev_final, scale, df),
        "holdout_margin_log_loss": margin_log_loss(holdout, scale, df),
        "holdout_coverage": coverage_report(holdout, scale, df),
        "dev_honesty": honesty_report(dev_final),
        "holdout_honesty": honesty_report(holdout),
    }
    store.write_processed(
        pd.DataFrame(results), "calibration", "search_history.parquet"
    )
    combined = pd.concat([dev_final, holdout], ignore_index=True)
    store.write_processed(combined, "calibration", "predictions.parquet")
    return summary
