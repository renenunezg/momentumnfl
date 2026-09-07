"""Walk-forward calibration. Every week of every evaluation season is refit
and projected strictly out-of-sample. Engine hyperparameters, output-layer
parameters, the preseason prior, and the score-distribution scale are
selected in sequence on development-season margin log loss; holdout seasons
are reported untouched.

The expensive pass (generate_walk_forward) produces per-game engine numbers;
layer parameters rescore those vectorially without refitting."""

import itertools
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from backend.config import (
    DEVELOPMENT_SEASONS,
    HISTORY_START_SEASON,
    HOLDOUT_SEASONS,
    STATIC_DIR,
)
from backend.etl import store
from backend.model import qb_adjustment as qb_layer
from backend.model.joint_scoring import JointScoringConfig, fit_joint_scoring
from backend.model.market_blend import (
    RESIDUAL_RANGE,
    blend_margin,
    fit_margin_residual_distribution,
)
from backend.model.preseason import (
    PreseasonConfig,
    build_preseason_prior,
    points_per_win,
)
from backend.model.projections import LayerConfig, rest_adjustment

EVAL_SEASONS = tuple(DEVELOPMENT_SEASONS) + tuple(HOLDOUT_SEASONS)
QB_SPANS = (250.0, 500.0, 1000.0)
# Near-Gaussian degrees of freedom while the location parameters are being
# selected; the tail shape is chosen last, on the winning configuration.
SELECTION_DF = 500.0
DENSITY_FLOOR = 1e-300


class WalkForwardData:
    """Config-independent inputs prepared once: game index, QB games,
    starters, per-span QB strength histories, and per-season win-total slopes."""

    def __init__(self, seasons: tuple[int, ...], qb_spans=QB_SPANS):
        history = list(range(HISTORY_START_SEASON, max(seasons) + 1))
        self.game_index = store.game_index(history)
        self.qb_games = store.qb_games(history)
        starters = self.qb_games[self.qb_games["started"]]
        self.starter_of = {
            (row.game_id, row.team): row.passer_player_id
            for row in starters.itertuples()
        }
        self.strength_history = {
            span: qb_layer.strength_history(self.qb_games, self.game_index, span)
            for span in qb_spans
        }
        # One slope per season from a long-memory reference config; the map
        # from win totals to points is not an engine-selection question.
        reference = JointScoringConfig(
            rating_half_life_weeks=float("inf"),
            strength_prior_sd_ppd=0.35,
            student_t_degrees_of_freedom=SELECTION_DF,
        )
        self.slopes = {
            season: points_per_win(list(range(HISTORY_START_SEASON, season)), reference)
            for season in seasons
        }
        self.season_games = {season: store.season_games(season) for season in seasons}
        self.previous_games = {
            season: store.season_games(season - 1) for season in seasons
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
    starter_of = data.starter_of

    rows = []
    for season in seasons:
        games = data.season_games[season]
        previous_games = data.previous_games[season]
        final_week = int(previous_games["model_week"].max()) + 1
        previous_fit = fit_joint_scoring(
            previous_games,
            final_week,
            datetime.now(UTC),
            engine_config,
        )
        prior = build_preseason_prior(
            season,
            as_of=datetime.now(UTC),
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
                except ValueError as error:
                    print(f"walk-forward {season} week {week} skipped: {error}")
                    continue
            contexts = {
                span: qb_layer.context_before_week(
                    data.strength_history[span],
                    season,
                    week,
                    engine_config.rating_half_life_weeks,
                )
                for span in qb_spans
            }
            for game in slate.itertuples():
                game_id = str(game.game_id)
                engine = fit.engine_projection(game)
                record = {
                    "season": season,
                    "model_week": int(week),
                    "week": int(game.week),
                    "season_type": game.season_type,
                    "game_id": game_id,
                    "neutral_site": bool(game.neutral_site),
                    "engine_margin": engine.expected_home - engine.expected_away,
                    "engine_total": engine.expected_home + engine.expected_away,
                    "margin_sd": engine.margin_sd,
                    "total_sd": engine.total_sd,
                    "correlation": engine.correlation,
                    # Clipped rest difference in days; the layer multiplies
                    # by its points-per-day coefficient.
                    "rest_diff": rest_adjustment(game.home_rest, game.away_rest, 1.0),
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
                        adj += sign * contexts[span].adjustment(team, passer)
                    record[f"qb_adj_{int(span)}"] = adj
                rows.append(record)
        if verbose:
            print(f"walk-forward {season}: {len(rows)} cumulative rows")
    return pd.DataFrame(rows)


def apply_layers(predictions: pd.DataFrame, config: LayerConfig) -> pd.DataFrame:
    """Rescore engine predictions with the production output layers."""
    out = predictions.copy()
    qb_column = f"qb_adj_{int(config.qb_span_dropbacks)}"
    if qb_column not in out.columns:
        raise KeyError(f"{qb_column} not precomputed")
    pure = (
        out["engine_margin"]
        + config.qb_adjustment_weight * out[qb_column]
        + config.rest_points_per_day * out["rest_diff"]
    ).to_numpy()
    out["pure_model_margin"] = pure
    out["model_margin"] = blend_margin(pure, out["market_margin"], config.market_weight)
    return out


def margin_log_loss(frame: pd.DataFrame, scale: float, df: float) -> float:
    z = (frame["actual_margin"] - frame["model_margin"]) / (frame["margin_sd"] * scale)
    density = student_t.pdf(z, df) / (frame["margin_sd"] * scale)
    return float(-np.mean(np.log(np.maximum(density, DENSITY_FLOOR))))


def honesty_report(frame: pd.DataFrame) -> pd.DataFrame:
    graded = frame[frame["closing_spread"].notna()].copy()
    graded["model_error"] = (graded["actual_margin"] - graded["model_margin"]).abs()
    graded["market_error"] = (graded["actual_margin"] + graded["closing_spread"]).abs()
    report = graded.groupby("season").agg(
        games=("game_id", "size"),
        model_mae=("model_error", "mean"),
        market_mae=("market_error", "mean"),
    )
    report["gap"] = report["model_mae"] - report["market_mae"]
    return report.round(3)


def coverage_report(frame: pd.DataFrame, scale: float, df: float) -> dict[str, float]:
    z = (frame["actual_margin"] - frame["model_margin"]) / (frame["margin_sd"] * scale)
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
        student_t_degrees_of_freedom=SELECTION_DF,
    )
    for half_life, prior_sd, shrinkage in itertools.product(
        (float("inf"), 12.0, 6.0), (0.25, 0.35, 0.45), (0.1, 0.5, 0.8)
    )
]

SD_GRID = list(itertools.product((0.85, 0.925, 1.0, 1.075), (7.0, 50.0, 500.0)))

LAYER_GRID = [
    LayerConfig(market_weight=w, rest_points_per_day=r)
    for w, r in itertools.product(
        (0.0, 0.15, 0.25, 0.35, 0.5), (0.0, 0.04, 0.08)
    )
]

PRESEASON_GRID = [
    PreseasonConfig(carryover=c, win_total_blend=q)
    for c, q in itertools.product((0.5, 0.6, 0.7), (0.2, 0.35, 0.5))
]

_lowest_loss = itemgetter(0)


def select_qb_layer(predictions: pd.DataFrame) -> LayerConfig:
    """Select QB stability and shrinkage on development pure-model log loss."""
    development = predictions[predictions["season"].isin(DEVELOPMENT_SEASONS)]
    if development.empty:
        raise ValueError("QB selection requires development seasons")
    candidates = []
    for span, weight in itertools.product(QB_SPANS, (0.0, 0.25, 0.5, 0.75, 1.0)):
        layer = LayerConfig(
            market_weight=0.0,
            qb_span_dropbacks=span,
            qb_adjustment_weight=weight,
        )
        loss = margin_log_loss(apply_layers(development, layer), 1.0, 7.0)
        candidates.append((loss, layer))
    return min(candidates, key=_lowest_loss)[1]


def rescore_qb_layer(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep frozen engine forecasts and replace only their QB deltas.

    Actual starters are supplied as lineup scenarios, as in the original
    conditional backtest. This does not evaluate starter-availability forecasts.
    """
    seasons = list(range(HISTORY_START_SEASON, int(predictions["season"].max()) + 1))
    index = store.game_index(seasons)
    games = store.qb_games(seasons)
    starters = {
        (row.game_id, row.team): row.passer_player_id
        for row in games[games["started"]].itertuples()
    }
    from backend.model.joint_scoring import DEFAULT_CONFIG

    out = predictions.copy()
    for span in QB_SPANS:
        history = qb_layer.strength_history(games, index, span)
        deltas = {}
        for (season, week), slate in out.groupby(["season", "model_week"]):
            context = qb_layer.context_before_week(
                history, season, week, DEFAULT_CONFIG.rating_half_life_weeks
            )
            for game in slate.itertuples():
                deltas[game.game_id] = context.adjustment(
                    game.home_team, starters.get((game.game_id, game.home_team))
                ) - context.adjustment(
                    game.away_team, starters.get((game.game_id, game.away_team))
                )
        out[f"qb_adj_{int(span)}"] = out["game_id"].map(deltas)
    return out


def run_calibration(verbose: bool = True) -> dict:
    development = tuple(DEVELOPMENT_SEASONS)
    results = []

    def log(message: str) -> None:
        if verbose:
            print(message)

    data = WalkForwardData(EVAL_SEASONS)
    log("prepared walk-forward inputs")

    # Stage 1: engine core + prior-means flag, scored on dev margin log loss
    # with the production layer defaults.
    scored = []
    for use_prior in (True, False):
        for config in ENGINE_GRID:
            predictions = generate_walk_forward(
                config, development, use_prior, data=data
            )
            layered = apply_layers(predictions, LayerConfig())
            loss = margin_log_loss(layered, 1.0, SELECTION_DF)
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
            scored.append((loss, config, use_prior))
            log(
                f"engine hl={config.rating_half_life_weeks} "
                f"sd={config.strength_prior_sd_ppd} "
                f"shr={config.covariance_shrinkage} prior={use_prior} "
                f"-> {loss:.4f}"
            )
    _, engine_config, use_prior_means = min(scored, key=_lowest_loss)

    # Stage 2: layers, on the winning engine's dev predictions.
    dev_predictions = generate_walk_forward(
        engine_config, development, use_prior_means, data=data
    )
    qb_layer_config = select_qb_layer(dev_predictions)
    scored = []
    for layer in LAYER_GRID:
        layer = replace(
            layer,
            qb_span_dropbacks=qb_layer_config.qb_span_dropbacks,
            qb_adjustment_weight=qb_layer_config.qb_adjustment_weight,
        )
        layered = apply_layers(dev_predictions, layer)
        loss = margin_log_loss(layered, 1.0, SELECTION_DF)
        mae = float((layered["actual_margin"] - layered["model_margin"]).abs().mean())
        results.append(
            {"stage": "layers", **asdict(layer), "margin_log_loss": loss, "mae": mae}
        )
        scored.append((loss, layer))
    _, layer_config = min(scored, key=_lowest_loss)
    log(f"selected layers: {layer_config}")

    # Stage 3: preseason carryover/blend, scored on weeks 1-4 only.
    scored = []
    for preseason_config in PRESEASON_GRID:
        predictions = generate_walk_forward(
            engine_config,
            development,
            use_prior_means,
            preseason_config,
            data=data,
        )
        early = apply_layers(predictions[predictions["model_week"] <= 4], layer_config)
        loss = margin_log_loss(early, 1.0, SELECTION_DF)
        results.append(
            {
                "stage": "preseason",
                **asdict(preseason_config),
                "early_margin_log_loss": loss,
            }
        )
        scored.append((loss, preseason_config))
        log(
            f"preseason c={preseason_config.carryover} "
            f"q={preseason_config.win_total_blend} -> {loss:.4f}"
        )
    _, preseason_config = min(scored, key=_lowest_loss)

    # Stage 4: sd scale and df on dev, then untouched holdout report.
    dev_predictions = generate_walk_forward(
        engine_config,
        development,
        use_prior_means,
        preseason_config,
        data=data,
    )
    dev_layered = apply_layers(dev_predictions, layer_config)
    scored = []
    for scale, df in SD_GRID:
        loss = margin_log_loss(dev_layered, scale, df)
        results.append(
            {"stage": "sd", "scale": scale, "df": df, "margin_log_loss": loss}
        )
        scored.append((loss, scale, df))
    _, scale, df = min(scored, key=_lowest_loss)
    engine_config = replace(
        engine_config,
        student_t_degrees_of_freedom=df,
        score_covariance_scale=scale,
    )
    log(f"selected engine: {engine_config}")
    log(f"selected preseason: {preseason_config}")

    holdout_predictions = generate_walk_forward(
        engine_config,
        tuple(HOLDOUT_SEASONS),
        use_prior_means,
        preseason_config,
        data=data,
    )
    holdout = apply_layers(holdout_predictions, layer_config)
    summary = {
        "engine_config": engine_config,
        "layer_config": layer_config,
        "preseason_config": preseason_config,
        "use_prior_means": use_prior_means,
        "dev_margin_log_loss": margin_log_loss(dev_layered, scale, df),
        "holdout_margin_log_loss": margin_log_loss(holdout, scale, df),
        "holdout_coverage": coverage_report(holdout, scale, df),
        "dev_honesty": honesty_report(dev_layered),
        "holdout_honesty": honesty_report(holdout),
    }
    store.write_processed(
        pd.DataFrame(results), "calibration", "search_history.parquet"
    )
    combined = pd.concat([dev_layered, holdout], ignore_index=True)
    store.write_processed(combined, "calibration", "predictions.parquet")

    # Freeze the discrete margin residual distribution as a small committed
    # artifact so cover/push pricing works on clean checkouts (CI) without
    # the full calibration output.
    distribution = fit_margin_residual_distribution(
        (combined["actual_margin"] - combined["model_margin"]).to_numpy()
    )
    pd.DataFrame(
        {
            "margin_offset": np.arange(-RESIDUAL_RANGE, RESIDUAL_RANGE + 1),
            "probability": distribution,
        }
    ).to_csv(STATIC_DIR / "margin_distribution.csv", index=False)
    return summary
