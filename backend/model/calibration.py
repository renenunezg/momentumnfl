"""Walk-forward calibration. Every week of every evaluation season is refit
and projected strictly out-of-sample. Engine hyperparameters, output-layer
parameters, the preseason prior, and the score-distribution scale are
selected in sequence on development-season margin log loss; previously
inspected validation seasons are labeled retrospective.

The expensive pass (generate_walk_forward) produces per-game engine numbers;
layer parameters rescore those vectorially without refitting."""

import itertools
from dataclasses import asdict, replace
from datetime import timedelta
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
from backend.features.qb import expected_starters
from backend.model import qb_adjustment as qb_layer
from backend.model.distributions import student_t_scale
from backend.model.fit_week import load_depth_charts
from backend.model.joint_scoring import JointScoringConfig, fit_joint_scoring
from backend.model.market_blend import (
    MARGINS,
    blend_margin,
    fit_margin_distribution,
    integer_margin_probabilities,
)
from backend.model.preseason import (
    PreseasonConfig,
    build_preseason_prior,
    load_qb_references,
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

    def __init__(self, seasons: tuple[int, ...], qb_spans=QB_SPANS, rebuild=False):
        history = list(range(HISTORY_START_SEASON, max(seasons) + 1))
        if rebuild:
            from backend.model.validation import rebuild_history

            all_games, self.qb_games = rebuild_history(history)
        else:
            all_games = {s: store.season_games(s) for s in history}
            self.qb_games = store.qb_games(history)
        self.all_games = all_games
        self.game_index = pd.concat(
            [
                g[["game_id", "season", "model_week", "start_date"]]
                for g in all_games.values()
            ],
            ignore_index=True,
        )
        self.depth_charts = {s: load_depth_charts(s) for s in seasons}
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
            season: points_per_win(
                list(range(HISTORY_START_SEASON, season)), reference, all_games
            )
            for season in seasons
        }
        self.season_games = {season: all_games[season] for season in seasons}
        self.previous_games = {season: all_games[season - 1] for season in seasons}
        self.forecast_inputs = {}
        self.contexts = {}

    def forecast_inputs_at(self, season, week, slate):
        key = season, week
        if key not in self.forecast_inputs:
            cutoff = pd.to_datetime(slate.start_date, utc=True).min() - timedelta(
                seconds=1
            )
            eligible = self.game_index[
                pd.to_datetime(self.game_index.start_date, utc=True).le(
                    cutoff - timedelta(days=1)
                )
            ]
            expected = expected_starters(
                self.qb_games,
                eligible,
                self.depth_charts[season],
                season,
                week,
                as_of=cutoff,
                use_overrides=False,
            )
            self.forecast_inputs[key] = cutoff.to_pydatetime(), eligible, expected
        return self.forecast_inputs[key]


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

    rows = []
    for season in seasons:
        games = data.season_games[season]
        previous_games = data.previous_games[season]
        final_week = int(previous_games["model_week"].max()) + 1
        preseason_cutoff = pd.to_datetime(games.start_date, utc=True).min() - timedelta(
            seconds=1
        )
        previous_fit = fit_joint_scoring(
            previous_games,
            final_week,
            preseason_cutoff.to_pydatetime(),
            engine_config,
        )
        prior = build_preseason_prior(
            season,
            as_of=preseason_cutoff.to_pydatetime(),
            config=preseason_config,
            engine_config=engine_config,
            previous_fit=previous_fit,
            slope=data.slopes[season],
        )
        qb_references = load_qb_references(season)
        weeks = sorted(games["model_week"].unique())
        for week in weeks:
            slate = games[games["model_week"].eq(week)]
            as_of, eligible, expected = data.forecast_inputs_at(season, week, slate)
            if week == weeks[0]:
                fit = prior.week1_fit()
                fit.as_of = as_of
            else:
                available = games[
                    games["game_id"].isin(eligible["game_id"])
                    | games["model_week"].ge(week)
                ]
                fit = fit_joint_scoring(
                    available,
                    week,
                    as_of,
                    engine_config,
                    strength_prior=prior.week1_fit() if use_prior_means else None,
                )
            context_key = (
                season,
                week,
                engine_config.rating_half_life_weeks,
                preseason_config.win_total_blend,
                qb_spans,
            )
            if context_key not in data.contexts:
                data.contexts[context_key] = {
                    span: qb_layer.context_before_week(
                        data.strength_history[span][
                            data.strength_history[span]["game_id"].isin(
                                eligible["game_id"]
                            )
                        ],
                        season,
                        week,
                        engine_config.rating_half_life_weeks,
                        qb_references,
                        preseason_config.win_total_blend,
                    )
                    for span in qb_spans
                }
            contexts = data.contexts[context_key]
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
                    "forecast_cutoff": as_of,
                    "home_expected_qb": expected.get(str(game.home_team)),
                    "away_expected_qb": expected.get(str(game.away_team)),
                    "market_input_basis": "closing_line_conditional_benchmark",
                    "qb_input_basis": "dated_snapshot_or_prior_week_proxy",
                    "model_version": "nfl_joint_scoring_qb_v4",
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
                        passer = expected.get(team)
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
    z = (frame["actual_margin"] - frame["model_margin"]) / student_t_scale(
        frame["margin_sd"] * scale, df
    )
    density = student_t.pdf(z, df) / student_t_scale(frame["margin_sd"] * scale, df)
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
    z = (frame["actual_margin"] - frame["model_margin"]) / student_t_scale(
        frame["margin_sd"] * scale, df
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
        student_t_degrees_of_freedom=SELECTION_DF,
    )
    for half_life, prior_sd, shrinkage in itertools.product(
        (float("inf"), 12.0, 6.0), (0.25, 0.35, 0.45), (0.1, 0.5, 0.8)
    )
]

SD_GRID = list(
    itertools.product((0.85, 0.925, 1.0, 1.075, 1.15, 1.25), (7.0, 50.0, 500.0))
)

LAYER_GRID = [
    LayerConfig(market_weight=w, rest_points_per_day=r)
    for w, r in itertools.product((0.0, 0.15, 0.25, 0.35, 0.5), (0.0, 0.04, 0.08))
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


def rescore_qb_layer(
    predictions: pd.DataFrame,
    anchor_preseason: bool = True,
    qb_spans: tuple[float, ...] = QB_SPANS,
    calendar_half_life: float = qb_layer.CALENDAR_HALF_LIFE_WEEKS,
    prior_dropbacks: float = qb_layer.SHRINK_DROPBACKS,
) -> pd.DataFrame:
    """Keep frozen engine forecasts and replace only their QB deltas.

    Starter identities must come from the original forecast cutoff. Legacy
    artifacts with outcome-derived starters must be regenerated first.
    """
    seasons = list(range(HISTORY_START_SEASON, int(predictions["season"].max()) + 1))
    index = store.game_index(seasons)
    games = store.qb_games(seasons)
    starters = frozen_starters(predictions)
    from backend.model.joint_scoring import DEFAULT_CONFIG

    out = predictions.copy()
    references = {
        season: load_qb_references(season) if anchor_preseason else {}
        for season in out["season"].unique()
    }
    for span in qb_spans:
        history = qb_layer.strength_history(
            games, index, span, calendar_half_life, prior_dropbacks
        )
        deltas = {}
        for (season, week), slate in out.groupby(["season", "model_week"]):
            cutoff = pd.to_datetime(slate["forecast_cutoff"], utc=True).min()
            context = qb_layer.context_before_week(
                history[
                    pd.to_datetime(history.start_date, utc=True).le(
                        cutoff - timedelta(days=1)
                    )
                ],
                season,
                week,
                DEFAULT_CONFIG.rating_half_life_weeks,
                references[season],
                PreseasonConfig().win_total_blend,
            )
            for game in slate.itertuples():
                deltas[game.game_id] = context.adjustment(
                    game.home_team, starters.get((game.game_id, game.home_team))
                ) - context.adjustment(
                    game.away_team, starters.get((game.game_id, game.away_team))
                )
        out[f"qb_adj_{int(span)}"] = out["game_id"].map(deltas)
    return out


def frozen_starters(predictions: pd.DataFrame) -> dict:
    required = {"forecast_cutoff", "home_expected_qb", "away_expected_qb"}
    if not required.issubset(predictions.columns):
        raise ValueError("Regenerate forecasts with cutoff-safe QB identities")
    return {
        (row.game_id, getattr(row, f"{side}_team")): (
            None
            if pd.isna(getattr(row, f"{side}_expected_qb"))
            else getattr(row, f"{side}_expected_qb")
        )
        for row in predictions.itertuples()
        for side in ("home", "away")
    }


def select_qb_memory(predictions: pd.DataFrame) -> dict:
    """Reproduce the memory search on development data at a fixed 500-DB span."""
    development = predictions[predictions["season"].isin(DEVELOPMENT_SEASONS)]
    choices = []
    for calendar, prior in itertools.product(
        (52.0, 104.0, 208.0), (50.0, 100.0, 200.0)
    ):
        rescored = rescore_qb_layer(
            development,
            qb_spans=(500.0,),
            calendar_half_life=calendar,
            prior_dropbacks=prior,
        )
        for weight in (0.5, 0.75, 1.0):
            layer = LayerConfig(
                market_weight=0.0, qb_span_dropbacks=500.0, qb_adjustment_weight=weight
            )
            loss = margin_log_loss(apply_layers(rescored, layer), 1.0, 7.0)
            choices.append((loss, calendar, prior, weight))
    loss, calendar, prior, weight = min(choices, key=_lowest_loss)
    return {
        "calendar_half_life": calendar,
        "prior_dropbacks": prior,
        "qb_weight": weight,
        "development_pure_nll": loss,
    }


def evaluate_qb_context(predictions: pd.DataFrame) -> dict:
    """Read-only development selection for the experimental context diagnostics."""
    from backend.features.qb_context import load_context_games
    from backend.model.qb_context import fit_context

    observations = load_context_games(int(predictions["season"].max()))
    seasons = list(range(HISTORY_START_SEASON, int(predictions["season"].max()) + 1))
    games = store.qb_games(seasons)
    index = store.game_index(seasons)
    layer = LayerConfig()
    history = qb_layer.strength_history(games, index, layer.qb_span_dropbacks)
    starters = frozen_starters(predictions)
    references = {s: load_qb_references(s) for s in predictions["season"].unique()}
    raw = rescore_qb_layer(predictions)
    raw_delta = raw[f"qb_adj_{int(layer.qb_span_dropbacks)}"]
    development = raw["season"].isin(DEVELOPMENT_SEASONS)
    candidates = []
    for qb_penalty, team_penalty in itertools.product((100.0, 300.0), repeat=2):
        deltas, variances = {}, {}
        for (season, week), slate in raw.groupby(["season", "model_week"]):
            fitted = fit_context(observations, season, week, qb_penalty, team_penalty)
            context = qb_layer.context_before_week(
                history,
                season,
                week,
                6.0,
                references[season],
                PreseasonConfig().win_total_blend,
                fitted.strengths,
            )
            contrasts = []
            for game in slate.itertuples():
                contrast = {}
                for sign, team in ((1, game.home_team), (-1, game.away_team)):
                    for passer, weight in context.contrast(
                        team, starters.get((game.game_id, team))
                    ).items():
                        contrast[passer] = contrast.get(passer, 0.0) + sign * weight
                deltas[game.game_id] = sum(
                    w * fitted.strengths.get(q, 0.0) for q, w in contrast.items()
                )
                contrasts.append(contrast)
            variances.update(zip(slate["game_id"], fitted.contrast_sd(contrasts) ** 2))
        context_delta = raw["game_id"].map(deltas)
        variance = raw["game_id"].map(variances)
        for share in (0.0, 0.25, 0.5, 0.75, 1.0):
            frame = raw.copy()
            frame["pure_model_margin"] = frame["engine_margin"] + (
                layer.qb_adjustment_weight
                * ((1 - share) * raw_delta + share * context_delta)
            )
            frame["model_margin"] = frame["pure_model_margin"]
            loss = margin_log_loss(frame[development], 1.0, 7.0)
            candidates.append((loss, share, qb_penalty, team_penalty, frame, variance))
    loss, share, qp, tp, frame, variance = min(candidates, key=_lowest_loss)
    frame["model_margin"] = blend_margin(
        frame["pure_model_margin"].to_numpy(),
        frame["market_margin"].to_numpy(),
        layer.market_weight,
    )
    market_weight = np.where(frame["market_margin"].notna(), layer.market_weight, 0)
    # Translate normal parameter variance into the Student-t scale convention.
    added_scale = (
        (5 / 7) * (1 - market_weight) ** 2 * layer.qb_adjustment_weight**2 * variance
    )
    scores = []
    for weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        trial = frame.copy()
        trial["margin_sd"] = np.sqrt(frame["margin_sd"] ** 2 + weight * added_scale)
        scores.append((margin_log_loss(trial[development], 1.0, 7.0), weight, trial))
    _, variance_weight, selected = min(scores, key=_lowest_loss)
    return {
        "context_share": share,
        "variance_weight": variance_weight,
        "qb_penalty": qp,
        "team_penalty": tp,
        "development_pure_nll": loss,
        "development_blended_nll": margin_log_loss(selected[development], 1.0, 7.0),
        "holdout_blended_nll": margin_log_loss(selected[~development], 1.0, 7.0),
    }


def run_calibration(
    verbose: bool = True,
    data: WalkForwardData | None = None,
    write_artifacts: bool = True,
) -> dict:
    development = tuple(DEVELOPMENT_SEASONS)
    results = []

    def log(message: str) -> None:
        if verbose:
            print(message)

    data = data or WalkForwardData(EVAL_SEASONS)
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
        replace(engine_config, score_covariance_scale=1.0),
        tuple(HOLDOUT_SEASONS),
        use_prior_means,
        preseason_config,
        data=data,
    )
    holdout = apply_layers(holdout_predictions, layer_config)
    return finish_calibration(
        dev_layered,
        holdout,
        engine_config,
        layer_config,
        preseason_config,
        use_prior_means,
        results,
        write_artifacts,
    )


def run_uncertainty_calibration(data: WalkForwardData, write_artifacts=False) -> dict:
    """Repair uncertainty calibration with the existing mean model held fixed."""
    from backend.model.joint_scoring import DEFAULT_CONFIG

    engine = replace(DEFAULT_CONFIG, score_covariance_scale=1.0)
    layer, preseason = LayerConfig(), PreseasonConfig()
    predictions = apply_layers(
        generate_walk_forward(engine, EVAL_SEASONS, data=data), layer
    )
    development = predictions[predictions.season.isin(DEVELOPMENT_SEASONS)]
    holdout = predictions[predictions.season.isin(HOLDOUT_SEASONS)]
    choices = [
        (margin_log_loss(development, scale, df), scale, df) for scale, df in SD_GRID
    ]
    _, scale, df = min(choices)
    results = [
        {"stage": "uncertainty", "margin_log_loss": loss, "scale": scale, "df": df}
        for loss, scale, df in choices
    ]
    return finish_calibration(
        development,
        holdout,
        replace(engine, score_covariance_scale=scale, student_t_degrees_of_freedom=df),
        layer,
        preseason,
        True,
        results,
        write_artifacts,
    )


def finish_calibration(
    development,
    validation,
    engine_config,
    layer_config,
    preseason_config,
    use_prior_means,
    results,
    write_artifacts,
):
    """Both inputs have unscaled SDs; scale once before scoring or persisting."""
    scale, df = (
        engine_config.score_covariance_scale,
        engine_config.student_t_degrees_of_freedom,
    )
    dev_layered, holdout = development.copy(), validation.copy()
    for frame in (dev_layered, holdout):
        frame[["margin_sd", "total_sd"]] *= scale
    summary = {
        "engine_config": engine_config,
        "layer_config": layer_config,
        "preseason_config": preseason_config,
        "use_prior_means": use_prior_means,
        "dev_margin_log_loss": margin_log_loss(dev_layered, 1.0, df),
        "holdout_margin_log_loss": margin_log_loss(holdout, 1.0, df),
        "holdout_coverage": coverage_report(holdout, 1.0, df),
        "dev_honesty": honesty_report(dev_layered),
        "holdout_honesty": honesty_report(holdout),
        "validation_basis": "retrospective; closing-line conditional benchmark",
    }
    combined = pd.concat([dev_layered, holdout], ignore_index=True)
    # Calibrate actual-margin key numbers on development only. The held-back
    # seasons never train this pricing layer.
    distribution = fit_margin_distribution(dev_layered, df)
    for name, weights in (
        ("discrete", distribution),
        ("smooth_discrete", np.ones(len(MARGINS))),
    ):
        probabilities = integer_margin_probabilities(
            holdout.model_margin.to_numpy(), holdout.margin_sd.to_numpy(), df, weights
        )
        actual = np.clip(holdout.actual_margin.to_numpy(int), MARGINS[0], MARGINS[-1])
        density = probabilities[np.arange(len(holdout)), actual - MARGINS[0]]
        summary[f"holdout_{name}_nll"] = float(
            -np.log(np.maximum(density, 1e-300)).mean()
        )
    if write_artifacts:
        store.write_processed(
            pd.DataFrame(results), "calibration", "search_history.parquet"
        )
        store.write_processed(combined, "calibration", "predictions.parquet")
        pd.DataFrame({"actual_margin": MARGINS, "weight": distribution}).to_csv(
            STATIC_DIR / "margin_distribution_v2.csv", index=False
        )
    return summary
