"""Weekly scoring of a season's traits with the frozen grade model.

The model and its expectation tables are only loaded, never refit, and a model
whose development seasons include the scored season is refused. The artifact
keeps every snapshot week; the publish reads the latest.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.etl import store
from backend.players import EPA_TRAIT, MODEL_VERSION, grade, season_stats, value, war
from backend.players.backtest import build_season_traits
from backend.players.context import load_games, load_season_context
from backend.players.expectations import (
    expectation_tables_path,
    load_expectation_tables,
)

log = logging.getLogger(__name__)

POSITION_TRAIT_COLUMNS = [
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
]
GRADE_ROW, WAR_ROW, SPREAD_ROW = "grade", "war", "spread_value"


def position_traits_artifact(season: int) -> tuple[str, ...]:
    return ("players", "position_traits", f"{season}.parquet")


def season_stats_artifact(season: int) -> tuple[str, ...]:
    return ("players", "season_stats", f"{season}.parquet")


def frozen_expectation_tables(model_path):
    return load_expectation_tables(
        expectation_tables_path(model_path.parent, grade.grade_cutoff(model_path))
    )


def _percentile(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    rank = values.groupby(keys).rank(method="average")
    n = values.groupby(keys).transform("size")
    return (100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0)


def _identity(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "athlete_id",
        "athlete_name",
        "headshot_url",
        "team",
        "position_group",
        "games",
    ]
    return (
        snapshot.astype({"athlete_id": str})
        .sort_values("games", ascending=False, kind="stable")
        .drop_duplicates(["week", "athlete_id", "position_group"])[["week", *columns]]
    )


def grade_rows(snapshot: pd.DataFrame, model: grade.GradeModel) -> pd.DataFrame:
    """One ``grade`` row per graded player-week."""
    parts = []
    for week, part in snapshot.groupby("week"):
        scored = grade.score_grades(part, model)
        parts.append(scored.assign(week=week))
    if not parts:
        return pd.DataFrame()
    scored = pd.concat(parts, ignore_index=True)
    out = scored.merge(
        _identity(snapshot).drop(columns="games"),
        on=["week", "athlete_id", "position_group"],
        how="left",
    )
    return out.assign(
        trait=GRADE_ROW,
        value=out["grade"],
        percentile=out["grade_percentile"],
        kind="counted",
        in_grade=True,
        qualified=True,
        points_above_replacement=np.nan,
    )


def war_rows(
    snapshot: pd.DataFrame,
    model: grade.GradeModel,
    credited: pd.DataFrame,
    projections: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    """One ``war`` row per qualified credited player-week."""
    special, kicked = war.special_credit(frames, snapshot)
    values = war.shrunk_credit(
        snapshot,
        model.epa_shrink_games,
        war.hurries(int(snapshot["season"].iloc[0])),
        special,
    )
    if values.empty:
        return pd.DataFrame()
    scored = war.wins_above_replacement(
        values, pd.concat([credited, kicked], ignore_index=True), projections
    )
    out = scored.merge(
        _identity(snapshot).drop(columns="games"),
        on=["week", "athlete_id", "position_group"],
        how="left",
    )
    out["value"] = out["wins_above_replacement"]
    out["percentile"] = _percentile(out["value"], [out["week"], out["position_group"]])
    return out.assign(
        trait=WAR_ROW,
        kind="counted",
        in_grade=False,
        qualified=True,
        uncertainty=np.nan,
    )


def spread_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    """One ``spread_value`` row per priced player-week: the points the team
    model's line gives up if he is ruled out of the next game, as the
    absence layer prices it. Quarterbacks have their own layer and the line
    is rated as a unit, so neither carries one."""
    from backend.model import absence_layer
    from backend.model.joint_scoring import DEFAULT_CONFIG
    from backend.model.projections import LayerConfig

    season = int(snapshot["season"].iloc[0])
    seasons = [s for s in (season - 1, season) if s >= absence_layer.FIRST_TRAIT_SEASON]
    identity = _identity(snapshot)
    identity = identity[identity["position_group"].isin(absence_layer.GROUPS)]
    values = absence_layer.spread_values(
        absence_layer.load_inputs(seasons),
        season,
        identity[["week", "athlete_id", "team"]],
        DEFAULT_CONFIG.rating_half_life_weeks,
        LayerConfig().absence_weight,
    )
    if values.empty:
        return pd.DataFrame()
    out = values.merge(identity, on=["week", "athlete_id"])
    out["percentile"] = _percentile(out["value"], [out["week"], out["position_group"]])
    return out.assign(
        trait=SPREAD_ROW,
        kind="counted",
        in_grade=False,
        qualified=True,
        uncertainty=np.nan,
        reliability=np.nan,
        points_above_replacement=np.nan,
    )


def value_rows(
    snapshot: pd.DataFrame, model: grade.GradeModel, war_scored: pd.DataFrame
) -> pd.DataFrame:
    """The latest week's ``player_value`` and ``trade_value`` rows."""
    if war_scored.empty:
        return pd.DataFrame()
    season = int(snapshot["season"].iloc[0])
    week = int(war_scored["week"].max())
    current = war_scored[war_scored["week"].eq(week)].rename(
        columns={"wins_above_replacement": "war"}
    )
    prior = value.season_war(season - 1, model.epa_shrink_games)
    scored = value.score_values(
        current,
        prior[prior["week"].eq(prior["week"].max())],
        value.load_value_model(season),
        season,
        week,
    )
    identity = _identity(snapshot)
    out = scored.merge(
        identity[identity["week"].eq(week)], on=["athlete_id", "position_group"]
    )
    out["percentile"] = _percentile(out["value"], [out["trait"], out["position_group"]])
    return out.assign(
        kind="counted",
        in_grade=False,
        qualified=True,
        reliability=np.nan,
        points_above_replacement=np.nan,
    )


def score_snapshot(
    snapshot, model, credited, frames, projections, as_of: datetime
) -> pd.DataFrame:
    """Trait rows plus the grade, war, spread value, player value and trade value
    rows, in ``POSITION_TRAIT_COLUMNS``."""
    weighted = model.weights[model.weights["weight"].ne(0)]
    weighted_keys = set(zip(weighted["position_group"], weighted["trait"]))
    traits = snapshot.astype({"athlete_id": str}).copy()
    traits["in_grade"] = [
        key in weighted_keys for key in zip(traits["position_group"], traits["trait"])
    ]
    traits["uncertainty"] = np.nan
    # The trait's pooled reliability at the grade's weight game count, so the
    # card's r means the same thing at every position.
    reliability = model.weights.set_index(["position_group", "trait"])["reliability"]
    traits["reliability"] = reliability.reindex(
        pd.MultiIndex.from_frame(traits[["position_group", "trait"]])
    ).to_numpy(float)
    traits["points_above_replacement"] = np.nan
    war_scored = war_rows(snapshot, model, credited, projections, frames)
    parts = [
        traits,
        grade_rows(snapshot, model),
        war_scored,
        spread_rows(snapshot),
        value_rows(snapshot, model, war_scored),
    ]
    out = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    out["reliability"] = out.get("reliability", np.nan)
    out["season"] = int(snapshot["season"].iloc[0])
    out["as_of"] = as_of.isoformat()
    out["model_version"] = MODEL_VERSION
    out["qualified"] = out["qualified"].eq(True)
    return out[[*POSITION_TRAIT_COLUMNS, "qualified"]].sort_values(
        ["week", "trait", "athlete_id"], ignore_index=True
    )


def run_position_traits(season: int, as_of: datetime) -> pd.DataFrame:
    """Score the season's weekly trait snapshots with the latest earlier grade model."""
    model_path = grade.latest_grade_model_path(season)
    model = grade.load_grade_model(model_path)
    grade.validate_grade_model(model, season)
    tables = frozen_expectation_tables(model_path)
    ctx = load_season_context(season)
    built = build_season_traits(season, tables=tables, ctx=ctx)
    snapshot = built["snapshots"]
    if snapshot.empty:
        log.info("position traits %s: no completed week to score", season)
        out = pd.DataFrame(columns=[*POSITION_TRAIT_COLUMNS, "qualified"])
        store.write_processed(out, *position_traits_artifact(season))
        return out
    from backend.awards import ingest

    games = load_games(season)
    projections = war.season_projections(
        season, games, ingest.read("schedules", season)
    )
    out = score_snapshot(
        snapshot, model, built["credited"], built["frames"], projections, as_of
    )
    out = with_fitted_ratings(out, season, as_of)
    store.write_processed(out, *position_traits_artifact(season))
    from backend.players.ratings import rating_games

    lines = season_stats.season_stat_lines(
        ctx, rating_games("QB", season)[1], latest_trait_week(season)
    )
    store.write_processed(lines, *season_stats_artifact(season))
    log.info("season stat lines %s: %d athletes", season, len(lines))
    latest = int(snapshot["week"].max())
    losses = grade.coverage_losses(
        grade.grade_eligibility(snapshot[snapshot["week"].eq(latest)], model)
    )
    for row in losses.itertuples():
        log.info(
            "position grade %s week %d %s: %d qualified, %d graded, "
            "%d without the primary trait, %d under coverage",
            season,
            latest,
            row.position_group,
            row.qualified,
            row.graded,
            row.no_primary,
            row.low_coverage,
        )
    return out


def with_fitted_ratings(
    out: pd.DataFrame, season: int, as_of: datetime
) -> pd.DataFrame:
    """Replace the rated positions' trait and grade rows with the fitted ratings,
    keeping their wins above replacement, spread value and contract value rows."""
    from backend.players.ratings import RATED_POSITIONS, score_ratings

    weeks = sorted(int(w) for w in out["week"].unique())
    headshots = (
        out.dropna(subset=["headshot_url"])
        .drop_duplicates("athlete_id")
        .set_index("athlete_id")["headshot_url"]
    )
    rated = score_ratings(season, weeks, as_of, headshots)
    if rated.empty:
        return out
    keep = ~out["position_group"].isin(RATED_POSITIONS) | out["trait"].isin(
        [WAR_ROW, SPREAD_ROW, value.PLAYER_VALUE_ROW, value.TRADE_VALUE_ROW]
    )
    rated = rated.reindex(columns=out.columns)
    return pd.concat([out[keep], rated], ignore_index=True).sort_values(
        ["week", "trait", "athlete_id"], ignore_index=True
    )


def season_stat_lines(season: int) -> pd.DataFrame:
    """The stat lines the last ``position-traits`` run wrote; none before one has."""
    path = PROCESSED_DIR.joinpath(*season_stats_artifact(season))
    if not path.exists():
        return pd.DataFrame(columns=season_stats.STAT_LINE_COLUMNS)
    return store.read_processed(*season_stats_artifact(season))


def latest_trait_week(season: int) -> pd.DataFrame:
    """The newest weekly snapshot, qualified players only; the artifact keeps all."""
    path = PROCESSED_DIR.joinpath(*position_traits_artifact(season))
    if not path.exists():
        return pd.DataFrame(columns=POSITION_TRAIT_COLUMNS)
    traits = store.read_processed(*position_traits_artifact(season))
    if traits.empty:
        return pd.DataFrame(columns=POSITION_TRAIT_COLUMNS)
    latest = traits["week"].eq(traits["week"].max()) & traits["qualified"].eq(True)
    return traits[latest].reset_index(drop=True)[POSITION_TRAIT_COLUMNS]


__all__ = [
    "EPA_TRAIT",
    "POSITION_TRAIT_COLUMNS",
    "latest_trait_week",
    "run_position_traits",
    "season_stat_lines",
]
