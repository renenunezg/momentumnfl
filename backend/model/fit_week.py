"""One production week: engine fit on games before the week, QB and rest
layers for the slate, market blend, ratings and projections out."""

from datetime import UTC, datetime

import pandas as pd

from backend.config import HISTORY_START_SEASON
from backend.etl import store
from backend.features import qb as qb_features
from backend.features.drives import kickoff_utc
from backend.features.scoring import closing_home_spread
from backend.model import absence_layer, seed_lock
from backend.model import qb_adjustment as qb_layer
from backend.model.joint_scoring import (
    DEFAULT_CONFIG,
    JointScoringConfig,
    JointScoringFit,
    fit_joint_scoring,
)
from backend.model.market_rating import with_market_ratings
from backend.model.outputs import GameProjection, TeamRating
from backend.model.preseason import (
    WIN_TOTAL_BLEND,
    build_preseason_prior,
    load_qb_references,
)
from backend.model.projections import LayerConfig, assemble_projections
from backend.model.totals import DEFAULT_TOTALS_CONFIG, TotalsConfig


def recency_by_game(
    games: pd.DataFrame, forecast_week: int, config: JointScoringConfig
) -> dict[str, float]:
    training = games[games["model_week"] < forecast_week]
    if training.empty:
        return {}
    latest = int(training["model_week"].max())
    return {
        str(row.game_id): 0.5
        ** ((latest - row.model_week) / config.rating_half_life_weeks)
        for row in training.itertuples()
    }


def forecast_qb_context(
    season,
    week,
    qb_games,
    depth_charts,
    config,
    layer_config,
    as_of=None,
    injuries=None,
):
    """Shared cutoff QB values, identities and reliability weights."""
    game_index = store.game_index(list(range(HISTORY_START_SEASON, season + 1)))
    eligible = game_index[
        (game_index["season"] < season)
        | (game_index["season"].eq(season) & (game_index["model_week"] < week))
    ]
    as_of = as_of or datetime.now(UTC)
    eligible = eligible[pd.to_datetime(eligible["start_date"], utc=True).lt(as_of)]
    qb_history = qb_games[qb_games["game_id"].isin(set(eligible["game_id"]))]
    history = qb_layer.strength_history(
        qb_history, eligible, layer_config.qb_span_dropbacks
    )
    context = qb_layer.context_before_week(
        history,
        season,
        week,
        config.rating_half_life_weeks,
        load_qb_references(season, as_of),
        WIN_TOTAL_BLEND,
    )
    expected = qb_features.expected_starters(
        qb_history,
        eligible,
        depth_charts,
        season,
        week,
        as_of=as_of,
        injuries=injuries,
    )

    unavailable = qb_features.ruled_out(injuries, season, week, as_of)
    basis = qb_features.identification_basis(depth_charts)
    weights = {
        team: layer_config.qb_weight(
            basis, context.confirmed_absence(team, passer, unavailable)
        )
        for team, passer in expected.items()
    }
    return context, expected, weights


def compute_qb_adjustments(
    season: int,
    week: int,
    slate: pd.DataFrame,
    qb_games: pd.DataFrame,
    depth_charts: pd.DataFrame,
    config: JointScoringConfig,
    layer_config: LayerConfig,
    as_of: datetime | None = None,
    injuries: pd.DataFrame | None = None,
) -> dict[str, tuple[float, float]]:
    """game_id -> (home_adj, away_adj) for the expected starters.

    QB values use career history across seasons (game_index); baselines use
    the recent QB workload, falling back to the previous season in preseason."""
    context, expected, weights = forecast_qb_context(
        season, week, qb_games, depth_charts, config, layer_config, as_of, injuries
    )
    adjustments: dict[str, tuple[float, float]] = {}
    for game in slate.itertuples():
        sides = []
        for team in (str(game.home_team), str(game.away_team)):
            passer = expected.get(team)
            if passer is None:
                sides.append(0.0)
                continue
            weight = weights[team]
            sides.append(context.adjustment(team, passer, weight))
        adjustments[str(game.game_id)] = (sides[0], sides[1])
    return adjustments


def qb_absence_values(
    season: int,
    week: int,
    qb_games: pd.DataFrame,
    depth_charts: pd.DataFrame,
    injuries: pd.DataFrame,
    config: JointScoringConfig,
    layer_config: LayerConfig,
    as_of: datetime,
) -> pd.DataFrame:
    """``team, athlete_id, value``: the points each team's projected margin
    gives up if its expected starter at ``as_of`` is ruled out, as the QB layer
    prices it: his adjustment less that of the starter it names with him listed
    Out, each at the weight it applies. A team without a QB baseline prices no
    substitution, and one whose starter is a verified override would refuse
    the absence, so neither has a row."""
    context, expected, weights = forecast_qb_context(
        season, week, qb_games, depth_charts, config, layer_config, as_of, injuries
    )
    overrides = qb_features.load_qb_overrides()
    overridden = set(
        overrides.loc[
            overrides["season"].eq(season) & overrides["week"].eq(week), "team_abbr"
        ]
    )
    starters = {
        team: passer
        for team, passer in expected.items()
        if team in context.baselines and team not in overridden
    }
    columns = ["team", "athlete_id", "value"]
    if not starters:
        return pd.DataFrame(columns=columns)
    _, replaced, replaced_weights = forecast_qb_context(
        season,
        week,
        qb_games,
        depth_charts,
        config,
        layer_config,
        as_of,
        qb_features.with_ruled_out(injuries, season, week, starters),
    )
    return pd.DataFrame(
        [
            (
                team,
                passer,
                context.adjustment(team, passer, weights[team])
                - context.adjustment(
                    team, replaced.get(team), replaced_weights.get(team, 0.0)
                ),
            )
            for team, passer in starters.items()
        ],
        columns=columns,
    )


def qb_spread_values(
    season: int,
    weeks,
    as_of: datetime,
    config: JointScoringConfig = DEFAULT_CONFIG,
    layer_config: LayerConfig = LayerConfig(),
) -> pd.DataFrame:
    """``week, team, athlete_id, value``: after week ``w``, ``qb_absence_values``
    for week ``w + 1`` at its forecast cutoff, the second before its first
    kickoff or ``as_of`` if earlier, with the injuries a forecast then reads."""
    qb_games = store.qb_games(list(range(HISTORY_START_SEASON, season + 1)))
    depth_charts = load_depth_charts(season)
    parts = []
    for week in weeks:
        target = int(week) + 1
        kickoffs = pd.to_datetime(week_slate(season, target)["start_date"], utc=True)
        cutoff = as_of
        if kickoffs.notna().any():
            cutoff = min(
                as_of, (kickoffs.min() - pd.Timedelta(seconds=1)).to_pydatetime()
            )
        values = qb_absence_values(
            season,
            target,
            qb_games,
            depth_charts,
            load_injuries(season, target, cutoff),
            config,
            layer_config,
            cutoff,
        )
        parts.append(values.assign(week=int(week)))
    if not parts:
        return pd.DataFrame(columns=["week", "team", "athlete_id", "value"])
    return pd.concat(parts, ignore_index=True)[["week", "team", "athlete_id", "value"]]


def load_depth_charts(season: int) -> pd.DataFrame:
    try:
        return store.read_raw("depth_charts", f"{season}.parquet")
    except FileNotFoundError:
        return pd.DataFrame(columns=["season", "week", "position", "team", "gsis_id"])


def load_injuries(season: int, week=None, as_of=None) -> pd.DataFrame:
    try:
        weekly = store.read_raw("injuries", f"{season}.parquet")
    except FileNotFoundError:
        weekly = pd.DataFrame(columns=["season", "week", "gsis_id", "report_status"])
    if week is None or as_of is None:
        return weekly
    from backend.availability import current_reports, merge_reports

    current, _ = current_reports(
        season, week, as_of, load_depth_charts(season), store.current_teams()
    )
    return merge_reports(weekly, current)


def week_slate(season: int, week: int) -> pd.DataFrame:
    schedules = store.read_raw("schedules.parquet")
    slate = schedules[
        schedules["season"].eq(season) & schedules["week"].eq(week)
    ].copy()
    slate["neutral_site"] = slate["location"].eq("Neutral")
    slate["start_date"] = kickoff_utc(slate)
    return slate


def regular_season(season: int) -> pd.DataFrame:
    """The season's regular-season schedule, with points where played."""
    schedules = store.read_raw("schedules.parquet")
    games = schedules[
        schedules["season"].eq(season) & schedules["game_type"].eq("REG")
    ].rename(columns={"home_score": "home_points", "away_score": "away_points"})
    return games.assign(start_date=kickoff_utc(games))


def market_home_spreads(slate: pd.DataFrame) -> dict[str, float]:
    """game_id -> sportsbook home line from the nflverse schedule."""
    return {
        str(row.game_id): float(closing_home_spread(row.spread_line))
        for row in slate.itertuples()
        if pd.notna(getattr(row, "spread_line", None))
    }


def market_totals(slate: pd.DataFrame) -> dict[str, float]:
    """game_id -> sportsbook total from the nflverse schedule."""
    return {
        str(row.game_id): float(row.total_line)
        for row in slate.itertuples()
        if pd.notna(getattr(row, "total_line", None))
    }


def fit_and_project(
    season: int,
    week: int,
    as_of: datetime | None = None,
    config: JointScoringConfig = DEFAULT_CONFIG,
    layer_config: LayerConfig = LayerConfig(),
    totals_config: TotalsConfig = DEFAULT_TOTALS_CONFIG,
) -> tuple[JointScoringFit, list[TeamRating], list[GameProjection]]:
    as_of = as_of or datetime.now(UTC)
    games = store.season_games(season)
    slate = week_slate(season, week)
    prior = build_preseason_prior(season, as_of=as_of, engine_config=config)
    fit = fit_joint_scoring(
        games,
        week,
        as_of,
        config,
        strength_prior=prior.week1_fit(),
        totals_config=totals_config,
    )
    team_names = store.team_names()
    qb_games = store.qb_games(list(range(HISTORY_START_SEASON, season + 1)))
    injuries = load_injuries(season, week, as_of)
    qb_adjustments = compute_qb_adjustments(
        season,
        week,
        slate,
        qb_games,
        load_depth_charts(season),
        config,
        layer_config,
        as_of=as_of,
        injuries=injuries,
    )
    absence_inputs = absence_layer.load_inputs([season - 1, season])
    absence_layer.require_coverage(absence_inputs, season, week)
    absence_points = absence_layer.points_lost(
        absence_inputs,
        slate.assign(forecast_cutoff=as_of),
        season,
        week,
        injuries,
        config.rating_half_life_weeks,
    )
    projections = assemble_projections(
        fit,
        slate,
        as_of,
        team_names,
        qb_adjustments,
        market_home_spreads(slate),
        layer_config,
        market_totals(slate),
        absence_points,
        seed_lock.locked_teams(regular_season(season), as_of, store.current_teams()),
    )
    # Published ratings are the fitted ratings: nothing from the coming
    # week's lines enters them, which live on the projection rows instead.
    return (
        fit,
        with_market_ratings(fit.ratings(team_names), season, week, as_of),
        projections,
    )
