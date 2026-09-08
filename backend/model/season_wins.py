"""Regular-season wins from the current engine, with correlated season outcomes.

Means use the engine's Student-t win probabilities without game-market blending.
A Gaussian copula preserves those marginals while sharing the engine's strength
uncertainty across games. Its season intervals are not yet coverage-calibrated.
Future ratings and today's expected QB remain fixed; future ties are omitted.
"""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import t as student_t

from backend.config import HISTORY_START_SEASON, STATIC_DIR
from backend.etl import store
from backend.features.drives import kickoff_utc
from backend.model.distributions import student_t_scale
from backend.model.fit_week import compute_qb_adjustments, load_depth_charts
from backend.model.joint_scoring import JointScoringFit, fit_joint_scoring
from backend.model.preseason import build_preseason_prior, load_win_totals
from backend.model.projections import LayerConfig, assemble_projections

MODEL_VERSION = "nfl_season_wins_v2"
SIMULATIONS = 100_000
SEED = 20260907


def regular_schedule(
    schedules: pd.DataFrame,
    season: int,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Refuse partial or inconsistent schedules rather than publish short totals."""
    if season < 2015:
        raise ValueError("Season wins require the supported 2015+ history")
    games_per_team = 17 if season >= 2021 else 16
    games = schedules[
        schedules["season"].eq(season) & schedules["game_type"].eq("REG")
    ].copy()
    # Restore the scheduled identity even when a final source omits it. The
    # cancellation cannot affect a forecast before it was publicly known.
    canceled_id = "2022_17_BUF_CIN"
    if season == 2022 and canceled_id not in set(games["game_id"]):
        games = pd.concat(
            [
                games,
                pd.DataFrame(
                    [
                        {
                            "game_id": canceled_id,
                            "season": 2022,
                            "week": 17,
                            "game_type": "REG",
                            "home_team": "CIN",
                            "away_team": "BUF",
                            "home_score": np.nan,
                            "away_score": np.nan,
                            "location": "Home",
                            "gameday": "2023-01-02",
                            "gametime": "20:30",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if games[["game_id", "home_team", "away_team", "week"]].isna().any().any():
        raise ValueError("Incomplete regular-season game identity")
    appearances = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    if (
        len(games) != 16 * games_per_team
        or games["game_id"].duplicated().any()
        or len(appearances) != 32
        or not appearances.eq(games_per_team).all()
        or games["home_team"].eq(games["away_team"]).any()
        or not games["week"].between(1, games_per_team + 1).all()
    ):
        raise ValueError(
            f"Expected {16 * games_per_team} unique regular-season games, "
            f"{games_per_team} per team"
        )
    # NFL cancellation announced Jan 5 Eastern; conservatively available
    # after that calendar day, Jan 6 05:00 UTC. Source is recorded in README.
    if season == 2022:
        canceled = games["game_id"].eq(canceled_id)
        games.loc[canceled, ["home_score", "away_score"]] = np.nan
        if as_of is None or as_of >= datetime(2023, 1, 6, 5, tzinfo=UTC):
            games = games[~canceled].copy()
    home_score, away_score = games["home_score"], games["away_score"]
    if (home_score.notna() != away_score.notna()).any():
        raise ValueError("A game has only one reported score")
    scores = games[["home_score", "away_score"]].dropna().to_numpy(float)
    if (
        not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores != np.floor(scores)).any()
    ):
        raise ValueError("Invalid completed-game scores")
    games["neutral_site"] = games["location"].eq("Neutral")
    games["start_date"] = kickoff_utc(games)
    # Week 18 kickoff times may still be TBD. Stable ordering keeps the seeded
    # simulation reproducible even if upstream changes row order.
    return games.sort_values(["week", "game_id"]).reset_index(drop=True)


def project_season(
    fit: JointScoringFit,
    schedule: pd.DataFrame,
    qb_adjustments: dict[str, tuple[float, float]],
    team_names: dict[str, str],
    as_of: datetime,
    simulations: int = SIMULATIONS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team summaries and per-game audit rows; one winner per unplayed game.

    Completed ties count as a tie for each team and as zero wins. Every draw's
    league wins equal 272 minus completed tied games. Marginal expected wins
    are analytic, avoiding Monte Carlo noise in the headline numbers.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if simulations < 1000:
        raise ValueError("At least 1000 season simulations are required")
    games = regular_schedule(schedule, fit.season, as_of)
    if set(fit.teams) != set(games["home_team"]) | set(games["away_team"]):
        raise ValueError("Fitted teams do not match the complete schedule")
    finished = games["home_score"].notna()
    if games.loc[finished, "start_date"].isna().any():
        raise ValueError("Completed games require valid kickoff timestamps")
    if (games.loc[finished, "start_date"] >= as_of).any():
        raise ValueError("Completed results cannot start at or after as_of")
    remaining = games[~finished]
    projections = assemble_projections(
        fit,
        remaining,
        as_of,
        team_names,
        qb_adjustments,
        config=LayerConfig(market_weight=0.0),
    )
    teams = sorted(fit.teams)
    index = {team: i for i, team in enumerate(teams)}
    actual = np.zeros((len(teams), 3), dtype=int)  # wins, losses, ties
    for game in games[finished].itertuples():
        home, away = index[game.home_team], index[game.away_team]
        if game.home_score == game.away_score:
            actual[[home, away], 2] += 1
        else:
            winner, loser = (
                (home, away) if game.home_score > game.away_score else (away, home)
            )
            actual[winner, 0] += 1
            actual[loser, 1] += 1
    draws = np.broadcast_to(actual[:, 0], (simulations, len(teams))).copy()
    expected = actual[:, 0].astype(float)
    audit = []
    if projections:
        design = np.array(
            [
                np.diff(fit.score_design(game), axis=0)[0] * -1
                for game in remaining.itertuples()
            ]
        )
        covariance = fit.parameter_covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues.min() < -1e-8:
            raise ValueError("Invalid strength covariance")
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        residual = float(
            np.array([1.0, -1.0])
            @ fit.score_residual_covariance
            @ np.array([1.0, -1.0])
        )
        if residual <= 0 or not np.isfinite(residual):
            raise ValueError("Invalid game residual variance")
        sd = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design) + residual)
        margins = np.array([p.pure_home_margin for p in projections], dtype=float)
        scale = student_t_scale(
            np.array([p.margin_sd for p in projections]),
            fit.config.student_t_degrees_of_freedom,
        )
        home_probability = student_t.cdf(
            margins / scale, fit.config.student_t_degrees_of_freedom
        )
        if not np.isfinite(home_probability).all():
            raise ValueError("Invalid game win probabilities")
        threshold = ndtri(1 - home_probability)
        rng = np.random.default_rng(seed)
        home_indices = np.array([index[p.home_team_abbr] for p in projections])
        away_indices = np.array([index[p.away_team_abbr] for p in projections])
        # Bound memory independently of the requested simulation count.
        for start in range(0, simulations, 5000):
            stop = min(start + 5000, simulations)
            shared = rng.standard_normal((stop - start, len(covariance))) @ root.T
            outcomes = (
                shared @ design.T
                + rng.standard_normal((stop - start, len(projections)))
                * np.sqrt(residual)
            ) / sd > threshold
            for column, (home, away) in enumerate(zip(home_indices, away_indices)):
                draws[start:stop, home] += outcomes[:, column]
                draws[start:stop, away] += ~outcomes[:, column]
        for p, probability, home, away in zip(
            projections, home_probability, home_indices, away_indices
        ):
            expected[home] += probability
            expected[away] += 1 - probability
            audit.append(
                {
                    "game_id": p.game_id,
                    "home_team": p.home_team_abbr,
                    "away_team": p.away_team_abbr,
                    "pure_home_margin": p.pure_home_margin,
                    "home_win_probability": probability,
                    "away_win_probability": 1 - probability,
                }
            )
    league_wins = len(games) - int(actual[:, 2].sum() // 2)
    if not np.all(draws.sum(axis=1) == league_wins) or not np.isclose(
        expected.sum(), league_wins
    ):
        raise ValueError("Season win accounting failed")
    bounds = np.quantile(draws, [0.1, 0.5, 0.9], axis=0, method="inverted_cdf")
    frame = pd.DataFrame(
        {
            "season": fit.season,
            "as_of": as_of,
            "model_version": MODEL_VERSION,
            "team_abbr": teams,
            "team": [team_names.get(t, t) for t in teams],
            "wins": actual[:, 0],
            "losses": actual[:, 1],
            "ties": actual[:, 2],
            "games_played": actual.sum(axis=1),
            "games_remaining": pd.concat([games["home_team"], games["away_team"]])
            .value_counts()
            .reindex(teams)
            .to_numpy()
            - actual.sum(axis=1),
            "projected_wins": expected,
            "remaining_expected_wins": expected - actual[:, 0],
            "wins_p10": bounds[0].astype(int),
            "wins_p50": bounds[1].astype(int),
            "wins_p90": bounds[2].astype(int),
            "simulation_count": simulations,
            "simulation_seed": seed,
        }
    )
    return frame, pd.DataFrame(audit)


def build_season_forecast(
    season: int, as_of: datetime | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = as_of or datetime.now(UTC)
    schedules = regular_schedule(store.read_raw("schedules.parquet"), season, as_of)
    remaining = schedules[schedules["home_score"].isna()]
    scheduled_week = (
        int(remaining.sort_values("start_date")["week"].iloc[0])
        if not remaining.empty
        else 19
    )
    prior = build_preseason_prior(season, as_of=as_of)
    try:
        games = store.season_games(season)
    except FileNotFoundError:
        games = pd.DataFrame(columns=["model_week", "game_id"])
    training = games[games["model_week"].lt(scheduled_week)]
    required_ids = set(
        schedules.loc[
            schedules.home_score.notna() & schedules.week.lt(scheduled_week), "game_id"
        ]
    )
    if set(training["game_id"]) != required_ids:
        raise ValueError(
            "Cached training games are incomplete or disagree with completed results"
        )
    if training.empty:
        fit = prior.week1_fit()
        through_week = 0
        through_date = None
    else:
        # Supply the complete catalog even if a postponed opener means some
        # teams have no current-season feature rows yet. These identity rows
        # are at the cutoff and cannot enter the training data.
        catalog = schedules[["home_team", "away_team"]].assign(
            model_week=scheduled_week
        )
        fit = fit_joint_scoring(
            pd.concat([training, catalog], ignore_index=True),
            scheduled_week,
            as_of,
            strength_prior=prior.week1_fit(),
        )
        through_week = int(training["model_week"].max())
        through_date = pd.to_datetime(training["start_date"], utc=True).max()
    charts = load_depth_charts(season)
    if "dt" in charts:
        charts = charts[pd.to_datetime(charts["dt"], utc=True).le(as_of)]
    qb_week = scheduled_week
    adjustments = compute_qb_adjustments(
        season,
        qb_week,
        remaining,
        store.qb_games(list(range(HISTORY_START_SEASON, season + 1))),
        charts,
        fit.config,
        LayerConfig(),
        as_of=as_of,
    )
    frame, audit = project_season(
        fit, schedules, adjustments, store.team_names(), as_of
    )
    metadata = store.current_teams().set_index("team_abbr")
    frame["conference"] = frame["team_abbr"].map(metadata["team_conf"])
    frame["division"] = frame["team_abbr"].map(metadata["team_division"])
    frame["ratings_through_week"] = through_week
    frame["ratings_through_date"] = through_date
    frame["schedule_fetched_at"] = datetime.fromtimestamp(
        (store.RAW_DIR / "schedules.parquet").stat().st_mtime, UTC
    )
    frame["depth_chart_as_of"] = (
        pd.to_datetime(charts["dt"], utc=True).max()
        if "dt" in charts and not charts.empty
        else None
    )
    frame["sportsbook_win_total"] = frame["team_abbr"].map(load_win_totals(season))
    sources = json.loads((STATIC_DIR / "win_total_sources.json").read_text())
    source = sources.get(str(season), {})
    for key in ("name", "date", "url"):
        frame[f"sportsbook_source_{key}"] = source.get(key)
    return frame.sort_values(
        ["projected_wins", "team_abbr"], ascending=[False, True]
    ), audit
