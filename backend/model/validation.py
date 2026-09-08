"""Read-only historical reconstructions and chronological season-win evaluation."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backend.config import DEVELOPMENT_SEASONS, STATIC_DIR
from backend.etl import store
from backend.features.drives import build_team_games
from backend.features.qb import build_qb_games, expected_starters
from backend.features.scoring import build_model_games
from backend.model import qb_adjustment
from backend.model.joint_scoring import DEFAULT_CONFIG, fit_joint_scoring
from backend.model.preseason import (
    PreseasonConfig,
    build_preseason_prior,
    load_qb_references,
    load_win_totals,
)
from backend.model.projections import LayerConfig
from backend.model.season_wins import project_season, regular_schedule

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "posteam",
    "epa",
    "play_type",
    "fixed_drive",
    "score_differential",
    "qtr",
    "posteam_score",
    "posteam_score_post",
    "defteam_score",
    "defteam_score_post",
    "qb_dropback",
    "passer_player_id",
    "passer_player_name",
]


def historical_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Final schedule reconstruction, explicitly not an archived schedule release."""
    schedules = games.rename(
        columns={
            "season_type": "game_type",
            "home_points": "home_score",
            "away_points": "away_score",
        }
    ).copy()
    local = pd.to_datetime(schedules["start_date"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    schedules["gameday"] = local.dt.strftime("%Y-%m-%d")
    schedules["gametime"] = local.dt.strftime("%H:%M")
    schedules["location"] = np.where(schedules["neutral_site"], "Neutral", "Home")
    return schedules


def rebuild_history(seasons: list[int]):
    """Rebuild versioned core inputs in memory; never overwrite forecast artifacts."""
    games, quarterbacks = {}, []
    for season in seasons:
        schedules = historical_schedule(store.season_games(season))
        pbp = store.read_raw("pbp", f"{season}.parquet", columns=PBP_COLUMNS)
        games[season] = build_model_games(build_team_games(pbp, schedules), schedules)
        if set(games[season]["game_id"]) != set(schedules["game_id"]):
            raise ValueError(f"Incomplete rebuilt feature coverage for {season}")
        quarterbacks.append(build_qb_games(pbp))
    return games, pd.concat(quarterbacks, ignore_index=True)


def schedule_at_cutoff(schedule: pd.DataFrame, season: int, cutoff: datetime):
    """Final results become usable 24 hours after actual kickoff, conservatively.

    Postponed games stay unplayed until their actual kickoff plus that lag.
    A recorded result_available_at timestamp, when present, supersedes the lag.
    """
    games = regular_schedule(schedule, season, cutoff)
    if "result_available_at" in games:
        available = pd.to_datetime(games["result_available_at"], utc=True)
    else:
        available = games["start_date"] + timedelta(days=1)
    observed = available.le(cutoff) & games["home_score"].notna()
    games.loc[~observed, ["home_score", "away_score"]] = np.nan
    # No future game line or rest information is needed by season projections.
    games["spread_line"] = np.nan
    games["home_rest"] = np.nan
    games["away_rest"] = np.nan
    return games


def validate_season_wins(data, seasons, weeks=(1, 9), simulations=10_000):
    """Return one row per season/cutoff and a complete team-level audit.

    Undated historical sportsbook totals are comparisons only, never prior inputs.
    Earlier weekly QB charts are availability proxies; revised schedule dates and
    corrected historical PBP prevent these seasons being a pristine live replay.
    """
    summaries, audits = [], []
    sources = json.loads((STATIC_DIR / "win_total_sources.json").read_text())
    layer = LayerConfig(market_weight=0)
    history = data.strength_history[layer.qb_span_dropbacks]
    for season in seasons:
        schedules = historical_schedule(data.season_games[season])
        final = regular_schedule(schedules, season)
        if final[["home_score", "away_score"]].isna().any().any():
            raise ValueError(f"Season {season} is not complete")
        final_wins = pd.concat(
            [
                final.loc[final.home_score > final.away_score, "home_team"],
                final.loc[final.away_score > final.home_score, "away_team"],
            ]
        ).value_counts()
        previous = data.previous_games[season]
        previous_fit = fit_joint_scoring(
            previous,
            int(previous.model_week.max()) + 1,
            datetime(season, 8, 1, tzinfo=UTC),
            DEFAULT_CONFIG,
        )
        opener = final.start_date.min().to_pydatetime()
        source = sources.get(str(season), {})
        source_date = pd.to_datetime(source.get("date"), utc=True)
        dated_preseason = pd.notna(source_date) and source_date < opener
        prior = build_preseason_prior(
            season,
            as_of=opener - timedelta(seconds=1),
            config=PreseasonConfig(
                win_total_blend=PreseasonConfig().win_total_blend
                if dated_preseason
                else 0
            ),
            previous_fit=previous_fit,
            slope=data.slopes[season],
        )
        for week in weeks:
            window = final[final.week.eq(week)]
            if window.empty:
                raise ValueError(f"No games at requested cutoff {season} week {week}")
            cutoff = window.start_date.min().to_pydatetime() - timedelta(seconds=1)
            schedule = schedule_at_cutoff(schedules, season, cutoff)
            completed = set(schedule.loc[schedule.home_score.notna(), "game_id"])
            training = data.season_games[season]
            training = training[training.game_id.isin(completed)].copy()
            # The chronological cutoff owns eligibility, including postponements.
            training["model_week"] = training.model_week.clip(upper=week - 1)
            if training.empty:
                fit = prior.week1_fit()
            else:
                catalog = schedule[["home_team", "away_team"]].assign(model_week=week)
                fit = fit_joint_scoring(
                    pd.concat([training, catalog], ignore_index=True),
                    week,
                    cutoff,
                    strength_prior=prior.week1_fit(),
                )
            eligible = data.game_index[
                data.game_index.season.lt(season)
                | data.game_index.game_id.isin(completed)
            ]
            eligible = eligible[
                pd.to_datetime(eligible.start_date, utc=True).le(
                    cutoff - timedelta(days=1)
                )
            ]
            starters = expected_starters(
                data.qb_games,
                eligible,
                data.depth_charts[season],
                season,
                week,
                as_of=cutoff,
                use_overrides=False,
            )
            context = qb_adjustment.context_before_week(
                history[history.game_id.isin(eligible.game_id)],
                season,
                week,
                fit.config.rating_half_life_weeks,
                load_qb_references(season, cutoff) if dated_preseason else {},
                PreseasonConfig().win_total_blend if dated_preseason else 0,
            )
            adjustments = {
                r.game_id: tuple(
                    context.adjustment(t, starters.get(t), layer.qb_adjustment_weight)
                    for t in (r.home_team, r.away_team)
                )
                for r in schedule.itertuples()
            }
            totals, _ = project_season(
                fit, schedule, adjustments, {}, cutoff, simulations=simulations
            )
            totals["actual_final_wins"] = totals.team_abbr.map(final_wins).fillna(0)
            totals["absolute_win_error"] = (
                totals.projected_wins - totals.actual_final_wins
            ).abs()
            totals["covered_80"] = totals.actual_final_wins.between(
                totals.wins_p10, totals.wins_p90
            )
            totals["interval_width"] = totals.wins_p90 - totals.wins_p10
            totals["forecast_week"] = week
            totals["expected_qb"] = totals.team_abbr.map(starters)
            totals["sportsbook_win_total"] = totals.team_abbr.map(
                load_win_totals(season)
            )
            totals["sportsbook_absolute_error"] = (
                totals.sportsbook_win_total - totals.actual_final_wins
            ).abs()
            totals["split"] = (
                "development"
                if season in DEVELOPMENT_SEASONS
                else "retrospective_validation"
            )
            totals["sportsbook_used_in_prior"] = dated_preseason
            totals["sportsbook_source_date"] = source.get("date")
            totals["sportsbook_source_url"] = source.get("url")
            totals["input_assumptions"] = (
                "final_schedule_reconstruction; results_lag_24h; "
                "dated_qb_or_prior_week_proxy; current_qb_fixed; future_ties_omitted; "
                "undated_sportsbook_comparison_only"
            )
            summaries.append(
                {
                    "season": season,
                    "forecast_week": week,
                    "cutoff": cutoff,
                    "split": totals.split.iloc[0],
                    "teams": len(totals),
                    "win_mae": totals.absolute_win_error.mean(),
                    "coverage_80": totals.covered_80.mean(),
                    "mean_interval_width": totals.interval_width.mean(),
                    "sportsbook_mae": totals.sportsbook_absolute_error.mean(),
                    "sportsbook_teams": totals.sportsbook_win_total.notna().sum(),
                    "sportsbook_used_in_prior": dated_preseason,
                }
            )
            audits.append(totals)
    return pd.DataFrame(summaries), pd.concat(audits, ignore_index=True)
