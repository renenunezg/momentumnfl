"""Market-implied team strength from earlier games' closing spreads.

A ridge rating (closing home margin = home - away + home field) with recent
games weighted more and the previous season's final rating, regressed, as the
prior. It is descriptive only: the fit, the blend, and picks never read it. The
gap between it and the fitted rating shows where the model and the market
disagree about a team.

Settings were chosen on the 2016-2021 walk-forward by next-week closing-line
MAE (2.477, against 2.671 for half-life 3, one prior game, carryover 0.6) and
scored once on 2022-2025 (2.399 against 2.557, 1,087 games). The short half-life
reflects that the latest line already prices what earlier lines knew.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backend.config import HISTORY_START_SEASON
from backend.etl import store
from backend.model.joint_scoring import solve_ridge
from backend.model.outputs import TeamRating

_HOME_FIELD_PRIOR_SD = 10.0


@dataclass(frozen=True, slots=True)
class MarketRatingConfig:
    half_life_weeks: float
    # The prior's weight, in full-weight games.
    prior_games: float
    # Share of last season's rating carried into this one.
    carryover: float


DEFAULT_MARKET_RATING_CONFIG = MarketRatingConfig(
    half_life_weeks=1.29, prior_games=0.105, carryover=0.82
)


@dataclass(frozen=True, slots=True)
class MarketRatingFit:
    rating: dict[str, float]
    home_field: float


def lined_games(games: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Games that kicked off before the cutoff and closed with a spread."""
    kickoff = pd.to_datetime(games["start_date"], utc=True)
    return games[kickoff.lt(as_of) & games["closing_spread"].notna()]


def fit_market_rating(
    games: pd.DataFrame,
    teams: list[str],
    prior: dict[str, float],
    latest_week: int,
    config: MarketRatingConfig,
) -> MarketRatingFit:
    index = {team: position for position, team in enumerate(teams)}
    rows = np.arange(len(games))
    design = np.zeros((len(games), len(teams) + 1))
    design[rows, games["home_team"].map(index).to_numpy(int)] = 1.0
    design[rows, games["away_team"].map(index).to_numpy(int)] = -1.0
    design[rows, -1] = (~games["neutral_site"].astype(bool)).to_numpy(float)
    recency = 0.5 ** (
        (latest_week - games["model_week"].to_numpy(float)) / config.half_life_weeks
    )
    prior_mean = np.array([prior.get(team, 0.0) for team in teams] + [0.0])
    prior_sd = np.full(len(teams) + 1, 1.0 / np.sqrt(config.prior_games))
    prior_sd[-1] = _HOME_FIELD_PRIOR_SD
    parameters, _ = solve_ridge(
        design,
        -games["closing_spread"].to_numpy(float),
        recency,
        prior_mean,
        prior_sd,
    )
    rating = parameters[:-1] - parameters[:-1].mean()
    return MarketRatingFit(
        dict(zip(teams, rating.astype(float))), float(parameters[-1])
    )


def carried_prior(
    seasons: list[pd.DataFrame], as_of: pd.Timestamp, config: MarketRatingConfig
) -> dict[str, float]:
    """Chain each finished season's final rating into the next one's prior."""
    prior: dict[str, float] = {}
    for games in seasons:
        lined = lined_games(games, as_of)
        teams = sorted(set(lined["home_team"]) | set(lined["away_team"]))
        final = fit_market_rating(
            lined, teams, prior, int(lined["model_week"].max()), config
        )
        prior = {team: config.carryover * r for team, r in final.rating.items()}
    return prior


def market_ratings(
    season: int,
    week: int,
    as_of,
    config: MarketRatingConfig = DEFAULT_MARKET_RATING_CONFIG,
) -> pd.DataFrame:
    """team_abbr, market_rating, market_rating_games for one forecast."""
    cutoff = pd.Timestamp(as_of)
    prior = carried_prior(
        [store.season_games(year) for year in range(HISTORY_START_SEASON, season)],
        cutoff,
        config,
    )
    current = pd.DataFrame()
    if str(season) in store.processed_names("team_games"):
        games = store.season_games(season)
        current = lined_games(games[games["model_week"].lt(week)], cutoff)
    if current.empty:
        return pd.DataFrame(
            {
                "team_abbr": list(prior),
                "market_rating": list(prior.values()),
                "market_rating_games": 0,
            }
        )
    fitted = fit_market_rating(current, sorted(prior), prior, week - 1, config)
    played = pd.concat([current["home_team"], current["away_team"]]).value_counts()
    return pd.DataFrame(
        {
            "team_abbr": list(fitted.rating),
            "market_rating": list(fitted.rating.values()),
            "market_rating_games": [int(played.get(team, 0)) for team in fitted.rating],
        }
    )


def with_market_ratings(
    ratings: list[TeamRating], season: int, week: int, as_of
) -> list[TeamRating]:
    market = market_ratings(season, week, as_of).set_index("team_abbr")
    return [
        replace(
            rating,
            market_rating=float(market.at[rating.team_abbr, "market_rating"]),
            market_rating_games=int(market.at[rating.team_abbr, "market_rating_games"]),
        )
        for rating in ratings
    ]
