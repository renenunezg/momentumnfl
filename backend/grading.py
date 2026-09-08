"""Grade genuinely published forecasts using completed nflverse schedule rows."""

import numpy as np
import pandas as pd

from backend.features.drives import kickoff_utc
from backend.publish import _prepare

RESULT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "start_date",
    "home_team_abbr",
    "away_team_abbr",
    "home_team",
    "away_team",
    "neutral_site",
    "home_points",
    "away_points",
    "closing_spread",
    "source",
    "source_fetched_at",
]


def result_frame(schedules, season, names, fetched_at) -> pd.DataFrame:
    if season < 2026:
        raise ValueError("Live grading starts in 2026; earlier years are backtests")
    games = schedules.loc[schedules["season"].eq(season)].copy()
    if games.empty or games["game_id"].duplicated().any():
        raise ValueError("Grading requires a nonempty, unique season schedule")
    # nflverse schedules expose completed scores and result, not live PBP.
    # Partial scores, unknown kickoff times, and canceled games do not qualify.
    games = games.loc[
        games["home_score"].notna()
        & games["away_score"].notna()
        & games["result"].notna()
        & games["gametime"].notna()
        & games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])
    ].copy()
    games["start_date"] = kickoff_utc(games)
    fetched_at = pd.Timestamp(fetched_at)
    if fetched_at.tzinfo is None or fetched_at > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Schedule receipt time must be a past UTC timestamp")
    games = games.loc[games["start_date"].lt(fetched_at)].copy()
    scores = games[["home_score", "away_score"]].to_numpy(dtype=float)
    if (
        not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores != np.floor(scores)).any()
        or not (games["home_score"] - games["away_score"]).eq(games["result"]).all()
    ):
        raise ValueError("Invalid or inconsistent completed-game scores")
    games = games.rename(
        columns={
            "game_type": "season_type",
            "home_team": "home_team_abbr",
            "away_team": "away_team_abbr",
            "home_score": "home_points",
            "away_score": "away_points",
        }
    )
    for side in ("home", "away"):
        games[f"{side}_team"] = games[f"{side}_team_abbr"].map(names)
        if games[f"{side}_team"].isna().any():
            raise ValueError("Unknown team in completed-game schedule")
    games["neutral_site"] = games["location"].eq("Neutral")
    # nflverse spread_line is the implied home margin; UI uses home spread.
    games["closing_spread"] = -pd.to_numeric(games["spread_line"])
    if np.isinf(games["closing_spread"].to_numpy(dtype=float)).any():
        raise ValueError("Invalid closing spread")
    games["source"] = "nflverse schedules"
    games["source_fetched_at"] = fetched_at
    return _prepare(games, RESULT_COLUMNS)


def recommendation_schedule(schedules, season, names, fetched_at):
    """All observed fixtures, including pending and explicit schedule changes.

    nflverse does not consistently provide cancellation status. Absent games
    remain pending; only an explicit status or changed known kickoff can void.
    """
    finals = result_frame(schedules, season, names, fetched_at).set_index("game_id")
    games = schedules[schedules.season.eq(season)].copy()
    games["start_date"] = kickoff_utc(games).where(games.gametime.notna())
    rows = []
    for game in games.itertuples():
        final = finals.loc[game.game_id] if game.game_id in finals.index else None
        status = str(getattr(game, "game_status", "scheduled")).lower()
        rows.append(
            dict(
                game_id=game.game_id,
                season=season,
                start_date=game.start_date,
                home_team=names[game.home_team],
                away_team=names[game.away_team],
                game_status=status,
                completed=final is not None,
                home_points=None if final is None else int(final.home_points),
                away_points=None if final is None else int(final.away_points),
                observed_at=fetched_at,
            )
        )
    return pd.DataFrame(rows)
