"""Model-ready games: chronological week index and market sign conventions."""

import pandas as pd


def max_regular_week(schedules: pd.DataFrame) -> pd.Series:
    """Per-season maximum regular-season week (17 through 2020, 18 after)."""
    regular = schedules[schedules["game_type"].eq("REG")]
    return regular.groupby("season")["week"].max()


def build_model_games(
    team_games: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    games = team_games.copy()
    # nflverse playoff weeks continue past the regular season (19-22 in the
    # 17-game era), so week is already a chronological index; the offset
    # only applies if a source ever restarts postseason weeks at 1.
    regular_max = games["season"].map(max_regular_week(schedules))
    postseason = ~games["season_type"].eq("REG")
    needs_offset = postseason & games["week"].le(regular_max)
    games["model_week"] = (
        games["week"] + needs_offset.astype(int) * regular_max
    ).astype(int)
    # nflverse spread_line is positive when the home team is favored (an
    # expected home margin); sportsbook home-line notation flips the sign.
    games["closing_spread"] = -games["spread_line"]
    games["actual_margin"] = games["home_points"] - games["away_points"]
    return games
