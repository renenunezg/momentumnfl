"""Players who changed teams between seasons, and how well each trait travels.

The NFL needs no name matching: a gsis id follows the player, so a mover is a
player whose first team of season ``t`` differs from his last team of season
``t - 1`` in the box score. ``portability`` then asks, trait by trait, whether
last season's value predicts this season's as well for movers as for stayers.
The ratio of the two slopes is the share of the trait that travels.
"""

import numpy as np
import pandas as pd

MOVER_COLUMNS = ["athlete_id", "season", "origin", "destination"]
PORTABILITY_COLUMNS = [
    "position_group",
    "trait",
    "stayers",
    "movers",
    "stay_slope",
    "move_slope",
    "move_slope_se",
    "portable_share",
]
MIN_MOVERS = 20


def _team_at(stats: pd.DataFrame, first: bool) -> pd.Series:
    rows = stats.sort_values("model_week").drop_duplicates(
        "athlete_id", keep="first" if first else "last"
    )
    return rows.set_index("athlete_id")["team"]


def team_changes(stats_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """``MOVER_COLUMNS`` for every player whose team changed between seasons."""
    rows = []
    seasons = sorted(stats_by_season)
    for prior, season in zip(seasons, seasons[1:]):
        if (
            season - prior != 1
            or stats_by_season[prior].empty
            or stats_by_season[season].empty
        ):
            continue
        before = _team_at(stats_by_season[prior], first=False)
        after = _team_at(stats_by_season[season], first=True)
        both = pd.concat(
            [before.rename("origin"), after.rename("destination")], axis=1
        ).dropna()
        moved = both[both["origin"].ne(both["destination"])]
        rows.append(moved.reset_index().assign(season=season)[MOVER_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=MOVER_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of ``y`` on ``x`` with an intercept, and the slope's standard error."""
    if len(x) < 2:
        return np.nan, np.nan
    dx = x - x.mean()
    sxx = float(dx @ dx)
    if sxx <= 0:
        return np.nan, np.nan
    slope = float(dx @ (y - y.mean())) / sxx
    if len(x) < 3:
        return slope, np.nan
    resid = (y - y.mean()) - slope * dx
    return slope, float(np.sqrt(float(resid @ resid) / (len(x) - 2) / sxx))


def _final_week(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.empty:
        frame = frame[frame["week"].eq(frame["week"].max())]
    frame = frame[frame["qualified"].astype(bool)]
    return frame[["athlete_id", "trait", "position_group", "value"]]


def portability(
    snapshots_by_season: dict[int, pd.DataFrame], movers: pd.DataFrame
) -> pd.DataFrame:
    """Stayer and mover year-over-year slopes per position group and trait.

    Consecutive seasons ``(t - 1, t)`` are pooled into one regression of the
    final-week value in ``t`` on the final-week value in ``t - 1``. A player is a
    mover when ``movers`` lists him for season ``t``. ``portable_share`` is the
    mover slope over the stayer slope, NaN when the stayer slope is not positive,
    fewer than ``MIN_MOVERS`` players moved, or the mover slope's standard error is
    at least the stayer slope.
    """
    pairs = []
    seasons = sorted(snapshots_by_season)
    for prior, season in zip(seasons, seasons[1:]):
        if season - prior != 1:
            continue
        before = _final_week(snapshots_by_season[prior])
        after = _final_week(snapshots_by_season[season])
        both = before.merge(
            after,
            on=["athlete_id", "trait", "position_group"],
            suffixes=("_before", ""),
        )
        both = both.dropna(subset=["value_before", "value"])
        moved = (
            set(movers.loc[movers["season"].eq(season), "athlete_id"])
            if len(movers)
            else set()
        )
        pairs.append(both.assign(mover=both["athlete_id"].isin(moved)))
    if not pairs:
        return pd.DataFrame(columns=PORTABILITY_COLUMNS)

    pooled = pd.concat(pairs, ignore_index=True)
    rows = []
    for (group, trait), part in pooled.groupby(["position_group", "trait"], sort=True):
        stay, move = part[~part["mover"]], part[part["mover"]]
        stay_slope, _ = _ols(
            stay["value_before"].to_numpy(float), stay["value"].to_numpy(float)
        )
        move_slope, move_se = _ols(
            move["value_before"].to_numpy(float), move["value"].to_numpy(float)
        )
        # The mover slope must be measured more precisely than the stayer slope is
        # large; otherwise the movers cannot tell full portability from none, and the
        # share is left unmeasured rather than read off a noisy estimate.
        usable = (
            len(move) >= MIN_MOVERS
            and stay_slope > 0
            and np.isfinite(move_se)
            and move_se < stay_slope
        )
        rows.append(
            {
                "position_group": group,
                "trait": trait,
                "stayers": len(stay),
                "movers": len(move),
                "stay_slope": stay_slope,
                "move_slope": move_slope,
                "move_slope_se": move_se,
                "portable_share": move_slope / stay_slope if usable else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=PORTABILITY_COLUMNS)
