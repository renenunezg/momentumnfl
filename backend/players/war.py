"""Wins above replacement: EPA credit above a positional replacement, in wins.

``epa.credit`` is a player's opponent-adjusted competitive EPA credit per
credited game. Per week and position group it is shrunk toward the group mean
by ``games / (games + k)``, with ``k`` the games at which the trait's split-half
reliability reaches one half on the development seasons (stored in the grade
model), and the replacement level is the ``REPLACEMENT_QUANTILE`` of the shrunk
values among the week's qualified players at the group. Points above
replacement are ``(shrunk - replacement) x games``.

A pressure that ends in neither a sack nor a quarterback hit tags nobody on the
play, so for wins above replacement each hurry a defender is charted with (PFR,
from 2018) adds ``HURRY_EPA`` times the hurry role weight to his credit:
``HURRY_EPA`` is the offense's EPA per dropback lost on such a pressure against
a clean pocket, 2016 to 2021 regular-season dropbacks at win probability 0.1 to
0.9 (nflverse participation pressure flag). The absence layer reads the credit
without it, since the charting starts in 2018.

Kickers and punters take their credit from their traits' per-game totals,
``SPECIAL_CREDIT``: field goals made above expectation at three points each,
and punt expected points above the league average.

A point is worth more wins in a close game than in a rout, so each credited
game is weighted by the win-probability slope per point at the team model's
pregame projection, ``leverage_g = phi(home_margin_g / margin_sd_g) /
margin_sd_g``, and ``wins_above_replacement = (shrunk - replacement) x sum of
leverage`` over the player's credited games through the week. A game without
a projection takes the season's mean leverage. Nothing is fitted.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.players import EPA_TRAIT
from backend.players.credit import ROLE_WEIGHTS
from backend.players.season_stats import pfr_ids

log = logging.getLogger(__name__)

REPLACEMENT_QUANTILE = 0.25
# Group: (trait whose per-game numerator is the credit, points per unit).
SPECIAL_CREDIT = {"K": ("k.field_goal", 3.0), "P": ("p.punt_epa", 1.0)}
HURRY_EPA = 0.166
WAR_COLUMNS = [
    "athlete_id",
    "position_group",
    "week",
    "games",
    "shrunk_per_game",
    "replacement_per_game",
    "points_above_replacement",
    "wins_above_replacement",
    "leverage_games",
]
PROJECTION_COLUMNS = ["game_id", "home_margin", "margin_sd"]


def game_leverage(projections: pd.DataFrame) -> pd.Series:
    """``phi(home_margin / margin_sd) / margin_sd`` per game id."""
    rows = projections.drop_duplicates("game_id")
    sd = rows["margin_sd"].astype(float)
    sd = sd.where(sd.gt(0))
    leverage = norm.pdf(rows["home_margin"].astype(float) / sd) / sd
    return pd.Series(leverage.to_numpy(float), index=rows["game_id"].astype(str))


def _calibration_projections(season: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "calibration" / "predictions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=PROJECTION_COLUMNS)
    rows = pd.read_parquet(
        path, columns=["season", "game_id", "model_margin", "margin_sd"]
    )
    rows = rows[rows["season"].eq(season)]
    return rows.rename(columns={"model_margin": "home_margin"})[PROJECTION_COLUMNS]


def _published_projections(season: int) -> pd.DataFrame:
    directory = PROCESSED_DIR / "projections"
    parts = []
    for path in (
        sorted(directory.glob(f"{season}_*.parquet")) if directory.is_dir() else []
    ):
        frame = pd.read_parquet(path, columns=["game_id", "home_margin", "margin_sd"])
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)
    # Each week's file projects that week's slate; a later reprojection of the
    # same game (a postponement) replaces the earlier one.
    return pd.concat(parts, ignore_index=True).drop_duplicates("game_id", keep="last")[
        PROJECTION_COLUMNS
    ]


def _typical_margin_sd() -> float:
    path = PROCESSED_DIR / "calibration" / "predictions.parquet"
    if not path.exists():
        return np.nan
    return float(pd.read_parquet(path, columns=["margin_sd"])["margin_sd"].median())


def season_projections(
    season: int, games: pd.DataFrame, schedule: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``game_id, home_margin, margin_sd`` for every completed game of ``season``.

    The calibration replay covers the development and validation seasons; the
    published weekly projections cover the live season. A game neither has takes
    the schedule's closing spread with the replay's typical margin sd, so the
    leverage weighting never depends on which seasons were replayed.
    """
    known = pd.concat(
        [_calibration_projections(season), _published_projections(season)],
        ignore_index=True,
    ).drop_duplicates("game_id", keep="first")
    completed = games[games["completed"]][["game_id"]]
    out = completed.merge(known, on="game_id", how="left")
    missing = out["home_margin"].isna() | out["margin_sd"].isna()
    if missing.any() and schedule is not None and "spread_line" in schedule:
        spreads = schedule.drop_duplicates("game_id").set_index("game_id")[
            "spread_line"
        ]
        out.loc[missing, "home_margin"] = (
            out.loc[missing, "game_id"].map(spreads).astype(float)
        )
        out.loc[missing, "margin_sd"] = _typical_margin_sd()
        log.info(
            "wins above replacement %s: %d games take the closing spread for leverage",
            season,
            int(missing.sum()),
        )
    return out[PROJECTION_COLUMNS]


def hurries(season: int) -> pd.DataFrame:
    """``athlete_id, week, hurries``: regular-season hurries through each week."""
    path = RAW_DIR / "pfr_def" / f"{season}.parquet"
    if not path.exists():
        return pd.DataFrame({"athlete_id": [], "week": [], "hurries": []}).astype(
            {"athlete_id": str, "week": int, "hurries": float}
        )
    rows = pd.read_parquet(
        path, columns=["game_type", "week", "pfr_player_id", "def_times_hurried"]
    )
    rows = rows[rows["game_type"].eq("REG")].assign(
        athlete_id=rows["pfr_player_id"].map(pfr_ids())
    )
    weekly = (
        rows.dropna(subset=["athlete_id"])
        .groupby(["athlete_id", "week"])["def_times_hurried"]
        .sum()
        .unstack()
        .reindex(columns=range(1, int(rows["week"].max()) + 1))
        .fillna(0.0)
        .cumsum(axis=1)
    )
    weekly.columns.name = "week"
    return weekly.stack().rename("hurries").reset_index()


def special_credit(
    frames: pd.DataFrame, snapshot: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kicker and punter credit: ``athlete_id, position_group, week, games,
    value`` per qualified player-week (points per game through the week), and
    their ``athlete_id, game_id, model_week`` credited games."""
    traits = {trait: group for group, (trait, _) in SPECIAL_CREDIT.items()}
    points = {trait: scale for trait, scale in SPECIAL_CREDIT.values()}
    rows = frames[frames["trait"].isin(traits)].astype({"athlete_id": str})
    rows = rows.dropna(subset=["numerator"]).assign(
        position_group=rows["trait"].map(traits),
        points=rows["numerator"] * rows["trait"].map(points),
    )
    credited = rows[["athlete_id", "game_id", "model_week"]].drop_duplicates(
        ignore_index=True
    )
    weekly = rows.groupby(["athlete_id", "position_group", "model_week"]).agg(
        points=("points", "sum"), games=("game_id", "nunique")
    )
    qualified = snapshot[
        snapshot["trait"].isin(traits) & snapshot["qualified"].eq(True)
    ].astype({"athlete_id": str})[["athlete_id", "position_group", "week"]]
    weeks = sorted(qualified["week"].unique())
    totals = []
    for column in ("points", "games"):
        wide = weekly[column].unstack().reindex(columns=weeks).fillna(0.0)
        wide.columns.name = "week"
        totals.append(wide.cumsum(axis=1).stack().rename(column))
    out = pd.concat(totals, axis=1).reset_index()
    out = out.merge(qualified, on=["athlete_id", "position_group", "week"])
    out = out[out["games"].gt(0)]
    return (
        out.assign(value=out["points"] / out["games"])[
            ["athlete_id", "position_group", "week", "games", "value"]
        ],
        credited,
    )


def shrunk_credit(
    snapshot: pd.DataFrame,
    shrink_games: dict[str, float],
    hurried: pd.DataFrame | None = None,
    special: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per qualified player-week: shrunk EPA credit per game and the replacement level.

    ``snapshot`` holds trait snapshot rows; only ``epa.credit`` rows of qualified
    players are read. The shrinkage ``k`` per group comes from ``shrink_games``
    (0 where absent). ``hurried`` (``hurries`` output) adds the weighted
    ``HURRY_EPA`` per hurry through the week to the credit; ``special``
    (``special_credit`` rows) adds kickers and punters.
    """
    rows = snapshot[snapshot["trait"].eq(EPA_TRAIT) & snapshot["qualified"].eq(True)]
    rows = rows.dropna(subset=["value"]).astype({"athlete_id": str})
    if hurried is not None:
        count = rows.merge(hurried, on=["athlete_id", "week"], how="left")["hurries"]
        rows = rows.assign(
            value=rows["value"]
            + HURRY_EPA
            * ROLE_WEIGHTS["hurry"]
            * count.fillna(0.0).to_numpy()
            / rows["games"].astype(float)
        )
    if special is not None:
        rows = pd.concat([rows, special], ignore_index=True)
    keys = ["week", "position_group"]
    mean = rows.groupby(keys)["value"].transform("mean")
    k = rows["position_group"].map(shrink_games).fillna(0.0).astype(float)
    games = rows["games"].astype(float)
    shrunk = (rows["value"] * games + mean * k) / (games + k)
    out = rows[["athlete_id", "position_group", "week", "games"]].assign(
        shrunk_per_game=shrunk
    )
    out["replacement_per_game"] = out.groupby(keys)["shrunk_per_game"].transform(
        lambda values: values.quantile(REPLACEMENT_QUANTILE)
    )
    return out.reset_index(drop=True)


def wins_above_replacement(
    values: pd.DataFrame, credited_games: pd.DataFrame, projections: pd.DataFrame
) -> pd.DataFrame:
    """``WAR_COLUMNS`` for every row of ``shrunk_credit`` output.

    ``credited_games`` holds one row per player and credited game with its
    ``model_week``; ``leverage_games`` sums the leverage of the player's credited
    games at or before the snapshot week.
    """
    leverage = game_leverage(projections)
    if leverage.isna().all():
        raise ValueError(
            "No game has a projection to weight wins by; run production-artifacts"
        )
    mean = float(leverage.mean())
    games = (
        credited_games[["athlete_id", "game_id", "model_week"]]
        .astype({"athlete_id": str})
        .drop_duplicates(["athlete_id", "game_id"])
    )
    games["leverage"] = games["game_id"].astype(str).map(leverage).fillna(mean)
    weeks = sorted(
        set(games["model_week"].astype(int)) | set(values["week"].astype(int))
    )
    cumulative = (
        games.groupby(["athlete_id", games["model_week"].astype(int)])["leverage"]
        .sum()
        .unstack()
        .reindex(columns=weeks)
        .fillna(0.0)
        .cumsum(axis=1)
    )
    cumulative.columns.name = "week"
    stacked = cumulative.stack().rename("leverage_games").reset_index()
    out = values.merge(stacked, on=["athlete_id", "week"], how="left")
    margin = out["shrunk_per_game"] - out["replacement_per_game"]
    out["points_above_replacement"] = margin * out["games"].astype(float)
    out["wins_above_replacement"] = margin * out["leverage_games"]
    return out[WAR_COLUMNS]
