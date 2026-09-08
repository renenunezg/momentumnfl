"""QB game logs and expected-starter detection."""

import pandas as pd

from backend.config import REPO_ROOT

# Optional manual starter overrides: season, week, team_abbr, gsis_id.
OVERRIDES_PATH = REPO_ROOT / "overrides" / "qb_starters.csv"


def build_qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, passer): dropbacks, EPA, team, starter flag."""
    dropbacks = pbp[pbp["qb_dropback"].eq(1) & pbp["passer_player_id"].notna()]
    qb = (
        dropbacks.groupby(["game_id", "posteam", "passer_player_id"], as_index=False)
        .agg(
            dropbacks=("epa", "size"),
            epa=("epa", "sum"),
            passer=("passer_player_name", "first"),
        )
        .rename(columns={"posteam": "team"})
    )
    starters = (
        dropbacks.sort_values(["game_id", "play_id"])
        .drop_duplicates(["game_id", "posteam"])[
            ["game_id", "posteam", "passer_player_id"]
        ]
        .rename(columns={"posteam": "team"})
        .assign(started=True)
    )
    qb = qb.merge(starters, on=["game_id", "team", "passer_player_id"], how="left")
    qb["started"] = qb["started"].notna() & qb["started"].eq(True)
    qb["feature_version"] = 2
    return qb


def _overrides() -> pd.DataFrame:
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"])
    return pd.read_csv(OVERRIDES_PATH)


def _depth_chart_qb1(
    depth_charts: pd.DataFrame, season: int, week: int, as_of=None
) -> pd.Series | None:
    """team -> depth-chart QB1 gsis_id, handling both nflverse formats:
    the weekly format (through 2024: season/week/position/depth_team) and
    the snapshot format (2025+: dt/pos_abb/pos_rank)."""
    if "pos_abb" in depth_charts.columns:
        if as_of is not None:
            depth_charts = depth_charts[
                pd.to_datetime(depth_charts["dt"], utc=True).le(as_of)
            ]
        charts = depth_charts[
            depth_charts["pos_abb"].eq("QB") & depth_charts["gsis_id"].notna()
        ]
        if charts.empty:
            return None
        latest = charts[charts["dt"].eq(charts.groupby("team")["dt"].transform("max"))]
        return (
            latest.sort_values("pos_rank").groupby("team")["gsis_id"].first().dropna()
        )
    if "season" not in depth_charts.columns:
        return None
    charts = depth_charts.rename(columns={"club_code": "team"})
    charts = charts[
        charts["season"].eq(season)
        & (charts["week"].lt(week) if as_of is not None else charts["week"].eq(week))
        & charts["position"].eq("QB")
    ]
    if "formation" in charts.columns:
        charts = charts[charts["formation"].eq("Offense")]
    if charts.empty or "depth_team" not in charts.columns:
        return None
    if as_of is not None:
        charts = charts[
            charts["week"].eq(charts.groupby("team")["week"].transform("max"))
        ]
    return (
        charts.assign(rank=pd.to_numeric(charts["depth_team"], errors="coerce"))
        .sort_values("rank")
        .groupby("team")["gsis_id"]
        .first()
        .dropna()
    )


def expected_starters(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    depth_charts: pd.DataFrame,
    season: int,
    week: int,
    as_of=None,
    use_overrides: bool = True,
) -> pd.Series:
    """team -> expected starter gsis_id for the given week.

    Priority: overrides file, then the depth-chart QB1 as of that week, then
    the team's most recent actual starter.
    """
    qb_meta = qb_games.merge(
        game_index[["game_id", "season", "model_week"]], on="game_id"
    )
    prior = qb_meta[
        (qb_meta["season"] < season)
        | ((qb_meta["season"] == season) & (qb_meta["model_week"] < week))
    ]
    if as_of is not None and "start_date" in game_index:
        eligible_ids = game_index.loc[
            pd.to_datetime(game_index["start_date"], utc=True).lt(as_of), "game_id"
        ]
        prior = prior[prior["game_id"].isin(eligible_ids)]
    # Latest known starter is the fallback; current depth charts supersede it.
    prior_starts = prior[prior["started"]].sort_values(["season", "model_week"])
    result = prior_starts.drop_duplicates("team", keep="last").set_index("team")[
        "passer_player_id"
    ]

    qb1 = _depth_chart_qb1(depth_charts, season, week, as_of)
    if qb1 is not None and not qb1.empty:
        # A named rookie starter has no NFL history yet. Keep that identity;
        # the projection layer supplies replacement value for an unseen QB.
        result = qb1.combine_first(result)

    overrides = (
        _overrides()
        if use_overrides
        else pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"])
    )
    overrides = overrides[overrides["season"].eq(season) & overrides["week"].eq(week)]
    for row in overrides.itertuples():
        result[row.team_abbr] = row.gsis_id
    return result
