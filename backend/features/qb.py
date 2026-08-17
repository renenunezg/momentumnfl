"""QB game logs and expected-starter detection."""

import pandas as pd

from backend.config import STATIC_DIR


def build_qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, passer): dropbacks, EPA, team, starter flag."""
    dropbacks = pbp[
        pbp["qb_dropback"].eq(1) & pbp["passer_player_id"].notna()
    ]
    qb = (
        dropbacks.groupby(
            ["game_id", "posteam", "passer_player_id"], as_index=False
        )
        .agg(
            dropbacks=("epa", "size"),
            epa=("epa", "sum"),
            passer=("passer_player_name", "first"),
        )
        .rename(columns={"posteam": "team"})
    )
    starters = qb.loc[
        qb.groupby(["game_id", "team"])["dropbacks"].idxmax(),
        ["game_id", "team", "passer_player_id"],
    ].assign(started=True)
    qb = qb.merge(starters, on=["game_id", "team", "passer_player_id"], how="left")
    qb["started"] = qb["started"].notna() & qb["started"].eq(True)
    return qb


def _overrides() -> pd.DataFrame:
    path = STATIC_DIR.parent.parent / "overrides" / "qb_starters.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["season", "week", "team_abbr", "gsis_id"]
        )
    return pd.read_csv(path)


def expected_starters(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    depth_charts: pd.DataFrame,
    season: int,
    week: int,
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
    prior_starts = prior[prior["started"]].sort_values(["season", "model_week"])
    result = prior_starts.groupby("team")["passer_player_id"].last()

    charts = depth_charts.rename(columns={"club_code": "team"})
    charts = charts[
        charts["season"].eq(season)
        & charts["week"].eq(week)
        & charts["position"].eq("QB")
    ]
    if "formation" in charts.columns:
        charts = charts[charts["formation"].eq("Offense")]
    if not charts.empty and "depth_team" in charts.columns:
        qb1 = (
            charts.assign(rank=pd.to_numeric(charts["depth_team"], errors="coerce"))
            .sort_values("rank")
            .groupby("team")["gsis_id"]
            .first()
            .dropna()
        )
        known = set(qb_games["passer_player_id"])
        qb1 = qb1[qb1.isin(known) | ~qb1.index.isin(result.index)]
        result.update(qb1)

    overrides = _overrides()
    overrides = overrides[
        overrides["season"].eq(season) & overrides["week"].eq(week)
    ]
    for row in overrides.itertuples():
        result[row.team_abbr] = row.gsis_id
    return result
