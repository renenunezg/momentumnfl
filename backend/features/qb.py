"""QB game logs and expected-starter detection."""

import pandas as pd

from backend.config import REPO_ROOT

# Optional manual starter overrides: season, week, team_abbr, gsis_id.
OVERRIDES_PATH = REPO_ROOT / "overrides" / "qb_starters.csv"
# Injury-report statuses that remove a quarterback from the expected lineup.
UNAVAILABLE_STATUSES = ("Out", "Doubtful")


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


def _depth_chart_qbs(
    depth_charts: pd.DataFrame, season: int, week: int, as_of=None
) -> pd.DataFrame | None:
    """(team, gsis_id) depth-chart quarterbacks in rank order, handling both
    nflverse formats: the weekly format (through 2024:
    season/week/position/depth_team) and the snapshot format (2025+:
    dt/pos_abb/pos_rank)."""
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
        return latest.sort_values("pos_rank")[["team", "gsis_id"]]
    if "season" not in depth_charts.columns:
        return None
    charts = depth_charts.rename(columns={"club_code": "team"})
    charts = charts[
        charts["season"].eq(season)
        & (charts["week"].lt(week) if as_of is not None else charts["week"].eq(week))
        & charts["position"].eq("QB")
        & charts["gsis_id"].notna()
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
        .drop_duplicates(["team", "gsis_id"])[["team", "gsis_id"]]
    )


def ruled_out(
    injuries: pd.DataFrame | None, season: int, week: int, as_of=None
) -> set[str]:
    """gsis_ids listed Out or Doubtful on the target week's injury report.

    Week numbers are unique across regular-season and playoff game types, so
    season and week identify the report. A report carrying date_modified
    counts only when it was modified before as_of. The 2025+ feed has no
    timestamp; its target-week report is treated as pregame because final
    game statuses are published before kickoff by league rule.
    """
    if injuries is None or injuries.empty:
        return set()
    reports = injuries[
        injuries["season"].eq(season)
        & injuries["week"].eq(week)
        & injuries["report_status"].isin(UNAVAILABLE_STATUSES)
    ]
    if as_of is not None and "date_modified" in reports.columns:
        dates = pd.to_datetime(reports["date_modified"], utc=True)
        untimestamped = reports.get(
            "untimestamped_weekly_report", pd.Series(False, index=reports.index)
        ).eq(True)
        reports = reports[(dates.isna() & untimestamped) | dates.le(as_of)]
    return set(reports["gsis_id"].dropna())


def depth_chart_starters(
    depth_charts: pd.DataFrame,
    season: int,
    week: int,
    as_of=None,
    unavailable: set[str] = frozenset(),
) -> pd.Series | None:
    """team -> highest-ranked depth-chart QB who is not ruled out."""
    charts = _depth_chart_qbs(depth_charts, season, week, as_of)
    if charts is None:
        return None
    available = charts[~charts["gsis_id"].isin(unavailable)]
    return available.groupby("team")["gsis_id"].first()


def lineup_starters(depth_charts, season, week, as_of, injuries, overrides):
    """Verified overrides and announcements precede the actual depth-chart order."""
    from backend.starter_announcements import confirmed

    unavailable = ruled_out(injuries, season, week, as_of)
    charted = depth_chart_starters(depth_charts, season, week, as_of, unavailable)
    selected = {} if charted is None else charted.to_dict()
    evidence = {team: dict(basis="depth_chart_fallback") for team in selected}
    for team, announcement in confirmed(
        injuries, season, week, as_of, unavailable
    ).items():
        selected[team] = announcement["gsis_id"]
        evidence[team] = dict(basis="announced_starter", **announcement)
    overrides = overrides[overrides.season.eq(season) & overrides.week.eq(week)]
    for row in overrides.itertuples():
        if row.gsis_id in unavailable:
            raise ValueError(
                f"QB override conflicts with confirmed absence: {row.team_abbr}"
            )
        announcement = evidence.get(row.team_abbr, {})
        if (
            announcement.get("basis") == "announced_starter"
            and selected[row.team_abbr] != row.gsis_id
        ):
            raise ValueError(
                f"QB override conflicts with announced starter: {row.team_abbr}"
            )
        selected[row.team_abbr] = row.gsis_id
        evidence[row.team_abbr] = dict(
            basis="verified_override", source_url=getattr(row, "source_url", None)
        )
    return pd.Series(selected, dtype=object), evidence


def expected_starters(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    depth_charts: pd.DataFrame,
    season: int,
    week: int,
    as_of=None,
    use_overrides: bool = True,
    injuries: pd.DataFrame | None = None,
) -> pd.Series:
    """team -> expected starter gsis_id for the given week.

    Priority: verified overrides, announcements, then available depth-chart order.
    Undated prior-week charts cannot supersede a newer current-season starter.
    The latest available actual starter is the fallback for missing charts.
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
    unavailable = ruled_out(injuries, season, week, as_of)
    # Latest available starter is the fallback; current depth charts supersede it.
    prior_starts = prior[
        prior["started"] & ~prior["passer_player_id"].isin(unavailable)
    ].sort_values(["season", "model_week"])
    result = prior_starts.drop_duplicates("team", keep="last").set_index("team")[
        "passer_player_id"
    ]

    charted = depth_chart_starters(depth_charts, season, week, as_of, unavailable)
    if charted is not None and not charted.empty:
        # A named rookie starter has no NFL history yet. Keep that identity;
        # the projection layer supplies replacement value for an unseen QB.
        result = charted.combine_first(result)
        if as_of is not None and "pos_abb" not in depth_charts.columns:
            # Undated prior-week charts predate that week's actual starter.
            # Keep the newer current-season observation unless he is ruled out.
            latest = prior[prior["started"] & prior["season"].eq(season)]
            latest = latest.sort_values(["season", "model_week"])
            latest = latest.drop_duplicates("team", keep="last")
            latest = latest[~latest["passer_player_id"].isin(unavailable)]
            recent = latest.set_index("team")["passer_player_id"]
            result = recent.combine_first(result)

    overrides = (
        _overrides()
        if use_overrides
        else pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"])
    )
    selected, evidence = lineup_starters(
        depth_charts, season, week, as_of, injuries, overrides
    )
    if injuries is not None and "current_status" in injuries:
        # Prospective inputs must not revive a former starter on a missing chart.
        return selected
    # Keep the historical fallback rules for undated weekly charts.
    for team, starter in selected.items():
        if (
            evidence[team]["basis"] != "depth_chart_fallback"
            or "pos_abb" in depth_charts
        ):
            result[team] = starter
    return result
