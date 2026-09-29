"""Observed source readiness for production awards, without a fixed time delay."""

import pandas as pd

from backend.config import RAW_DIR
from backend.features.drives import kickoff_utc


def resolve_week(schedule: pd.DataFrame, as_of) -> int:
    regular = schedule[schedule.game_type.eq("REG")]
    if regular.empty:
        raise ValueError("Awards schedule has no regular-season games")
    started = regular[kickoff_utc(regular).le(pd.Timestamp(as_of))]
    return int(started.week.max()) if not started.empty else 0


def validate(played: pd.DataFrame, stats: pd.DataFrame, season: int) -> dict:
    """Require both teams' stats and final, scored play-by-play for every game."""
    required = {
        "game_id",
        "team",
        "attempts",
        "carries",
        "targets",
        "def_tackles_solo",
        "def_tackle_assists",
    }
    if missing_columns := required - set(stats):
        raise ValueError(f"Awards missing statistic columns: {sorted(missing_columns)}")
    expected = pd.MultiIndex.from_frame(
        pd.concat(
            [
                played[["game_id", side]].rename(columns={side: "team"})
                for side in ("home_team", "away_team")
            ],
            ignore_index=True,
        )
    )
    present = pd.MultiIndex.from_frame(stats[["game_id", "team"]].drop_duplicates())
    missing = expected.difference(present)
    if len(missing):
        raise ValueError(f"Awards waiting for player statistics: {missing.tolist()}")
    # A partially published offensive-only or defensive-only file is not ready.
    for label, columns in (
        ("offense", ["attempts", "carries", "targets"]),
        ("defense", ["def_tackles_solo", "def_tackle_assists"]),
    ):
        active = stats[stats[columns].sum(axis=1, min_count=1).gt(0)]
        covered = pd.MultiIndex.from_frame(
            active[["game_id", "team"]].drop_duplicates()
        )
        missing = expected.difference(covered)
        if len(missing):
            raise ValueError(
                f"Awards waiting for {label} statistics: {missing.tolist()}"
            )
    path = RAW_DIR / "pbp" / f"{season}.parquet"
    if not path.exists():
        raise ValueError(f"Awards waiting for {season} play-by-play")
    pbp = pd.read_parquet(
        path,
        columns=[
            "game_id",
            "desc",
            "total_home_score",
            "total_away_score",
            "posteam",
            "epa",
        ],
    )
    pbp = pbp[pbp.game_id.isin(played.game_id)]
    final = pbp[pbp.desc.fillna("").str.strip().eq("END GAME")]
    final = final.drop_duplicates("game_id", keep="last").set_index("game_id")
    scores = played.set_index("game_id")
    final = final.reindex(scores.index)
    complete = final.total_home_score.eq(scores.home_score) & final.total_away_score.eq(
        scores.away_score
    )
    if not complete.all():
        raise ValueError(
            "Awards waiting for final play-by-play matching schedule scores: "
            f"{scores.index[~complete].tolist()}"
        )
    credit = pbp[pbp.epa.notna()].rename(columns={"posteam": "team"})
    covered = pd.MultiIndex.from_frame(credit[["game_id", "team"]].drop_duplicates())
    missing = expected.difference(covered)
    if len(missing):
        raise ValueError(f"Awards waiting for play-by-play EPA: {missing.tolist()}")
    return {"games_checked": len(played), "team_games_checked": len(expected)}
