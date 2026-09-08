"""Descriptive competitive EPA credit, separate from voter-facing full totals."""

import pandas as pd

from backend.config import PROCESSED_DIR, RAW_DIR
from backend.etl.store import write_parquet
from backend.features.drives import competitive_drive_mask


def build_credit(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[
        competitive_drive_mask(pbp)
        & pbp.epa.notna()
        & pbp.season_type.eq("REG")
        & pbp.play_type.isin(["pass", "run"])
    ].copy()
    frames = []
    for role in ("passer", "receiver", "rusher"):
        selected = plays[plays[f"{role}_player_id"].notna()].copy()
        if role == "rusher":
            selected = selected[selected.play_type.eq("run")]
            share = pd.Series(1.0, index=selected.index)
        else:
            selected = selected[selected.play_type.eq("pass")]
            share = pd.Series(0.5, index=selected.index)
            if role == "passer":
                share = share.where(selected.receiver_player_id.notna(), 1.0)
        frame = selected[["game_id", "week"]].copy()
        frame["candidate_id"] = selected[f"{role}_player_id"]
        frame["competitive_epa"] = selected.epa * share
        frame["credit_opportunities"] = share
        frames.append(frame)
    return (
        pd.concat(frames)
        .groupby(["game_id", "week", "candidate_id"], as_index=False)[
            ["competitive_epa", "credit_opportunities"]
        ]
        .sum()
    )


def attach(
    frame: pd.DataFrame, season: int, game_ids: pd.Series, *, write_cache: bool = False
) -> pd.DataFrame:
    frame = frame.copy()
    path = PROCESSED_DIR / "awards" / "competitive_credit" / f"{season}.parquet"
    raw = RAW_DIR / "pbp" / f"{season}.parquet"
    credit = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if raw.exists() and (
        not path.exists() or path.stat().st_mtime < raw.stat().st_mtime
    ):
        columns = [
            "game_id",
            "play_id",
            "week",
            "season_type",
            "posteam",
            "epa",
            "play_type",
            "fixed_drive",
            "score_differential",
            "qtr",
            "passer_player_id",
            "receiver_player_id",
            "rusher_player_id",
        ]
        credit = build_credit(pd.read_parquet(raw, columns=columns))
        if write_cache:
            write_parquet(credit, path)
    frame["competitive_epa"] = float("nan")
    frame["credit_opportunities"] = float("nan")
    if not credit.empty:
        credit = (
            credit[credit.game_id.isin(game_ids)]
            .groupby("candidate_id")[["competitive_epa", "credit_opportunities"]]
            .sum()
        )
        for column in credit:
            frame[column] = frame.candidate_id.map(credit[column])
    rate = frame.competitive_epa / frame.credit_opportunities.clip(lower=1)
    qualified = rate.where(frame.credit_opportunities.ge(10))
    replacement = qualified.groupby(frame.position).transform(
        lambda s: s.quantile(0.25)
    )
    reliability = frame.credit_opportunities / (frame.credit_opportunities + 150)
    frame["performance_score"] = (
        frame.competitive_epa - replacement * frame.credit_opportunities
    ) * reliability
    return frame
