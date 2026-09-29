"""Acceptance gates for award cutoffs, eligibility, and isolated publication."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import publish
from backend.awards import AWARDS, features, model, pipeline, readiness, value


def test_award_snapshot_excludes_future_stats_and_checks_eligibility(
    monkeypatch, tmp_path
):
    schedule = pd.read_parquet(
        Path(__file__).parent / "fixtures/schedule_2022_regular.parquet"
    )
    schedule["home_coach"] = schedule.home_team + " coach"
    schedule["away_coach"] = schedule.away_team + " coach"
    first = schedule[schedule.week.eq(1)].iloc[0]
    stats = pd.DataFrame(
        [
            {
                **dict.fromkeys(features.STATS, 0.0),
                "player_id": "rookie",
                "game_id": first.game_id,
                "season": 2022,
                "week": 1,
                "season_type": "REG",
                "player_display_name": "Rookie",
                "position": "WR",
                "team": first.home_team,
                "receiving_yards": 100.0,
            },
            {
                **dict.fromkeys(features.STATS, 0.0),
                "player_id": "veteran",
                "game_id": first.game_id,
                "season": 2022,
                "week": 1,
                "season_type": "REG",
                "player_display_name": "Veteran",
                "position": "WR",
                "team": first.away_team,
                "receiving_yards": 200.0,
            },
        ]
    )
    # Complete schedule, but only one completed game at this recorded cutoff.
    schedule.loc[schedule.game_id.ne(first.game_id), ["home_score", "away_score"]] = (
        np.nan
    )
    prior_schedule = schedule.copy()
    prior_schedule[["home_score", "away_score"]] = [20, 10]
    roster = pd.DataFrame(
        {"gsis_id": ["rookie", "veteran"], "rookie_year": [2022, 2020]}
    )
    prior_stats = stats.copy()

    def read(name, season):
        if name == "schedules":
            return (schedule if season == 2022 else prior_schedule).copy()
        if name == "rosters":
            return roster.copy()
        if name == "stats":
            return (stats if season == 2022 else prior_stats).copy()
        raise AssertionError(name)

    monkeypatch.setattr(features.ingest, "read", read)
    monkeypatch.setattr(features.ingest, "provenance", lambda _: {})
    monkeypatch.setattr(features.value, "attach", lambda frame, *_: frame)
    before, coaches, audit = features.snapshot(2022, 1, "2022-09-14T12:00:00Z")
    rookie_pool = features.candidates(before, coaches, "OROY")
    assert rookie_pool.candidate_id.tolist() == ["rookie"]
    assert audit["games_included"] == 1
    # Future production and postseason records must never alter the snapshot.
    future = stats.iloc[[0]].copy()
    future["game_id"], future["week"], future["receiving_yards"] = "future", 18, 1e8
    stats = pd.concat([stats, future], ignore_index=True)
    after, _, _ = features.snapshot(2022, 1, "2022-09-14T12:00:00Z")
    pd.testing.assert_frame_equal(before, after)
    context = pd.DataFrame(
        [
            {
                "season": 2022,
                "candidate_id": "veteran",
                "candidate_name": "Veteran",
                "team": first.away_team,
                "position": "WR",
                "known_at": "2022-09-20T00:00:00Z",
                "source_url": "https://www.nfl.com/news/example",
                "reason": "Injury return",
                "eligible": True,
            }
        ]
    )
    context.to_csv(tmp_path / "award_comeback_context.csv", index=False)
    monkeypatch.setattr(features, "STATIC_DIR", tmp_path)
    assert features.candidates(before, coaches, "CPOY").empty
    assert len(features.comeback_context(2022, "2022-09-21T00:00:00Z")) == 1
    early, _, _ = features.snapshot(2022, 1, "2022-09-01T12:00:00Z")
    assert early.empty
    stats = stats[stats.game_id.ne(first.game_id)]
    with pytest.raises(ValueError, match="every completed game"):
        features.snapshot(2022, 1, "2022-09-14T12:00:00Z")

    # Production on Tuesday must include MNF once its actual sources are ready.
    schedule.loc[schedule.week.eq(1), ["home_score", "away_score"]] = [20, 10]
    games = schedule[schedule.week.eq(1)]
    from backend.features.drives import kickoff_utc

    monday = games.loc[kickoff_utc(games).idxmax()]
    tuesday = "2022-09-13T16:00:00Z"
    assert (
        pd.Timestamp(tuesday).value - kickoff_utc(games).max().value < 24 * 3600 * 10**9
    )
    stats = pd.DataFrame(
        [
            {
                **dict.fromkeys(features.STATS, 0.0),
                "player_id": f"player:{team}",
                "game_id": game.game_id,
                "season": 2022,
                "week": 1,
                "season_type": "REG",
                "player_display_name": team,
                "position": "QB",
                "team": team,
                "attempts": 20,
                "def_tackles_solo": 5,
                "def_tackle_assists": 5,
            }
            for game in games.itertuples()
            for team in (game.home_team, game.away_team)
        ]
    )
    pbp = pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "desc": "END GAME",
                "total_home_score": game.home_score,
                "total_away_score": game.away_score,
                "posteam": team,
                "epa": 0.0,
            }
            for game in games.itertuples()
            for team in (game.home_team, game.away_team)
        ]
    )
    monkeypatch.setattr(readiness, "RAW_DIR", tmp_path)
    raw = tmp_path / "pbp" / "2022.parquet"
    raw.parent.mkdir()
    pbp.to_parquet(raw)
    monkeypatch.setattr(pipeline, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(pipeline, "history_for", lambda *args: ({"COY": []}, {}))

    class TuesdayClock:
        @staticmethod
        def now(tz):
            return pd.Timestamp(tuesday).to_pydatetime()

    monkeypatch.setattr(pipeline, "datetime", TuesdayClock)

    def production(*, write=True):
        return pipeline.build(2022, -1, awards=["COY"], write=write)

    _, meta, _ = production(write=False)
    audit = json.loads(meta.provenance.iloc[0])
    assert meta.week.tolist() == [1]
    assert audit["games_included"] == len(games)
    assert audit["readiness"]["team_games_checked"] == len(games) * 2
    _, _, historical = features.snapshot(2022, 1, tuesday)
    assert historical["games_included"] == len(games) - 1
    assert historical["readiness"] is None
    _, historical_meta, _ = pipeline.build(
        2022, 1, tuesday, awards=["COY"], write=False
    )
    assert (
        json.loads(historical_meta.provenance.iloc[0])["games_included"]
        == len(games) - 1
    )
    assert readiness.resolve_week(schedule, "2022-09-01T12:00:00Z") == 0
    # A missing Monday result or postponed game must not select an older week.
    schedule.loc[schedule.game_id.eq(monday.game_id), "away_score"] = np.nan
    with pytest.raises(ValueError, match="waiting for completed games"):
        production()
    schedule.loc[schedule.game_id.eq(monday.game_id), "away_score"] = 10
    original_day = monday.gameday
    schedule.loc[schedule.game_id.eq(monday.game_id), "gameday"] = "2022-09-15"
    with pytest.raises(ValueError, match="waiting for completed games"):
        production()
    schedule.loc[schedule.game_id.eq(monday.game_id), "gameday"] = original_day
    complete_stats = stats.copy()
    stats = stats[stats.team.ne(monday.away_team)]
    with pytest.raises(ValueError, match="waiting for player statistics"):
        production()
    stats = complete_stats.copy()
    stats.loc[stats.team.eq(monday.away_team), "def_tackles_solo"] = 0
    stats.loc[stats.team.eq(monday.away_team), "def_tackle_assists"] = 0
    with pytest.raises(ValueError, match="waiting for defense statistics"):
        production()
    stats = complete_stats
    for incomplete in (
        pbp[pbp.game_id.ne(monday.game_id)],
        pbp.assign(desc="END QUARTER"),
        pbp.assign(total_home_score=999),
    ):
        incomplete.to_parquet(raw)
        with pytest.raises(ValueError, match="waiting for final play-by-play"):
            production()
    pbp.assign(epa=np.nan).to_parquet(raw)
    with pytest.raises(ValueError, match="waiting for play-by-play EPA"):
        production()
    assert not (tmp_path / "processed").exists()


def test_competitive_credit_and_chronological_missing_winner_gate():
    pbp = pd.read_parquet(
        Path(__file__).parent / "fixtures/pbp_2023_07_DET_BAL.parquet"
    )
    credit = value.build_credit(pbp)
    changed = pbp.copy()
    from backend.features.drives import competitive_drive_mask

    changed.loc[~competitive_drive_mask(changed), "epa"] = 1e9
    pd.testing.assert_frame_equal(credit, value.build_credit(changed))
    races = []
    for year in range(2010, 2017):
        frame = pd.DataFrame(0.0, index=range(3), columns=features.features_for("COY"))
        frame["season"], frame["week"], frame["as_of"] = (
            year,
            8,
            f"{year}-11-01T00:00:00Z",
        )
        frame["winner"] = [1.0, 0.0, 0.0] if year != 2015 else [0.0, 0.0, 0.0]
        frame["win_pct"] = [0.8, 0.5, 0.3]
        races.append(frame)
    report = model.evaluate(races, "COY")
    assert (report.training_through < report.season).all()
    omitted = report[report.season.eq(2015)].iloc[0]
    assert not omitted.winner_in_pool and not omitted.winner_hit
    assert omitted.log_loss > 30 and omitted.brier >= 1
    original = report[report.season.eq(2014)].copy()
    races[-1]["win_pct"] = [100, -100, 0]
    pd.testing.assert_frame_equal(
        original, model.evaluate(races, "COY").query("season == 2014")
    )


def test_awards_publication_rejects_false_probability_claim_before_connecting(
    monkeypatch,
):
    stamp = "2026-09-15T12:00:00Z"
    meta = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "award": award,
                "as_of": stamp,
                "model_version": pipeline.MODEL_VERSION,
                "status": "ready" if award == "MVP" else "awaiting_games",
                "candidate_count": 1 if award == "MVP" else 0,
                "training_seasons": "[2020,2021,2022,2023]",
                "validation": json.dumps({"probabilities_publishable": False}),
            }
            for award in AWARDS
        ]
    )
    board = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "award": "MVP",
                "as_of": stamp,
                "model_version": pipeline.MODEL_VERSION,
                "candidate_id": "candidate",
                "predicted_rank": 1,
                "win_probability": 1.0,
            }
        ]
    )
    monkeypatch.setattr(
        publish.store,
        "read_processed",
        lambda *parts: board if "boards" in parts else meta,
    )
    with pytest.raises(ValueError, match="Unvalidated"):
        publish.publish_awards(None, 2026, 1)
