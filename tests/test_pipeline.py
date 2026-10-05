"""Production pipeline regressions for cached history and special game days."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from backend import pipeline
from backend.etl import ingest, store
from backend.features.qb import expected_starters


def test_upcoming_games_does_not_assume_a_weekday():
    schedules = pd.DataFrame(
        [
            {
                "game_id": "2026_17_WED_GAME",
                "season": 2026,
                "home_score": None,
                "gameday": "2026-12-23",
                "gametime": "13:00",
            }
        ]
    )
    now = datetime(2026, 12, 23, 15, 0, tzinfo=UTC)

    games = pipeline.upcoming_games(schedules, 2026, hours=4, now=now)

    assert games["game_id"].tolist() == ["2026_17_WED_GAME"]


def test_incremental_write_replaces_game_and_keeps_prior_weeks(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path)
    existing = pd.DataFrame(
        [
            {"game_id": "W2", "value": 2, "start_date": "2026-09-08"},
            {"game_id": "W3", "value": 3, "start_date": "2026-09-15"},
        ]
    )
    store.write_processed(existing, "team_games", "2026.parquet")
    schedules = pd.DataFrame(
        [
            {"game_id": "W1", "week": 1, "home_score": 20},
            {"game_id": "W2", "week": 2, "home_score": 20},
            {"game_id": "W3", "week": 3, "home_score": 20},
            {"game_id": "W4", "week": 4, "home_score": None},
        ]
    )
    selected = pipeline.incremental_game_ids(
        schedules, [{"W1", "W2", "W3"}, set(existing.game_id)], lookback_weeks=1
    )
    rebuilt = pd.DataFrame(
        [
            {"game_id": "W1", "value": 10, "start_date": "2026-09-01"},
            {"game_id": "W2", "value": 20, "start_date": "2026-09-08"},
            {"game_id": "W3", "value": 30, "start_date": "2026-09-15"},
            {"game_id": "W4", "value": 40, "start_date": "2026-09-22"},
        ]
    )
    pipeline.write_incremental_features(
        rebuilt[rebuilt.game_id.isin(selected)],
        "team_games",
        2026,
        ["start_date", "game_id"],
    )
    saved = store.read_processed("team_games", "2026.parquet")
    assert saved[["game_id", "value"]].to_dict("records") == [
        {"game_id": "W1", "value": 10},
        {"game_id": "W2", "value": 2},
        {"game_id": "W3", "value": 30},
    ]


def test_preseason_runs_load_current_starters_before_pbp_opens(monkeypatch, tmp_path):
    from backend import source_inputs

    monkeypatch.setattr(source_inputs, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(source_inputs, "ARCHIVE", tmp_path / "source_archive")
    monkeypatch.setattr(ingest, "_current_injuries", lambda season: [])
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        ingest.nflreadpy,
        "get_current_season",
        lambda roster=False: 2026 if roster else 2025,
    )
    monkeypatch.setattr(ingest.data, "load_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest.data, "load_teams", pd.DataFrame)
    charts = pd.DataFrame(
        [
            ("2026-09-06", "MIN", "former", 1),
            ("2026-09-07", "MIN", "transfer", 1),
            ("2026-09-06", "LV", "rookie", 1),
            ("2026-09-07", "NEW", "new_team_qb", 1),
        ],
        columns=["dt", "team", "gsis_id", "pos_rank"],
    ).assign(pos_abb="QB")
    monkeypatch.setattr(ingest.data, "load_depth_charts", lambda seasons: charts)

    def game_data_unavailable(*args, **kwargs):
        raise AssertionError("Preseason must not request unavailable game data")

    for loader in ("load_pbp", "load_injuries", "load_pfr_advstats"):
        monkeypatch.setattr(ingest.data, loader, game_data_unavailable)

    for run in (ingest.ingest_season, ingest.ingest_projection_inputs):
        assert run(2026) == []
        loaded = store.read_raw("depth_charts", "2026.parquet")
        pd.testing.assert_frame_equal(loaded, charts)
        (tmp_path / "depth_charts" / "2026.parquet").unlink()

    index = pd.DataFrame([{"game_id": "old", "season": 2025, "model_week": 18}])
    history = pd.DataFrame(
        [("old", "MIN", "former"), ("old", "LV", "incumbent")],
        columns=["game_id", "team", "passer_player_id"],
    ).assign(dropbacks=30, started=True)
    starters = expected_starters(history, index, loaded, 2026, 1)
    assert starters.to_dict() == {
        "MIN": "transfer",
        "LV": "rookie",
        "NEW": "new_team_qb",
    }


@pytest.mark.parametrize("home_qb_adjustment", [-1.25, -100.0])
def test_early_totals_stabilization_preserves_published_spread_and_moneyline(
    home_qb_adjustment,
):
    """One high-scoring opening week must not replace the preseason environment."""
    from dataclasses import replace

    from backend.model.joint_scoring import (
        DEFAULT_CONFIG,
        JointScoringFit,
        fit_joint_scoring,
    )
    from backend.model.market_blend import MARGINS
    from backend.model.projections import (
        align_ratings_to_forecast,
        assemble_projections,
    )
    from backend.model.totals import TotalsConfig
    from backend.recommendations import _h2h_projection, _probabilities

    as_of = datetime(2026, 9, 15, tzinfo=UTC)
    teams = [f"T{i:02d}" for i in range(32)]
    prior = JointScoringFit(
        season=2026,
        week=1,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        teams=teams,
        offense_ppd=np.linspace(-0.15, 0.15, 32),
        defense_ppd=np.linspace(0.1, -0.1, 32),
        pace=np.zeros(32),
        base_ppd=2.2,
        base_drives=10.0,
        hfa_ppd=0.2,
        parameter_covariance=np.eye(65) * 0.01,
        score_residual_covariance=np.eye(2) * 50,
        config=DEFAULT_CONFIG,
    )
    games = pd.DataFrame(
        [
            dict(
                game_id=f"2026_01_{i}",
                season=2026,
                week=1,
                model_week=1,
                home_team=teams[i],
                away_team=teams[-i - 1],
                neutral_site=False,
                home_points=28.0 + i % 3,
                away_points=23.0 - i % 3,
                game_drives=11.0,
                competitive_drives=10.0,
                home_competitive_points=26.0 + i % 3,
                away_competitive_points=21.0 - i % 3,
                home_epa_per_drive=0.1 + i / 100,
                away_epa_per_drive=-0.1,
                feature_version=2,
                start_date=datetime(2026, 9, 10, tzinfo=UTC),
            )
            for i in range(16)
        ]
    )
    legacy = fit_joint_scoring(
        games,
        2,
        as_of,
        strength_prior=prior,
        totals_config=TotalsConfig(scoring_prior_games=0, pace_prior_games=0),
    )
    fixed = fit_joint_scoring(games, 2, as_of, strength_prior=prior)
    assert (
        2 * prior.base_ppd * prior.base_drives
        < (2 * fixed.totals_base_ppd * fixed.totals_base_drives)
        < 2 * legacy.base_ppd * legacy.base_drives
    )
    assert [r.to_record() for r in fixed.ratings({})] == [
        r.to_record() for r in legacy.ratings({})
    ]
    slate = games.assign(
        week=2, model_week=2, start_date=datetime(2026, 9, 20, tzinfo=UTC)
    )
    # The extreme substitution reaches the existing zero-score clamp.
    # Applying the total shift before that clamp would alter the margin.
    qb_adjustments = {g: (home_qb_adjustment, 0.5) for g in slate.game_id}
    market_spreads = {g: -3.5 for g in slate.game_id}
    before, after = [
        assemble_projections(fit, slate, as_of, {}, qb_adjustments, market_spreads)
        for fit in (legacy, fixed)
    ]
    distribution = np.ones(len(MARGINS))
    uncalibrated = assemble_projections(
        replace(
            fixed,
            totals_config=replace(
                fixed.totals_config, location_slope=1.0, location_intercept=0.0
            ),
        ),
        slate,
        as_of,
        {},
        qb_adjustments,
        market_spreads,
    )
    # The published total and scores move toward the market total; the total
    # that pick pricing reads and the published margin do not.
    published = assemble_projections(
        fixed,
        slate,
        as_of,
        {},
        qb_adjustments,
        market_spreads,
        market_totals={g: 40.0 for g in slate.game_id},
    )
    for base, row in zip(after, published, strict=True):
        assert row.model_total == pytest.approx(base.model_total)
        assert row.market_informed_total == pytest.approx(
            max(0.5 * (base.model_total + 40.0), abs(base.home_margin))
        )
        assert (
            row.market_informed_home_points - row.market_informed_away_points
        ) == pytest.approx(base.home_margin)
        assert base.market_informed_total == pytest.approx(base.model_total)
    # Published ratings reproduce every published line.
    aligned = {
        rating.team_abbr: rating
        for rating in align_ratings_to_forecast(fixed.ratings({}), after)
    }
    for row in after:
        assert (
            aligned[row.home_team_abbr].power_rating
            - aligned[row.away_team_abbr].power_rating
            + row.home_field_points
        ) == pytest.approx(row.home_margin)
    for old, new, raw in zip(before, after, uncalibrated, strict=True):
        # Pin the selected location correction after QB/clipping, including
        # the boundary where a total reduction would change the pure margin.
        assert new.model_total == pytest.approx(
            max(
                45 - 0.19747920925833432 + 0.7163893796380143 * (raw.model_total - 45),
                abs(raw.pure_home_margin),
                abs(raw.home_margin),
            )
        )
        assert new.model_version.endswith("totals_v2")
        if home_qb_adjustment > -100:
            assert 0 < new.model_total < old.model_total
        else:
            assert new.model_total == pytest.approx(old.model_total)
        assert min(new.expected_home_points, new.expected_away_points) > 0
        assert new.pure_home_margin == pytest.approx(old.pure_home_margin)
        assert new.home_margin == pytest.approx(old.home_margin)
        assert new.margin_sd == old.margin_sd
        assert new.total_sd == old.total_sd
        old_row = next(pd.DataFrame([old.to_record()]).itertuples())
        new_row = next(pd.DataFrame([new.to_record()]).itertuples())
        for market in ("spreads", "h2h"):
            offer = dict(market=market, side="home", point=-3.5)
            old_priced, new_priced = (
                (_h2h_projection(old_row), _h2h_projection(new_row))
                if market == "h2h"
                else (old_row, new_row)
            )
            assert _probabilities(new_priced, offer, distribution) == pytest.approx(
                _probabilities(old_priced, offer, distribution)
            )


def test_production_win_total_slope_uses_the_calibration_reference(monkeypatch):
    """The blend weight was selected with this slope; a shrunk engine halves it."""
    from backend.model import calibration, preseason

    seen = []
    monkeypatch.setattr(
        preseason, "load_win_totals", lambda season: pd.Series({"A": 9.0, "B": 8.0})
    )
    monkeypatch.setattr(
        preseason.store,
        "season_games",
        lambda season: pd.DataFrame({"model_week": [1]}),
    )

    def fit(games, week, as_of, config):
        seen.append(config)
        raise KeyError

    monkeypatch.setattr(preseason, "fit_joint_scoring", fit)
    with pytest.raises(KeyError):
        preseason.points_per_win([2025])

    assert seen == [preseason.SLOPE_REFERENCE_CONFIG]
    assert calibration.SLOPE_REFERENCE_CONFIG is preseason.SLOPE_REFERENCE_CONFIG
    assert seen[0].rating_half_life_weeks == float("inf")


def test_calibration_identity_includes_availability_and_missingness(
    tmp_path, monkeypatch
):
    from backend.model import artifacts

    monkeypatch.setattr(artifacts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "STATIC_DIR", tmp_path / "static")
    monkeypatch.setattr(artifacts, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(artifacts, "PROCESSED_DIR", tmp_path / "processed")
    missing = artifacts.fingerprint()
    path = tmp_path / "raw/depth_charts" / f"{artifacts.DEVELOPMENT_SEASONS[0]}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"starter": ["A"]}).to_parquet(path)
    present = artifacts.fingerprint()
    assert present != missing and artifacts.fingerprint() == present
    pd.DataFrame({"starter": ["B"]}).to_parquet(path)
    assert artifacts.fingerprint() != present
    path.unlink()
    assert artifacts.fingerprint() == missing
    with pytest.raises(FileNotFoundError, match="bootstrap-history"):
        artifacts.ensure()
