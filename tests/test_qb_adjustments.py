"""Production regressions: QB changes are substitutions, with one forecast cut."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from backend.features import qb as qb_features
from backend.model import qb_adjustment as qb
from backend.model.fit_week import compute_qb_adjustments
from backend.model.joint_scoring import DEFAULT_CONFIG, JointScoringFit
from backend.model.projections import LayerConfig, assemble_projections


def test_starter_swap_changes_projection_without_double_counting(monkeypatch):
    index = pd.DataFrame(
        [
            {
                "game_id": "old1",
                "season": 2025,
                "model_week": 17,
                "start_date": datetime(2025, 12, 21, tzinfo=UTC),
            },
            {
                "game_id": "old2",
                "season": 2025,
                "model_week": 18,
                "start_date": datetime(2025, 12, 28, tzinfo=UTC),
            },
        ]
    )
    logs = pd.DataFrame(
        [
            (game, team, passer, epa)
            for game in ("old1", "old2")
            for team, passer, epa in (("A", "starter", 12.0), ("B", "backup", -4.0))
        ],
        columns=["game_id", "team", "passer_player_id", "epa"],
    ).assign(dropbacks=40, started=True)
    slate = pd.DataFrame(
        [
            {
                "game_id": "new",
                "season": 2026,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "neutral_site": False,
            }
        ]
    )
    monkeypatch.setattr(
        "backend.model.fit_week.store.game_index", lambda seasons: index
    )
    overrides = pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"])
    monkeypatch.setattr(qb_features, "_overrides", lambda: overrides)
    layer = LayerConfig(market_weight=0)
    healthy = compute_qb_adjustments(
        2026, 1, slate, logs, pd.DataFrame(), DEFAULT_CONFIG, layer
    )
    assert healthy["new"] == pytest.approx((0, 0))
    context = qb.context_before_week(
        qb.strength_history(logs, index, layer.qb_span_dropbacks), 2026, 1, 6
    )
    overrides.loc[0] = [2026, 1, "A", "backup"]
    injured = compute_qb_adjustments(
        2026, 1, slate, logs, pd.DataFrame(), DEFAULT_CONFIG, layer
    )
    expected_loss = layer.qb_adjustment_weight * (
        context.strengths["backup"] - context.strengths["starter"]
    )
    assert injured["new"] == pytest.approx((expected_loss, 0))
    assert expected_loss < 0
    fit = JointScoringFit(
        season=2026,
        week=1,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        teams=["A", "B"],
        offense_ppd=np.array([0.3, -0.3]),
        defense_ppd=np.array([0.1, -0.1]),
        pace=np.zeros(2),
        base_ppd=2.2,
        base_drives=10.0,
        hfa_ppd=0.2,
        parameter_covariance=np.eye(5) * 0.01,
        score_residual_covariance=np.eye(2) * 50.0,
        config=DEFAULT_CONFIG,
    )
    before_ratings = [r.to_record() for r in fit.ratings({})]
    before = assemble_projections(fit, slate, fit.as_of, {}, healthy, config=layer)[0]
    after = assemble_projections(fit, slate, fit.as_of, {}, injured, config=layer)[0]
    assert after.pure_home_margin - before.pure_home_margin == pytest.approx(
        expected_loss
    )
    assert after.expected_away_points == before.expected_away_points
    assert [r.to_record() for r in fit.ratings({})] == before_ratings


def test_backtest_and_production_qb_context_ignore_target_week_outcomes():
    index = pd.DataFrame(
        [
            {
                "game_id": "prior",
                "season": 2025,
                "model_week": 18,
                "start_date": datetime(2025, 12, 28, tzinfo=UTC),
            },
            {
                "game_id": "target",
                "season": 2026,
                "model_week": 1,
                "start_date": datetime(2026, 9, 9, tzinfo=UTC),
            },
        ]
    )
    logs = pd.DataFrame(
        [
            {
                "game_id": "prior",
                "team": "A",
                "passer_player_id": "starter",
                "dropbacks": 40,
                "epa": 10.0,
                "started": True,
            },
            {
                "game_id": "target",
                "team": "A",
                "passer_player_id": "starter",
                "dropbacks": 40,
                "epa": 9999.0,
                "started": True,
            },
        ]
    )
    full = qb.context_before_week(qb.strength_history(logs, index), 2026, 1, 6)
    cut = qb.context_before_week(qb.strength_history(logs.iloc[:1], index), 2026, 1, 6)
    assert full == cut
    assert full.adjustment("A", "starter") == pytest.approx(0)
    assert full.adjustment("A", "rookie") < 0
