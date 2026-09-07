"""Production regressions: QB changes are substitutions, with one forecast cut."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from backend.features import qb as qb_features
from backend.features.qb_context import build_context_games
from backend.model import qb_adjustment as qb
from backend.model.fit_week import compute_qb_adjustments
from backend.model.joint_scoring import DEFAULT_CONFIG, JointScoringFit
from backend.model.projections import LayerConfig, assemble_projections
from backend.model.qb_context import fit_context


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

    # Half of a preseason prior already prices the new reference QB. A later
    # override must still move the forecast by the full calibrated QB gap.
    monkeypatch.setattr(
        "backend.model.fit_week.load_qb_references", lambda season: {"A": "backup"}
    )
    overrides.drop(overrides.index, inplace=True)
    anchored_healthy = compute_qb_adjustments(
        2026, 1, slate, logs, pd.DataFrame(), DEFAULT_CONFIG, layer
    )
    overrides.loc[0] = [2026, 1, "A", "backup"]
    anchored_swap = compute_qb_adjustments(
        2026, 1, slate, logs, pd.DataFrame(), DEFAULT_CONFIG, layer
    )
    assert anchored_swap["new"][0] == pytest.approx(expected_loss * 0.5)
    assert anchored_swap["new"][0] - anchored_healthy["new"][0] == pytest.approx(
        expected_loss
    )


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

    # A team whose opener is postponed still needs its previous-season
    # baseline after other teams have played week one.
    other = logs.iloc[[1]].assign(team="B", passer_player_id="other")
    bye_history = qb.strength_history(pd.concat([logs.iloc[:1], other]), index)
    bye = qb.context_before_week(bye_history, 2026, 2, 6)
    same_cut = qb.context_before_week(
        qb.strength_history(logs.iloc[:1], index), 2026, 2, 6
    )
    assert bye.baselines["A"] == same_cut.baselines["A"]
    stale = qb.context_before_week(
        qb.strength_history(logs.iloc[:1], index), 2027, 1, 6
    )
    assert stale.effective_dropbacks["starter"] == pytest.approx(
        0.5 * cut.effective_dropbacks["starter"]
    )
    assert abs(stale.strengths["starter"]) < abs(cut.strengths["starter"])

    observations = logs.merge(index, on="game_id").assign(opponent="B", hit_sacks=2)
    full_context = fit_context(observations, 2026, 1)
    cut_context = fit_context(observations.iloc[:1], 2026, 1)
    assert full_context.strengths == cut_context.strengths
    swap = [{"rookie": 1, "starter": -1}]
    assert full_context.contrast_sd(swap) == pytest.approx(
        cut_context.contrast_sd(swap)
    )
    assert full_context.contrast_sd(swap)[0] > 0

    # The context input must credit scrambles to the runner, remove a
    # receiver's fumble penalty, and discard blowout observations.
    plays = pd.DataFrame(
        [
            ("starter", None, 0, -10.0, 3.0, 0),
            (None, "starter", 1, 2.0, 2.0, 0),
            ("starter", None, 0, 9999.0, 9999.0, 40),
        ],
        columns=[
            "passer_player_id",
            "rusher_player_id",
            "qb_scramble",
            "epa",
            "qb_epa",
            "score_differential",
        ],
    )
    plays = plays.assign(
        game_id="prior",
        posteam="A",
        defteam="B",
        qb_dropback=1,
        qb_hit=0,
        sack=0,
        qtr=4,
    )
    built = build_context_games(plays)
    assert built["passer_player_id"].tolist() == ["starter"]
    assert built["dropbacks"].tolist() == [2]
    assert built["epa"].tolist() == [5.0]
