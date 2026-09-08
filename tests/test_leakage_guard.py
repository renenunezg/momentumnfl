"""The fit must refuse to train on games at or after as_of."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.model.joint_scoring import fit_joint_scoring


def _toy_games() -> pd.DataFrame:
    rows = []
    for week in (1, 2, 3, 4, 5):
        rows.append(
            {
                "game_id": f"2023_{week:02d}_A_B",
                "season": 2023,
                "model_week": week,
                "home_team": "A",
                "away_team": "B",
                "neutral_site": False,
                "home_points": 24.0,
                "away_points": 20.0,
                "game_drives": 22.0,
                "competitive_drives": 22.0,
                "home_competitive_points": 24.0,
                "away_competitive_points": 20.0,
                "feature_version": 2,
                "home_epa_per_drive": 0.2,
                "away_epa_per_drive": -0.1,
                "start_date": datetime(2023, 9, week, tzinfo=UTC),
            }
        )
    return pd.DataFrame(rows)


def test_fit_refuses_future_games():
    games = _toy_games()
    with pytest.raises(ValueError):
        fit_joint_scoring(
            games,
            forecast_week=5,
            as_of=datetime(2023, 9, 1, tzinfo=UTC),
        )


def test_fit_accepts_clean_cut():
    fit = fit_joint_scoring(
        games=_toy_games(),
        forecast_week=5,
        as_of=datetime(2023, 9, 10, tzinfo=UTC),
    )
    assert fit.week == 5
    index = fit.team_index
    assert fit.offense_ppd[index["A"]] > fit.offense_ppd[index["B"]]
    from backend.model.preseason import PreseasonPrior

    prior = PreseasonPrior(
        2023,
        fit.as_of,
        fit,
        pd.DataFrame(
            {
                "team_abbr": ["B", "A"],
                "power_rating": [0, 0],
                "scoring_environment": [0, 0],
                "expected_drives": [22, 22],
                "power_rating_sd": [6, 2],
            }
        ),
        2.7,
    ).week1_fit()
    posterior = fit_joint_scoring(
        _toy_games(), 2, datetime(2023, 9, 10, tzinfo=UTC), strength_prior=prior
    )
    # Preserve covariance in points even if team catalog ordering differs.
    row = _toy_games().iloc[0]
    contrast = np.array([1, -1])
    before = contrast @ prior.score_design(row)
    after = contrast @ posterior.score_design(row)
    assert after @ posterior.parameter_covariance @ after <= (
        before @ prior.parameter_covariance @ before
    )


def test_calibration_uses_one_sd_contract_and_keeps_pricing_holdout_free(
    monkeypatch,
    tmp_path,
):
    from backend.model import calibration as c
    from backend.model.distributions import student_t_scale
    from backend.model.joint_scoring import JointScoringConfig
    from backend.model.preseason import PreseasonConfig
    from backend.model.projections import LayerConfig

    engine = JointScoringConfig()

    def replay(config, seasons, *args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "game_id": f"{season}-{i}",
                    "model_week": 1,
                    "engine_margin": 0.0,
                    "market_margin": 0.0,
                    "closing_spread": 0.0,
                    "actual_margin": actual,
                    "margin_sd": 10 * config.score_covariance_scale,
                    "total_sd": 12 * config.score_covariance_scale,
                    "rest_diff": 0.0,
                    **{f"qb_adj_{int(span)}": 0.0 for span in c.QB_SPANS},
                }
                for season in seasons
                for i, actual in enumerate([-10, -3, 0, 3, 10])
            ]
        )

    monkeypatch.setattr(c, "generate_walk_forward", replay)
    monkeypatch.setattr(c, "ENGINE_GRID", [engine])
    monkeypatch.setattr(c, "LAYER_GRID", [LayerConfig()])
    monkeypatch.setattr(c, "PRESEASON_GRID", [PreseasonConfig()])
    monkeypatch.setattr(c, "SD_GRID", [(0.85, 7)])
    from backend.model import artifacts

    monkeypatch.setattr(
        artifacts, "PRICING_PATH", tmp_path / "margin_distribution_v2.csv"
    )
    written = {}
    monkeypatch.setattr(
        c.store,
        "write_processed",
        lambda frame, *parts: written.update({parts[-1]: frame}),
    )
    summary = c.run_calibration(verbose=False, data=object())
    predictions = written["predictions.parquet"]
    assert predictions.margin_sd.eq(8.5).all()
    assert predictions.total_sd.eq(10.2).all()
    holdout = predictions[predictions.season.isin(c.HOLDOUT_SEASONS)]
    assert summary["holdout_margin_log_loss"] == pytest.approx(
        c.margin_log_loss(holdout, 1, 7)
    )
    assert summary["holdout_coverage"] == c.coverage_report(holdout, 1, 7)
    # The marginal scale used by the season simulation and calibration is identical.
    from scipy.stats import t

    expected = -t.logpdf(holdout.actual_margin, 7, scale=student_t_scale(8.5, 7)).mean()
    assert summary["holdout_margin_log_loss"] == pytest.approx(expected)
    artifact = pd.read_csv(tmp_path / "margin_distribution_v2.csv")
    development = predictions[predictions.season.isin(c.DEVELOPMENT_SEASONS)]
    assert artifact.weight.to_numpy() == pytest.approx(
        c.fit_margin_distribution(development, 7)
    )


def test_source_receipts_preserve_bytes_and_refuse_later_revisions(
    tmp_path, monkeypatch
):
    """A receipt timestamp cannot be substituted with a source's own date."""
    import json

    import pandas as pd

    from backend import source_inputs

    monkeypatch.setattr(source_inputs, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(source_inputs, "ARCHIVE", tmp_path / "archive")
    path = tmp_path / "depth.parquet"
    path.write_bytes(b"original expected QB")
    assert source_inputs.receipt_for(path) is None
    first = source_inputs.archive_source(path)
    cutoff = pd.Timestamp.now(tz="UTC")
    assert source_inputs.archive_source(path) == first
    path.write_bytes(b"revised expected QB")
    assert source_inputs.receipt_for(path) is None
    second = source_inputs.archive_source(path)
    assert second["observed_at"] > first["observed_at"]
    assert (tmp_path / first["archive"]).read_bytes() == b"original expected QB"
    (tmp_path / second["archive"]).write_bytes(b"corrupted archive")
    assert source_inputs.receipt_for(path) is None
    repaired = source_inputs.archive_source(path)
    assert repaired["observed_at"] > second["observed_at"]
    assert (tmp_path / repaired["archive"]).read_bytes() == path.read_bytes()
    sources = {
        name: first
        for name in (
            "schedule",
            "depth_charts",
            "qb_overrides",
            "win_totals",
            "win_total_sources",
            "preseason_qbs",
        )
    }
    assert source_inputs.source_reason(json.dumps(sources), cutoff, cutoff) is None
    sources["win_total_sources"] = second
    assert (
        source_inputs.source_reason(
            json.dumps(sources), cutoff, pd.Timestamp.now(tz="UTC")
        )
        == "source_after_forecast"
    )


def test_recommendation_qb_freshness_uses_the_selected_player(tmp_path, monkeypatch):
    """A fresh empty QB row must not clear an older selected starter."""
    import json

    from backend import source_inputs
    from backend.model.market_blend import MARGINS
    from backend.recommendations import build_recommendations

    now = pd.Timestamp.now(tz="UTC")
    monkeypatch.setattr(source_inputs, "RAW_DIR", tmp_path)
    monkeypatch.setattr(source_inputs, "receipt_for", lambda path: None)
    monkeypatch.setattr(
        source_inputs,
        "_overrides",
        lambda: pd.DataFrame(columns=["season", "week", "team_abbr", "gsis_id"]),
    )
    (tmp_path / "depth_charts").mkdir()
    pd.DataFrame(
        [
            dict(team=team, dt=stamp, pos_abb="QB", pos_rank=1, gsis_id=player)
            for team in ("H", "A")
            for stamp, player in (
                (now - timedelta(days=4), "old-qb"),
                (now - timedelta(minutes=10), None),
            )
        ]
    ).to_parquet(tmp_path / "depth_charts" / "2026.parquet")
    frame = pd.DataFrame(
        [
            dict(
                game_id="fresh-empty-chart",
                season=2026,
                week=1,
                as_of=now,
                start_date=now + timedelta(days=1),
                model_version="test",
                home_team_abbr="H",
                away_team_abbr="A",
                home_team="Home",
                away_team="Away",
                pure_home_margin=8.0,
                model_total=45.0,
                margin_sd=10.0,
                total_sd=10.0,
                degrees_of_freedom=7.0,
            )
        ]
    )
    attached = source_inputs.attach_sources(frame)
    assert attached.home_missing_input_count.iloc[0] == 1
    flags = json.loads(attached.data_flags.iloc[0])
    assert pd.Timestamp(flags["home_qb_source_at"]) == now - timedelta(days=4)
    decisions = build_recommendations(
        attached, pd.DataFrame(), np.ones(len(MARGINS)), decision_at=now
    )
    assert decisions.status.eq("no_play").all()
    assert decisions.reason.eq("missing_model_inputs").all()
