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
    # Historical fits without a preseason anchor and the synthetic opening
    # forecast retain their original scoring environment.
    assert fit.totals_base_ppd is None
    assert fit.totals_base_drives is None
    assert prior.totals_base_ppd is None
    assert prior.totals_base_drives is None
    assert prior.base_ppd == fit.base_ppd
    assert prior.base_drives == fit.base_drives


@pytest.mark.parametrize("forecast_week", [3, 20])
def test_totals_prior_uses_only_earlier_model_weeks(forecast_week):
    """The same chronology contract applies after regular-season week 18."""
    from dataclasses import replace

    from backend.model.joint_scoring import DEFAULT_CONFIG

    games = _toy_games()
    previous_season = games.assign(
        season=2022, start_date=datetime(2022, 9, 1, tzinfo=UTC)
    )
    preseason = replace(
        fit_joint_scoring(previous_season, 6, datetime(2023, 8, 31, tzinfo=UTC)),
        season=2023,
        week=1,
        base_ppd=1.0,
        base_drives=20.0,
    )
    games["model_week"] = np.arange(forecast_week - 2, forecast_week + 3)
    games["start_date"] = [
        datetime(2023, 9, 1, tzinfo=UTC) + timedelta(weeks=int(week) - 1)
        for week in games.model_week
    ]
    # Vary the earlier outcomes so weighting by observations, team-rows, or
    # calendar week instead of the configured recency changes the answer.
    games["home_points"] = [21.0, 35.0, 14.0, 28.0, 10.0]
    games["game_drives"] = [20.0, 24.0, 22.0, 22.0, 22.0]
    config = replace(DEFAULT_CONFIG, rating_half_life_weeks=3.0)
    cutoff = datetime(2023, 9, 1, tzinfo=UTC) + timedelta(weeks=forecast_week - 1)
    fixed = fit_joint_scoring(
        games, forecast_week, cutoff, config, totals_prior=preseason
    )
    effective_games = 1.0 + 0.5 ** (1 / 3)
    prior_games = 64 * 0.5 ** ((forecast_week - 1) / 6)
    assert fixed.totals_base_ppd == pytest.approx(
        (effective_games * fixed.base_ppd + prior_games * preseason.base_ppd)
        / (effective_games + prior_games)
    )
    assert fixed.totals_base_drives == pytest.approx(
        (effective_games * fixed.base_drives + prior_games * preseason.base_drives)
        / (effective_games + prior_games)
    )
    poisoned = games.copy()
    outcome_fields = [
        "home_points",
        "away_points",
        "game_drives",
        "competitive_drives",
        "home_competitive_points",
        "away_competitive_points",
        "home_epa_per_drive",
        "away_epa_per_drive",
    ]
    poisoned.loc[poisoned.model_week.ge(forecast_week), outcome_fields] = 1000.0
    replay = fit_joint_scoring(
        poisoned, forecast_week, cutoff, config, totals_prior=preseason
    )
    assert replay.totals_base_ppd == fixed.totals_base_ppd
    assert replay.totals_base_drives == fixed.totals_base_drives
    assert replay.engine_projection(games.iloc[-1]) == fixed.engine_projection(
        games.iloc[-1]
    )
    # An earlier week label cannot make an outcome at the cutoff available.
    poisoned.loc[0, "start_date"] = cutoff
    with pytest.raises(ValueError, match="training games must start before as_of"):
        fit_joint_scoring(
            poisoned, forecast_week, cutoff, config, totals_prior=preseason
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
                    "engine_total": 44.0,
                    "actual_total": 44.0 + actual,
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
    totals = summary["holdout_engine_totals"]
    assert totals["forecast_basis"] == "engine_only_before_qb_and_market"
    assert totals["mae"] == pytest.approx(5.2)
    assert totals["log_loss"] == pytest.approx(
        -t.logpdf(holdout.actual_total - 44, 7, scale=student_t_scale(10.2, 7)).mean()
    )
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
                home_margin=8.0,
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
