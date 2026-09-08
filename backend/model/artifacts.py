"""Reproducible production artifacts from the approved, fixed model settings."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd

from backend.config import (
    DEVELOPMENT_SEASONS,
    HISTORY_START_SEASON,
    HOLDOUT_SEASONS,
    PROCESSED_DIR,
    REPO_ROOT,
    STATIC_DIR,
)

PRICING_PATH = PROCESSED_DIR / "calibration" / "margin_distribution_v2.csv"
MANIFEST_PATH = PROCESSED_DIR / "calibration" / "production_manifest.json"


def fingerprint() -> str:
    paths = [REPO_ROOT / "pyproject.toml", REPO_ROOT / "backend/config.py"]
    for directory in ("backend/model", "backend/features", "backend/etl"):
        paths.extend(sorted((REPO_ROOT / directory).glob("*.py")))
    paths.extend(sorted(STATIC_DIR.glob("win_total*")))
    paths.append(STATIC_DIR / "preseason_qbs.csv")
    for kind in ("team_games", "qb_games"):
        paths.extend(
            PROCESSED_DIR / kind / f"{season}.parquet"
            for season in range(HISTORY_START_SEASON, max(HOLDOUT_SEASONS) + 1)
        )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure(force: bool = False) -> dict:
    from backend.model.calibration import (
        EVAL_SEASONS,
        WalkForwardData,
        apply_layers,
        finish_calibration,
        generate_walk_forward,
    )
    from backend.model.joint_scoring import DEFAULT_CONFIG
    from backend.model.preseason import PreseasonConfig
    from backend.model.projections import LayerConfig

    source_hash = fingerprint()
    paths = [PRICING_PATH, PROCESSED_DIR / "calibration/predictions.parquet"]
    if not force and MANIFEST_PATH.exists() and all(p.exists() for p in paths):
        manifest = json.loads(MANIFEST_PATH.read_text())
        if manifest.get("source_hash") == source_hash and all(
            manifest.get("artifacts", {}).get(p.name)
            == hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        ):
            return manifest
    data = WalkForwardData(EVAL_SEASONS)
    base = replace(DEFAULT_CONFIG, score_covariance_scale=1.0)
    layer = LayerConfig()
    predictions = apply_layers(
        generate_walk_forward(base, EVAL_SEASONS, data=data), layer
    )
    development = predictions[predictions.season.isin(DEVELOPMENT_SEASONS)]
    validation = predictions[~predictions.season.isin(DEVELOPMENT_SEASONS)]
    summary = finish_calibration(
        development,
        validation,
        DEFAULT_CONFIG,
        layer,
        PreseasonConfig(),
        True,
        [],
        True,
    )
    manifest = {
        "source_hash": source_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "training_seasons": list(DEVELOPMENT_SEASONS),
        "fixed_model_config": True,
        "games": len(predictions),
        "validation_basis": summary["validation_basis"],
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_pricing():
    import numpy as np

    from backend.model.market_blend import MARGINS

    if not MANIFEST_PATH.exists():
        raise ValueError("Run production-artifacts before pricing markets")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (
        manifest.get("source_hash") != fingerprint()
        or manifest.get("artifacts", {}).get(PRICING_PATH.name)
        != hashlib.sha256(PRICING_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError("Pricing artifact is stale; run production-artifacts")
    frame = pd.read_csv(PRICING_PATH)
    if (
        not np.array_equal(frame.actual_margin, MARGINS)
        or not np.isfinite(frame.weight).all()
        or not frame.weight.gt(0).all()
    ):
        raise ValueError("Invalid actual-margin pricing artifact")
    return frame.weight.to_numpy()
