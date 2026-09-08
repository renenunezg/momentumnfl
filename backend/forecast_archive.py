"""Immutable, self-contained prospective forecast inputs and offline replay."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backend.config import HISTORY_START_SEASON, PROCESSED_DIR, REPO_ROOT
from backend.source_inputs import receipt_for

ARCHIVE = PROCESSED_DIR / "forecast_archive"
REPLAY_CUTOFF = "MOMENTUMNFL_REPLAY_CUTOFF"
PACKAGES = ("numpy", "pandas", "scipy", "pyarrow")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    """Capture exact available bytes before assigning the forecast cutoff."""
    if hasattr(args, "forecast_cutoff"):
        return
    if os.getenv(REPLAY_CUTOFF):
        args.forecast_cutoff = datetime.fromisoformat(os.environ[REPLAY_CUTOFF])
        args.forecast_archive = None
        return
    paths = list((REPO_ROOT / "backend").rglob("*.py"))
    paths += list((REPO_ROOT / "backend/data_static").glob("*"))
    paths += [REPO_ROOT / "pyproject.toml", REPO_ROOT / "poetry.lock"]
    paths += [REPO_ROOT / "overrides/qb_starters.csv"]
    paths += [
        REPO_ROOT / "backend/data/raw" / name
        for name in (
            "schedules.parquet",
            "teams.parquet",
            f"depth_charts/{args.season}.parquet",
        )
    ]
    for directory in ("team_games", "qb_games"):
        paths += [
            PROCESSED_DIR / directory / f"{season}.parquet"
            for season in range(HISTORY_START_SEASON, args.season + 1)
        ]
    files = {}
    for path in sorted(set(paths)):
        name = str(path.relative_to(REPO_ROOT))
        if not path.is_file():
            files[name] = {
                "present": False,
                "captured_at": datetime.now(UTC).isoformat(),
            }
            continue
        content = path.read_bytes()
        digest = _digest(content)
        target = ARCHIVE / "objects" / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as output:
                output.write(content)
        receipt = receipt_for(path)
        files[name] = {
            "present": True,
            "sha256": digest,
            "captured_at": datetime.now(UTC).isoformat(),
            "source_receipt": receipt,
        }
    args.forecast_cutoff = datetime.now(UTC)
    args.forecast_archive = {
        "format": 1,
        "season": args.season,
        "week": getattr(args, "week", 1),
        "cutoff": args.forecast_cutoff.isoformat(),
        "files": files,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def finish(args, forecasts):
    if args.forecast_archive is None:
        return
    manifest = args.forecast_archive.copy()
    for name, entry in manifest["files"].items():
        path = REPO_ROOT / name
        if path.is_file() != entry["present"] or (
            entry["present"] and _digest(path.read_bytes()) != entry["sha256"]
        ):
            raise ValueError(f"Forecast inputs changed during computation: {name}")
    manifest["week"] = int(forecasts.iloc[0].week)
    manifest["forecasts"] = json.loads(
        forecasts.to_json(orient="records", double_precision=15)
    )
    stamp = args.forecast_cutoff.strftime("%Y%m%dT%H%M%S%fZ")
    path = ARCHIVE / "runs" / f"{args.season}_{manifest['week']:02d}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(manifest, output, sort_keys=True)
    print(f"forecast archive: {path}")


def replay(path: Path):
    """Restore only archived bytes in a disposable workspace; never use live data."""
    manifest = json.loads(path.read_text())
    cutoff = pd.Timestamp(manifest["cutoff"])
    if cutoff.tzinfo is None or manifest["format"] != 1:
        raise ValueError("Invalid forecast archive cutoff or format")
    for package, version in manifest["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Replay requires {package}=={version}")
    season, week = manifest["season"], manifest["week"]
    for source in ("schedules.parquet", f"depth_charts/{season}.parquet"):
        entry = manifest["files"].get(f"backend/data/raw/{source}", {})
        receipt = entry.get("source_receipt") or {}
        observed = pd.to_datetime(receipt.get("observed_at"), utc=True)
        if pd.isna(observed) or observed > cutoff or not entry.get("present"):
            raise ValueError(f"Missing pre-cutoff source coverage: {source}")
        if receipt.get("sha256") != entry["sha256"]:
            raise ValueError(f"Source receipt mismatch: {source}")
    root = path.parent.parent
    with tempfile.TemporaryDirectory(prefix="nfl-replay-") as temporary:
        workspace = Path(temporary).resolve()
        for name, entry in manifest["files"].items():
            target = (workspace / name).resolve()
            if not target.is_relative_to(workspace) or name.startswith("/"):
                raise ValueError("Unsafe archive path")
            if pd.Timestamp(entry["captured_at"]) > cutoff:
                raise ValueError(f"Input observed after forecast cutoff: {name}")
            if not entry["present"]:
                continue
            content = (root / "objects" / entry["sha256"]).read_bytes()
            if _digest(content) != entry["sha256"]:
                raise ValueError(f"Corrupt archived input: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        from backend.features.qb import _depth_chart_qb1

        depth = pd.read_parquet(
            workspace / f"backend/data/raw/depth_charts/{season}.parquet"
        )
        if not {"dt", "pos_abb"}.issubset(depth.columns):
            raise ValueError(
                "Replay requires timestamped QB charts, not weekly proxies"
            )
        qb1 = _depth_chart_qb1(depth, season, week, cutoff)
        starters = {} if qb1 is None else qb1.to_dict()
        override_path = workspace / "overrides/qb_starters.csv"
        if override_path.exists():
            overrides = pd.read_csv(override_path)
            overrides = overrides[overrides.season.eq(season) & overrides.week.eq(week)]
            starters.update(dict(zip(overrides.team_abbr, overrides.gsis_id)))
        for row in manifest["forecasts"]:
            for side in ("home", "away"):
                if pd.isna(starters.get(row[f"{side}_team_abbr"])):
                    raise ValueError(
                        "Missing archived expected QB; proxy replay refused"
                    )
        # Block network access even if a future model accidentally adds a fetch.
        (workspace / "sitecustomize.py").write_text(
            "import socket\n"
            "def denied(*a, **k): raise RuntimeError('Replay forbids network')\n"
            "socket.socket.connect = denied\n"
            "socket.create_connection = denied\n"
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "LANG", "TMPDIR"}
        }
        env.update(
            {
                REPLAY_CUTOFF: str(cutoff),
                "PYTHONPATH": str(workspace),
                "MOMENTUMNFL_DB_WRITES": "0",
                "GITHUB_ACTIONS": "false",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "backend",
                "fit",
                "--season",
                str(season),
                "--week",
                str(week),
                "--projections-only",
            ],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            raise ValueError(f"Offline replay failed: {result.stderr[-2000:]}")
        actual = (
            pd.read_parquet(
                workspace
                / f"backend/data/processed/projections/{season}_{week:02d}.parquet"
            )
            .set_index("game_id")
            .sort_index()
        )
        expected = pd.DataFrame(manifest["forecasts"]).set_index("game_id").sort_index()
        pd.testing.assert_frame_equal(
            actual[expected.columns],
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-8,
        )
    return {
        "season": season,
        "week": week,
        "cutoff": str(cutoff),
        "games_reproduced": len(expected),
        "status": "matched",
    }


def publication_bundle(season, week, forecasts):
    cutoff = pd.Timestamp(forecasts.iloc[0].as_of)
    stamp = cutoff.strftime("%Y%m%dT%H%M%S%fZ")
    path = ARCHIVE / "runs" / f"{season}_{week:02d}_{stamp}.json"
    if not path.exists():
        raise ValueError(
            "Missing frozen input bundle; regenerate forecasts before publish"
        )
    manifest = json.loads(path.read_text())
    expected = pd.DataFrame(manifest["forecasts"]).set_index("game_id").sort_index()
    actual = forecasts.set_index("game_id").sort_index()
    pd.testing.assert_frame_equal(
        actual[expected.columns],
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-8,
    )
    return manifest


def persist(connection, manifest):
    """Archive in the same transaction as publication; no artifact expiry."""
    from sqlalchemy import text

    digests = sorted(
        {entry["sha256"] for entry in manifest["files"].values() if entry["present"]}
    )
    existing = set(
        connection.execute(
            text(
                "select sha256 from nfl.forecast_input_objects where sha256 = any(:ids)"
            ),
            {"ids": digests},
        ).scalars()
    )
    for digest in digests:
        if digest in existing:
            continue
        content = (ARCHIVE / "objects" / digest).read_bytes()
        if _digest(content) != digest:
            raise ValueError("Corrupt input archive; publication refused")
        connection.execute(
            text(
                "insert into nfl.forecast_input_objects(sha256,compressed_content) "
                "values (:digest,:content) on conflict do nothing"
            ),
            {"digest": digest, "content": zlib.compress(content)},
        )
    serialized = json.dumps(manifest, sort_keys=True)
    connection.execute(
        text(
            "insert into nfl.forecast_input_runs "
            "(run_id,season,week,forecast_as_of,manifest) "
            "values (:id,:season,:week,:cutoff,cast(:manifest as jsonb)) "
            "on conflict do nothing"
        ),
        {
            "id": _digest(serialized.encode()),
            "season": manifest["season"],
            "week": manifest["week"],
            "cutoff": manifest["cutoff"],
            "manifest": serialized,
        },
    )


def download(run_id, destination):
    """Export a durable DB bundle; database access is explicitly read-only."""
    from sqlalchemy import text

    from backend.db import engine

    destination = Path(destination)
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        manifest = connection.execute(
            text("select manifest from nfl.forecast_input_runs where run_id=:id"),
            {"id": run_id},
        ).scalar_one()
        for entry in manifest["files"].values():
            if not entry["present"]:
                continue
            digest = entry["sha256"]
            path = destination / "objects" / digest
            if path.exists() and _digest(path.read_bytes()) == digest:
                continue
            content = connection.execute(
                text(
                    "select compressed_content from nfl.forecast_input_objects "
                    "where sha256=:digest"
                ),
                {"digest": digest},
            ).scalar_one()
            raw = zlib.decompress(content)
            if _digest(raw) != digest:
                raise ValueError("Corrupt database input object")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(raw)
    path = destination / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(manifest, output, sort_keys=True)
    return path
