"""Versioned public inputs, separate from the team model's raw cache."""

import hashlib
import json
from datetime import UTC, datetime

import nflreadpy
import pandas as pd

from backend.config import RAW_DIR
from backend.etl.store import write_parquet
from backend.nflverse.data import normalize_teams


def ingest(seasons: list[int], refresh: bool = False) -> None:
    for season in sorted(set(seasons)):
        loaders = {
            "schedules": nflreadpy.load_schedules,
            "rosters": nflreadpy.load_rosters,
            "stats": nflreadpy.load_player_stats,
        }
        if season >= 2012:
            loaders["snaps"] = nflreadpy.load_snap_counts
        for name, loader in loaders.items():
            path = RAW_DIR / "awards" / name / f"{season}.parquet"
            manifest = path.with_suffix(".json")
            if path.exists() and manifest.exists() and not refresh:
                continue
            no_games = False
            if name in {"stats", "snaps"}:
                schedule = read("schedules", season)
                no_games = not (
                    schedule.game_type.eq("REG") & schedule.home_score.notna()
                ).any()
            frame = (
                pd.DataFrame()
                if no_games
                else normalize_teams(loader([season]).to_pandas())
            )
            if frame.empty and name != "snaps" and not no_games:
                raise ValueError(f"No {name} data for {season}")
            stamp = datetime.now(UTC).isoformat()
            write_parquet(frame, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = path.parent / "versions" / str(season) / f"{digest}.parquet"
            if not archive.exists():
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(path.read_bytes())
                archive.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "observed_at": stamp,
                            "sha256": digest,
                            "source": f"nflreadpy.{loader.__name__}",
                            "rows": len(frame),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            manifest.write_text(archive.with_suffix(".json").read_text())
            print(f"awards inputs {season} {name}: {len(frame)} rows", flush=True)


def read(name: str, season: int) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "awards" / name / f"{season}.parquet")


def provenance(season: int) -> dict:
    result = {}
    for name in ("stats", "rosters", "schedules", "snaps"):
        path = RAW_DIR / "awards" / name / f"{season}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    return result
