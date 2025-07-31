"""
Load NFL play‑by‑play data using nfl_data_py.

Usage (from backend/):
    poetry run python etl/load_pbp.py --season 2024 2023
"""

# ─── env / path bootstrap ────────────────────────────────
import sys
from pathlib import Path
from dotenv import load_dotenv
import math

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")
# ─────────────────────────────────────────────────────────

import argparse
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from utils.db import supabase
import json

def shortlist_and_upsert(df, supabase_client):
    keep = [
        "game_id", "play_id", "season", "week", "posteam", "defteam", "play_type",
        "qtr", "down", "ydstogo", "yardline_100", "shotgun", "no_huddle",
        "goal_to_go", "score_differential", "epa", "air_yards",
        "yards_after_catch", "rush_attempt", "pass_attempt", "sack",
        "interception", "touchdown", "penalty_yards", "success",
        "wp", "home_wp", "away_wp", "vegas_wp", "vegas_home_wp",
        "vegas_wpa", "vegas_home_wpa"
    ]
    sub = df.loc[:, [c for c in keep if c in df.columns]].copy()
    sub = sub.replace([np.inf, -np.inf], None)
    sub = sub.where(pd.notnull(sub), None)
    sub = sub.replace({np.nan: None})

    # normalize boolean-ish float columns to actual bool or None (prefer explicit 1/0 floats)
    bool_cols = [
        "shotgun", "no_huddle", "goal_to_go",
        "rush_attempt", "pass_attempt", "sack",
        "interception", "touchdown", "success",
    ]
    for col in bool_cols:
        if col in sub.columns:
            # map 1/0 or truthy floats/strings to bool, else None
            def to_bool_or_none(x):
                if x is None or x is pd.NA or (isinstance(x, float) and not math.isfinite(x)):
                    return None
                if isinstance(x, (float, int)):
                    if x == 1 or x == 1.0:
                        return True
                    if x == 0 or x == 0.0:
                        return False
                if isinstance(x, (str,)):
                    if x.lower() in ("true", "1"):
                        return True
                    if x.lower() in ("false", "0"):
                        return False
                return None
            sub[col] = sub[col].apply(to_bool_or_none)

    # coerce integer-like columns to plain Python int or None
    int_cols = [
        "qtr",
        "down",
        "ydstogo",
        "yardline_100",
        "season",
        "week",
        "score_differential",
        "air_yards",
        "yards_after_catch",
        "penalty_yards",
    ]
    for col in int_cols:
        if col in sub.columns:
            coerced = pd.to_numeric(sub[col], errors="coerce")
            def to_int_or_none(x):
                if pd.isna(x):
                    return None
                try:
                    return int(x)
                except Exception:
                    return None
            # direct apply the conversion, avoiding any string like "1.0" slipping through
            sub[col] = coerced.apply(to_int_or_none)

    def clean_record(rec):
        cleaned = {}
        for k, v in rec.items():
            # pandas NA or numpy NaN / infinities become None
            if v is pd.NA:
                cleaned[k] = None
            elif isinstance(v, float):
                # If it's not a finite number, drop it.
                if not math.isfinite(v):
                    cleaned[k] = None
                # Cast float‐integers like 0.0 or -3.0 to real ints so Postgres smallint columns
                # don't receive `"0.0"` or `"-3.0"`.
                elif v.is_integer():
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = v
            elif isinstance(v, (np.integer,)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                cleaned[k] = bool(v)
            elif isinstance(v, str):
                # convert stringified integer-like "123.0" to int
                if v.endswith('.0') and v.replace('.', '', 1).isdigit():
                    try:
                        cleaned[k] = int(float(v))
                        continue
                    except Exception:
                        pass
                cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned
    records = [clean_record(r) for r in sub.to_dict("records")]

    chunk_size = 500

    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]

        # convert any residual pandas-specific types into plain Python
        chunk = json.loads(json.dumps(chunk, default=lambda o: None))

        # ensure any string like "123.0" in chunk is normalized to int
        for rec in chunk:
            for key, val in list(rec.items()):
                if isinstance(val, str) and val.endswith('.0') and val.replace('.', '', 1).isdigit():
                    try:
                        rec[key] = int(float(val))
                    except Exception:
                        pass

        # sanity check before sending
        for rec in chunk:
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    print("Bad value still present before upsert:", k, v)
                    break
                if isinstance(v, str) and v.replace('.', '', 1).isdigit() and v.endswith('.0'):
                    print("Bad stringified numeric still present before upsert:", k, v)
                    break

        resp = supabase_client.table("pbp").upsert(
            chunk, on_conflict="game_id,play_id"
        ).execute()

        if hasattr(resp, 'error') and resp.error:
            print(f"Upsert error chunk {i // chunk_size + 1}:", resp.error)
        else:
            print(f"Upserted chunk {i // chunk_size + 1} ({len(chunk)} records)")


def fetch_pbp(seasons: list[int]) -> pd.DataFrame:
    """
    Fetch play‑by‑play data for the given NFL seasons.

    Parameters
    ----------
    seasons : list[int]
        List of season years, e.g. [2024, 2023]

    Returns
    -------
    pandas.DataFrame
        Raw play‑by‑play records combined for all seasons.
    """
    df = nfl.import_pbp_data(seasons)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NFL play‑by‑play with nfl_data_py and inspect."
    )
    parser.add_argument(
        "--season",
        type=int,
        nargs="+",
        required=True,
        help="Season year(s) to pull, e.g. 2024 2023",
    )
    args = parser.parse_args()

    df = fetch_pbp(args.season)
    print(f"\nFetched {len(df):,} rows for seasons {args.season}.")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nSample rows:")
    print(df.head())
    shortlist_and_upsert(df, supabase)
    print("Upsert complete.")


if __name__ == "__main__":
    main()
