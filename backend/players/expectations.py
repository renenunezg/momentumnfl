"""Situation-based expectations fitted on earlier seasons.

A player's rushing yards, conversions and field goals are judged against what
an average play in the same situation produced. The tables are fit only on
seasons before the one being rated, so a season never sets its own baseline,
and the serving tables are frozen beside the grade model.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DOWN_KEYS = ["down_bucket", "distance_bucket", "field_bucket"]
FG_KEYS = ["fg_bucket"]
TABLE_KEYS = {"rush_yards": DOWN_KEYS, "conversion": DOWN_KEYS, "field_goal": FG_KEYS}
INTEGER_KEYS = ("down_bucket", "fg_bucket")
SOURCE_SEASONS = b"source_seasons"


@dataclass
class ExpectationTables:
    rush_yards: pd.DataFrame
    conversion: pd.DataFrame
    field_goal: pd.DataFrame
    source_seasons: list[int]


def bucket(plays: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``plays`` with the down, distance, field and field goal buckets.

    Each group is added only when its source columns are present.
    """
    out = plays.copy()
    if {"down", "ydstogo", "yardline_100"} <= set(out.columns):
        down = pd.to_numeric(out["down"], errors="coerce")
        distance = pd.to_numeric(out["ydstogo"], errors="coerce")
        yards_to_goal = pd.to_numeric(out["yardline_100"], errors="coerce")
        out["down_bucket"] = down.where(down.isin([1, 2, 3, 4])).astype("Int64")
        out["distance_bucket"] = np.select(
            [distance.le(2), distance.le(6), distance.le(10), distance.gt(10)],
            ["1-2", "3-6", "7-10", "11+"],
            default=None,
        )
        out["field_bucket"] = np.select(
            [
                yards_to_goal.ge(80),
                yards_to_goal.ge(40),
                yards_to_goal.ge(20),
                yards_to_goal.lt(20),
            ],
            ["own", "mid", "plus", "red"],
            default=None,
        )
    if "kick_distance" in out:
        distance = pd.to_numeric(out["kick_distance"], errors="coerce")
        out["fg_bucket"] = (5 * (distance // 5)).astype("Int64")
    return out


def _table(frame: pd.DataFrame, keys: list[str], value: pd.Series) -> pd.DataFrame:
    grouped = value.groupby([frame[key] for key in keys], observed=True)
    table = grouped.agg(expected="mean", plays="size").reset_index()
    table.columns = [*keys, "expected", "plays"]
    return table


def fit_expectation_tables(
    plays_by_season: dict[int, pd.DataFrame], before_season: int
) -> ExpectationTables:
    """Tables from every season in ``plays_by_season`` before ``before_season``."""
    seasons = sorted(season for season in plays_by_season if season < before_season)
    if not seasons:
        raise ValueError(f"no season before {before_season} to fit expectations on")
    plays = bucket(
        pd.concat([plays_by_season[season] for season in seasons], ignore_index=True)
    )
    competitive = plays["is_competitive"].astype(bool)
    rushes = plays[
        plays["rush_attempt"].eq(1) & competitive & plays["yards_gained"].notna()
    ]
    scrimmage = plays[plays["play_type"].isin(["pass", "run"]) & competitive]
    kicks = plays[plays["play_type"].eq("field_goal") & plays["fg_bucket"].notna()]
    kicks = kicks[kicks["field_goal_result"].isin(["made", "missed", "blocked"])]
    return ExpectationTables(
        rush_yards=_table(rushes, DOWN_KEYS, rushes["yards_gained"].astype(float)),
        conversion=_table(scrimmage, DOWN_KEYS, scrimmage["is_success"].astype(float)),
        field_goal=_table(
            kicks, FG_KEYS, kicks["field_goal_result"].eq("made").astype(float)
        ),
        source_seasons=seasons,
    )


def _lookup(table: pd.DataFrame, plays: pd.DataFrame, keys: list[str]) -> pd.Series:
    """The table's mean for each play's bucket; the table's global mean where unseen."""
    if not set(keys) <= set(plays.columns):
        plays = bucket(plays)
    merged = (
        plays[keys]
        .reset_index(drop=True)
        .merge(table[[*keys, "expected"]], on=keys, how="left")
    )
    mean = (
        (table["expected"] * table["plays"]).sum() / table["plays"].sum()
        if len(table)
        else np.nan
    )
    return pd.Series(merged["expected"].fillna(mean).to_numpy(float), index=plays.index)


def expected_rush_yards(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.rush_yards, plays, DOWN_KEYS)


def expected_conversion(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.conversion, plays, DOWN_KEYS)


def expected_field_goal(tables: ExpectationTables, plays: pd.DataFrame) -> pd.Series:
    return _lookup(tables.field_goal, plays, FG_KEYS)


# The frozen serving artifact


def expectation_tables_path(models_dir: Path, cutoff_season: int) -> Path:
    """The frozen tables saved beside ``position_grade_<cutoff>.json``."""
    return Path(models_dir) / f"position_grade_expectations_{cutoff_season}.parquet"


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column in INTEGER_KEYS:
            out[column] = pd.to_numeric(out[column]).astype("Int64")
        elif column in DOWN_KEYS:
            out[column] = (
                out[column]
                .map(lambda value: None if pd.isna(value) else str(value))
                .astype(object)
            )
    return out


def tables_frame(tables: ExpectationTables) -> pd.DataFrame:
    keys = list(dict.fromkeys(key for names in TABLE_KEYS.values() for key in names))
    parts = [getattr(tables, name).assign(table=name) for name in TABLE_KEYS]
    frame = pd.concat(parts, ignore_index=True).reindex(
        columns=["table", *keys, "expected", "plays"]
    )
    return _typed(frame).astype({"expected": float, "plays": "int64"})


def save_expectation_tables(tables: ExpectationTables, path: Path) -> Path:
    table = pa.Table.from_pandas(tables_frame(tables), preserve_index=False)
    metadata = {
        **(table.schema.metadata or {}),
        SOURCE_SEASONS: json.dumps(
            [int(season) for season in tables.source_seasons]
        ).encode(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), temporary)
    temporary.replace(path)
    return path


def load_expectation_tables(source) -> ExpectationTables:
    table = pq.read_table(source)
    metadata = table.schema.metadata or {}
    if SOURCE_SEASONS not in metadata:
        raise ValueError("expectation tables have no source_seasons metadata")
    frame = _typed(table.to_pandas())
    parts = {
        name: frame.loc[
            frame["table"].eq(name), [*keys, "expected", "plays"]
        ].reset_index(drop=True)
        for name, keys in TABLE_KEYS.items()
    }
    return ExpectationTables(
        **parts,
        source_seasons=[int(season) for season in json.loads(metadata[SOURCE_SEASONS])],
    )
