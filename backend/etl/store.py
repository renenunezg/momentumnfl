import os

import pandas as pd
import pyarrow.parquet

from backend.config import PROCESSED_DIR, RAW_DIR


def write_parquet(df: pd.DataFrame, path) -> None:
    """Atomic parquet write: tmp file then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_raw(*parts: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR.joinpath(*parts))


def write_processed(df: pd.DataFrame, *parts: str) -> None:
    write_parquet(df, PROCESSED_DIR.joinpath(*parts))


def read_processed(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR.joinpath(*parts), columns=columns)


def processed_names(*parts: str) -> list[str]:
    """Parquet file stems stored under a processed directory, if it exists."""
    directory = PROCESSED_DIR.joinpath(*parts)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def processed_columns(*parts: str) -> list[str]:
    """Column names of a processed artifact, from parquet metadata only."""
    return list(
        pyarrow.parquet.read_schema(PROCESSED_DIR.joinpath(*parts)).names
    )
