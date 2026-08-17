"""Season pulls into backend/data/raw parquet. All writes are atomic."""

from backend.config import RAW_DIR
from backend.etl.store import write_parquet
from backend.nflverse import data


def ingest_season(season: int) -> list[str]:
    """Each source pulls independently so a season with no pbp yet (August)
    still gets depth charts and injuries. Returns per-source problems."""
    sources = [
        ("pbp", lambda: data.load_pbp([season]), RAW_DIR / "pbp"),
        (
            "depth_charts",
            lambda: data.load_depth_charts([season]),
            RAW_DIR / "depth_charts",
        ),
        (
            "injuries",
            lambda: data.load_injuries([season]),
            RAW_DIR / "injuries",
        ),
    ]
    if season >= 2018:
        sources.append(
            (
                "pfr_pass",
                lambda: data.load_pfr_advstats([season], "pass"),
                RAW_DIR / "pfr_pass",
            )
        )
    problems = []
    for name, loader, directory in sources:
        try:
            write_parquet(loader(), directory / f"{season}.parquet")
        except Exception as error:  # noqa: BLE001 - report and continue
            problems.append(f"{season} {name}: {error}")
    return problems


def ingest_shared(seasons: list[int]) -> None:
    write_parquet(data.load_schedules(seasons), RAW_DIR / "schedules.parquet")
    write_parquet(data.load_teams(), RAW_DIR / "teams.parquet")
    write_parquet(
        data.load_nextgen_stats("passing"), RAW_DIR / "ngs_passing.parquet"
    )
