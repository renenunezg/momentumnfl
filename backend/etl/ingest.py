"""Season pulls into backend/data/raw parquet. All writes are atomic."""

import nflreadpy

from backend.config import RAW_DIR
from backend.etl.store import write_parquet as _write_parquet
from backend.nflverse import data
from backend.source_inputs import archive_source


def write_parquet(frame, path):
    _write_parquet(frame, path)
    if path.name == "schedules.parquet" or path.parent.name in (
        "depth_charts",
        "injuries",
    ):
        archive_source(path, refresh=True)


def _published(season: int, *, roster: bool = False) -> bool:
    """nflreadpy rejects seasons it has not opened yet; before the season
    starts that is expected, not a pipeline failure."""
    return season <= nflreadpy.get_current_season(roster=roster)


def ingest_season(season: int) -> list[str]:
    """Each source pulls independently so a season with no pbp yet (August)
    still gets depth charts and injuries. Returns per-source problems."""
    sources = [
        (
            "depth_charts",
            lambda: data.load_depth_charts([season]),
            RAW_DIR / "depth_charts",
        ),
    ]
    if not _published(season, roster=True):
        print(f"note: {season} depth_charts not yet published upstream")
        sources = []
    if _published(season):
        sources.extend(
            [
                ("pbp", lambda: data.load_pbp([season]), RAW_DIR / "pbp"),
                (
                    "injuries",
                    lambda: data.load_injuries([season]),
                    RAW_DIR / "injuries",
                ),
            ]
        )
    else:
        print(f"note: {season} game data not yet published upstream")
    if season >= 2018 and _published(season):
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
    # nflreadpy rejects NGS for a season until the Thursday after Labor Day,
    # so the Tuesday runs before kickoff pull only the seasons it accepts.
    published = [s for s in seasons if _published(s)]
    if published:
        write_parquet(
            data.load_nextgen_stats(published, "passing"),
            RAW_DIR / "ngs_passing.parquet",
        )
    else:
        print(f"note: {seasons[-1]} ngs_passing not yet published upstream")


def ingest_projection_inputs(season: int) -> list[str]:
    """Refresh only inputs that can change an unplayed game's projection."""
    write_parquet(data.load_schedules([season]), RAW_DIR / "schedules.parquet")
    write_parquet(data.load_teams(), RAW_DIR / "teams.parquet")
    sources = []
    if _published(season, roster=True):
        sources.append(("depth_charts", data.load_depth_charts))
    else:
        print(f"note: {season} depth_charts not yet published upstream")
    if _published(season):
        sources.append(("injuries", data.load_injuries))
    problems = []
    for name, loader in sources:
        try:
            write_parquet(loader([season]), RAW_DIR / name / f"{season}.parquet")
        except Exception as error:  # noqa: BLE001 - caller reports all sources
            problems.append(f"{season} {name}: {error}")
    return problems
