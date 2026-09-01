"""The publish column lists are the frontend contract; they must match the
checked-in DDL exactly, column for column."""

import re
from pathlib import Path

from backend import publish

DDL = "\n".join(
    path.read_text()
    for path in sorted((Path(__file__).parent.parent / "sql").glob("*.sql"))
)

CONTRACTS = {
    "teams": publish.TEAMS_COLUMNS,
    "team_ratings": publish.TEAM_RATINGS_COLUMNS,
    "team_unit_ratings": publish.TEAM_UNIT_RATINGS_COLUMNS,
    "game_projections": publish.GAME_PROJECTIONS_COLUMNS,
    "market_comparisons": publish.MARKET_COMPARISONS_COLUMNS,
    "backtest_predictions": publish.BACKTEST_COLUMNS,
    "market_snapshots": publish.MARKET_SNAPSHOTS_COLUMNS,
}


def _ddl_columns(table: str) -> list[str]:
    match = re.search(
        rf"create table nfl\.{table} \((.*?)\);", DDL, re.DOTALL
    )
    assert match, f"table {table} not in DDL"
    columns = []
    for line in match.group(1).splitlines():
        line = line.split("--")[0].strip().rstrip(",")
        if not line or line.startswith("primary key"):
            continue
        columns.append(line.split()[0])
    return columns


def test_publish_columns_match_ddl():
    for table, contract in CONTRACTS.items():
        assert list(contract) == _ddl_columns(table), table
