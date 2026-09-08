"""The publish column lists are the frontend contract; they must match the
checked-in DDL exactly, column for column."""

import re
from pathlib import Path

import pandas as pd

from backend import publish
from backend.awards.pipeline import BOARD_COLUMNS, META_COLUMNS
from backend.etl import store
from backend.grading import RESULT_COLUMNS

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
    "season_win_totals": publish.SEASON_WIN_TOTALS_COLUMNS,
    "game_results": RESULT_COLUMNS,
    "award_boards": BOARD_COLUMNS,
    "award_model_meta": META_COLUMNS,
}


def _ddl_columns(table: str) -> list[str]:
    match = re.search(rf"create table nfl\.{table} \((.*?)\);", DDL, re.DOTALL)
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


def test_published_names_and_model_inputs_use_current_franchises(monkeypatch):
    raw = pd.DataFrame(
        [
            ("LA", "Los Angeles Rams"),
            ("LAC", "Los Angeles Chargers"),
            ("LV", "Las Vegas Raiders"),
            ("LA", "St. Louis Rams"),
            ("LAC", "San Diego Chargers"),
            ("LV", "Oakland Raiders"),
        ],
        columns=["team_abbr", "team_name"],
    ).assign(team_color="#000000", team_color2="#ffffff", team_logo_espn="logo")
    monkeypatch.setattr(store, "read_raw", lambda *parts: raw)
    expected = {
        "LA": "Los Angeles Rams",
        "LAC": "Los Angeles Chargers",
        "LV": "Las Vegas Raiders",
    }
    assert store.team_names() == expected
    published = publish.build_teams_frame().set_index("team_abbr")["team"].to_dict()
    assert published == expected
