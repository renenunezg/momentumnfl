"""OverTheCap contracts as one row per player and season, in millions.

Each player keeps his latest contract row, whose season history covers his
career, and the total value of the contract he signed in his draft year. The
acquiring team of a trade pays cash only: signing bonus money already paid stays
with the trading team.
"""

import numpy as np
import pandas as pd

from backend.config import RAW_DIR

CONTRACTS_FILE = "contracts.parquet"
CONTRACT_SEASON_COLUMNS = [
    "gsis_id",
    "position",
    "is_active",
    "draft_year",
    "draft_overall",
    "season",
    "base_salary",
    "guaranteed_salary",
    "cap_number",
    "cash_paid",
    "rookie_value",
]
MONEY_COLUMNS = ["base_salary", "guaranteed_salary", "cap_number", "cash_paid"]
PAY_WEEKS = 18
# League-announced salary cap.
SALARY_CAP_MUSD = {
    2016: 155.27,
    2017: 167.0,
    2018: 177.2,
    2019: 188.2,
    2020: 198.2,
    2021: 182.5,
    2022: 208.2,
    2023: 224.8,
    2024: 255.4,
    2025: 279.2,
    2026: 301.2,
}
# CBA rookie minimum salary; later seasons use the latest.
LEAGUE_MINIMUM_MUSD = {
    2016: 0.450,
    2017: 0.465,
    2018: 0.480,
    2019: 0.495,
    2020: 0.610,
    2021: 0.660,
    2022: 0.705,
    2023: 0.750,
    2024: 0.795,
    2025: 0.840,
    2026: 0.885,
    2027: 0.930,
    2028: 0.975,
    2029: 1.020,
    2030: 1.065,
}


def league_minimum(seasons) -> np.ndarray:
    last = max(LEAGUE_MINIMUM_MUSD)
    return np.array([LEAGUE_MINIMUM_MUSD[min(int(s), last)] for s in np.ravel(seasons)])


def contract_seasons(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=["gsis_id"])
    rookie_value = (
        raw[raw["year_signed"].eq(raw["draft_year"])]
        .sort_values("year_signed")
        .drop_duplicates("gsis_id")
        .set_index("gsis_id")["value"]
        .astype(float)
    )
    latest = raw.sort_values(["is_active", "year_signed"]).drop_duplicates(
        "gsis_id", keep="last"
    )
    rows = (
        latest[
            [
                "gsis_id",
                "position",
                "is_active",
                "draft_year",
                "draft_overall",
                "season_history",
            ]
        ]
        .explode("season_history")
        .dropna(subset=["season_history"])
        .reset_index(drop=True)
    )
    history = pd.json_normalize(rows.pop("season_history").tolist())
    out = rows.join(history[["year", *MONEY_COLUMNS]])
    out = out[out["year"].notna() & out["year"].ne("Total")]
    out["season"] = out["year"].astype(int)
    out[MONEY_COLUMNS] = out[MONEY_COLUMNS].astype(float).fillna(0.0)
    out["rookie_value"] = out["gsis_id"].map(rookie_value)
    return out[CONTRACT_SEASON_COLUMNS].reset_index(drop=True)


def read_contract_seasons() -> pd.DataFrame:
    path = RAW_DIR / CONTRACTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run player-sources")
    return pd.read_parquet(path)


def season_caps(through: int) -> pd.Series:
    """Salary cap in millions by season through ``through``, grown past the last
    known season at the known seasons' geometric mean growth."""
    caps = pd.Series(SALARY_CAP_MUSD, dtype=float)
    last = caps.index[-1]
    growth = (caps.iloc[-1] / caps.iloc[0]) ** (1 / (len(caps) - 1))
    future = [
        caps.iloc[-1] * growth ** (s - last) for s in range(last + 1, through + 1)
    ]
    return pd.concat([caps, pd.Series(future, index=range(last + 1, through + 1))])


def acquirer_schedule(
    rows: pd.DataFrame, season: int, pay_weeks_left: int
) -> pd.DataFrame:
    """``gsis_id, season, cash, guaranteed`` an acquiring team owes from
    ``season`` on: the current season's base salary pro rata, then each later
    season's cash, of which the guaranteed salary is owed even after a cut."""
    rows = rows[rows["is_active"] & rows["season"].ge(season)]
    current = rows["season"].eq(season)
    share = pay_weeks_left / PAY_WEEKS
    cash = np.where(current, rows["base_salary"] * share, rows["cash_paid"])
    guaranteed = np.where(
        current, rows["guaranteed_salary"] * share, rows["guaranteed_salary"]
    )
    out = pd.DataFrame(
        {
            "gsis_id": rows["gsis_id"].to_numpy(),
            "season": rows["season"].to_numpy(),
            "cash": cash,
            "guaranteed": np.minimum(guaranteed, cash),
        }
    )
    return out[out["cash"].gt(0)].reset_index(drop=True)
