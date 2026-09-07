"""QB strengths and a shared forecast-time baseline for lineup substitutions."""

from dataclasses import dataclass, field
from math import isfinite

import pandas as pd

REPLACEMENT_EPA_PER_DROPBACK = -0.08
LEAGUE_DROPBACKS_PER_GAME = 34.0
SHRINK_DROPBACKS = 200.0
DEFAULT_SPAN_DROPBACKS = 1000.0
CALENDAR_HALF_LIFE_WEEKS = 52.0


def strength_history(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
    calendar_half_life: float = CALENDAR_HALF_LIFE_WEEKS,
    prior_dropbacks: float = SHRINK_DROPBACKS,
) -> pd.DataFrame:
    """QB strength after each game, in raw points above a fixed replacement.

    The fixed origin avoids using future league averages. A forecast must
    select only prior weeks through context_before_week before using it.
    """
    ordered = qb_games.merge(
        game_index[["game_id", "season", "model_week", "start_date"]], on="game_id"
    ).sort_values(["passer_player_id", "start_date"])
    if not all(isfinite(x) and x > 0 for x in (
        span_dropbacks, calendar_half_life, prior_dropbacks
    )):
        raise ValueError("QB memory and prior sizes must be finite and positive")
    epa_states, weight_states = [], []
    for _, group in ordered.groupby("passer_player_id", sort=False):
        epa = weight = 0.0
        previous_clock = None
        for row in group.itertuples():
            clock = row.season * 52 + row.model_week
            elapsed = 0 if previous_clock is None else clock - previous_clock
            decay = 0.5 ** (
                row.dropbacks / span_dropbacks + elapsed / calendar_half_life
            )
            epa = decay * epa + row.epa
            weight = decay * weight + row.dropbacks
            epa_states.append(epa)
            weight_states.append(weight)
            previous_clock = clock
    ordered["weighted_epa"] = epa_states
    ordered["effective_dropbacks"] = weight_states
    ordered["calendar_half_life"] = calendar_half_life
    ordered["prior_dropbacks"] = prior_dropbacks
    ordered["strength_points"] = (
        (
            ordered["weighted_epa"]
            - REPLACEMENT_EPA_PER_DROPBACK * ordered["effective_dropbacks"]
        )
        / (ordered["effective_dropbacks"] + prior_dropbacks)
        * LEAGUE_DROPBACKS_PER_GAME
    )
    return ordered.sort_values("start_date")


@dataclass(frozen=True)
class QbContext:
    strengths: dict[str, float]
    baselines: dict[str, float]
    effective_dropbacks: dict[str, float] = field(default_factory=dict)
    mixtures: dict[str, dict[str, float]] = field(default_factory=dict)

    def contrast(self, team: str, passer: str | None) -> dict[str, float]:
        """QB coefficients in a lineup change, for uncertainty calculations."""
        if passer is None or team not in self.mixtures:
            return {}
        coefficients = {q: -w for q, w in self.mixtures[team].items()}
        coefficients[passer] = coefficients.get(passer, 0.0) + 1.0
        return coefficients

    def adjustment(self, team: str, passer: str | None, weight: float = 1.0) -> float:
        """Only replace the QB contribution represented by the team's mix.

        An unseen QB starts at replacement strength. With no team history
        there is no defensible baseline to remove, so adjustment is neutral.
        """
        if passer is None or team not in self.baselines:
            return 0.0
        return weight * (self.strengths.get(passer, 0.0) - self.baselines[team])


def context_before_week(
    history: pd.DataFrame,
    season: int,
    week: int,
    half_life_weeks: float,
    preseason_references: dict[str, str] | None = None,
    win_total_weight: float = 0.0,
    strengths_override: dict[str, float] | None = None,
) -> QbContext:
    """Value both the starter and the team's QB mix at the same forecast cut.

    Use every passer's dropback share with the engine's time decay. Before
    the first played week, carry the previous season's mix with its actual
    weeks intact. The market-funded preseason share already represents its
    fixed reference QB; later lineup changes must not change that reference.
    """
    eligible = history[
        (history["season"] < season)
        | (history["season"].eq(season) & (history["model_week"] < week))
    ]
    latest = eligible.drop_duplicates("passer_player_id", keep="last").copy()
    age = (season - latest["season"]) * 52 + week - latest["model_week"]
    decay = 0.5 ** (age / latest["calendar_half_life"])
    effective = latest["effective_dropbacks"] * decay
    latest["effective_dropbacks"] = effective
    latest["strength_points"] = (
        (latest["weighted_epa"] * decay - REPLACEMENT_EPA_PER_DROPBACK * effective)
        / (effective + latest["prior_dropbacks"])
        * LEAGUE_DROPBACKS_PER_GAME
    )
    evidence = latest.set_index("passer_player_id")["effective_dropbacks"].to_dict()
    strengths = (
        latest.set_index("passer_player_id")["strength_points"].to_dict()
        if strengths_override is None else strengths_override
    )
    window = eligible[eligible["season"].eq(season)].copy()
    played_teams = set(window["team"])
    previous = eligible[
        eligible["season"].eq(season - 1) & ~eligible["team"].isin(played_teams)
    ]
    window = pd.concat([window, previous], ignore_index=True)
    if window.empty:
        return QbContext(strengths, {}, evidence)
    recency = 0.5 ** (
        (window.groupby("team")["model_week"].transform("max") - window["model_week"])
        / half_life_weeks
    )
    window["weight"] = window["dropbacks"] * recency
    window["weighted_strength"] = (
        window["passer_player_id"].map(strengths) * window["weight"]
    )
    totals = window.groupby("team")[["weight", "weighted_strength"]].sum()
    totals = totals[totals["weight"] > 0]
    baselines = (totals["weighted_strength"] / totals["weight"]).to_dict()
    mixtures = {
        team: (group.groupby("passer_player_id")["weight"].sum()
               / group["weight"].sum()).to_dict()
        for team, group in window.groupby("team") if group["weight"].sum() > 0
    }
    if not 0 <= win_total_weight <= 1:
        raise ValueError("win_total_weight must be between 0 and 1")
    for team, passer in (preseason_references or {}).items():
        if team in baselines and team not in played_teams:
            baselines[team] = (
                (1 - win_total_weight) * baselines[team]
                + win_total_weight * strengths.get(passer, 0.0)
            )
            mixtures[team] = {
                q: w * (1 - win_total_weight) for q, w in mixtures[team].items()
            }
            mixtures[team][passer] = mixtures[team].get(passer, 0.0) + win_total_weight
    return QbContext(strengths, baselines, evidence, mixtures)
