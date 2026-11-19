"""QB strengths and a shared forecast-time baseline for lineup substitutions."""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
import pandas as pd

# The level an unseen QB plays at, and the mean every QB's record shrinks
# toward: what QBs with under 150 effective dropbacks produced in their starts,
# net of the opponent, fitted on development seasons 2016-2021 at the
# 200-dropback prior (407 starts; standard error 0.021). It was -0.08, above
# those starts' raw -0.101.
REPLACEMENT_EPA_PER_DROPBACK = -0.1193
LEAGUE_DROPBACKS_PER_GAME = 34.0
SHRINK_DROPBACKS = 200.0
DEFAULT_SPAN_DROPBACKS = 1000.0
CALENDAR_HALF_LIFE_WEEKS = 52.0
# Shrinkage of the opponent-defense term, in dropbacks: fitted on development
# seasons 2016-2021 to predict each QB-game's EPA per dropback from pregame
# records (dropback-weighted squared error 0.11489 without it, 0.11218 with).
DEFENSE_PRIOR_DROPBACKS = 320.0


def strength_history(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
    calendar_half_life: float = CALENDAR_HALF_LIFE_WEEKS,
    prior_dropbacks: float = SHRINK_DROPBACKS,
    defense_prior: float = DEFENSE_PRIOR_DROPBACKS,
) -> pd.DataFrame:
    """QB strength after each game, in raw points above a fixed replacement.

    Each game's EPA is credited to the passer net of the opponent: its
    defense's pass EPA allowed per dropback above the league's, from games
    before that week, shrunk toward zero by ``defense_prior`` dropbacks.
    Defense and league records decay like a passer's. The fixed origin avoids
    using future league averages. A forecast must select only prior weeks
    through context_before_week before using it.
    """
    ordered = qb_games.merge(
        game_index[["game_id", "season", "model_week", "start_date"]], on="game_id"
    ).sort_values(["season", "model_week", "start_date", "game_id"])
    sizes = (span_dropbacks, calendar_half_life, prior_dropbacks, defense_prior)
    if not all(isfinite(x) and x > 0 for x in sizes):
        raise ValueError("QB memory and prior sizes must be finite and positive")
    clock = (ordered["season"] * 52 + ordered["model_week"]).to_numpy()
    passers = ordered["passer_player_id"].to_numpy()
    dropbacks = ordered["dropbacks"].to_numpy(dtype=float)
    epa = ordered["epa"].to_numpy(dtype=float)
    opponents = _opponents(ordered)
    # name -> [decayed EPA, decayed dropbacks, clock of the last update]
    qbs: dict[str, list] = {}
    defenses: dict[str, list] = {}
    league = [0.0, 0.0, None]

    def aged(state, now):
        """A record's EPA and dropbacks at ``now`` under calendar decay."""
        if state is None or state[2] is None:
            return 0.0, 0.0
        factor = 0.5 ** ((now - state[2]) / calendar_half_life)
        return state[0] * factor, state[1] * factor

    def add(records, key, gained, plays, now):
        state = records.setdefault(key, [0.0, 0.0, None])
        elapsed = 0 if state[2] is None else now - state[2]
        decay = 0.5 ** (plays / span_dropbacks + elapsed / calendar_half_life)
        state[:] = [decay * state[0] + gained, decay * state[1] + plays, now]

    opponent = np.zeros(len(ordered))
    weighted, effective = np.zeros(len(ordered)), np.zeros(len(ordered))
    week_starts = np.flatnonzero(np.r_[True, clock[1:] != clock[:-1]])
    for first, stop in zip(week_starts, np.r_[week_starts[1:], len(ordered)]):
        now = clock[first]
        league_epa, league_dropbacks = aged(league, now)
        mean = league_epa / league_dropbacks if league_dropbacks > 0 else 0.0
        for i in range(first, stop):
            allowed, faced = aged(defenses.get(opponents[i]), now)
            opponent[i] = (allowed + mean * defense_prior) / (
                faced + defense_prior
            ) - mean
        faced_by: dict[str, list[float]] = {}
        for i in range(first, stop):
            add(qbs, passers[i], epa[i] - dropbacks[i] * opponent[i], dropbacks[i], now)
            weighted[i], effective[i] = qbs[passers[i]][:2]
            totals = faced_by.setdefault(opponents[i], [0.0, 0.0])
            totals[0] += epa[i]
            totals[1] += dropbacks[i]
        for defense, (gained, plays) in faced_by.items():
            add(defenses, defense, gained, plays, now)
            league_epa, league_dropbacks = aged(league, now)
            league[:] = [league_epa + gained, league_dropbacks + plays, now]
    ordered["opponent_epa_per_dropback"] = opponent
    ordered["weighted_epa"] = weighted
    ordered["effective_dropbacks"] = effective
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
    return ordered.sort_values("start_date", kind="stable")


def _opponents(games: pd.DataFrame) -> np.ndarray:
    """Each row's opponent: the other team with dropbacks in its game."""
    sides = games.groupby("game_id")["team"].unique()
    if not sides.map(len).le(2).all():
        raise ValueError("A QB game lists more than two teams")
    lookup = {
        (game, team): next((t for t in pair if t != team), None)
        for game, pair in sides.items()
        for team in pair
    }
    return np.array([lookup[key] for key in zip(games["game_id"], games["team"])])


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

    def confirmed_absence(
        self, team: str, passer: str | None, unavailable: set[str]
    ) -> bool:
        """The dominant recent QB is explicitly ruled unavailable and replaced."""
        mixture = self.mixtures.get(team, {})
        if not mixture or passer is None:
            return False
        incumbent = max(mixture, key=mixture.get)
        return incumbent != passer and incumbent in unavailable

    def absence_delta(
        self, team: str, passer: str | None, unavailable: set[str]
    ) -> float:
        """Substitution from the dominant recent QB, explicitly ruled unavailable."""
        if not self.confirmed_absence(team, passer, unavailable):
            return 0.0
        return self.adjustment(team, passer)

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
        if strengths_override is None
        else strengths_override
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
        team: (
            group.groupby("passer_player_id")["weight"].sum() / group["weight"].sum()
        ).to_dict()
        for team, group in window.groupby("team")
        if group["weight"].sum() > 0
    }
    if not 0 <= win_total_weight <= 1:
        raise ValueError("win_total_weight must be between 0 and 1")
    for team, passer in (preseason_references or {}).items():
        if team in baselines and team not in played_teams:
            baselines[team] = (1 - win_total_weight) * baselines[
                team
            ] + win_total_weight * strengths.get(passer, 0.0)
            mixtures[team] = {
                q: w * (1 - win_total_weight) for q, w in mixtures[team].items()
            }
            mixtures[team][passer] = mixtures[team].get(passer, 0.0) + win_total_weight
    return QbContext(strengths, baselines, evidence, mixtures)
