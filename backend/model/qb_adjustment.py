"""QB strengths and a shared forecast-time baseline for lineup substitutions."""

from dataclasses import dataclass

import pandas as pd

REPLACEMENT_EPA_PER_DROPBACK = -0.08
LEAGUE_DROPBACKS_PER_GAME = 34.0
SHRINK_DROPBACKS = 100.0
DEFAULT_SPAN_DROPBACKS = 500.0


def _shrunk_value(ew_value: float, ew_weight: float, seen: float) -> float:
    if ew_weight <= 0:
        return REPLACEMENT_EPA_PER_DROPBACK
    shrink = seen / (seen + SHRINK_DROPBACKS)
    raw = ew_value / ew_weight
    return REPLACEMENT_EPA_PER_DROPBACK + shrink * (raw - REPLACEMENT_EPA_PER_DROPBACK)


def _value_trajectory(
    games: pd.DataFrame, span_dropbacks: float
) -> tuple[list[tuple[float, float]], float]:
    """games are one passer's rows in chronological order. Returns the
    (value, dropbacks_before) pair entering each game and the value after the
    last one, as exponentially decayed EPA per dropback shrunk to replacement."""
    entering = []
    ew_value = ew_weight = seen = 0.0
    for row in games.itertuples():
        entering.append((_shrunk_value(ew_value, ew_weight, seen), seen))
        decay = 0.5 ** (row.dropbacks / span_dropbacks)
        ew_value = ew_value * decay + row.epa
        ew_weight = ew_weight * decay + row.dropbacks
        seen += row.dropbacks
    return entering, _shrunk_value(ew_value, ew_weight, seen)


def strength_history(
    qb_games: pd.DataFrame,
    game_index: pd.DataFrame,
    span_dropbacks: float = DEFAULT_SPAN_DROPBACKS,
) -> pd.DataFrame:
    """QB strength after each game, in raw points above a fixed replacement.

    The fixed origin avoids using future league averages. A forecast must
    select only prior weeks through context_before_week before using it.
    """
    ordered = qb_games.merge(
        game_index[["game_id", "season", "model_week", "start_date"]], on="game_id"
    ).sort_values(["passer_player_id", "start_date"])
    strengths = []
    for _, group in ordered.groupby("passer_player_id", sort=False):
        entering, last = _value_trajectory(group, span_dropbacks)
        after = [value for value, _ in entering[1:]] + [last]
        strengths.extend(
            (value - REPLACEMENT_EPA_PER_DROPBACK) * LEAGUE_DROPBACKS_PER_GAME
            for value in after
        )
    return ordered.assign(strength_points=strengths).sort_values("start_date")


@dataclass(frozen=True)
class QbContext:
    strengths: dict[str, float]
    baselines: dict[str, float]

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
) -> QbContext:
    """Value both the starter and the team's QB mix at the same forecast cut.

    Use every passer's dropback share with the engine's time decay. Before
    the first played week, carry the previous season's mix with its actual
    weeks intact. Same-QB teams therefore receive no extra QB adjustment.
    """
    eligible = history[
        (history["season"] < season)
        | (history["season"].eq(season) & (history["model_week"] < week))
    ]
    latest = eligible.drop_duplicates("passer_player_id", keep="last")
    strengths = latest.set_index("passer_player_id")["strength_points"].to_dict()
    window = eligible[eligible["season"].eq(season)].copy()
    if window.empty:
        window = eligible[eligible["season"].eq(season - 1)].copy()
    if window.empty:
        return QbContext(strengths, {})
    recency = 0.5 ** (
        (window["model_week"].max() - window["model_week"]) / half_life_weeks
    )
    window["weight"] = window["dropbacks"] * recency
    window["weighted_strength"] = (
        window["passer_player_id"].map(strengths) * window["weight"]
    )
    totals = window.groupby("team")[["weight", "weighted_strength"]].sum()
    totals = totals[totals["weight"] > 0]
    baselines = (totals["weighted_strength"] / totals["weight"]).to_dict()
    return QbContext(strengths, baselines)
