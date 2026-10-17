"""With-without estimator for traits only a unit result can reveal."""

import pandas as pd

MIN_PRESENT_GAMES = 3


def with_without(
    presence: pd.DataFrame,
    unit_games: pd.DataFrame,
    value_column: str,
    shrink_games: float,
) -> pd.DataFrame:
    """Compare a team's unit value in games a player played and games he missed.

    ``value_column`` is a quantity where lower is better (EPA allowed), so
    ``raw_difference`` is the absent mean minus the present mean and a positive
    value means the team did better with the player. ``impact`` shrinks it by
    ``present * absent / (present * absent + shrink_games)``. Both are NaN when
    the player never missed a game or played in fewer than three.
    """
    games = presence.merge(
        unit_games[["team", "game_id", value_column]],
        on=["team", "game_id"],
        how="inner",
    ).dropna(subset=[value_column])
    games["present"] = games["present"].astype(bool)
    value = games[value_column]
    grouped = games.assign(
        present_value=value.where(games["present"]),
        absent_value=value.where(~games["present"]),
        absent=~games["present"],
    ).groupby("athlete_id")
    out = pd.DataFrame(
        {
            "present_games": grouped["present"].sum(),
            "absent_games": grouped["absent"].sum(),
            "present_mean": grouped["present_value"].mean(),
            "absent_mean": grouped["absent_value"].mean(),
        }
    ).reset_index()
    present = out["present_games"]
    absent = out["absent_games"]
    estimable = (absent > 0) & (present >= MIN_PRESENT_GAMES)
    out["raw_difference"] = (out["absent_mean"] - out["present_mean"]).where(estimable)
    weight = present * absent / (present * absent + shrink_games)
    out["impact"] = (out["raw_difference"] * weight).where(estimable)
    out[["present_games", "absent_games"]] = out[
        ["present_games", "absent_games"]
    ].astype(int)
    return out[
        ["athlete_id", "present_games", "absent_games", "raw_difference", "impact"]
    ]
