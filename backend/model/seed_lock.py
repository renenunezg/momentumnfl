"""Teams with nothing to play for in the final regular-season week.

A team is locked when it is in the playoff field and wins or loses to the same
seed whatever the other open games do. Such teams rest starters, which no
rating carries: the rating describes the team that earned the seed.

The check enumerates every outcome of the open games that involve the team's
own conference, seeded with the league tiebreakers (``playoffs.seeding``).
Games between two teams of the other conference reach a conference's seeding
only through strength of victory and strength of schedule, so they are
sampled, not enumerated. Open games are treated as decisive; a tie is not
enumerated.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from backend.playoffs.seeding import Standings

# Seven teams a conference from 2020, six before.
EXPANDED_FIELD_SEASON = 2020
# Assignments of the other conference's open games per own-conference outcome:
# all home wins, all away wins, and seeded random draws.
OTHER_CONFERENCE_DRAWS = 8


def _other_assignments(count: int, rng) -> np.ndarray:
    if 2**count <= OTHER_CONFERENCE_DRAWS:
        codes = np.arange(2**count)
        return ((codes[:, None] >> np.arange(count)) & 1).astype(float)
    draws = rng.integers(0, 2, size=(OTHER_CONFERENCE_DRAWS, count)).astype(float)
    draws[0], draws[1] = 1.0, 0.0
    return draws


def locked_teams(
    games: pd.DataFrame, as_of: datetime, teams: pd.DataFrame
) -> frozenset[str]:
    """Teams whose seed is locked entering their open final-week game.

    ``games``: one season's regular-season rows with season, week, start_date,
    home_team, away_team, home_points and away_points. A result counts when its
    game kicked off by ``as_of``; every other row is open. Nothing is locked
    until the only open games are in the last regular-season week.
    ``teams``: team_abbr, team_conf, team_division for the 32 franchises.
    """
    kickoff = pd.to_datetime(games["start_date"], utc=True)
    decided = (
        kickoff.le(pd.Timestamp(as_of))
        & games["home_points"].notna()
        & games["away_points"].notna()
    ).to_numpy()
    week = games["week"].to_numpy()
    if decided.all() or (week[~decided] != week.max()).any():
        return frozenset()
    seasons = games["season"].unique()
    if len(seasons) != 1:
        raise ValueError("Seed locks require one season's games")
    names = teams["team_abbr"].tolist()
    index = {name: i for i, name in enumerate(names)}
    if len(names) != 32 or not set(games["home_team"]) | set(games["away_team"]) <= set(
        index
    ):
        raise ValueError("Seed locks require the 32 current franchises")
    conferences = teams["team_conf"].to_numpy()
    home = games["home_team"].map(index).to_numpy()
    away = games["away_team"].map(index).to_numpy()
    standings = Standings(
        names, conferences, teams["team_division"].to_numpy(), home, away
    )
    margin = (games["home_points"] - games["away_points"]).to_numpy()
    outcomes = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))
    pending = np.flatnonzero(~decided)
    field_size = 7 if int(seasons[0]) >= EXPANDED_FIELD_SEASON else 6
    rng = np.random.default_rng(0)
    locked = set()
    for slot, conference in enumerate(("AFC", "NFC")):
        inside = (conferences[home[pending]] == conference) | (
            conferences[away[pending]] == conference
        )
        own, other = pending[inside], pending[~inside]
        codes = np.arange(2 ** len(own))
        assignments = _other_assignments(len(other), rng)
        seeds = np.zeros((len(assignments), len(codes), 32), dtype=np.int8)
        for draw, assignment in enumerate(assignments):
            outcomes[other] = assignment
            for code in codes:
                outcomes[own] = (code >> np.arange(len(own))) & 1
                field = standings.seed(outcomes, rng)[0][slot][:field_size]
                seeds[draw, code, field] = np.arange(1, field_size + 1)
        for bit, game in enumerate(own):
            reversed_result = codes ^ (1 << bit)
            for team in (home[game], away[game]):
                seed = seeds[:, :, team]
                if (
                    conferences[team] == conference
                    and seed.all()
                    and (seed == seed[:, reversed_result]).all()
                ):
                    locked.add(names[team])
    return frozenset(locked)
