"""NFL seeding through strength of schedule, with explicit deep-tie uncertainty.

Rules: https://www.nfl.com/standings/tie-breaking-procedures
Scoring-based tiebreakers are not inferred from simulated binary outcomes.
"""

import numpy as np


class Standings:
    def __init__(self, teams, conferences, divisions, home, away):
        self.teams = teams
        self.conferences = np.asarray(conferences)
        self.divisions = np.asarray(divisions)
        self.home, self.away = np.asarray(home), np.asarray(away)
        self.meetings = np.zeros((len(teams), len(teams)), dtype=int)
        np.add.at(self.meetings, (home, away), 1)
        np.add.at(self.meetings, (away, home), 1)
        self.played = self.meetings.sum(axis=1)
        self.division_groups = [
            np.flatnonzero(self.divisions == d).tolist() for d in sorted(set(divisions))
        ]
        self.conference_groups = [
            np.flatnonzero(self.conferences == c).tolist() for c in ("AFC", "NFC")
        ]
        self.common_cache = {}

    def seed(self, outcomes, rng):
        """outcomes: 1 home win, .5 tie, 0 away win, including actual results."""
        credit = np.zeros(self.meetings.shape)
        np.add.at(credit, (self.home, self.away), outcomes)
        np.add.at(credit, (self.away, self.home), 1 - outcomes)
        points = credit.sum(axis=1)
        percentage = points / self.played
        victories = np.zeros(self.meetings.shape)
        np.add.at(victories, (self.home, self.away), outcomes == 1)
        np.add.at(victories, (self.away, self.home), outcomes == 0)
        deep_ties = set()

        def record(t, opponents):
            count = self.meetings[t, opponents].sum()
            return credit[t, opponents].sum() / count if count else -1

        def choose(group, division):
            if len(group) == 1:
                return group[0]
            top = max(percentage[t] for t in group)
            group = [t for t in group if percentage[t] == top]
            if len(group) == 1:
                return group[0]
            # Restart the applicable procedure whenever the tied set shrinks.
            h2h = None
            if division or len(group) == 2:
                if all(self.meetings[t, group].sum() for t in group):
                    h2h = [record(t, group) for t in group]
            else:
                sweep = [
                    t
                    for t in group
                    if all(credit[t, o] > 0 for o in group if o != t)
                    and record(t, group) == 1
                ]
                if sweep:
                    return sweep[0]
                survivors = [
                    t
                    for t in group
                    if not (
                        all(self.meetings[t, o] for o in group if o != t)
                        and record(t, group) == 0
                    )
                ]
                if len(survivors) < len(group):
                    return choose(survivors, division)
            key = tuple(sorted(group))
            if key not in self.common_cache:
                self.common_cache[key] = np.flatnonzero(
                    (self.meetings[group] > 0).all(axis=0)
                )
            common = self.common_cache[key]
            common_record = (
                [record(t, common) for t in group]
                if len(common)
                and (
                    division or all(self.meetings[t, common].sum() >= 4 for t in group)
                )
                else None
            )
            conference_record = [
                record(t, np.flatnonzero(self.conferences == self.conferences[t]))
                for t in group
            ]
            criteria = [h2h]
            if division:
                criteria += [
                    [
                        record(t, np.flatnonzero(self.divisions == self.divisions[t]))
                        for t in group
                    ],
                    common_record,
                    conference_record,
                ]
            else:
                criteria += [conference_record, common_record]
            for weights in (victories, self.meetings):
                criteria.append(
                    [
                        float(weights[t] @ points / (weights[t] @ self.played))
                        if weights[t].sum()
                        else -1
                        for t in group
                    ]
                )
            for values in criteria:
                if values is None:
                    continue
                best = max(values)
                reduced = [
                    t
                    for t, value in zip(group, values)
                    if np.isclose(value, best, atol=1e-12, rtol=0)
                ]
                if len(reduced) < len(group):
                    return choose(reduced, division)
            deep_ties.update(group)
            return int(rng.choice(group))

        # Freeze each division's internal order before comparing wild cards.
        division_orders = []
        for group in self.division_groups:
            remaining, order = group.copy(), []
            while remaining:
                winner = choose(remaining, True)
                order.append(winner)
                remaining.remove(winner)
            division_orders.append(order)
        fields = []
        for conference in self.conference_groups:
            orders = [o.copy() for o in division_orders if o[0] in conference]
            champions = [o.pop(0) for o in orders]
            field = []
            while champions:
                winner = choose(champions, False)
                field.append(winner)
                champions.remove(winner)
            for _ in range(3):
                winner = choose([o[0] for o in orders if o], False)
                field.append(winner)
                next(o for o in orders if o and o[0] == winner).pop(0)
            fields.append(field)
        return fields, deep_ties
