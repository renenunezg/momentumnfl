"""Offline NFL season and playoff forecast; no database or network operations."""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backend.etl import store
from backend.model.fit_week import market_home_spreads
from backend.model.projections import MODEL_VERSION
from backend.model.season_wins import (
    prepare_season_forecast,
    regular_schedule,
    season_outcomes,
)
from backend.playoffs.availability import (
    RETURN_WINDOWS,
    prepare_availability,
    project_with_availability,
    return_share,
)
from backend.playoffs.seeding import Standings

MODEL = f"nfl_playoffs_v2:{MODEL_VERSION}"


def matchup_probabilities(
    fit, names, conferences, as_of, qb_by_team, reports=(), kickoff=None
):
    """Evaluate each possible venue/matchup once, outside the simulation loop."""
    rows = []
    for h, home in enumerate(fit.teams):
        for a, away in enumerate(fit.teams):
            if home == away:
                continue
            rows.append(
                dict(
                    game_id=f"playoff_{home}_{away}",
                    season=fit.season,
                    week=19,
                    start_date=kickoff or as_of,
                    home_team=home,
                    away_team=away,
                    neutral_site=conferences[h] != conferences[a],
                )
            )
    adjustments = {
        row["game_id"]: (
            qb_by_team.get(row["home_team"], 0.0),
            qb_by_team.get(row["away_team"], 0.0),
        )
        for row in rows
    }
    projections, probabilities = project_with_availability(
        fit, pd.DataFrame(rows), as_of, names, adjustments, reports
    )
    matrix = np.full((32, 32), np.nan)
    index = {t: i for i, t in enumerate(fit.teams)}
    for p, probability in zip(projections, probabilities):
        matrix[index[p.home_team_abbr], index[p.away_team_abbr]] = probability
    return matrix


def play_bracket(fields, probabilities, rng=None):
    """Reseed by original seed every round; the Super Bowl is neutral-site."""
    games, conference_winners = [], []

    def play(home, away, round_name, conference, seeds):
        p = float(
            (
                probabilities[round_name]
                if isinstance(probabilities, dict)
                else probabilities
            )[home, away]
        )
        winner = home if (p >= 0.5 if rng is None else rng.random() < p) else away
        games.append(
            dict(
                round=round_name,
                conference=conference,
                home=home,
                away=away,
                home_seed=seeds.get(home),
                away_seed=seeds.get(away),
                home_win_probability=p,
                winner=winner,
            )
        )
        return winner

    for conf, field in zip(("AFC", "NFC"), fields):
        seeds = {t: i + 1 for i, t in enumerate(field)}
        remaining = [field[0]] + [
            play(field[h - 1], field[a - 1], "Wild Card", conf, seeds)
            for h, a in ((2, 7), (3, 6), (4, 5))
        ]
        remaining.sort(key=seeds.get)
        finalists = [
            play(remaining[0], remaining[3], "Divisional", conf, seeds),
            play(remaining[1], remaining[2], "Divisional", conf, seeds),
        ]
        finalists.sort(key=seeds.get)
        conference_winners.append(play(*finalists, "Conference", conf, seeds))
    play(*conference_winners, "Super Bowl", "NFL", {})
    return games


def forecast(
    fit,
    schedule,
    adjustments,
    metadata,
    as_of,
    draws=10_000,
    seed=20261018,
    qb_availability=(),
):
    if draws < 1000 or as_of.tzinfo is None or fit.as_of > as_of:
        raise ValueError("Require at least 1000 draws and a valid aware cutoff")
    if fit.season < 2021:
        raise ValueError("Only the current 17-game, 14-team format is supported")
    games = regular_schedule(schedule, fit.season, as_of)
    if set(fit.teams) != set(metadata.team_abbr) or len(fit.teams) != 32:
        raise ValueError("Require exactly 32 fitted franchises")
    teams = fit.teams
    index = {t: i for i, t in enumerate(teams)}
    metadata = metadata.set_index("team_abbr").loc[teams]
    conferences = metadata.team_conf.tolist()
    divisions = metadata.team_division.tolist()
    if (
        set(conferences) != {"AFC", "NFC"}
        or any(conferences.count(c) != 16 for c in set(conferences))
        or len(set(divisions)) != 8
        or any(divisions.count(d) != 4 for d in set(divisions))
    ):
        raise ValueError("Invalid NFL conference/division membership")
    finished = games.home_score.notna().to_numpy()
    if (
        games.loc[finished, "start_date"].isna().any()
        or (games.loc[finished, "start_date"] >= as_of).any()
    ):
        raise ValueError("Completed results require kickoff before cutoff")
    # This edition is a regular-season outlook, never silently rerun playoffs.
    postseason = schedule[
        (schedule.season == fit.season) & (schedule.game_type != "REG")
    ]
    if postseason.home_score.notna().any():
        raise ValueError("Postseason has begun; a results-aware edition is required")
    remaining = games.loc[~finished]
    names = metadata.team_name.to_dict()
    projections, probability = project_with_availability(
        fit,
        remaining,
        as_of,
        names,
        adjustments,
        qb_availability,
        market_home_spreads(remaining),
    )
    qb_by_team = {}
    for game in games.itertuples():
        if game.game_id not in adjustments:
            continue
        for team, adjustment in zip(
            (game.home_team, game.away_team), adjustments[game.game_id]
        ):
            if team in qb_by_team and not np.isclose(qb_by_team[team], adjustment):
                raise ValueError(
                    "Playoff outlook requires one current QB assumption per team"
                )
            qb_by_team[team] = adjustment
    round_dates = {}
    for round_name, kind, days in (
        ("Wild Card", "WC", 6),
        ("Divisional", "DIV", 13),
        ("Conference", "CON", 20),
        ("Super Bowl", "SB", 34),
    ):
        scheduled = postseason[postseason.game_type.eq(kind)]
        dates = (
            pd.to_datetime(scheduled.gameday, utc=True)
            if not scheduled.empty
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        kickoff = (
            dates.min() + timedelta(hours=18)
            if len(dates)
            else games.start_date.max() + timedelta(days=days)
        )
        round_dates[round_name] = kickoff
    matrix = (
        {
            name: matchup_probabilities(
                fit, names, conferences, as_of, qb_by_team, qb_availability, date
            )
            for name, date in round_dates.items()
        }
        if qb_availability
        else matchup_probabilities(fit, names, conferences, as_of, qb_by_team)
    )
    home = games.home_team.map(index).to_numpy()
    away = games.away_team.map(index).to_numpy()
    standings = Standings(teams, conferences, divisions, home, away)
    outcomes = np.where(
        games.home_score > games.away_score,
        1.0,
        np.where(games.home_score == games.away_score, 0.5, 0.0),
    )
    actual = np.zeros((32, 3), dtype=int)
    for h, a, result in zip(home[finished], away[finished], outcomes[finished]):
        if result == 0.5:
            actual[[h, a], 2] += 1
        else:
            winner, loser = (h, a) if result else (a, h)
            actual[winner, 0] += 1
            actual[loser, 1] += 1
    expected = actual[:, 0].astype(float)
    np.add.at(expected, home[~finished], probability)
    np.add.at(expected, away[~finished], 1 - probability)
    # Columns: playoffs, division, bye, divisional, conference game, SB, title.
    counts = np.zeros((32, 7), dtype=int)
    seed_counts = np.zeros((32, 7), dtype=int)
    unresolved = np.zeros(32, dtype=int)
    fields_seen = []
    rng = np.random.default_rng(seed + 1)
    for _, batch in season_outcomes(
        fit, remaining, projections, draws, seed, probabilities=probability
    ):
        for row in batch:
            outcomes[~finished] = row
            fields, deep = standings.seed(outcomes, rng)
            unresolved[list(deep)] += 1
            fields_seen.append(fields)
            for field in fields:
                counts[field, 0] += 1
                counts[field[:4], 1] += 1
                counts[field[0], 2] += 1
                counts[field[0], 3] += 1
                for s, team in enumerate(field):
                    seed_counts[team, s] += 1
            for game in play_bracket(fields, matrix, rng):
                col = {
                    "Wild Card": 3,
                    "Divisional": 4,
                    "Conference": 5,
                    "Super Bowl": 6,
                }[game["round"]]
                counts[game["winner"], col] += 1

    # Choose a legal realized field closest to the marginal seed frequencies.
    def field_score(fields):
        return sum(
            np.log(max(seed_counts[t, s] / draws, 1 / draws))
            for field in fields
            for s, t in enumerate(field)
        )

    field = max(fields_seen, key=field_score)
    path = play_bracket(field, matrix)
    for g in path:
        for key in ("home", "away", "winner"):
            g[key] = teams[g[key]]
    keys = [
        "playoff",
        "division_title",
        "bye",
        "divisional",
        "conference_game",
        "super_bowl",
        "champion",
    ]
    rows = []
    for i, team in enumerate(teams):
        rows.append(
            dict(
                id=team,
                team=names[team],
                conference=conferences[i],
                division=divisions[i],
                wins=int(actual[i, 0]),
                losses=int(actual[i, 1]),
                ties=int(actual[i, 2]),
                expected_wins=float(expected[i]),
                **dict(zip(keys, (counts[i] / draws).tolist())),
                seed_probabilities=(seed_counts[i] / draws).tolist(),
                deep_tiebreak_rate=float(unresolved[i] / draws),
            )
        )
    payload = dict(
        schema_version=1,
        season=fit.season,
        model=MODEL,
        model_as_of=fit.as_of.isoformat(),
        results_as_of=as_of.isoformat(),
        draws=draws,
        seed=seed,
        teams=rows,
        bracket=[
            dict(conference=c, seed=s + 1, team_id=teams[t])
            for c, f in zip(("AFC", "NFC"), field)
            for s, t in enumerate(f)
        ],
        path=path,
        games=[
            dict(
                id=p.game_id,
                home=p.home_team_abbr,
                away=p.away_team_abbr,
                week=p.week,
                date=p.start_date.isoformat() if p.start_date else None,
                neutral=p.neutral_site,
                home_win_probability=float(prob),
            )
            for p, prob in zip(projections, probability)
        ],
        qb_availability=[
            {
                **{k: v for k, v in report.items() if not k.endswith("_adjustment")},
                "games": [
                    dict(
                        game_id=g.game_id,
                        week=int(g.week),
                        date=g.start_date.isoformat(),
                        return_share=return_share(report, g.start_date),
                    )
                    for g in remaining.itertuples()
                    if report["team"] in (g.home_team, g.away_team)
                ],
                "postseason": [
                    dict(
                        round=name,
                        date=date.isoformat(),
                        return_share=return_share(report, date),
                    )
                    for name, date in round_dates.items()
                ],
            }
            for report in qb_availability
        ],
        limitations=[
            "Regular-season outlook; the bracket is one plausible field, not "
            "locked seeds.",
            "Future regular-season ties are omitted; completed ties count in "
            "standings.",
            "Tiebreakers follow NFL rules through strength of schedule. Remaining ties "
            "are randomized; each team's deep-tiebreak rate reports exposure.",
            "QB returns use the midpoint of the latest sourced range, not medical "
            "recovery probabilities. Unknown ETAs leave baseline QB assumptions "
            "unchanged beyond confirmed absences; those projections are provisional. "
            "Unscheduled playoff rounds use approximate dates.",
            "Regular-season outcomes share strength uncertainty. Playoff games use "
            "independent matchup draws; title odds have not been separately "
            "calibrated.",
            "Available regular-season sportsbook lines enter the existing "
            "published blend. "
            "Hypothetical postseason matchups have no market line.",
        ],
    )
    validate(payload)
    return payload


def validate(payload):
    if payload["schema_version"] != 1 or len(payload["teams"]) != 32:
        raise ValueError("Invalid playoff contract")
    rows = payload["teams"]
    if len({t["id"] for t in rows}) != 32:
        raise ValueError("Duplicate teams")
    targets = dict(
        playoff=14,
        division_title=8,
        bye=2,
        divisional=8,
        conference_game=4,
        super_bowl=2,
        champion=1,
    )
    for key, target in targets.items():
        values = [t[key] for t in rows]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid {key} probabilities")
        if not np.isclose(sum(values), target, atol=1e-10):
            raise ValueError(f"Invalid {key} probability mass")
    for t in rows:
        if not (
            t["champion"]
            <= t["super_bowl"]
            <= t["conference_game"]
            <= t["divisional"]
            <= t["playoff"]
            and t["bye"] <= t["division_title"] <= t["playoff"]
        ):
            raise ValueError("Inconsistent nested probabilities")
    for conf in ("AFC", "NFC"):
        slots = [s for s in payload["bracket"] if s["conference"] == conf]
        if sorted(s["seed"] for s in slots) != list(range(1, 8)):
            raise ValueError("Invalid conference seeds")
        lookup = {t["id"]: t for t in rows if t["conference"] == conf}
        if len({s["team_id"] for s in slots}) != 7:
            raise ValueError("Duplicate bracket team")
        if any(s["team_id"] not in lookup for s in slots):
            raise ValueError("Wrong conference")
        if (
            len({lookup[s["team_id"]]["division"] for s in slots if s["seed"] <= 4})
            != 4
        ):
            raise ValueError("Require four distinct division winners")
    ids = {t["id"] for t in rows}
    games = payload["games"]
    if len({g["id"] for g in games}) != len(games):
        raise ValueError("Duplicate remaining game")
    for g in games:
        if (
            g["home"] not in ids
            or g["away"] not in ids
            or g["home"] == g["away"]
            or not 0 <= g["home_win_probability"] <= 1
        ):
            raise ValueError("Invalid remaining matchup")
    for conf in ("AFC", "NFC"):
        members = [t for t in rows if t["conference"] == conf]
        if len(members) != 16:
            raise ValueError("Incomplete conference")
        for key, target in targets.items():
            if key != "champion" and not np.isclose(
                sum(t[key] for t in members), target / 2, atol=1e-10
            ):
                raise ValueError("Invalid conference probability mass")
        for seed in range(7):
            if not np.isclose(sum(t["seed_probabilities"][seed] for t in members), 1):
                raise ValueError("Invalid seed mass")
    for t in rows:
        if (
            len(t["seed_probabilities"]) != 7
            or not all(0 <= v <= 1 for v in t["seed_probabilities"])
            or not np.isclose(sum(t["seed_probabilities"]), t["playoff"])
            or not 0 <= t["deep_tiebreak_rate"] <= 1
        ):
            raise ValueError("Invalid team seed probabilities")
    # Reconstruct the projected favorite path to verify all pairings and reseeding.
    index = {t["id"]: i for i, t in enumerate(rows)}
    fields = [
        [
            index[s["team_id"]]
            for s in sorted(
                (s for s in payload["bracket"] if s["conference"] == conf),
                key=lambda s: s["seed"],
            )
        ]
        for conf in ("AFC", "NFC")
    ]
    matrix = np.full((32, 32), np.nan)
    if len(payload["path"]) != 13:
        raise ValueError("Incomplete postseason path")
    for game in payload["path"]:
        if not 0 <= game["home_win_probability"] <= 1:
            raise ValueError("Invalid postseason probability")
        matrix[index[game["home"]], index[game["away"]]] = game["home_win_probability"]
    expected_path = play_bracket(fields, matrix)
    for game in expected_path:
        for key in ("home", "away", "winner"):
            game[key] = rows[game[key]]["id"]
    if payload["path"] != expected_path:
        raise ValueError("Invalid postseason path or reseeding")
    model = datetime.fromisoformat(payload["model_as_of"])
    results = datetime.fromisoformat(payload["results_as_of"])
    if model.tzinfo is None or results.tzinfo is None or model > results:
        raise ValueError("Invalid forecast cutoff")
    json.dumps(payload, allow_nan=False)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--qb-returns", type=Path, default=RETURN_WINDOWS)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-current-injuries", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    as_of = datetime.now(UTC)
    if args.require_current_injuries:
        from backend.availability import current_reports
        from backend.model.fit_week import load_depth_charts

        teams = store.current_teams()
        _, covered = current_reports(
            args.season, 0, as_of, load_depth_charts(args.season), teams
        )
        missing = set(teams.team_abbr) - covered
        if missing:
            raise ValueError(
                "Playoff forecast requires fresh injury coverage: "
                + ", ".join(sorted(missing))
            )
    fit, schedule, adjustments, charts, _, _ = prepare_season_forecast(
        args.season, as_of
    )
    payload = forecast(
        fit,
        store.read_raw("schedules.parquet"),
        adjustments,
        store.current_teams(),
        as_of,
        args.draws,
        qb_availability=prepare_availability(
            fit, schedule, charts, as_of, args.qb_returns
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(data)
    temporary.replace(args.output)
    print(
        json.dumps(
            dict(
                seconds=round(time.perf_counter() - start, 2),
                bytes=len(data.encode()),
                draws=args.draws,
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
