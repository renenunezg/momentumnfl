"""Cutoff-specific candidates and shrinkage projections from regular-season games."""

import re
import unicodedata
from datetime import timedelta

import pandas as pd

from backend.awards import ingest, value
from backend.config import STATIC_DIR
from backend.features.drives import kickoff_utc

OFFENSE = {"QB", "RB", "FB", "WR", "TE"}
DEFENSE = {
    "DE",
    "DT",
    "DL",
    "NT",
    "EDGE",
    "LB",
    "OLB",
    "ILB",
    "MLB",
    "CB",
    "DB",
    "S",
    "FS",
    "SS",
}
STATS = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "attempts",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "receptions",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_tackles_solo",
    "def_tds",
]
OFFENSIVE_FEATURES = [
    "projected_passing_yards",
    "projected_passing_tds",
    "projected_passing_interceptions",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "epa_per_opportunity",
]
DEFENSIVE_FEATURES = [f"projected_{s}" for s in STATS if s.startswith("def_")]
CONTEXT = ["win_pct", "projected_team_wins", "games", "remaining_games"]
POSITIONS = ["is_QB", "is_RB", "is_WR", "is_TE", "is_DL", "is_LB", "is_DB"]


def name_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", value.lower())
    aliases = {
        "dariusleonard": "shaquilleleonard",
        "ahmadgardner": "saucegardner",
        "patsurtain": "patricksurtain",
    }
    key = re.sub("[^a-z]", "", value)
    return aliases.get(key, key)


def features_for(award: str) -> list[str]:
    if award == "COY":
        return [
            "win_pct",
            "projected_team_wins",
            "win_improvement",
            "wins_above_preseason_baseline",
            "point_margin",
            "games",
            "remaining_games",
        ]
    core = DEFENSIVE_FEATURES if award in {"DPOY", "DROY"} else OFFENSIVE_FEATURES
    if award == "CPOY":
        core = core + DEFENSIVE_FEATURES + ["prior_missed_games"]
    return core + CONTEXT + POSITIONS


def team_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, other in (("home", "away"), ("away", "home")):
        frame = schedule[["game_id", "season", "week", "start_date"]].copy()
        frame["team"] = schedule[f"{side}_team"]
        frame["coach"] = schedule[f"{side}_coach"]
        frame["win"] = (schedule[f"{side}_score"] > schedule[f"{other}_score"]).astype(
            float
        )
        frame["tie"] = (schedule[f"{side}_score"] == schedule[f"{other}_score"]).astype(
            float
        )
        frame["point_margin"] = schedule[f"{side}_score"] - schedule[f"{other}_score"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def snapshot(
    season: int, week: int, as_of=None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    schedule = ingest.read("schedules", season)
    schedule = schedule[schedule.game_type.eq("REG")].copy()
    schedule["start_date"] = kickoff_utc(schedule).astype("datetime64[ns, UTC]")
    if not 0 <= week <= (18 if season >= 2021 else 17):
        raise ValueError(f"Invalid regular-season week {week} for {season}")
    cutoff = (
        pd.Timestamp(as_of)
        if as_of is not None
        else (
            schedule.loc[schedule.week.le(week), "start_date"].max()
            + timedelta(hours=24)
        )
    )
    if cutoff.tzinfo is None:
        raise ValueError("Award forecast cutoff must include a timezone")
    played = schedule[
        schedule.week.le(week)
        & schedule.home_score.notna()
        & schedule.away_score.notna()
        & schedule.start_date.astype("int64").le(
            cutoff.value - 24 * 60 * 60 * 1_000_000_000
        )
    ]
    scheduled = pd.concat([schedule.home_team, schedule.away_team]).value_counts()
    expected = 17 if season >= 2021 else 16
    # BUF-CIN was canceled in 2022; the final reconstructed schedule has 16
    # completed games for these teams. A forecast before cancellation retains it.
    counts = scheduled.copy()
    required_counts = pd.Series(expected, index=counts.index)
    if season == 2022:
        if not set(("BUF", "CIN")).issubset(counts.index):
            raise ValueError("Missing canceled-game teams")
        canceled_known = cutoff >= pd.Timestamp("2023-01-06T05:00:00Z")
        for team in ("BUF", "CIN"):
            if counts[team] == 16 and not canceled_known:
                counts[team] = 17
            elif counts[team] == 17 and canceled_known:
                counts[team] = 16
            required_counts[team] = 16 if canceled_known else 17
    if len(counts) != 32 or not counts.eq(required_counts).all():
        raise ValueError(f"Incomplete {season} regular-season schedule")
    if played.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "cutoff": cutoff.isoformat(),
                "reason": "No completed regular-season games",
            },
        )
    appearances = team_rows(played)
    teams = appearances.groupby("team").agg(
        games=("game_id", "nunique"),
        wins=("win", "sum"),
        ties=("tie", "sum"),
        point_margin=("point_margin", "mean"),
    )
    teams["remaining_games"] = counts - teams.games
    teams["win_pct"] = (teams.wins + teams.ties * 0.5) / teams.games
    # Beta(4,4) shrinks early records. No closing odds or future game outcomes.
    teams["projected_team_wins"] = teams.wins + teams.remaining_games * (
        (teams.wins + teams.ties * 0.5 + 4) / (teams.games + 8)
    )
    previous_schedule = ingest.read("schedules", season - 1)
    previous_schedule = previous_schedule[
        previous_schedule.game_type.eq("REG") & previous_schedule.home_score.notna()
    ].copy()
    previous_schedule["start_date"] = kickoff_utc(previous_schedule)
    prior_teams = (
        team_rows(previous_schedule)
        .groupby("team")
        .agg(wins=("win", "sum"), games=("game_id", "nunique"))
    )
    teams["win_improvement"] = teams.win_pct - (prior_teams.wins / prior_teams.games)
    # Freeze a simple preseason expectation from the prior year's record.
    # This baseline is market-free and available at every historical cutoff.
    teams["preseason_baseline"] = (
        expected * (prior_teams.wins + 4) / (prior_teams.games + 8)
    )
    teams["wins_above_preseason_baseline"] = (
        teams.projected_team_wins - teams.preseason_baseline
    )
    # Individual coach tenure: an interim coach gets only his own completed games.
    coaches = (
        appearances.groupby(["coach", "team"])
        .agg(
            games=("game_id", "nunique"),
            wins=("win", "sum"),
            ties=("tie", "sum"),
            point_margin=("point_margin", "mean"),
        )
        .reset_index()
        .rename(columns={"coach": "candidate_name"})
    )
    coaches = coaches.merge(
        teams[
            [
                "remaining_games",
                "projected_team_wins",
                "win_improvement",
                "wins_above_preseason_baseline",
            ]
        ],
        on="team",
        validate="many_to_one",
    )
    coaches["win_pct"] = (coaches.wins + 0.5 * coaches.ties) / coaches.games
    coaches["candidate_id"] = "coach:" + coaches.candidate_name.map(name_key)
    coaches["position"] = "HC"
    coaches["performance_score"] = coaches.win_improvement * 100

    stats = ingest.read("stats", season)
    stats = stats[
        stats.season_type.eq("REG") & stats.game_id.isin(played.game_id)
    ].copy()
    if set(stats.game_id) != set(played.game_id):
        raise ValueError("Player statistics do not cover every completed game")
    if stats.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Duplicate player-game statistics")
    missing = set(STATS) - set(stats)
    if missing:
        raise ValueError(f"Missing award statistics: {sorted(missing)}")
    identity = stats.sort_values("week").drop_duplicates("player_id", keep="last")
    totals = stats.groupby("player_id")[STATS].sum(min_count=1)
    totals["games"] = stats.groupby("player_id").game_id.nunique()
    totals = totals.join(
        identity.set_index("player_id")[["player_display_name", "team", "position"]]
    )
    totals = totals.rename(columns={"player_display_name": "candidate_name"})
    totals["position"] = totals.position.replace({"HB": "RB"})
    rosters = ingest.read("rosters", season)
    eligibility = rosters.dropna(subset=["gsis_id", "rookie_year"])
    if eligibility.groupby("gsis_id").rookie_year.nunique().gt(1).any():
        raise ValueError("Conflicting rookie eligibility records")
    totals["rookie_year"] = totals.index.map(
        eligibility.drop_duplicates("gsis_id").set_index("gsis_id").rookie_year
    )
    totals = totals.join(
        teams[["remaining_games", "win_pct", "projected_team_wins"]], on="team"
    )
    previous = ingest.read("stats", season - 1)
    previous = previous[previous.season_type.eq("REG")]
    prior = previous.groupby("player_id")[STATS].sum(min_count=1)
    prior_games = previous.groupby("player_id").game_id.nunique()
    totals["prior_missed_games"] = (
        17 if season - 1 >= 2021 else 16
    ) - totals.index.map(prior_games).fillna(0)
    for stat in STATS:
        # Position priors are computed only from completed games at the cutoff.
        rate = totals[stat].fillna(0) / totals.games
        position_prior = rate.groupby(totals.position).transform("mean")
        prior_rate = totals.index.map(prior[stat] / prior_games)
        prior_rate = pd.Series(prior_rate, index=totals.index).fillna(position_prior)
        projected_rate = (totals[stat].fillna(0) + 4 * prior_rate) / (totals.games + 4)
        totals[f"projected_{stat}"] = (
            totals[stat].fillna(0) + projected_rate * totals.remaining_games
        )
    totals["epa_per_opportunity"] = (
        totals.passing_epa.fillna(0)
        + totals.rushing_epa.fillna(0)
        + totals.receiving_epa.fillna(0)
    ) / (totals.attempts + totals.carries + totals.targets).clip(lower=1)
    # Descriptive offensive efficiency, not causal credit or replacement value.
    totals["performance_score"] = totals.epa_per_opportunity
    for position, members in {
        "QB": {"QB"},
        "RB": {"RB", "FB"},
        "WR": {"WR"},
        "TE": {"TE"},
        "DL": {"DL", "DE", "DT", "NT", "EDGE"},
        "LB": {"LB", "OLB", "ILB", "MLB"},
        "DB": {"DB", "CB", "S", "FS", "SS"},
    }.items():
        totals[f"is_{position}"] = totals.position.isin(members).astype(float)
    totals = totals.reset_index().rename(columns={"player_id": "candidate_id"})
    totals = value.attach(totals, season, played.game_id)
    for frame in (totals, coaches):
        frame["season"], frame["week"], frame["as_of"] = (
            season,
            week,
            cutoff.isoformat(),
        )
    return (
        totals,
        coaches,
        {
            "cutoff": cutoff.isoformat(),
            "games_included": len(played),
            "source_basis": (
                "reconstructed historical sources; results available after 24h"
            ),
            "projection_basis": (
                "four-game player rate prior; beta(4,4) team record prior"
            ),
            "market_inputs": False,
            "sources": ingest.provenance(season),
        },
    )


def candidates(
    players: pd.DataFrame, coaches: pd.DataFrame, award: str
) -> pd.DataFrame:
    if award == "COY":
        return coaches.copy()
    if players.empty:
        return players.copy()
    pool = players.copy()
    if award in {"OPOY", "OROY"}:
        pool = pool[pool.position.isin(OFFENSE)]
    elif award in {"DPOY", "DROY"}:
        pool = pool[pool.position.isin(DEFENSE)]
    if award in {"OROY", "DROY"}:
        pool = pool[pool.rookie_year.eq(pool.season)]
    if award == "CPOY":
        context = comeback_context(int(pool.season.iloc[0]), pool.as_of.iloc[0])
        pool = pool.merge(
            context[["candidate_id", "reason", "source_url"]],
            on="candidate_id",
            validate="one_to_one",
        )
    if award in {"DPOY", "DROY"} and not pool.empty:
        rates = pool[[s for s in STATS if s.startswith("def_")]].div(pool.games, axis=0)
        pool["performance_score"] = (
            (rates - rates.mean()) / rates.std().replace(0, 1)
        ).mean(axis=1)
    return pool.reset_index(drop=True)


def comeback_context(season: int, as_of) -> pd.DataFrame:
    columns = [
        "season",
        "candidate_id",
        "candidate_name",
        "team",
        "position",
        "known_at",
        "source_url",
        "reason",
        "eligible",
    ]
    path = STATIC_DIR / "award_comeback_context.csv"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    context = pd.read_csv(path)
    if not set(columns).issubset(context):
        raise ValueError("CPOY context requires dated, sourced eligibility")
    context = context[context.season.eq(season)].copy()
    context = context[
        pd.to_datetime(context.known_at, utc=True).le(pd.Timestamp(as_of))
    ]
    context = context.sort_values("known_at").drop_duplicates(
        "candidate_id", keep="last"
    )
    return context[
        context.eligible.eq(True)
        & context.source_url.str.startswith("https://")
        & context.reason.notna()
    ].copy()
