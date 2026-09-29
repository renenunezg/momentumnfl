"""Conservative recognition of coach-attributed, matchup-specific QB news.

Only definite named-starter statements qualify. Unsupported wording stays a
depth-chart fallback, never an inferred announcement or model-strength ranking.
The raw news remains in the archived current-injury response.
"""

import re
from datetime import timedelta

import pandas as pd

from backend.features.drives import kickoff_utc

UNCERTAIN = re.compile(
    r"\b(?:if|unless|could|would|should|might|may|expected|likely|presumably|"
    r"appears|possibly|potentially|not|never|unsure|whether|undecided|considering|"
    r"rumor|rumour|false|denied|declined|refused|speculation|unconfirmed|premature)\b"
    r"|n't\b|\?",
    re.I,
)
COACH = re.compile(
    r"\bcoach\b.{0,100}\b(?:said|says|announced|confirmed|named)\b", re.I
)


def retraction(entry, week):
    """A definite newer benching report invalidates a retained announcement."""
    athlete = entry["athlete"]
    name = athlete.get("displayName", "")
    if not name:
        return False
    surname = athlete.get("lastName") or name.split()[-1]
    text = entry.get("shortComment") or ""
    weeks = {int(n) for n in re.findall(r"\bWeek\s+(\d+)\b", text, re.I)}
    if weeks and weeks != {week}:
        return False
    if re.search(
        r"\b(?:if|unless|could|would|might|may|expected|likely|unsure|whether)\b",
        text,
        re.I,
    ):
        return False
    subject = rf"\b(?:{re.escape(name)}|{re.escape(surname)})\s+"
    return bool(
        COACH.search(text)
        and re.search(
            subject + r"(?:will not start|won't start|will (?:serve|remain) as "
            r"[^.!?]{0,50}backup)\b",
            text,
            re.I,
        )
    )


def announcement(entry, team, gsis, season, week, observed, depth, teams, schedules):
    required = {"season", "week", "game_id", "home_team", "away_team"}
    if schedules is None or not required.issubset(schedules.columns):
        return None
    games = schedules[
        schedules.season.eq(season)
        & (schedules.home_team.eq(team) | schedules.away_team.eq(team))
    ].copy()
    games["kickoff"] = (
        pd.to_datetime(games.start_date, utc=True)
        if "start_date" in games
        else kickoff_utc(games)
    )
    target = games[games.week.eq(week) & games.kickoff.gt(observed)]
    if len(target) != 1:
        return None
    game = target.iloc[0]
    published = pd.Timestamp(entry["date"]).to_pydatetime()
    prior = games.loc[games.kickoff.lt(game.kickoff), "kickoff"].max()
    # A previous game's announcement cannot become next week's starter evidence.
    if published > observed or (
        pd.notna(prior) and published <= prior.to_pydatetime() + timedelta(hours=6)
    ):
        return None
    if published < game.kickoff.to_pydatetime() - timedelta(days=7):
        return None
    athlete = entry["athlete"]
    full_name = athlete.get("displayName", "")
    if not full_name:
        return None
    surname = athlete.get("lastName") or full_name.split()[-1]
    subject = rf"\b(?:{re.escape(full_name)}|{re.escape(surname)})\s+"
    definite = re.compile(
        subject
        + (
            r"(?:will (?:start (?:at quarterback|under center|for |against |versus |"
            r"in Week |Week |(?:this )?(?:Sunday|Monday|Tuesday|Wednesday|Thursday|"
            r"Friday|Saturday)\b)|(?:step in|take over) as "
            r"[^.!?]{0,70}starting quarterback\b)"
            r"|(?:has been|was) named [^.!?]{0,50}starting quarterback\b)"
        ),
        re.I,
    )
    headline = entry.get("shortComment") or ""
    if (
        not COACH.search(headline)
        or UNCERTAIN.search(headline)
        or not definite.search(headline)
    ):
        return None
    mentioned_weeks = {int(n) for n in re.findall(r"\bWeek\s+(\d+)\b", headline, re.I)}
    if mentioned_weeks and mentioned_weeks != {week}:
        return None
    opponent = game.away_team if game.home_team == team else game.home_team
    opponent_names = teams.loc[teams.team_abbr.eq(opponent), "team_name"]
    text = headline + " " + (entry.get("longComment") or "")
    if not mentioned_weeks and not any(
        re.search(rf"\b{re.escape(name.split()[-1])}\b", text, re.I)
        for name in opponent_names
    ):
        return None
    conditional_on = None
    condition = re.search(r"\bwhile (.+?) (?:is|remains) out\b", headline, re.I)
    if "while" in headline.lower():
        if condition is None or "player_name" not in depth:
            return None
        match = (
            depth[
                depth.team.eq(team)
                & depth.pos_abb.eq("QB")
                & depth.player_name.str.casefold().eq(condition[1].casefold())
                & pd.to_datetime(depth.dt, utc=True).le(observed)
            ]
            .gsis_id.dropna()
            .unique()
        )
        if len(match) != 1:
            return None
        conditional_on = match[0]
    links = athlete.get("links", [])
    url = next(
        (link.get("href") for link in links if "playercard" in link.get("rel", [])),
        None,
    )
    if not url:
        return None
    return dict(
        game_id=game.game_id,
        gsis_id=gsis,
        player_name=full_name,
        source_url=url,
        quote=headline,
        published_at=published.isoformat(),
        observed_at=observed.isoformat(),
        conditional_on=conditional_on,
    )


def confirmed(injuries, season, week, as_of, unavailable):
    if injuries is None or "starter_announcement" not in injuries:
        return {}
    candidates = {}
    for row in injuries[
        injuries.season.eq(season) & injuries.week.eq(week)
    ].itertuples():
        evidence = row.starter_announcement
        if not isinstance(evidence, dict) or row.gsis_id in unavailable:
            continue
        if evidence["conditional_on"] and evidence["conditional_on"] not in unavailable:
            continue
        if as_of is not None and any(
            pd.Timestamp(evidence[key]) > pd.Timestamp(as_of)
            for key in ("published_at", "observed_at")
        ):
            continue
        candidates.setdefault(row.team, []).append(evidence)
    selected = {}
    for team, evidence in candidates.items():
        latest = max(pd.Timestamp(item["published_at"]) for item in evidence)
        newest = [
            item for item in evidence if pd.Timestamp(item["published_at"]) == latest
        ]
        if len({item["gsis_id"] for item in newest}) != 1:
            raise ValueError(f"Conflicting announced quarterbacks: {team}")
        selected[team] = newest[0]
    return selected
