"""Publish live win probabilities across the Supabase boundary.

One scoreboard call covers the whole slate. None is made before a tracked
game's scheduled kickoff or after every tracked game is terminal, and the
worker exits as soon as nothing is left to watch. Each state is scored by the
in-game model against the projection published before kickoff, which is
frozen into the row at first sight; pregame tables are never written here.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backend.features.ingame import KICKOFF, REGULATION_SECONDS, SCRIMMAGE
from backend.live_feed import LiveState
from backend.model.ingame_nflfastr import MODEL_VERSION, TreeEnsemble, win_probability
from backend.publish import SCHEMA

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBABILITY_SOURCE = "pregame_anchored_game_state"
TERMINAL_STATES = {"Final", "Off"}
PERIOD_SECONDS = 900
# The dispatcher treats a row newer than four minutes as a living worker.
HEARTBEAT_SECONDS = 120
SCHEDULE_REFRESH_SECONDS = 300
KICKOFF_LEAD = timedelta(minutes=15)
# Matches the dispatcher window, so a recovered worker resumes the same games.
KICKOFF_LOOKBACK = timedelta(hours=6)
# A game still unstarted this long after kickoff is abandoned, not polled.
NO_START_LIMIT = timedelta(hours=3)
# The side with the ball this early in a scoreless game took the opening
# kickoff, so the other side receives after halftime.
OPENING_DRIVE_SECONDS = 120
MAX_BACKOFF_SECONDS = 300
VOLATILE_FIELDS = {"fetched_at", "worker_expires_at"}


@dataclass(frozen=True, slots=True)
class Game:
    game_id: str
    season: int
    week: int
    start: datetime
    home_abbr: str
    away_abbr: str
    home_team: str
    away_team: str
    home_margin: float
    as_of: datetime
    model_version: str

    def anchor(self) -> dict:
        return {
            "home_margin": self.home_margin,
            "as_of": self.as_of.isoformat(),
            "model_version": self.model_version,
        }


def _probability(
    game: Game,
    model: TreeEnsemble,
    state: LiveState,
    second_half_receiver: str | None,
) -> float:
    overtime = state.period > 4
    half_seconds = state.clock_seconds + (
        PERIOD_SECONDS if state.period in (1, 3) else 0
    )
    receives = (
        np.nan
        if second_half_receiver is None
        else float(state.period <= 2 and state.possession == second_half_receiver)
    )

    def number(value) -> float:
        return np.nan if value is None else float(value)

    row = {
        "offense_is_home": state.possession == "home",
        # Possession without a down is a kickoff or the start of a drive.
        "play_kind": KICKOFF if state.down is None else SCRIMMAGE,
        "period": state.period,
        "seconds_remaining": state.clock_seconds
        if overtime
        else (4 - state.period) * PERIOD_SECONDS + state.clock_seconds,
        "half_seconds_remaining": half_seconds,
        "score_margin": float(state.home_score - state.away_score),
        "pregame_margin": game.home_margin,
        "offense_receives_second_half": receives,
        "down": number(state.down),
        "distance": number(state.distance),
        "yards_to_goal": number(state.yards_to_goal),
        "home_timeouts": number(state.home_timeouts),
        "away_timeouts": number(state.away_timeouts),
    }
    return float(win_probability(pd.DataFrame([row]), model)[0])


def _pregame_probability(game: Game, model: TreeEnsemble) -> float:
    """The opening kickoff averaged over which side receives it."""
    return float(
        np.mean(
            [
                _probability(
                    game,
                    model,
                    LiveState(
                        "live",
                        period=1,
                        clock_seconds=PERIOD_SECONDS,
                        possession=side,
                        home_timeouts=3,
                        away_timeouts=3,
                    ),
                    "away" if side == "home" else "home",
                )
                for side in ("home", "away")
            ]
        )
    )


def _point(elapsed: int, home: int, away: int, probability: float) -> dict:
    return {"s": elapsed, "h": home, "a": away, "p": round(probability, 4)}


def score_game(
    game: Game,
    state: LiveState | None,
    model: TreeEnsemble,
    saved: dict | None,
    now: datetime,
    *,
    board_fetched: bool,
) -> dict:
    """One game's serving payload from its live state and frozen anchor."""
    saved = saved or {}
    history = saved.get("history") or [
        _point(0, 0, 0, _pregame_probability(game, model))
    ]
    receiver = saved.get("second_half_receiver")
    last = history[-1]["p"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "game_id": game.game_id,
        "season": game.season,
        "week": game.week,
        "abstract_state": "Pre",
        "status": "scheduled",
        "home_team": game.home_team,
        "away_team": game.away_team,
        "fetched_at": now.isoformat(),
        "probability_source": PROBABILITY_SOURCE,
        "model_version": MODEL_VERSION,
        "anchor": game.anchor(),
        "second_half_receiver": receiver,
        "home_win_probability": last,
        "away_win_probability": 1 - last,
        "history": history,
    }

    def unavailable(reason: str) -> dict:
        result.update(
            home_win_probability=None,
            away_win_probability=None,
            unavailable_reason=reason,
        )
        return result

    if game.as_of >= game.start:
        return unavailable("Pregame projection was not frozen before kickoff")
    if not board_fetched:
        return result
    if state is not None and state.status == "off":
        result.update(abstract_state="Off", status="off")
        return unavailable("Game was called off")
    if state is None or state.status == "scheduled":
        if now - game.start > NO_START_LIMIT:
            result["abstract_state"] = "Off"
            return unavailable("Game did not start")
        return result
    home, away = state.home_score, state.away_score
    result.update(status=state.status, home_score=home, away_score=away)
    if state.status == "final":
        result["abstract_state"] = "Final"
        # A regular-season game can end tied; neither side won it.
        probability = 0.5 if home == away else float(home > away)
        elapsed = int(REGULATION_SECONDS)
    else:
        result["abstract_state"] = "Live"
        clock = state.clock_seconds
        if state.period is None or clock is None or not 0 <= clock <= PERIOD_SECONDS:
            return unavailable("Scoreboard is missing the period or clock")
        elapsed = int(REGULATION_SECONDS) - (
            0 if state.period > 4 else (4 - state.period) * PERIOD_SECONDS + clock
        )
        if (
            receiver is None
            and state.possession
            and home == away == 0
            and elapsed <= OPENING_DRIVE_SECONDS
        ):
            receiver = "away" if state.possession == "home" else "home"
            result["second_half_receiver"] = receiver
        result.update(
            period=state.period,
            clock=f"{clock // 60}:{clock % 60:02d}",
            possession=state.possession,
            down=state.down,
            distance=state.distance,
            yards_to_goal=state.yards_to_goal,
        )
        # Between plays the feed names no side with the ball; the last
        # probability stands until it does.
        probability = (
            _probability(game, model, state, receiver) if state.possession else last
        )
    result.update(
        home_win_probability=probability, away_win_probability=1 - probability
    )
    point = _point(elapsed, home, away, probability)
    if history[-1] != point:
        result["history"] = [*history, point]
    return result


def _signature(payload: dict) -> str:
    content = {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class LivePublisher:
    """Tracks the games in the kickoff window and writes only what changed."""

    def __init__(
        self,
        model: TreeEnsemble,
        *,
        load_games,
        load_saved,
        fetch_states,
        write=None,
        expires_at: datetime | None = None,
    ):
        self.model = model
        self.load_games = load_games
        self.load_saved = load_saved
        self.fetch_states = fetch_states
        self.write = write
        self.expires_at = expires_at.isoformat() if expires_at else None
        self.games: dict[str, Game] = {}
        self.payloads: dict[str, dict] = {}
        self.written: dict[str, tuple[str, datetime]] = {}
        self.done: set[str] = set()
        self.next_schedule: datetime | None = None

    def _refresh_schedule(self, now: datetime) -> None:
        if self.next_schedule and now < self.next_schedule:
            return
        window = {
            game.game_id: game
            for game in self.load_games(now - KICKOFF_LOOKBACK, now + KICKOFF_LEAD)
        }
        unseen = [game_id for game_id in window if game_id not in self.games]
        saved = self.load_saved(unseen) if unseen and self.write else {}
        for game_id in unseen:
            game, payload = window[game_id], saved.get(game_id)
            if payload:
                # A projection republished mid-game must not move the anchor.
                anchor = payload["anchor"]
                game = replace(
                    game,
                    home_margin=float(anchor["home_margin"]),
                    as_of=datetime.fromisoformat(anchor["as_of"]),
                    model_version=anchor["model_version"],
                )
                self.payloads[game_id] = payload
                if payload["abstract_state"] in TERMINAL_STATES:
                    self.done.add(game_id)
            self.games[game_id] = game
        for game_id in set(self.games) - set(window):
            del self.games[game_id]
            self.payloads.pop(game_id, None)
            self.written.pop(game_id, None)
            self.done.discard(game_id)
        self.next_schedule = now + timedelta(seconds=SCHEDULE_REFRESH_SECONDS)

    def poll(self, now: datetime) -> bool:
        """Score the open games once; False when nothing is left to watch."""
        self._refresh_schedule(now)
        open_games = [g for g in self.games.values() if g.game_id not in self.done]
        if not open_games:
            return False
        # Before the first scheduled kickoff the database alone is enough.
        board_fetched = any(game.start <= now for game in open_games)
        states = self.fetch_states() if board_fetched else {}
        changed, finished = [], []
        for game in open_games:
            payload = score_game(
                game,
                states.get((game.home_abbr, game.away_abbr)),
                self.model,
                self.payloads.get(game.game_id),
                now,
                board_fetched=board_fetched and game.start <= now,
            )
            payload["worker_expires_at"] = self.expires_at
            self.payloads[game.game_id] = payload
            signature = _signature(payload)
            previous = self.written.get(game.game_id)
            if (
                previous
                and previous[0] == signature
                and (now - previous[1]).total_seconds() < HEARTBEAT_SECONDS
            ):
                continue
            changed.append((payload, signature))
            if payload["abstract_state"] in TERMINAL_STATES:
                finished.append(game.game_id)
        if changed and self.write:
            self.write([payload for payload, _ in changed])
        # Acknowledge only after the batch commits, so a failed final retries.
        for payload, signature in changed:
            self.written[payload["game_id"]] = (signature, now)
            log.info(
                "%s %s @ %s: %s, home WP %s%s",
                payload["game_id"],
                payload["away_team"],
                payload["home_team"],
                payload["abstract_state"],
                payload["home_win_probability"],
                " (published)" if self.write else " (read only)",
            )
        self.done.update(finished)
        return len(self.done) < len(self.games)


def run(
    publisher: LivePublisher,
    *,
    watch: bool,
    interval: int,
    duration: int | None = None,
    sleep=time.sleep,
    monotonic=time.monotonic,
    now=lambda: datetime.now(UTC),
) -> None:
    deadline = monotonic() + duration if duration else float("inf")
    failures = 0
    while True:
        started = monotonic()
        try:
            more = publisher.poll(now())
            failures = 0
        except Exception:  # noqa: BLE001 - a watch worker outlives a bad poll
            if not watch:
                raise
            log.exception("Refresh failed; previous snapshots age visibly on the site")
            more, failures = True, failures + 1
        if not more:
            log.info("No live or imminent games; stopping until the next dispatch")
            return
        if not watch or monotonic() >= deadline:
            return
        # A refused or failing feed is asked less often, not hammered.
        wait = min(MAX_BACKOFF_SECONDS, interval * 2**failures)
        sleep(max(1, min(deadline - monotonic(), wait - (monotonic() - started))))
        if monotonic() >= deadline:
            return


def load_games(start: datetime, end: datetime) -> list[Game]:
    from sqlalchemy import text

    from backend.db import engine

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT game_id, season, week, start_date, home_team_abbr, "
                "away_team_abbr, home_team, away_team, home_margin, as_of, "
                f"model_version FROM {SCHEMA}.game_projections "
                "WHERE start_date BETWEEN :start AND :end"
            ),
            {"start": start, "end": end},
        ).mappings()
        return [
            Game(
                game_id=row["game_id"],
                season=int(row["season"]),
                week=int(row["week"]),
                start=row["start_date"],
                home_abbr=row["home_team_abbr"],
                away_abbr=row["away_team_abbr"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_margin=float(row["home_margin"]),
                as_of=row["as_of"],
                model_version=row["model_version"],
            )
            for row in rows
        ]


def load_saved(game_ids: list[str]) -> dict[str, dict]:
    from sqlalchemy import text

    from backend.db import engine

    with engine.connect() as conn:
        return dict(
            conn.execute(
                text(
                    f"SELECT game_id, payload FROM {SCHEMA}.live_win_probability "
                    "WHERE game_id = ANY(:game_ids)"
                ),
                {"game_ids": game_ids},
            )
            .tuples()
            .all()
        )


def write_snapshots(payloads: list[dict]) -> None:
    from sqlalchemy import text

    from backend.db import engine

    rows = [
        {"game_id": p["game_id"], "updated_at": p["fetched_at"], "payload": p}
        for p in payloads
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {SCHEMA}.live_win_probability "
                "(game_id, updated_at, payload) "
                "SELECT game_id, updated_at, payload "
                "FROM jsonb_to_recordset(CAST(:snapshots AS jsonb)) "
                "AS incoming(game_id text, updated_at timestamptz, payload jsonb) "
                "ON CONFLICT (game_id) DO UPDATE SET "
                "updated_at = EXCLUDED.updated_at, payload = EXCLUDED.payload "
                f"WHERE {SCHEMA}.live_win_probability.updated_at "
                "< EXCLUDED.updated_at"
            ),
            {"snapshots": json.dumps(rows, allow_nan=False)},
        )
