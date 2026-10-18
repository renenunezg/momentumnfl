"""Explicit publication of one validated NFL playoff snapshot."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from backend.playoffs.forecast import validate


def publish(path):
    payload = validate(json.loads(path.read_text()))
    cutoff = datetime.fromisoformat(payload["results_as_of"])
    now = datetime.now(UTC)
    if not now - timedelta(hours=24) <= cutoff <= now + timedelta(minutes=5):
        raise ValueError("Regenerate stale or future-dated forecasts before publishing")
    from backend.db import engine

    with engine.begin() as conn:
        changed = conn.execute(
            text("""
                insert into nfl.playoff_forecasts(season,payload)
                values(:season,cast(:payload as jsonb))
                on conflict(season) do update
                set payload=excluded.payload, published_at=now()
                where (nfl.playoff_forecasts.payload->>'results_as_of')::timestamptz
                  <= (excluded.payload->>'results_as_of')::timestamptz
                  and nfl.playoff_forecasts.payload is distinct from excluded.payload
                returning season
            """),
            {
                "season": payload["season"],
                "payload": json.dumps(payload, separators=(",", ":"), allow_nan=False),
            },
        ).scalar_one_or_none()
    print(json.dumps({"published": changed is not None, "season": payload["season"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--publish", action="store_true", required=True)
    publish(parser.parse_args().path)
