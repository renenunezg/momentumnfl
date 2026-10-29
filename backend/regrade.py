"""Regrade the NFL performance page from the public tables alone.

Reads nfl.graded_games and nfl.recommendations through the site's public
(anonymous) Supabase REST endpoint, recomputes the forecast accuracy with the
same code that publishes nfl.performance_metrics, recomputes the prediction
record in Python, and prints both. The record deliberately does not call the
nfl.recommendation_dashboard RPC: it mirrors the overall rules of
nfl.recommendation_summary (sql/010_recommendation_h2h_weight_floor.sql) so the
page's record is checked against an independent computation. No database
credentials are needed:

    python -m backend.regrade --season 2026 --url https://<project>.supabase.co \
        --anon-key <public anon key>

SUPABASE_URL and SUPABASE_ANON_KEY may be set instead of the two flags.
"""

import argparse
import os

import pandas as pd
import requests

from backend.performance import compute_performance_metrics

PAGE_ROWS = 1000
SETTLED = ("win", "loss", "push")
PICK_COLUMNS = ["status", "outcome", "stake_units", "profit_units"]


def read_public(
    url: str, key: str, table: str, season: int, order: str
) -> pd.DataFrame:
    """Every row of one public nfl table for the season, paged in ``order``
    (a unique key, so pages neither overlap nor skip rows)."""
    rows, start = [], 0
    while True:
        response = requests.get(
            f"{url.rstrip('/')}/rest/v1/{table}",
            params={"season": f"eq.{season}", "select": "*", "order": order},
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept-Profile": "nfl",
                "Range": f"{start}-{start + PAGE_ROWS - 1}",
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < PAGE_ROWS:
            return pd.DataFrame(rows)
        start += PAGE_ROWS


def pick_record(picks: pd.DataFrame) -> pd.Series:
    """The page's record rules: ROI counts pushes and excludes pending and
    void picks; win rate excludes pushes and voids."""
    recommended = picks[picks["status"].eq("recommended")]
    settled = recommended[recommended["outcome"].isin(SETTLED)]
    wins = int(settled["outcome"].eq("win").sum())
    losses = int(settled["outcome"].eq("loss").sum())
    staked = settled["stake_units"].sum()
    profit = settled["profit_units"].sum()
    return pd.Series(
        {
            "picks": len(recommended),
            "wins": wins,
            "losses": losses,
            "pushes": int(settled["outcome"].eq("push").sum()),
            "pending": int(recommended["outcome"].eq("pending").sum()),
            "voids": int(recommended["outcome"].eq("void").sum()),
            "profit_units": round(float(profit), 2),
            "roi": round(float(profit / staked), 4) if staked else None,
            "win_rate": round(wins / (wins + losses), 4) if wins + losses else None,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--anon-key", default=os.getenv("SUPABASE_ANON_KEY"))
    args = parser.parse_args()
    if not args.url or not args.anon_key:
        parser.error("--url and --anon-key (or SUPABASE_URL, SUPABASE_ANON_KEY)")

    graded = read_public(
        args.url, args.anon_key, "graded_games", args.season, "game_id"
    )
    for column in ("start_date", "forecast_as_of", "forecast_recorded_at"):
        graded[column] = pd.to_datetime(graded[column], utc=True)
    metrics = compute_performance_metrics(graded)
    overall = metrics[metrics["segment_kind"].eq("overall")].set_index(
        "prediction_source"
    )
    columns = [
        "games",
        "margin_mae",
        "games_with_market",
        "market_mae",
        "model_minus_market_mae",
        "model_minus_market_mae_ci_low",
        "model_minus_market_mae_ci_high",
        "closer_than_market_share",
        "total_games",
        "total_mae",
        "coverage_80",
        "probability_games",
        "brier_score",
        "log_loss",
    ]
    print(f"Forecast accuracy, {args.season}, all graded games")
    print(overall[columns].T.round(4).to_string())

    picks = read_public(
        args.url,
        args.anon_key,
        "recommendations",
        args.season,
        "game_id,market,decision_at",
    )
    print(f"\nPrediction record, {args.season}")
    print(pick_record(picks).to_string())
    print("\nBy policy version")
    print(picks.groupby("policy_version")[PICK_COLUMNS].apply(pick_record).to_string())
    if "odds_source" in picks:
        print("\nBy odds source")
        print(picks.groupby("odds_source")[PICK_COLUMNS].apply(pick_record).to_string())


if __name__ == "__main__":
    main()
