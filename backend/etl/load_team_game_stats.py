# backend/etl/load_team_game_stats.py

import os
import sys
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import math
import numpy as np
import nfl_data_py as nfl

def load_pbp_from_supabase(supabase: Client) -> pd.DataFrame:
    # Pull all PBP rows with simple pagination (PostgREST caps page size to ~1k)
    PAGE_SIZE = 1000
    offset = 0
    frames: list[pd.DataFrame] = []

    columns = (
        "season,week,game_id,"
        "posteam,defteam,"
        "play_type,qtr,down,ydstogo,yardline_100,"
        "shotgun,no_huddle,goal_to_go,"
        "score_differential,"
        "epa,air_yards,yards_after_catch,success,"
        "rush_attempt,pass_attempt,sack,interception,touchdown,"
        "penalty_yards,"
        "wp,home_wp,away_wp,"
        "vegas_wp,vegas_home_wp,vegas_wpa,vegas_home_wpa,"
        "home_team,away_team,home_score,away_score,"
        "play_id"
    )

    while True:
        query = (
            supabase
            .from_("pbp")
            .select(columns)
            .range(offset, offset + PAGE_SIZE - 1)
        )
        result = query.execute()
        rows = result.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"🔍  Supabase returned {len(df)} PBP rows (paginated)")
    return df

def aggregate_team_game(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    pbp["rush_attempt"] = pbp["rush_attempt"].fillna(False).astype(bool)
    pbp["pass_attempt"] = pbp["pass_attempt"].fillna(False).astype(bool)
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce")
    pbp = pbp[pbp["week"].between(1, 18)]
    pbp["eparush"] = pbp["epa"].where(pbp["rush_attempt"], 0.0)
    pbp["epapass"] = pbp["epa"].where(pbp["pass_attempt"], 0.0)

    # Distinct game metadata (one row per game)
    games = (
        pbp.groupby("game_id", as_index=False)
           .agg(
               season=("season", "last"),
               week=("week", "last"),
               home_team=("home_team", "last"),
               away_team=("away_team", "last"),
               total_home_score=("home_score", "max"),
               total_away_score=("away_score", "max"),
           )
    )

    # Expand to exactly two rows per game: home and away
    home_rows = games.assign(posteam=games["home_team"], defteam=games["away_team"])
    away_rows = games.assign(posteam=games["away_team"], defteam=games["home_team"])
    teams = pd.concat([home_rows, away_rows], ignore_index=True)

    # Offensive aggregates by posteam within game
    off = (
        pbp.groupby(["season","week","game_id","posteam"], as_index=False)
           .agg(
               total_epa=("epa","sum"),
               avg_epa=("epa","mean"),
               rush_epa=("eparush","sum"),
               pass_epa=("epapass","sum"),
               wp_end=("wp","last"),
           )
    )

    df = teams.merge(off, on=["season","week","game_id","posteam"], how="left")

    # Coerce numerics and fill missing aggregates with zeros
    for col in ["total_epa","avg_epa","rush_epa","pass_epa","wp_end"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)

    # Map team division using nfl_data_py
    try:
        teams_meta = nfl.import_teams()
    except Exception:
        # Fallback if import_teams is unavailable in this version
        teams_meta = nfl.import_team_desc()

    # Normalize expected column names
    possible_key_cols = ["team_abbr", "team"]
    possible_division_cols = ["team_division", "division"]
    key_col = next((c for c in possible_key_cols if c in teams_meta.columns), None)
    div_col = next((c for c in possible_division_cols if c in teams_meta.columns), None)
    if key_col is not None and div_col is not None:
        division_map = dict(zip(teams_meta[key_col], teams_meta[div_col]))
        df["division"] = df["posteam"].map(division_map)
    else:
        # If mapping failed, create empty division column to keep schema alignment
        df["division"] = None

    # Fallback mapping for nflverse abbreviations used in pbp (covers LA, etc.)
    fallback_divisions = {
        "ARI": "NFC West", "ATL": "NFC South", "BAL": "AFC North", "BUF": "AFC East",
        "CAR": "NFC South", "CHI": "NFC North", "CIN": "AFC North", "CLE": "AFC North",
        "DAL": "NFC East", "DEN": "AFC West", "DET": "NFC North", "GB": "NFC North",
        "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "KC": "AFC West",
        "LA": "NFC West", "LAC": "AFC West", "LV": "AFC West", "MIA": "AFC East",
        "MIN": "NFC North", "NE": "AFC East", "NO": "NFC South", "NYG": "NFC East",
        "NYJ": "AFC East", "PHI": "NFC East", "PIT": "AFC North", "SEA": "NFC West",
        "SF": "NFC West", "TB": "NFC South", "TEN": "AFC South", "WAS": "NFC East",
    }
    df["division"] = df["division"].where(df["division"].notna(), df["posteam"].map(fallback_divisions))

    return df[[
        "season","week","game_id",
        "posteam","defteam","division",
        "home_team","away_team","total_home_score","total_away_score",
        "total_epa","avg_epa","rush_epa","pass_epa","wp_end"
    ]]

def upsert_team_game_stats(df: pd.DataFrame, supabase: Client):
    # align DataFrame columns with table schema
    # prime the client’s schema cache so upsert() can see your columns
    supabase.table("team_game_stats") \
        .select("season") \
        .limit(1) \
        .execute()
    df = df.rename(columns={
        "posteam": "team_id",
        "defteam": "opponent_id",
    })
    # mark who was at home
    df["is_home"] = df["team_id"] == df["home_team"]
    # assign final points for/against
    df["points_for"] = df.apply(lambda r: r["total_home_score"] if r["is_home"] else r["total_away_score"], axis=1)
    df["points_against"] = df.apply(lambda r: r["total_away_score"] if r["is_home"] else r["total_home_score"], axis=1)
    # drop helper columns
    df.drop(columns=["home_team","away_team","total_home_score","total_away_score"], inplace=True)
    # ensure all aggregate stats are numeric, fill missing with zero, and round to 2 decimals
    for col in ["total_epa", "avg_epa", "rush_epa", "pass_epa", "wp_end"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)
    # convert NaNs to None so they become SQL NULLs where appropriate
    df = df.where(pd.notnull(df), None)
    # reorder columns to match team_game_stats schema
    df = df[[
        "season", "week", "game_id",
        "team_id", "opponent_id", "is_home", "division",
        "points_for", "points_against",
        "total_epa", "avg_epa", "rush_epa", "pass_epa",
        "wp_end"
    ]]
    # split into chunks, then upsert into your new table
    CHUNK = 500
    print(f"Prepared {len(df)} team-game rows across {df['game_id'].nunique()} games")
    for i in range(0, len(df), CHUNK):
        chunk = df.iloc[i : i + CHUNK].to_dict(orient="records")
        # convert any remaining NaNs to None so JSON serialization won't fail
        for rec in chunk:
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
        # ensure no None remains in aggregate fields
        for rec in chunk:
            for col in ["total_epa", "avg_epa", "rush_epa", "pass_epa", "wp_end"]:
                if rec.get(col) is None:
                    rec[col] = 0.0
        resp = supabase.table("team_game_stats").upsert(
            chunk, on_conflict="season,week,game_id,team_id"
        ).execute()
        if getattr(resp, "error", None):
            print("Upsert error:", resp.error)
        else:
            print(f"Upserted chunk {(i//CHUNK)+1} ({len(chunk)} rows)")

def main():
    load_dotenv()
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    supabase = create_client(url, key)

    pbp = load_pbp_from_supabase(supabase)
    team_game = aggregate_team_game(pbp)
    upsert_team_game_stats(team_game, supabase)

if __name__ == "__main__":
    main()