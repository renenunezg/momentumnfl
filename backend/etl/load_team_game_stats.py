# backend/etl/load_team_game_stats.py

import os
import sys
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import math

def load_pbp_from_supabase(supabase: Client) -> pd.DataFrame:
    # NOTE: This pulls raw play-by-play data from the `pbp` table (not the `team_game_stats` table).
    result = (
        supabase
        .from_("pbp")
        .select(
            "season","week","game_id",
            "posteam","defteam",
            "play_type","qtr","down","ydstogo","yardline_100",
            "shotgun","no_huddle","goal_to_go",
            "score_differential",
            "epa","air_yards","yards_after_catch","success",
            "rush_attempt","pass_attempt","sack","interception","touchdown",
            "penalty_yards",
            "wp","home_wp","away_wp",
            "vegas_wp","vegas_home_wp","vegas_wpa","vegas_home_wpa",
            "home_team","away_team","home_score","away_score"
        )
        .range(0, 100000)  # fetch up to 100k rows for full pagination
        .execute()
    )
    print(f"🔍  Supabase returned {len(result.data)} PBP rows")  # debug count
    return pd.DataFrame(result.data)

def aggregate_team_game(pbp: pd.DataFrame) -> pd.DataFrame:
    # Ensure boolean columns and compute epa for rush and pass only
    pbp["rush_attempt"] = pbp["rush_attempt"].fillna(False).astype(bool)
    pbp["pass_attempt"] = pbp["pass_attempt"].fillna(False).astype(bool)
    # Compute epa contributions: epa when attempt, else zero
    pbp["eparush"] = pbp["epa"].where(pbp["rush_attempt"], 0)
    pbp["epapass"] = pbp["epa"].where(pbp["pass_attempt"], 0)
    df = (
        pbp
        .groupby(
            ["season","week","game_id","posteam","defteam","home_team","away_team"],
            as_index=False
        )
        .agg(
            total_epa      = ("epa", "sum"),
            avg_epa        = ("epa", "mean"),
            rush_epa       = ("eparush", "sum"),
            pass_epa       = ("epapass", "sum"),
            wp_end         = ("wp", "last"),
            home_team      = ("home_team", "last"),
            away_team      = ("away_team", "last"),
            total_home_score = ("home_score", "max"),
            total_away_score = ("away_score", "max"),
        )
    )
    # Compute is_home and round numeric columns
    for col in ["total_epa","avg_epa","rush_epa","pass_epa","wp_end"]:
        df[col] = df[col].round(2)
    return df

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
        "team_id", "opponent_id",
        "is_home", "points_for", "points_against",
        "total_epa", "avg_epa", "rush_epa", "pass_epa",
        "wp_end"
    ]]
    # split into chunks, then upsert into your new table
    CHUNK = 500
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
        supabase.table("team_game_stats").upsert(chunk).execute()
        print(f"Upserted chunk {(i//CHUNK)+1}")

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