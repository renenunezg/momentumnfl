"""Season inputs every trait builder reads.

Regular-season nflverse play-by-play gives the plays, their situation and the
players each play tags; the awards inputs (player stats, rosters, snap counts
and schedules) give the box score, identities, positions and who was in the
building each week. Nothing here is fitted: the context is the evidence, and
``backend.players.credit`` turns it into per-play credit.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from backend.awards import ingest
from backend.config import RAW_DIR
from backend.features.drives import competitive_drive_mask, kickoff_utc
from backend.players import POSITION_GROUPS, UNIT_PREFIX

log = logging.getLogger(__name__)

PLAY_COLUMNS = [
    "game_id",
    "play_id",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "qb_dropback",
    "qb_scramble",
    "pass_attempt",
    "rush_attempt",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "sack",
    "qb_hit",
    "fumble",
    "fumble_lost",
    "fumbled_1_player_id",
    "fumbled_1_team",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "quarter_seconds_remaining",
    "score_differential",
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "xyac_mean_yardage",
    "cpoe",
    "success",
    "epa",
    "wpa",
    "fixed_drive",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "sack_player_id",
    "half_sack_1_player_id",
    "half_sack_2_player_id",
    "interception_player_id",
    "pass_defense_1_player_id",
    "pass_defense_2_player_id",
    "forced_fumble_player_1_player_id",
    "forced_fumble_player_1_team",
    "forced_fumble_player_2_player_id",
    "forced_fumble_player_2_team",
    "tackle_for_loss_1_player_id",
    "tackle_for_loss_2_player_id",
    "qb_hit_1_player_id",
    "qb_hit_2_player_id",
    "solo_tackle_1_player_id",
    "solo_tackle_1_team",
    "solo_tackle_2_player_id",
    "solo_tackle_2_team",
    "assist_tackle_1_player_id",
    "assist_tackle_1_team",
    "assist_tackle_2_player_id",
    "assist_tackle_2_team",
    "assist_tackle_3_player_id",
    "assist_tackle_3_team",
    "assist_tackle_4_player_id",
    "assist_tackle_4_team",
    "tackle_with_assist_1_player_id",
    "tackle_with_assist_1_team",
    "tackle_with_assist_2_player_id",
    "tackle_with_assist_2_team",
    "kick_distance",
    "field_goal_result",
    "field_goal_attempt",
    "punt_attempt",
    "kicker_player_id",
    "punter_player_id",
]
GAME_COLUMNS = [
    "game_id",
    "model_week",
    "home_team",
    "away_team",
    "completed",
    "start_date",
]
PRESENCE_COLUMNS = ["athlete_id", "team", "game_id", "model_week", "present"]
IDENTITY_COLUMNS = ["athlete_id", "athlete_name", "team", "headshot_url"]
BOX_COLUMNS = {
    "def_tackles_solo": "solo",
    "def_tackles_with_assist": "tackles_with_assist",
    "def_tackle_assists": "assists",
    "def_tackles_for_loss": "tfl",
    "def_sacks": "sacks",
    "def_qb_hits": "qb_hits",
    "def_interceptions": "interceptions",
    "def_pass_defended": "passes_defended",
    "def_fumbles_forced": "forced_fumbles",
}


@dataclass
class SeasonContext:
    season: int
    plays: pd.DataFrame
    games: pd.DataFrame
    stats: pd.DataFrame
    roster: pd.DataFrame
    positions: pd.DataFrame
    presence: pd.DataFrame
    identity: pd.DataFrame
    # Filled by ``backend.players.credit.with_credit``.
    credit: pd.DataFrame | None = None


def _pbp_path(season: int):
    return RAW_DIR / "pbp" / f"{season}.parquet"


def load_plays(season: int) -> pd.DataFrame:
    """Regular-season scrimmage and kicking plays with ``is_competitive`` and
    ``model_week``; ``play_id`` is a string key."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(_pbp_path(season)).names)
    columns = [column for column in PLAY_COLUMNS if column in available]
    missing = set(PLAY_COLUMNS) - available
    if missing:
        log.info("play-by-play %s lacks %s", season, sorted(missing))
    pbp = pd.read_parquet(_pbp_path(season), columns=columns)
    for column in missing:
        pbp[column] = pd.NA
    pbp = pbp[pbp["season_type"].eq("REG") & pbp["posteam"].notna()].copy()
    pbp["is_competitive"] = competitive_drive_mask(pbp).to_numpy()
    pbp["model_week"] = pbp["week"].astype(int)
    # nflverse play ids restart in every game, so the key carries the game.
    pbp["play_id"] = (
        pbp["game_id"].astype(str) + ":" + pbp["play_id"].astype("int64").astype(str)
    )
    pbp["is_success"] = pbp["success"].eq(1)
    return pbp.reset_index(drop=True)


def load_games(season: int) -> pd.DataFrame:
    """Every regular-season game with whether it has a final score."""
    schedule = ingest.read("schedules", season)
    schedule = schedule[schedule["game_type"].eq("REG")].copy()
    out = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "model_week": schedule["week"].astype(int),
            "home_team": schedule["home_team"],
            "away_team": schedule["away_team"],
            "completed": schedule["home_score"].notna()
            & schedule["away_score"].notna(),
            "start_date": kickoff_utc(schedule),
        }
    )
    return out.sort_values(["model_week", "game_id"], ignore_index=True)[GAME_COLUMNS]


def load_stats(season: int) -> pd.DataFrame:
    stats = ingest.read("stats", season)
    if stats.empty:
        return stats
    stats = stats[stats["season_type"].eq("REG")].copy()
    stats["athlete_id"] = stats["player_id"].astype(str)
    stats["model_week"] = stats["week"].astype(int)
    return stats.reset_index(drop=True)


def opponents(games: pd.DataFrame) -> pd.DataFrame:
    """``game_id, team, opponent`` for both sides of every game."""
    sides = []
    for team, opponent in (("home", "away"), ("away", "home")):
        sides.append(
            games[["game_id", f"{team}_team", f"{opponent}_team"]].set_axis(
                ["game_id", "team", "opponent"], axis=1
            )
        )
    return pd.concat(sides, ignore_index=True)


def season_positions(stats: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, position_group`` for every player seen in the season.

    The box score's modal position over the season comes first (``HB`` and ``FB``
    are backs, ``SAF`` a safety); a player with no box row takes his roster
    depth-chart position, then his roster position. Players at positions the
    grade does not rate (long snappers) are left out.
    """
    parts = []
    if not stats.empty:
        modal = (
            stats.dropna(subset=["position"])
            .groupby(["athlete_id", "position"])
            .size()
            .reset_index(name="rows")
            .sort_values(["athlete_id", "rows"], ascending=[True, False])
            .drop_duplicates("athlete_id")
        )
        parts.append(modal[["athlete_id", "position"]])
    rostered = roster.dropna(subset=["gsis_id"]).copy()
    rostered["athlete_id"] = rostered["gsis_id"].astype(str)
    rostered["position"] = rostered["depth_chart_position"].where(
        rostered["depth_chart_position"].isin(POSITION_GROUPS), rostered["position"]
    )
    parts.append(
        rostered.drop_duplicates("athlete_id", keep="last")[["athlete_id", "position"]]
    )
    out = pd.concat(parts, ignore_index=True).drop_duplicates("athlete_id")
    out["position_group"] = out["position"].map(POSITION_GROUPS)
    return out.dropna(subset=["position_group"])[["athlete_id", "position_group"]]


def snap_appearances(season: int, roster: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, team, game_id`` for every snap-count row that maps to a gsis id.

    Snap counts carry Pro Football Reference ids; the roster ties about four in
    five of them to a gsis id. The rest contribute nothing here, which is why
    presence is a union of sources rather than snap counts alone.
    """
    try:
        snaps = ingest.read("snaps", season)
    except FileNotFoundError:
        return pd.DataFrame(columns=["athlete_id", "team", "game_id"])
    if snaps.empty:
        return pd.DataFrame(columns=["athlete_id", "team", "game_id"])
    snaps = snaps[
        snaps["game_type"].eq("REG")
        & (snaps["offense_snaps"].fillna(0) + snaps["defense_snaps"].fillna(0)).gt(0)
    ]
    ids = roster.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    lookup = dict(zip(ids["pfr_id"], ids["gsis_id"].astype(str)))
    out = snaps.assign(athlete_id=snaps["pfr_player_id"].map(lookup)).dropna(
        subset=["athlete_id"]
    )
    return out[["athlete_id", "team", "game_id"]].drop_duplicates()


def player_presence(seen: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per player and completed game of each team the player was seen with.

    ``seen`` holds ``athlete_id, team, game_id`` from every source (box rows, snap
    counts, credited plays). A player is present where any source has him.
    """
    seen = (
        seen[["athlete_id", "team", "game_id"]]
        .astype({"athlete_id": str})
        .drop_duplicates()
    )
    completed = games[games["completed"]]
    team_games = pd.concat(
        [
            completed[["game_id", "model_week", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    )
    pairs = seen[["athlete_id", "team"]].drop_duplicates()
    out = pairs.merge(team_games, on="team", how="inner")
    out = out.merge(
        seen.assign(present=True), on=["athlete_id", "team", "game_id"], how="left"
    )
    out["present"] = out["present"].eq(True)
    return out[PRESENCE_COLUMNS].reset_index(drop=True)


ESPN_HEADSHOT = "https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/{espn}.png&w=112"


def espn_headshots(roster: pd.DataFrame) -> pd.Series:
    """ESPN's sized portrait per gsis id, from the season roster and the player
    crosswalk. The NFL CDN serves 3,400-pixel originals that take seconds to
    show on a card; ESPN's combiner returns a 112-pixel image in a few
    kilobytes."""
    parts = [roster[["gsis_id", "espn_id"]]]
    crosswalk = RAW_DIR / "players.parquet"
    if crosswalk.exists():
        parts.append(pd.read_parquet(crosswalk, columns=["gsis_id", "espn_id"]))
    ids = pd.concat(parts, ignore_index=True).dropna(subset=["gsis_id", "espn_id"])
    ids = ids.drop_duplicates("gsis_id")
    espn = ids["espn_id"].astype(float).astype("int64").astype(str)
    return pd.Series(
        [ESPN_HEADSHOT.format(espn=e) for e in espn],
        index=ids["gsis_id"].astype(str),
    )


def season_identity(
    stats: pd.DataFrame, roster: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Name, latest team and headshot per athlete, plus one row per OL unit."""
    parts = []
    if not stats.empty:
        latest = stats.sort_values("model_week").drop_duplicates(
            "athlete_id", keep="last"
        )
        parts.append(
            pd.DataFrame(
                {
                    "athlete_id": latest["athlete_id"],
                    "athlete_name": latest["player_display_name"],
                    "team": latest["team"],
                    "headshot_url": latest["headshot_url"],
                }
            )
        )
    rostered = roster.dropna(subset=["gsis_id"])
    parts.append(
        pd.DataFrame(
            {
                "athlete_id": rostered["gsis_id"].astype(str),
                "athlete_name": rostered["full_name"],
                "team": rostered["team"],
                "headshot_url": rostered["headshot_url"],
            }
        ).drop_duplicates("athlete_id", keep="last")
    )
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    parts.append(
        pd.DataFrame(
            {
                "athlete_id": [f"{UNIT_PREFIX}{team}" for team in teams],
                "athlete_name": [f"{team} offensive line" for team in teams],
                "team": teams,
                "headshot_url": None,
            }
        )
    )
    out = pd.concat(parts, ignore_index=True).drop_duplicates("athlete_id")
    espn = out["athlete_id"].astype(str).map(espn_headshots(roster))
    out["headshot_url"] = espn.fillna(out["headshot_url"])
    return out[IDENTITY_COLUMNS].reset_index(drop=True)


def load_season_context(season: int) -> SeasonContext:
    plays = load_plays(season)
    games = load_games(season)
    stats = load_stats(season)
    roster = ingest.read("rosters", season)
    positions = season_positions(stats, roster)
    sources = [snap_appearances(season, roster)]
    if not stats.empty:
        sources.append(stats[["athlete_id", "team", "game_id"]])
    presence = player_presence(pd.concat(sources, ignore_index=True), games)
    return SeasonContext(
        season=season,
        plays=plays,
        games=games,
        stats=stats,
        roster=roster,
        positions=positions,
        presence=presence,
        identity=season_identity(stats, roster, games),
    )
