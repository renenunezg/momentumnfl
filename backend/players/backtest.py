"""Build every season's traits, set the grade's weights on the development
seasons, and judge the grade once on the validation season.

Every choice is made on development pairs ``(t, t + 1)`` with both seasons in
the development set. The measure is the Spearman correlation of a player's
final-week grade in ``t`` with his final-week EPA credit per game in ``t + 1``,
over players qualified at the same group in both seasons, averaged over the
pairs. EPA credit alone is the baseline every variant must beat. The
validation pair is computed once, after every choice is fixed.
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import HISTORY_START_SEASON, PROCESSED_DIR, RAW_DIR
from backend.etl import store
from backend.players import EPA_TRAIT, TRAITS, UNIT_PREFIX, grade, snapshots
from backend.players.context import load_plays, load_season_context, load_stats
from backend.players.credit import credited_games, epa_credit_frames, with_credit
from backend.players.expectations import (
    ExpectationTables,
    expectation_tables_path,
    fit_expectation_tables,
    save_expectation_tables,
)
from backend.players.movers import MOVER_COLUMNS, portability, team_changes
from backend.players.snapshots import MIN_QUALIFIED_PLAYERS
from backend.players.traits import (
    defense_traits,
    line_traits,
    qb_traits,
    rb_traits,
    receiver_traits,
    special_traits,
)

log = logging.getLogger(__name__)

RELIABILITY_GAMES = (4, 8, 12)
SEASON_OUTPUTS = ("frames", "snapshots", "reliability", "credited")
# The fewest players a development pair needs for its Spearman or its ridge fit
# to count. Thirty-two teams make NFL pools smaller than FBS ones, so this sits
# below the pooled reliability floor; a sample floor, not a tuned value.
MIN_PAIR_PLAYERS = 20
MIN_FIT_ROWS = MIN_PAIR_PLAYERS
NEXT_SEASON_COLUMNS = [
    "season",
    "next_season",
    "position_group",
    "players",
    "spearman_grade_next_grade",
    "spearman_grade_next_epa",
    "spearman_epa_next_epa",
    "spearman_standard_next_epa",
    "holdout",
]
DEV_PAIR_COLUMNS = ["variant", *NEXT_SEASON_COLUMNS[:-1]]
SELECTION_COLUMNS = [
    "position_group",
    "current",
    "epa_trait",
    "fitted",
    "shrinkage",
    "epa_alone",
    "epa_weight_share",
    "epa_kept",
    "fitted_kept",
    "fitted_penalty",
    "shrink_games",
    "shrink_kept",
]
SPLIT_COLUMNS = ["position_group", "games", "players", "correlation"]
# In-season checkpoints: the grade at these weeks against the player's EPA credit
# over the rest of the season, so a mid-season grade is judged on what it is for.
CHECKPOINT_WEEKS = (4, 6, 9, 12)
IN_SEASON_COLUMNS = [
    "season",
    "week",
    "position_group",
    "players",
    "spearman_grade_rest_epa",
    "spearman_epa_rest_epa",
    "spearman_standard_rest_epa",
    "holdout",
]
# The box-score stat a scout would rank each group by without the grade, as a
# second baseline beside EPA credit: (trait frame it is read from, how). ``share``
# is the trait's own numerator over denominator; ``numerator`` and
# ``denominator`` are that column's sum per game present. Target share for
# receivers, sacks plus quarterback hits for linemen, tackles for linebackers,
# interceptions plus passes defended for defensive backs.
STANDARD_BASELINES = {
    "WR": ("rec.target_share", "share"),
    "TE": ("rec.target_share", "share"),
    "DL": ("def.finishing", "denominator"),
    "LB": ("def.tackle_share", "numerator"),
    "DB": ("def.ball_production", "numerator"),
}
PRESENCE_TRAIT = "def.every_week"
BAND_COLUMNS = [
    "position_group",
    "week",
    "players",
    "mean_uncertainty",
    "sd_change_to_final",
    "within_1",
    "within_2",
]
COVERAGE_COLUMNS = ["season", *grade.COVERAGE_LOSS_COLUMNS]
TABLES = (
    "selection",
    "weights",
    "dev_pairs",
    "next_season",
    "in_season",
    "band_calibration",
    "reliability",
    "portability",
    "movers_count",
    "grade_coverage",
)


def traits_artifact(name: str) -> tuple[str, ...]:
    return ("players", "position_traits", f"{name}.parquet")


def season_artifact(output: str, season: int) -> tuple[str, ...]:
    return traits_artifact(f"{output}_{season}")


def play_seasons_on_disk() -> list[int]:
    directory = RAW_DIR / "pbp"
    return sorted(
        int(path.stem)
        for path in directory.glob("*.parquet")
        if path.stem.isdigit() and int(path.stem) >= HISTORY_START_SEASON
    )


def ensure_play_seasons(seasons: list[int]) -> list[int]:
    """Download regular-season play-by-play for any season missing on disk.

    The weekly runner keeps only the active season's plays, so a cold fit pulls
    the development seasons from nflverse once and the cache keeps them after.
    """
    from backend.etl.store import write_parquet
    from backend.nflverse.data import load_pbp

    fetched = []
    for season in sorted(set(seasons)):
        path = RAW_DIR / "pbp" / f"{season}.parquet"
        if path.exists():
            continue
        log.info("position traits: downloading %s play-by-play", season)
        write_parquet(load_pbp([season]), path)
        fetched.append(season)
    return fetched


def fit_tables_before(season: int) -> ExpectationTables:
    """Expectation tables from every play season on disk before ``season``."""
    plays = {
        earlier: load_plays(earlier)
        for earlier in play_seasons_on_disk()
        if earlier < season
    }
    return fit_expectation_tables(plays, season)


def build_season_traits(
    season: int, tables: ExpectationTables | None = None
) -> dict[str, pd.DataFrame]:
    """Every trait frame, weekly snapshot and reliability table for ``season``.

    With ``tables`` (the frozen serving tables) no expectation is refit; a table
    whose source seasons include ``season`` or a later one is refused. Without
    it the tables are fit on every earlier season on disk, as the backtest does.
    """
    started = time.perf_counter()
    if tables is None:
        tables = fit_tables_before(season)
    elif any(int(source) >= season for source in tables.source_seasons):
        raise ValueError(
            f"expectation tables from {tables.source_seasons} cannot rate {season}"
        )
    ctx = with_credit(load_season_context(season))
    frames = pd.concat(
        [
            epa_credit_frames(ctx.credit, ctx.positions),
            qb_traits(ctx, tables),
            rb_traits(ctx, tables),
            receiver_traits(ctx, tables),
            defense_traits(ctx),
            line_traits(ctx, tables),
            special_traits(ctx, tables),
        ],
        ignore_index=True,
    ).astype({"athlete_id": str})
    frames["game_id"] = frames["game_id"].where(frames["game_id"].notna(), None)
    weekly = snapshots.weekly_trait_snapshots(
        frames, ctx.positions, ctx.games, ctx.identity, season, datetime.now(UTC)
    )
    outputs = {
        "frames": frames,
        "snapshots": weekly,
        "reliability": snapshots.split_half_reliability(
            frames, ctx.positions, RELIABILITY_GAMES
        ),
        "credited": credited_games(ctx.credit),
    }
    for output, frame in outputs.items():
        store.write_processed(frame, *season_artifact(output, season))
    log.info(
        "position traits %s: %d frame rows, %d snapshot rows in %.1fs",
        season,
        len(frames),
        len(weekly),
        time.perf_counter() - started,
    )
    return outputs


def load_season(season: int, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """The season's written outputs, built when any is missing or ``rebuild``."""
    stored = set(store.processed_names("players", "position_traits"))
    names = {output: f"{output}_{season}" for output in SEASON_OUTPUTS}
    if rebuild or not set(names.values()) <= stored:
        return build_season_traits(season)
    log.info("position traits %s: reading cached artifacts", season)
    return {
        output: store.read_processed(*season_artifact(output, season))
        for output in names
    }


# Pooled development evidence


def _season_keys(values: pd.Series, season: int) -> pd.Series:
    """Season-suffixed ids, so one athlete's seasons stay apart when pooled."""
    return values.astype(str) + f"@{season}"


def _people(snapshot: pd.DataFrame) -> pd.DataFrame:
    """``position_group`` and ``team`` per athlete of a season's snapshot."""
    rows = snapshot.astype({"athlete_id": str}).drop_duplicates(
        "athlete_id", keep="last"
    )
    return rows.set_index("athlete_id")[["position_group", "team"]]


def pooled_frames(
    dev: dict[int, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every development season's frames and positions with season-suffixed ids."""
    frames, positions = [], []
    for season, data in dev.items():
        rows = data["frames"].astype({"athlete_id": str})
        people = _people(data["snapshots"]).reset_index()
        frames.append(rows.assign(athlete_id=_season_keys(rows["athlete_id"], season)))
        positions.append(
            people[["athlete_id", "position_group"]].assign(
                athlete_id=_season_keys(people["athlete_id"], season)
            )
        )
    return pd.concat(frames, ignore_index=True), pd.concat(positions, ignore_index=True)


def pooled_trait_reliability(dev, games_at: tuple[int, ...]) -> pd.DataFrame:
    frames, positions = pooled_frames(dev)
    # Unit ids carry a season suffix here, so the unit prefix check in
    # ``position_groups`` still finds them; the roster group is OL regardless.
    return snapshots.split_half_reliability(frames, positions, games_at)


def _final_epa(snapshot: pd.DataFrame) -> pd.Series:
    """``epa.credit`` at the snapshot's final week, by athlete id."""
    rows = snapshot[snapshot["trait"].eq(EPA_TRAIT)]
    if rows.empty:
        return pd.Series(dtype=float)
    final = rows[rows["week"].eq(rows["week"].max())].astype({"athlete_id": str})
    return final.drop_duplicates("athlete_id").set_index("athlete_id")["value"]


def final_week_rows(dev) -> pd.DataFrame:
    """Qualified final-week rows of every dev season with the player's EPA credit."""
    parts = []
    for season, data in dev.items():
        snapshot = data["snapshots"]
        final = snapshot[
            snapshot["week"].eq(snapshot["week"].max()) & snapshot["qualified"].eq(True)
        ]
        final = final[["athlete_id", "position_group", "trait", "value"]].astype(
            {"athlete_id": str}
        )
        parts.append(final.assign(epa=final["athlete_id"].map(_final_epa(snapshot))))
    return pd.concat(parts, ignore_index=True)


def _alternate(games: pd.DataFrame) -> pd.DataFrame:
    games = (
        games[["athlete_id", "game_id", "model_week"]]
        .dropna(subset=["game_id"])
        .drop_duplicates(["athlete_id", "game_id"])
        .sort_values(["athlete_id", "model_week", "game_id"], kind="stable")
    )
    order = games.groupby("athlete_id").cumcount()
    return games.assign(
        half=order % 2,
        slot=order // 2,
        played=games.groupby("athlete_id")["game_id"].transform("size"),
    )


def _half_games(alternated: pd.DataFrame, n: int) -> pd.DataFrame:
    chosen = alternated[alternated["played"].ge(n) & alternated["slot"].lt(n // 2)]
    return chosen[["athlete_id", "game_id", "half"]]


def _pearson(left: pd.Series, right: pd.Series) -> float:
    if not (left.std() > 0 and right.std() > 0):
        return np.nan
    return float(left.corr(right))


def half_composites(frames, people, chosen, n, model) -> pd.DataFrame:
    """Each chosen athlete's composite in half 0 and half 1, indexed by athlete."""
    rows = frames.merge(chosen, on=["athlete_id", "game_id"])
    rows = rows.assign(
        aggregate=rows["trait"].map(
            {name: spec.aggregate for name, spec in TRAITS.items()}
        )
    )
    values = snapshots.aggregate(rows, ["athlete_id", "trait", "half"]).reset_index()
    half_snapshot = values.assign(
        position_group=values["athlete_id"].map(people["position_group"]),
        team=values["athlete_id"].map(people["team"]),
        games=n // 2,
        qualified=True,
    ).dropna(subset=["position_group"])
    columns = {}
    for half in (0, 1):
        scored = grade.score_grades(
            half_snapshot[half_snapshot["half"].eq(half)], model
        )
        columns[half] = scored.set_index("athlete_id")["composite"]
    return pd.DataFrame(columns)


def composite_split_half(seasons, model, games_at) -> pd.DataFrame:
    """Reliability of the grade's composite per group at each game count: the
    correlation of its odd-game and even-game halves, stepped up to the full
    count by ``spearman_brown``."""
    by_games: dict[int, list[pd.DataFrame]] = {n: [] for n in games_at}
    for season, data in seasons.items():
        frames = data["frames"].astype({"athlete_id": str})
        people = _people(data["snapshots"])
        alternated = _alternate(frames)
        for n in games_at:
            chosen = _half_games(alternated, n)
            if n < 2 or chosen.empty:
                continue
            halves = half_composites(frames, people, chosen, n, model)
            halves["position_group"] = halves.index.map(people["position_group"])
            by_games[n].append(
                halves.set_axis(_season_keys(halves.index.to_series(), season))
            )
    rows = []
    for n, parts in by_games.items():
        if not parts:
            continue
        wide = pd.concat(parts).dropna()
        for group, frame in wide.groupby("position_group"):
            if len(frame) < MIN_QUALIFIED_PLAYERS:
                continue
            rows.append(
                {
                    "position_group": group,
                    "games": n,
                    "players": len(frame),
                    "correlation": snapshots.spearman_brown(
                        _pearson(frame[0], frame[1])
                    ),
                }
            )
    out = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    return out.sort_values(["position_group", "games"], ignore_index=True)


def reliability_table(dev, pooled_traits, composite) -> pd.DataFrame:
    parts = [
        data["reliability"].assign(season=str(season)) for season, data in dev.items()
    ]
    parts.append(pooled_traits.assign(season="pooled"))
    parts.append(composite.assign(trait="grade", season="pooled"))
    columns = ["season", *snapshots.RELIABILITY_COLUMNS]
    return pd.concat(
        [part[columns] for part in parts if len(part)], ignore_index=True
    ).reindex(columns=columns)


def epa_shrink_games(pooled: pd.DataFrame) -> dict[str, float]:
    """Per group, the games at which ``epa.credit`` pooled reliability reaches 0.5."""
    rows = pooled[pooled["trait"].eq(EPA_TRAIT)][
        ["position_group", "games", "correlation"]
    ]
    return grade.shrink_games_from(rows)


# Movers


def movers_for(seasons: list[int]) -> pd.DataFrame:
    stats = {season: load_stats(season) for season in sorted(seasons)}
    movers = team_changes(stats)
    if movers.empty:
        return pd.DataFrame(columns=MOVER_COLUMNS)
    return movers


def movers_count_table(movers: pd.DataFrame) -> pd.DataFrame:
    if movers.empty:
        return pd.DataFrame(columns=["season", "movers"])
    return movers.groupby("season").size().rename("movers").reset_index()


# Next season


def final_week_grades(snapshot: pd.DataFrame, model: grade.GradeModel) -> pd.DataFrame:
    final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
    return grade.score_grades(final, model)[
        ["athlete_id", "position_group", "games", "grade"]
    ]


def _spearman(left: pd.Series, right: pd.Series) -> float:
    both = left.notna() & right.notna()
    if both.sum() < MIN_PAIR_PLAYERS:
        return np.nan
    return _pearson(left[both].rank(), right[both].rank())


def _qualified_final_epa(snapshot: pd.DataFrame) -> pd.Series:
    rows = snapshot[snapshot["trait"].eq(EPA_TRAIT) & snapshot["qualified"].eq(True)]
    if rows.empty:
        return pd.Series(dtype=float)
    final = rows[rows["week"].eq(rows["week"].max())].astype({"athlete_id": str})
    return final.drop_duplicates("athlete_id").set_index("athlete_id")["value"]


def standard_values(frames: pd.DataFrame, week: int | None = None) -> pd.Series:
    """Each player's ``STANDARD_BASELINES`` stat through ``week`` (every week when
    None), indexed by ``athlete_id`` and ``position_group``."""
    rows = frames.astype({"athlete_id": str})
    if week is not None:
        rows = rows[rows["model_week"].le(week)]
    present = (
        rows[rows["trait"].eq(PRESENCE_TRAIT)]
        .groupby("athlete_id")["game_id"]
        .nunique()
    )
    parts = []
    for group, (trait, how) in STANDARD_BASELINES.items():
        sums = (
            rows[rows["trait"].eq(trait)]
            .groupby("athlete_id")[["numerator", "denominator"]]
            .sum(min_count=1)
        )
        if how == "share":
            value = sums["numerator"] / sums["denominator"].where(
                sums["denominator"] > 0
            )
        else:
            games = present.reindex(sums.index)
            value = sums[how] / games.where(games > 0)
        parts.append(value.to_frame("value").assign(position_group=group).reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.set_index(["athlete_id", "position_group"])["value"]


def _standard_for(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    keys = pd.MultiIndex.from_frame(frame[["athlete_id", "position_group"]])
    return pd.Series(values.reindex(keys).to_numpy(), index=frame.index)


def next_season_table(seasons, model, validation_season: int) -> pd.DataFrame:
    """Does a final-week grade predict next season, compared with EPA credit alone?

    For each consecutive pair ``(t, t + 1)`` with ``t + 1 <= validation_season``,
    over players graded at the final week of both seasons at the same group:
    the Spearman of the season-``t`` grade with the ``t + 1`` grade and with the
    ``t + 1`` EPA credit per game (qualified at the final week), and of the
    season-``t`` EPA credit and ``STANDARD_BASELINES`` stat with the ``t + 1``
    EPA credit (the baselines).
    ``model`` may map each pair's first season to the model that grades both of
    its seasons (cross-fitted weights).
    """
    epa = {
        season: _qualified_final_epa(data["snapshots"])
        for season, data in seasons.items()
    }
    rows = []
    for season in sorted(seasons):
        later = season + 1
        if later not in seasons or later > validation_season:
            continue
        pair_model = model[season] if isinstance(model, dict) else model
        now = final_week_grades(seasons[season]["snapshots"], pair_model)
        nxt = final_week_grades(seasons[later]["snapshots"], pair_model)
        both = now.merge(
            nxt, on=["athlete_id", "position_group"], suffixes=("", "_next")
        )
        both = both.assign(
            epa=both["athlete_id"].map(epa[season]),
            epa_next=both["athlete_id"].map(epa[later]),
        )
        both["standard"] = _standard_for(
            both, standard_values(seasons[season]["frames"])
        )
        for group, frame in both.groupby("position_group"):
            complete = frame.dropna(subset=["grade", "grade_next", "epa", "epa_next"])
            used = complete if len(complete) >= MIN_PAIR_PLAYERS else frame
            rows.append(
                {
                    "season": season,
                    "next_season": later,
                    "position_group": group,
                    "players": len(used),
                    "spearman_grade_next_grade": _spearman(
                        used["grade"], used["grade_next"]
                    ),
                    "spearman_grade_next_epa": _spearman(
                        used["grade"], used["epa_next"]
                    ),
                    "spearman_epa_next_epa": _spearman(used["epa"], used["epa_next"]),
                    "spearman_standard_next_epa": _spearman(
                        used["standard"], used["epa_next"]
                    ),
                    "holdout": later == validation_season
                    and season == validation_season - 1,
                }
            )
    return pd.DataFrame(rows, columns=NEXT_SEASON_COLUMNS)


def _week_grades(
    snapshot: pd.DataFrame, model: grade.GradeModel, week: int
) -> pd.DataFrame:
    rows = snapshot[snapshot["week"].eq(week)]
    if rows.empty:
        return pd.DataFrame(columns=grade.OUTPUT_COLUMNS)
    return grade.score_grades(rows, model)


def _rest_of_season_epa(frames: pd.DataFrame, after_week: int) -> pd.Series:
    """EPA credit per credited game over weeks after ``after_week``, by athlete."""
    rows = frames[frames["trait"].eq(EPA_TRAIT) & frames["model_week"].gt(after_week)]
    rows = rows.dropna(subset=["numerator"]).astype({"athlete_id": str})
    return rows.groupby("athlete_id")["numerator"].mean()


def _epa_to_date(snapshot: pd.DataFrame, week: int) -> pd.Series:
    rows = snapshot[snapshot["trait"].eq(EPA_TRAIT) & snapshot["week"].eq(week)]
    rows = rows.astype({"athlete_id": str}).drop_duplicates("athlete_id")
    return rows.set_index("athlete_id")["value"]


def in_season_table(seasons, model, validation_season: int) -> pd.DataFrame:
    """Does a checkpoint-week grade predict the rest of the season?

    Per season, checkpoint week and group, over players graded at that week with
    at least three credited games afterwards: the Spearman of the grade with the
    player's EPA credit per game over the remaining weeks, and of his EPA credit to
    date and ``STANDARD_BASELINES`` stat to date with the same (the baselines).
    ``holdout`` marks the validation season.
    """
    rows = []
    for season in sorted(seasons):
        data = seasons[season]
        snapshot, frames = data["snapshots"], data["frames"]
        final_week = int(snapshot["week"].max())
        for week in CHECKPOINT_WEEKS:
            if week >= final_week:
                continue
            graded = _week_grades(snapshot, model, week)
            if graded.empty:
                continue
            later = frames[
                frames["trait"].eq(EPA_TRAIT) & frames["model_week"].gt(week)
            ]
            later = later.dropna(subset=["numerator"]).astype({"athlete_id": str})
            enough = later.groupby("athlete_id")["game_id"].nunique()
            rest = _rest_of_season_epa(frames, week)
            to_date = _epa_to_date(snapshot, week)
            graded = graded.assign(
                rest=graded["athlete_id"].map(rest),
                epa=graded["athlete_id"].map(to_date),
                later_games=graded["athlete_id"].map(enough).fillna(0),
            )
            graded = graded[graded["later_games"].ge(3)]
            graded = graded.assign(
                standard=_standard_for(graded, standard_values(frames, week))
            )
            for group, frame in graded.groupby("position_group"):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "position_group": group,
                        "players": int(frame["rest"].notna().sum()),
                        "spearman_grade_rest_epa": _spearman(
                            frame["grade"], frame["rest"]
                        ),
                        "spearman_epa_rest_epa": _spearman(frame["epa"], frame["rest"]),
                        "spearman_standard_rest_epa": _spearman(
                            frame["standard"], frame["rest"]
                        ),
                        "holdout": season == validation_season,
                    }
                )
    return pd.DataFrame(rows, columns=IN_SEASON_COLUMNS)


def band_calibration_table(seasons, model) -> pd.DataFrame:
    """Is the card's uncertainty band the right size?

    For each checkpoint week, the change from a player's week-``w`` grade to his
    final-week grade, standardised by the two bands combined in quadrature
    (``sqrt(u_w^2 + u_final^2)``). If the band is one standard deviation of
    measurement noise, about 68 percent of changes fall within one unit and 95
    percent within two. Pooled over the development seasons, per group.
    """
    parts = []
    for season in sorted(seasons):
        snapshot = seasons[season]["snapshots"]
        final_week = int(snapshot["week"].max())
        final = _week_grades(snapshot, model, final_week).set_index(
            ["athlete_id", "position_group"]
        )
        for week in CHECKPOINT_WEEKS:
            if week >= final_week:
                continue
            early = _week_grades(snapshot, model, week).set_index(
                ["athlete_id", "position_group"]
            )
            both = early.join(final, rsuffix="_final", how="inner")
            scale = np.sqrt(both["uncertainty"] ** 2 + both["uncertainty_final"] ** 2)
            both = both.assign(
                week=week,
                z=(both["grade_final"] - both["grade"]) / scale.where(scale.gt(0)),
                change=both["grade_final"] - both["grade"],
            )
            parts.append(both.reset_index())
    if not parts:
        return pd.DataFrame(columns=BAND_COLUMNS)
    pooled = pd.concat(parts, ignore_index=True).dropna(subset=["z"])
    rows = []
    for (group, week), frame in pooled.groupby(["position_group", "week"]):
        rows.append(
            {
                "position_group": group,
                "week": int(week),
                "players": len(frame),
                "mean_uncertainty": float(frame["uncertainty"].mean()),
                "sd_change_to_final": float(frame["change"].std()),
                "within_1": float(frame["z"].abs().le(1).mean()),
                "within_2": float(frame["z"].abs().le(2).mean()),
            }
        )
    return pd.DataFrame(rows, columns=BAND_COLUMNS)


def grade_coverage_table(seasons, model) -> pd.DataFrame:
    rows = []
    for season, data in sorted(seasons.items()):
        snapshot = data["snapshots"]
        final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
        rows.append(
            grade.coverage_losses(grade.grade_eligibility(final, model)).assign(
                season=season
            )
        )
    return pd.concat(rows, ignore_index=True)[COVERAGE_COLUMNS]


# Model selection


def dev_pair_table(dev, model, variant: str) -> pd.DataFrame:
    table = next_season_table(dev, model, max(dev))
    return table.assign(variant=variant)[DEV_PAIR_COLUMNS]


def pair_means(
    table: pd.DataFrame, column: str = "spearman_grade_next_epa"
) -> pd.Series:
    return table.groupby("position_group")[column].mean()


def pair_values(
    table: pd.DataFrame, column: str = "spearman_grade_next_epa"
) -> pd.Series:
    """The development pairs' values, indexed by group and season."""
    return table.set_index(["position_group", "season"])[column]


def clearly_better(candidate: pd.Series, reference: pd.Series, group: str) -> bool:
    """Whether ``candidate`` beats ``reference`` at ``group`` by more than one
    standard error of their paired difference over the development pairs, so a
    change is not adopted on a gain the pairs cannot tell from noise. One
    standard error is an evidence bar, not a tuned value."""
    if group not in candidate.index.get_level_values(0) or group not in (
        reference.index.get_level_values(0)
    ):
        return False
    both = pd.concat(
        [candidate.xs(group), reference.xs(group)], axis=1, keys=["new", "old"]
    ).dropna()
    if len(both) < 2:
        return False
    difference = both["new"] - both["old"]
    return bool(difference.mean() > difference.std(ddof=1) / np.sqrt(len(difference)))


def _kept_pairs(kept: dict[str, bool], new: pd.Series, old: pd.Series) -> pd.Series:
    """Per group, the development pairs of ``new`` where it was kept, else ``old``."""
    return pd.concat(
        [
            source[source.index.get_level_values(0) == g]
            for g, source in ((g, new if chosen else old) for g, chosen in kept.items())
        ]
    )


def _replace_groups(base: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    kept = base[~base["position_group"].isin(parts)]
    return pd.concat([kept, *parts.values()], ignore_index=True)[grade.WEIGHT_COLUMNS]


def _model(weights: pd.DataFrame, dev_seasons: list[int], **fields) -> grade.GradeModel:
    return grade.GradeModel(
        weights=weights.reset_index(drop=True), dev_seasons=dev_seasons, **fields
    )


def regression_pairs(dev, base, group: str, candidates: pd.DataFrame):
    """``(X, y)`` per development pair for ``group``: capped oriented z-scores on
    ``candidates`` for players graded at the final week of ``t``, against their
    next-season final-week EPA credit (qualified), z-scored over the sample."""
    out = {}
    for season in sorted(dev):
        later = season + 1
        if later not in dev:
            continue
        snapshot = dev[season]["snapshots"]
        final = snapshot[snapshot["week"].eq(snapshot["week"].max())]
        graded = grade.score_grades(final, base)
        graded = graded[graded["position_group"].eq(group)]["athlete_id"]
        nxt = dev[later]["snapshots"]
        nxt = nxt[
            nxt["trait"].eq(EPA_TRAIT)
            & nxt["qualified"].eq(True)
            & nxt["position_group"].eq(group)
        ]
        nxt = (
            nxt[nxt["week"].eq(nxt["week"].max())]
            .astype({"athlete_id": str})
            .drop_duplicates("athlete_id")
        )
        target = nxt.set_index("athlete_id")["value"].reindex(graded).dropna()
        if len(target) < MIN_FIT_ROWS or not target.std() > 0:
            continue
        x = (
            grade.trait_z_frame(final, candidates, group)
            .reindex(target.index)
            .fillna(0.0)
        )
        out[season] = (x, (target - target.mean()) / target.std())
    return out


def _candidates(weights: pd.DataFrame, group: str, epa_kept: bool) -> pd.DataFrame:
    rows = weights[weights["position_group"].eq(group) & weights["reliability"].notna()]
    if not epa_kept:
        rows = rows[rows["trait"].ne(EPA_TRAIT)]
    return rows[["position_group", "trait", "sign"]]


def _lower_bounds(candidates: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            0.0 if TRAITS[trait].higher_is_better is True else -np.inf
            for trait in candidates["trait"]
        ],
        index=candidates["trait"].to_list(),
    )


def select_grade_model(dev, pooled, portability_rows, orientation):
    """The grade model, chosen position by position on the development pairs.

    Starting from the current grade (reliability x portability weights without
    EPA credit), each change is kept at a group only where it is
    ``clearly_better`` on the development pairs: (1) ``epa.credit`` as a trait;
    (2) weights fitted by a bounded ridge of next-season EPA credit on the
    traits, evaluated leave-one-pair-out; (3) shrinkage of the composite by
    games / (games + k_g).
    """
    dev_seasons = sorted(dev)
    with_epa = grade.trait_weights(pooled, portability_rows, orientation)
    without = grade.trait_weights(
        pooled, portability_rows, orientation[orientation["trait"].ne(EPA_TRAIT)]
    )
    current = _model(without, dev_seasons)
    epa_model = _model(with_epa, dev_seasons)
    tables = [
        dev_pair_table(dev, current, "current"),
        dev_pair_table(dev, epa_model, "epa_trait"),
    ]
    base_mean, epa_mean = pair_means(tables[0]), pair_means(tables[1])
    base_pairs, epa_pairs = pair_values(tables[0]), pair_values(tables[1])
    groups = sorted(grade.graded_groups(epa_model) | grade.graded_groups(current))
    epa_kept = {g: clearly_better(epa_pairs, base_pairs, g) for g in groups}

    epa_rows = with_epa[with_epa["trait"].eq(EPA_TRAIT)]
    dropped_epa = epa_rows.assign(weight=0.0, status="not kept")
    step1_parts = {}
    for g in groups:
        if epa_kept[g]:
            step1_parts[g] = with_epa[with_epa["position_group"].eq(g)]
        else:
            step1_parts[g] = pd.concat(
                [
                    without[without["position_group"].eq(g)],
                    dropped_epa[dropped_epa["position_group"].eq(g)],
                ]
            )
    step1 = _replace_groups(
        pd.concat([without, dropped_epa], ignore_index=True), step1_parts
    )
    model1 = _model(step1, dev_seasons)
    step1_pairs = _kept_pairs(epa_kept, epa_pairs, base_pairs)

    designs, lowers = {}, {}
    for g in groups:
        candidates = _candidates(step1, g, epa_kept[g])
        if candidates.empty:
            continue
        pairs = regression_pairs(dev, model1, g, candidates)
        if len(pairs) < 2:
            continue
        designs[g], lowers[g] = pairs, _lower_bounds(candidates)
    pair_seasons = [season for season in dev_seasons if season + 1 in dev]
    cross_models = {}
    for held in pair_seasons:
        parts = {}
        for g, pairs in designs.items():
            training = {
                season: frame for season, frame in pairs.items() if season != held
            }
            if not training:
                continue
            coefficients, _ = grade.fit_ridge_weights(training, lowers[g])
            parts[g] = grade.fitted_group_weights(step1, g, coefficients)
        cross_models[held] = _model(_replace_groups(step1, parts), dev_seasons)
    fitted_table = dev_pair_table(dev, cross_models, "fitted")
    fitted_mean, fitted_pairs = pair_means(fitted_table), pair_values(fitted_table)
    fitted_kept = {
        g: g in designs and clearly_better(fitted_pairs, step1_pairs, g) for g in groups
    }
    final_parts, penalties = {}, {}
    for g in groups:
        if fitted_kept[g]:
            coefficients, penalties[g] = grade.fit_ridge_weights(designs[g], lowers[g])
            final_parts[g] = grade.fitted_group_weights(step1, g, coefficients)
    step2 = _replace_groups(step1, final_parts)
    model2 = _model(step2, dev_seasons)
    step2_pairs = _kept_pairs(fitted_kept, fitted_pairs, step1_pairs)
    composite = composite_split_half(dev, model2, RELIABILITY_GAMES)
    reliability = composite.dropna(subset=["correlation"])[
        grade.RELIABILITY_COLUMNS
    ].reset_index(drop=True)

    shrink = grade.shrink_games_from(reliability)
    shrink_models = {}
    for held in pair_seasons:
        cross = cross_models[held].weights
        parts = {
            g: cross[cross["position_group"].eq(g)] for g in groups if fitted_kept[g]
        }
        shrink_models[held] = _model(
            _replace_groups(step2, parts),
            dev_seasons,
            reliability=reliability,
            shrink_games=shrink,
        )
    shrink_table = dev_pair_table(dev, shrink_models, "shrinkage")
    shrink_mean = pair_means(shrink_table)
    shrink_kept = {
        g: clearly_better(pair_values(shrink_table), step2_pairs, g) for g in groups
    }
    model = _model(
        step2,
        dev_seasons,
        reliability=reliability,
        shrink_games={g: shrink[g] for g in groups if shrink_kept[g] and g in shrink},
        epa_shrink_games=epa_shrink_games(pooled),
    )

    epa_alone = pair_means(tables[0], "spearman_epa_next_epa")
    share = epa_rows.set_index("position_group")["weight"]
    selection = pd.DataFrame(
        [
            {
                "position_group": g,
                "current": base_mean.get(g, np.nan),
                "epa_trait": epa_mean.get(g, np.nan),
                "fitted": fitted_mean.get(g, np.nan) if g in designs else np.nan,
                "shrinkage": shrink_mean.get(g, np.nan),
                "epa_alone": epa_alone.get(g, np.nan),
                "epa_weight_share": share.get(g, np.nan),
                "epa_kept": epa_kept[g],
                "fitted_kept": fitted_kept[g],
                "fitted_penalty": penalties.get(g, np.nan),
                "shrink_games": shrink.get(g, np.nan),
                "shrink_kept": shrink_kept[g],
            }
            for g in groups
        ],
        columns=SELECTION_COLUMNS,
    )
    dev_pairs = pd.concat([*tables, fitted_table, shrink_table], ignore_index=True)
    for row in selection.itertuples():
        log.info(
            "position grade %s: dev-pair mean current %.4f, +epa %.4f, +fitted %.4f, "
            "+shrinkage %.4f, epa alone %.4f; kept epa %s, fitted %s, "
            "shrinkage %s (k %.2f)",
            row.position_group,
            row.current,
            row.epa_trait,
            row.fitted,
            row.shrinkage,
            row.epa_alone,
            row.epa_kept,
            row.fitted_kept,
            row.shrink_kept,
            row.shrink_games,
        )
    return model, composite, dev_pairs, selection


# The backtest


@dataclass
class BacktestResult:
    selection: pd.DataFrame
    weights: pd.DataFrame
    dev_pairs: pd.DataFrame
    next_season: pd.DataFrame
    in_season: pd.DataFrame
    band_calibration: pd.DataFrame
    reliability: pd.DataFrame
    portability: pd.DataFrame
    movers_count: pd.DataFrame
    grade_coverage: pd.DataFrame
    model: grade.GradeModel
    expectations: ExpectationTables
    dev_seasons: list[int]
    validation_season: int


def run_backtest(
    dev_seasons: list[int],
    validation_season: int,
    rebuild: bool = False,
    score_holdout: bool = True,
) -> BacktestResult:
    """Set the grade's weights on ``dev_seasons`` and judge it on ``validation_season``.

    Reliability, portability and orientation read the development seasons only;
    the validation season enters no weight and is only scored. The result also
    carries the serving expectation tables, fit on every play season through
    ``validation_season``, for the weekly runner to score later seasons with.
    """
    dev_seasons = sorted({int(season) for season in dev_seasons})
    if validation_season <= max(dev_seasons):
        raise ValueError(
            f"validation season {validation_season} must follow every dev season "
            f"{dev_seasons}"
        )
    seasons = {
        season: load_season(season, rebuild)
        for season in [*dev_seasons, validation_season]
    }
    dev = {season: seasons[season] for season in dev_seasons}
    snapshots_by_season = {
        season: data["snapshots"] for season, data in seasons.items()
    }

    pooled = pooled_trait_reliability(dev, RELIABILITY_GAMES)
    movers = movers_for(list(seasons))
    dev_portability = portability(
        {season: snapshots_by_season[season] for season in dev_seasons},
        movers[movers["season"].isin(dev_seasons)],
    )
    orientation = grade.orientations(final_week_rows(dev))
    for row in orientation[orientation["sign_source"].eq("epa_spearman")].itertuples():
        log.info(
            "position grade: %s %s takes sign %+.0f "
            "(Spearman with EPA credit %.3f over %d players)",
            row.position_group,
            row.trait,
            row.sign,
            row.spearman_epa,
            row.players,
        )
    model, composite, dev_pairs, selection = select_grade_model(
        dev, pooled, dev_portability, orientation
    )
    return BacktestResult(
        selection=selection,
        weights=model.weights,
        dev_pairs=dev_pairs,
        next_season=(
            next_season_table(seasons, model, validation_season)
            if score_holdout
            else pd.DataFrame(columns=NEXT_SEASON_COLUMNS)
        ),
        in_season=in_season_table(
            seasons if score_holdout else dev, model, validation_season
        ),
        band_calibration=band_calibration_table(dev, model),
        reliability=reliability_table(dev, pooled, composite),
        portability=dev_portability,
        movers_count=movers_count_table(movers[movers["season"].isin(dev_seasons)]),
        grade_coverage=grade_coverage_table(seasons, model),
        model=model,
        expectations=fit_tables_before(validation_season + 1),
        dev_seasons=dev_seasons,
        validation_season=validation_season,
    )


# Output


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float | np.floating):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _column_type(values: pd.Series) -> pd.Series:
    present = values.dropna()
    if present.empty:
        return values.astype(float)
    if present.map(lambda value: isinstance(value, bool | np.bool_)).all():
        return values.astype("boolean")
    if present.map(
        lambda value: isinstance(value, int | float | np.number)
        and not isinstance(value, bool)
    ).all():
        return values.astype(float)
    return values.map(lambda value: None if pd.isna(value) else str(value)).astype(
        object
    )


def evaluation_frame(result: BacktestResult) -> pd.DataFrame:
    stacked = pd.concat(
        [getattr(result, name).assign(table=name).astype(object) for name in TABLES],
        ignore_index=True,
    )
    stacked = stacked[
        ["table", *[column for column in stacked.columns if column != "table"]]
    ]
    return stacked.apply(
        lambda column: column if column.name == "table" else _column_type(column)
    )


def report_markdown(result: BacktestResult) -> str:
    """The model record: what the grade is, how it was chosen, and every table."""
    dev = result.dev_seasons
    pairs = ", ".join(f"{s} to {s + 1}" for s in dev if s + 1 in dev)
    kept = result.selection
    lines = [
        "# Position traits v1: per-position trait ratings and the grade",
        "",
        "Generated by `position-traits-fit`; every table is copied from the evaluation",
        f"artifact of the run on {datetime.now(UTC).date().isoformat()}. "
        "Edit the code, not this file.",
        "",
        "## What the model is",
        "",
        "Every NFL player gets a set of named traits at his position and two numbers.",
        "",
        "- The **grade** is a 20 to 80 scouting number built from his traits, which at",
        "  most positions include his EPA credit. It has mean 50 and sd 10 among",
        "  graded qualified players at his position that week. It is built to",
        "  predict how good",
        "  he will be next season.",
        "- **Wins above replacement** is his opponent-adjusted competitive EPA credit",
        "  above the positional replacement level, weighted by how much a point was",
        "  worth in each game he played. Nothing in it is fitted.",
        "",
        "A trait is a per-game rate built from tagged plays or box score counts,",
        "adjusted for the opponent through the previous week where the data allows,",
        "and turned into a position percentile among qualified players that week.",
        "Traits are `counted` when the feed tags the player directly and `inferred`",
        "when they come from unit outcomes or teammate comparisons. Nothing from the",
        "week being rated enters its own rating.",
        "",
        "## How the grade was chosen",
        "",
        f"Development seasons are {dev[0]} through {dev[-1]} (pairs {pairs}).",
        f"{result.validation_season} is the held-out check, computed once after every",
        "choice was fixed. The measure is the Spearman correlation of a player's",
        "final-week grade in season t with his final-week EPA credit per game in",
        "season t + 1, per position, averaged over the development pairs, among",
        "players qualified at the same position in both seasons. EPA credit alone",
        "(season t against t + 1) is the baseline, beside the box-score stat each",
        "group is usually ranked by (`spearman_standard_next_epa`: target share for",
        "receivers, sacks plus quarterback hits per game for linemen, tackles per",
        "game for linebackers, interceptions plus passes defended per game for",
        "defensive backs).",
        "",
        "The starting grade weights each trait by its reliability at 8 games (odd",
        "against even games, stepped up to 8 games by Spearman-Brown, pooled over",
        "the development seasons; under 30 players counts as",
        "unmeasured) times its portability (the mover slope over the stayer slope",
        "of this season's value on last season's; imputed where fewer than 20",
        "players moved). Three changes were then tried in order and each kept at a",
        "position only where it raised the dev-pair mean there: EPA credit as a",
        "trait, weights fitted by a bounded ridge of next-season EPA credit on the",
        "trait z-scores (penalty chosen leave-one-pair-out from 61 log-spaced",
        "values), and shrinkage of the composite by games / (games + k) with k the",
        "games at which the composite's reliability reaches one half. A change is",
        "kept only where its mean gain over the development pairs exceeds one",
        "standard error of the paired differences.",
        "",
        "### Selection",
        "",
        markdown_table(kept),
        "",
        "### Weights",
        "",
        markdown_table(result.weights),
        "",
        "### Development pairs",
        "",
        markdown_table(result.dev_pairs),
        "",
        "## Held-out check",
        "",
        "The pair ending in the validation season is `holdout`. Nothing was changed",
        "after it was computed.",
        "",
        markdown_table(result.next_season),
        "",
        "## In-season check",
        "",
        "A checkpoint-week grade against the player's EPA credit per game over the",
        "rest of that season, for players with at least three credited games",
        "afterwards, beside his EPA credit to date as the baseline. `holdout` rows",
        "are the validation season.",
        "",
        markdown_table(result.in_season),
        "",
        "## Band calibration",
        "",
        "The change from the checkpoint-week grade to the final-week grade,",
        "standardised by both bands in quadrature. A one-sd band puts about 68",
        "percent of changes within 1 and 95 percent within 2.",
        "",
        markdown_table(result.band_calibration),
        "",
        "## Reliability",
        "",
        "Odd-game against even-game correlations stepped up by Spearman-Brown to",
        "the full game count, per season and pooled; `grade` rows are the",
        "composite's curve that sets the card's uncertainty band.",
        "",
        markdown_table(result.reliability),
        "",
        "## Portability",
        "",
        markdown_table(result.portability),
        "",
        markdown_table(result.movers_count),
        "",
        "## Grade coverage",
        "",
        "Qualified players at the final week of each season, how many are graded,",
        "and how many lose the grade for lacking the primary trait or for low weight",
        "coverage.",
        "",
        markdown_table(result.grade_coverage),
        "",
        "## Known limits",
        "",
        "- Defensive credit covers tagged disruptions and tackles on stops only;",
        "  coverage is never tagged, so defensive back traits lean on with-without",
        "  comparisons and ball production.",
        "- 3-4 outside linebackers are grouped as LB from the box score position,",
        "  so edge rushers split between DL and LB.",
        "- Snap counts map to about four in five players by Pro Football Reference",
        "  id; presence is a union of box rows, snaps and credited plays, so a",
        "  defender with none of the three in a game is treated as absent.",
        "- The offensive line is one unit per team; no lineman is tagged on any play.",
        "- A defender's EPA credit is one-sided: he shares the value of plays he is",
        "  tagged on that cost the offense, and nothing is charged to him for a gain",
        "  he allowed, so defensive credit and wins above replacement grow with",
        "  playing time. Defensive grades are judged against next-season credit built",
        "  from the same tags.",
        "- Punters have no production outcome to validate against; their grade is",
        "  expected points per punt above the league, and kickers are ungraded",
        "  because field goal accuracy does not repeat over 8 games.",
        "",
    ]
    return "\n".join(lines)


def write_backtest(result: BacktestResult, cutoff_season: int) -> Path:
    """Save the grade model JSON, the frozen expectation tables beside it, the long
    evaluation frame and the Markdown report, all under the ignored data directory;
    print every table."""
    path = grade.save_grade_model(result.model, grade.grade_model_path(cutoff_season))
    tables = save_expectation_tables(
        result.expectations, expectation_tables_path(path.parent, cutoff_season)
    )
    log.info(
        "position grade expectation tables from %s written to %s",
        result.expectations.source_seasons,
        tables,
    )
    store.write_processed(
        evaluation_frame(result), *traits_artifact(f"evaluation_{cutoff_season}")
    )
    report = PROCESSED_DIR.joinpath(
        *traits_artifact(f"report_{cutoff_season}")
    ).with_suffix(".md")
    report.write_text(report_markdown(result))
    log.info("position grade report written to %s", report)
    for name in TABLES:
        print(f"\n## {name}\n\n{markdown_table(getattr(result, name))}")
    return path


__all__ = [
    "UNIT_PREFIX",
    "run_backtest",
    "write_backtest",
    "build_season_traits",
    "load_season",
]
