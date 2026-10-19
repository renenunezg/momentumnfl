"""Weekly position trait snapshots, percentiles and split-half reliability.

The input is every builder's trait frame concatenated: one row per player, game
and trait (or per player and week for ``latest`` traits, whose ``game_id`` is
missing). Each snapshot week aggregates the rows of that week and earlier by the
trait's registered aggregate, so a value is what was knowable at the time.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from backend.players import MODEL_VERSION, TRAITS, UNIT_PREFIX

TRAIT_SNAPSHOT_COLUMNS = [
    "season",
    "week",
    "as_of",
    "model_version",
    "athlete_id",
    "athlete_name",
    "headshot_url",
    "team",
    "position_group",
    "games",
    "trait",
    "value",
    "percentile",
    "kind",
    "qualified",
]
RELIABILITY_COLUMNS = ["position_group", "trait", "games", "players", "correlation"]

# A player qualifies at a week once he has played week // 2 games, between one
# and this many: the same ladder as the team model's early-season handling.
QUALIFYING_GAMES = 6
# The fewest players a pooled reliability may rest on; a sample floor.
MIN_QUALIFIED_PLAYERS = 30
FLOOR_QUANTILE = 0.2
FLOOR_MIN_GAMES = 4
UNIT_GROUP = "OL"
KEYS = ["athlete_id", "trait"]


def _trait_attribute(traits: pd.Series, attribute: str) -> pd.Series:
    unknown = set(traits.unique()) - set(TRAITS)
    if unknown:
        raise ValueError(f"unregistered traits: {sorted(unknown)}")
    return traits.map({name: getattr(spec, attribute) for name, spec in TRAITS.items()})


def position_groups(frames: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """``athlete_id, position_group`` for every athlete in ``frames``.

    The roster group comes first. An athlete the roster misses is grouped by the
    traits he carries: each trait row splits one vote evenly across the groups
    the trait is registered for, and the group with the most votes wins. OL unit
    ids are always ``OL``.
    """
    rows = frames[KEYS].astype({"athlete_id": str})
    votes = pd.DataFrame(
        [
            (name, group, 1 / len(spec.position_groups))
            for name, spec in TRAITS.items()
            for group in spec.position_groups
        ],
        columns=["trait", "group", "vote"],
    )
    dominant = (
        rows.merge(votes, on="trait")
        .groupby(["athlete_id", "group"])["vote"]
        .sum()
        .reset_index()
        .sort_values("vote", ascending=False, kind="stable")
        .drop_duplicates("athlete_id")
    )
    roster = (
        positions[["athlete_id", "position_group"]]
        .astype({"athlete_id": str})
        .dropna(subset=["position_group"])
        .drop_duplicates("athlete_id")
    )
    out = pd.DataFrame({"athlete_id": rows["athlete_id"].unique()})
    out = out.merge(roster, on="athlete_id", how="left").merge(
        dominant[["athlete_id", "group"]], on="athlete_id", how="left"
    )
    out["position_group"] = out["position_group"].fillna(out.pop("group"))
    out.loc[out["athlete_id"].str.startswith(UNIT_PREFIX), "position_group"] = (
        UNIT_GROUP
    )
    return out


def in_registered_group(frames: pd.DataFrame, groups: pd.DataFrame) -> pd.Series:
    """Whether each frame row's trait is registered for the athlete's group.

    A kicker who threw a pass on a fake has a passer credit row; it is not an
    ``epa.credit`` row at K, so it is dropped here rather than ranked.
    """
    group = (
        frames["athlete_id"]
        .astype(str)
        .map(groups.set_index("athlete_id")["position_group"])
    )
    registered = pd.DataFrame(
        [(name, g) for name, spec in TRAITS.items() for g in spec.position_groups],
        columns=["trait", "position_group"],
    )
    keys = pd.MultiIndex.from_arrays([frames["trait"], group])
    return pd.Series(
        keys.isin(pd.MultiIndex.from_frame(registered)), index=frames.index
    )


def _ratio(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    sums = rows.groupby(keys)[["numerator", "denominator"]].sum()
    return sums["numerator"] / sums["denominator"].where(sums["denominator"].ne(0))


def _quantile20(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    grouped = rows.groupby(keys)["numerator"]
    return grouped.quantile(FLOOR_QUANTILE).where(grouped.size().ge(FLOOR_MIN_GAMES))


def _latest(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    last = rows.sort_values("model_week", kind="stable").drop_duplicates(
        keys, keep="last"
    )
    return last.set_index(keys)["numerator"]


AGGREGATORS = {
    "rate": _ratio,
    "share_of_games": _ratio,
    "quantile20": _quantile20,
    "latest": _latest,
}


def aggregate(rows: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Each group's value by its trait's aggregate; all-NaN numerators give NaN."""
    index = pd.MultiIndex.from_frame(rows[keys].drop_duplicates())
    scored = rows[rows["numerator"].notna()]
    parts = [
        AGGREGATORS[name](group, keys)
        for name, group in scored.groupby("aggregate")
        if not group.empty
    ]
    values = pd.concat(parts) if parts else pd.Series(dtype=float)
    return values.reindex(index).rename("value")


def snapshot_weeks(games: pd.DataFrame) -> list[tuple[int, int]]:
    """(week, data cutoff) pairs from the first to the last completed week.

    A week without a completed game takes the prior completed week's cutoff, so
    its rows repeat that week's values.
    """
    weeks = sorted(int(week) for week in games["model_week"].dropna().unique())
    completed = sorted(
        int(week)
        for week in games.loc[games["completed"].eq(True), "model_week"].unique()
    )
    if not completed:
        return []
    out, cutoff = [], completed[0]
    for week in weeks:
        if week < completed[0] or week > completed[-1]:
            continue
        if week in completed:
            cutoff = week
        out.append((week, cutoff))
    return out


def _cumulative(rows: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    through = rows[rows["model_week"].le(cutoff)]
    values = aggregate(through, KEYS).reset_index()
    played = (
        through.dropna(subset=["game_id"]).groupby("athlete_id")["game_id"].nunique()
    )
    values["games"] = values["athlete_id"].map(played).fillna(0).astype(int)
    return values


def percentiles(out: pd.DataFrame) -> pd.Series:
    """0 to 100 among qualified players with a value at the group, trait and week."""
    orientation = _trait_attribute(out["trait"], "higher_is_better")
    sign = orientation.map({True: 1.0, False: -1.0})
    ranked = out["qualified"] & out["value"].notna() & sign.notna()
    signed = (out["value"] * sign)[ranked]
    groups = [out.loc[ranked, column] for column in ["week", "trait", "position_group"]]
    rank = signed.groupby(groups).rank(method="average")
    n = signed.groupby(groups).transform("size")
    percentile = (100 * (rank - 1) / (n - 1)).where(n.gt(1), 100.0)
    return percentile.reindex(out.index)


def qualified_at(week: pd.Series, games: pd.Series) -> pd.Series:
    threshold = (week // 2).clip(lower=1, upper=QUALIFYING_GAMES)
    return games.ge(threshold)


def weekly_trait_snapshots(
    frames: pd.DataFrame,
    positions: pd.DataFrame,
    games: pd.DataFrame,
    identity: pd.DataFrame,
    season: int,
    as_of: datetime,
) -> pd.DataFrame:
    """One row per player, snapshot week and trait the player has a row for by then."""
    rows = frames.astype({"athlete_id": str})
    groups = position_groups(rows, positions)
    rows = rows[in_registered_group(rows, groups)]
    rows = rows.assign(aggregate=_trait_attribute(rows["trait"], "aggregate"))
    by_cutoff: dict[int, pd.DataFrame] = {}
    weekly = []
    for week, cutoff in snapshot_weeks(games):
        if cutoff not in by_cutoff:
            by_cutoff[cutoff] = _cumulative(rows, cutoff)
        weekly.append(by_cutoff[cutoff].assign(week=week))
    if not weekly:
        return pd.DataFrame(columns=TRAIT_SNAPSHOT_COLUMNS)
    out = pd.concat(weekly, ignore_index=True)

    people = (
        identity[["athlete_id", "athlete_name", "team", "headshot_url"]]
        .astype({"athlete_id": str})
        .drop_duplicates("athlete_id", keep="last")
    )
    out = out.merge(people, on="athlete_id", how="left")
    out = out.merge(groups, on="athlete_id", how="left")
    out["qualified"] = qualified_at(out["week"], out["games"])
    out["percentile"] = percentiles(out)
    out["kind"] = _trait_attribute(out["trait"], "kind")
    out["season"] = season
    out["as_of"] = as_of.isoformat()
    out["model_version"] = MODEL_VERSION
    out = out.sort_values(["week", "trait", "athlete_id"], ignore_index=True)
    return out[TRAIT_SNAPSHOT_COLUMNS]


def spearman_brown(half: float) -> float:
    """The reliability of a full sample from the correlation of its two halves.

    Two halves of ``n // 2`` games correlate at the reliability of ``n // 2``
    games; ``2r / (1 + r)`` steps that up to ``n`` games.
    """
    if not np.isfinite(half) or half <= -1.0:
        return np.nan
    return float(2.0 * half / (1.0 + half))


def split_half_reliability(
    frames: pd.DataFrame,
    positions: pd.DataFrame,
    games_at: tuple[int, ...] = (4, 8, 12),
) -> pd.DataFrame:
    """A trait's reliability at ``n`` games per group, from its odd and even games.

    ``n`` in ``games_at`` counts the player's total games: a player needs ``n``
    games with a numerator for the trait; his games in week order alternate
    between the halves, and each half aggregates its first ``n // 2`` games.
    ``correlation`` is the halves' correlation stepped up to ``n`` games by
    ``spearman_brown``. ``latest`` traits have no games and are skipped.
    """
    rows = frames.astype({"athlete_id": str})
    groups = position_groups(rows, positions)
    rows = rows[in_registered_group(rows, groups)]
    rows = rows.assign(aggregate=_trait_attribute(rows["trait"], "aggregate"))
    rows = rows[
        rows["numerator"].notna()
        & rows["game_id"].notna()
        & rows["aggregate"].ne("latest")
    ]
    rows = rows.sort_values([*KEYS, "model_week", "game_id"], kind="stable")
    order = rows.groupby(KEYS).cumcount()
    rows = rows.assign(
        half=order % 2,
        slot=order // 2,
        played=rows.groupby(KEYS)["game_id"].transform("size"),
    )

    results = []
    for n in games_at:
        sample = rows[rows["played"].ge(n) & rows["slot"].lt(n // 2)]
        if sample.empty:
            continue
        wide = aggregate(sample, [*KEYS, "half"]).unstack("half").dropna()
        if wide.empty:
            continue
        wide = wide.reset_index().merge(groups, on="athlete_id", how="left")
        for (group, trait), frame in wide.groupby(["position_group", "trait"]):
            if len(frame) < MIN_QUALIFIED_PLAYERS:
                continue
            results.append(
                {
                    "position_group": group,
                    "trait": trait,
                    "games": n,
                    "players": len(frame),
                    "correlation": spearman_brown(float(frame[0].corr(frame[1]))),
                }
            )
    out = pd.DataFrame(results, columns=RELIABILITY_COLUMNS)
    return out.sort_values(["position_group", "trait", "games"], ignore_index=True)
