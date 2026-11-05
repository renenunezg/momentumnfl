"""Player value and trade value: projected wins above replacement priced in cap share.

A player's per-game WAR is shrunk toward his group's development mean, his
previous season counted at a fitted decay, and aged by a fitted linear term each
later season. A win costs the cap the league spends above the minimum per net
win on the groups WAR covers. Worth is the minimum salary plus priced wins; the
trade value is worth less the acquiring team's cash, each later season kept only
while keeping beats cutting him and paying the rest of his guarantees.
"""

import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.awards import ingest
from backend.config import PROCESSED_DIR, RAW_DIR
from backend.players import EPA_TRAIT, war
from backend.players.context import load_games
from backend.players.contracts import (
    PAY_WEEKS,
    acquirer_schedule,
    league_minimum,
    read_contract_seasons,
    season_caps,
)
from backend.players.grade import MODELS_DIR

DISCOUNT_RATE = 0.08
RANGE_Z = 1.2816
PRIME_AGE = 27.0
SEASON_GAMES = 17
ROS_WEEK = 9
ROOKIE_SEASONS = 4
FIFTH_YEAR = 4
FIRST_ROUND_PICKS = 32
OPTION_TOP_PICKS = 10
DRAFT_PICKS = 256
PICK_WINDOW = 16
BOOTSTRAP_DRAWS = 2000
CLEARLY_WORSE = 0.95
UNRATED_POSITIONS = ("LT", "LG", "C", "RG", "RT", "K", "P", "LS")
VALUED_GROUPS = ("QB", "RB", "WR", "TE", "DL", "LB", "DB")
PLAYER_VALUE_ROW, TRADE_VALUE_ROW = "player_value", "trade_value"
TRAITS_DIR = PROCESSED_DIR / "players" / "position_traits"


@dataclass(frozen=True)
class GroupFit:
    k: float
    decay: float
    k_ros: float
    decay_ros: float
    age_a: float
    age_b: float
    mean_rate: float
    resid_sd: float
    games_a: float
    games_b: float
    passes: bool


@dataclass(frozen=True)
class ValueModel:
    last_dev_season: int
    price: float
    groups: dict[str, GroupFit]
    evaluation: dict
    pick_values: list[float]


def season_war(season: int, shrink_games: dict[str, float]) -> pd.DataFrame:
    """``athlete_id, position_group, team, week, games, war`` for every snapshot
    week of a scored season."""
    snapshot = pd.read_parquet(TRAITS_DIR / f"snapshots_{season}.parquet")
    credited = pd.read_parquet(TRAITS_DIR / f"credited_{season}.parquet")
    projections = war.season_projections(
        season, load_games(season), ingest.read("schedules", season)
    )
    scored = war.wins_above_replacement(
        war.shrunk_credit(snapshot, shrink_games), credited, projections
    )
    teams = snapshot.loc[
        snapshot["trait"].eq(EPA_TRAIT), ["week", "athlete_id", "team"]
    ].astype({"athlete_id": str})
    return scored.merge(
        teams.drop_duplicates(["week", "athlete_id"]), on=["week", "athlete_id"]
    ).rename(columns={"wins_above_replacement": "war"})[
        ["athlete_id", "position_group", "team", "week", "games", "war"]
    ]


@cache
def _birth_dates() -> pd.Series:
    players = pd.read_parquet(
        RAW_DIR / "players.parquet", columns=["gsis_id", "birth_date"]
    )
    return pd.to_datetime(
        players.drop_duplicates("gsis_id").set_index("gsis_id")["birth_date"]
    )


def ages(season: int, athlete_ids: pd.Series) -> np.ndarray:
    """Age on September 1 of ``season``; NaN without a birth date."""
    birth = athlete_ids.map(_birth_dates())
    return ((pd.Timestamp(season, 9, 1) - birth).dt.days / 365.25).to_numpy(float)


def _shrunk(fit_k, decay, mean, r0, g0, r1, g1):
    return mean + (g0 * (r0 - mean) + decay * g1 * (r1 - mean)) / (
        fit_k + g0 + decay * g1
    )


def _aging(age_a, age_b, age, years: int) -> np.ndarray:
    """Cumulative aging over ``years`` seasons after the current one, zero
    without an age."""
    age_a, age_b = np.atleast_1d(age_a)[:, None], np.atleast_1d(age_b)[:, None]
    steps = age_a + age_b * (age[:, None] + np.arange(years) - PRIME_AGE)
    steps = np.where(np.isnan(steps), 0.0, steps)
    return np.concatenate([np.zeros((len(age), 1)), steps.cumsum(axis=1)], axis=1)


def _group_params(groups: dict[str, GroupFit]) -> pd.DataFrame:
    return pd.DataFrame({group: asdict(fit) for group, fit in groups.items()}).T.astype(
        float
    )


def projected_rates(
    frame: pd.DataFrame, groups: dict[str, GroupFit], seasons: int
) -> np.ndarray:
    """Per-game WAR, players x ``seasons`` from the rest of the current season on,
    that column with the within-season fit. ``frame`` has ``position_group,
    rate_0, games_0, rate_1, games_1, age``."""
    fit = _group_params(groups).loc[frame["position_group"]]
    inputs = [
        frame[c].to_numpy(float) for c in ["rate_0", "games_0", "rate_1", "games_1"]
    ]
    mean = fit["mean_rate"].to_numpy()
    rest = _shrunk(fit["k_ros"].to_numpy(), fit["decay_ros"].to_numpy(), mean, *inputs)
    later = _shrunk(fit["k"].to_numpy(), fit["decay"].to_numpy(), mean, *inputs)
    aging = _aging(
        fit["age_a"].to_numpy(),
        fit["age_b"].to_numpy(),
        frame["age"].to_numpy(float),
        seasons - 1,
    )
    out = later[:, None] + aging
    out[:, 0] = rest
    return out


def fair_values(
    worth: np.ndarray, cash: np.ndarray, guaranteed: np.ndarray, discount: float
) -> np.ndarray:
    """Present value of each row's contract seasons to an acquiring team, which
    keeps a season only while worth less cash plus what follows beats cutting."""
    factor = 1 / (1 + discount)
    value = np.zeros(len(worth))
    owed = np.zeros(len(worth))
    for t in range(worth.shape[1] - 1, -1, -1):
        owed = guaranteed[:, t] + factor * owed
        value = np.maximum(worth[:, t] - cash[:, t] + factor * value, -owed)
    return value


def _team_games(season: int) -> tuple[pd.Series, pd.Series]:
    """Regular-season games played and remaining per team."""
    games = load_games(season)
    sides = pd.concat(
        [
            games[["home_team", "completed"]].rename(columns={"home_team": "team"}),
            games[["away_team", "completed"]].rename(columns={"away_team": "team"}),
        ]
    )
    played = sides.groupby("team")["completed"].sum()
    return played, sides.groupby("team").size() - played


def _expected_games(
    frame: pd.DataFrame, fit: pd.DataFrame, season: int, seasons: int
) -> np.ndarray:
    """Games per season: the team's remaining games at the player's share so far,
    then the fitted next-season games from the season before."""
    played, remaining = _team_games(season)
    share = (frame["games_0"] / frame["team"].map(played)).clip(upper=1.0).fillna(0.0)
    out = np.empty((len(frame), seasons))
    out[:, 0] = (frame["team"].map(remaining).fillna(0.0) * share).to_numpy()
    full = frame["games_0"].to_numpy(float) + out[:, 0]
    for t in range(1, seasons):
        full = np.clip(
            fit["games_a"].to_numpy() + fit["games_b"].to_numpy() * full,
            0,
            SEASON_GAMES,
        )
        out[:, t] = full
    return out


def score_values(
    current: pd.DataFrame,
    prior: pd.DataFrame,
    model: ValueModel,
    season: int,
    week: int,
) -> pd.DataFrame:
    """``athlete_id, position_group, trait, value, uncertainty`` rows, in
    millions of ``season`` dollars, for players at groups that passed holdout with
    an active contract. ``current`` and ``prior`` are this season's latest and last
    season's final WAR rows."""
    passed = {g: fit for g, fit in model.groups.items() if fit.passes}
    inputs = _inputs(current, prior, season)
    inputs = (
        inputs[inputs["position_group"].isin(passed)]
        .sort_values("games_0", ascending=False, kind="stable")
        .drop_duplicates("athlete_id")
        .reset_index(drop=True)
    )
    pay_weeks_left = max(0, PAY_WEEKS - week)
    schedule = acquirer_schedule(read_contract_seasons(), season, pay_weeks_left)
    schedule = schedule[schedule["gsis_id"].isin(inputs["athlete_id"])]
    if schedule.empty:
        raise ValueError(f"no active contract matches a valued {season} player")
    inputs = inputs[inputs["athlete_id"].isin(schedule["gsis_id"])].reset_index(
        drop=True
    )
    seasons = int(schedule["season"].max()) - season + 1
    years = np.arange(season, season + seasons)
    owed = schedule.pivot(index="gsis_id", columns="season").reindex(
        index=inputs["athlete_id"],
        columns=pd.MultiIndex.from_product([["cash", "guaranteed"], years]),
    )
    in_contract = owed["cash"].notna().to_numpy()
    caps = season_caps(int(years[-1])).loc[years].to_numpy()
    cash = owed["cash"].fillna(0.0).to_numpy() / caps
    guaranteed = owed["guaranteed"].fillna(0.0).to_numpy() / caps
    fit = _group_params(passed).loc[inputs["position_group"]]
    games = _expected_games(inputs, fit, season, seasons)
    minimum = league_minimum(years) / caps
    minimum[0] *= pay_weeks_left / PAY_WEEKS
    discount = (1 + DISCOUNT_RATE) ** -np.arange(seasons)
    rates = projected_rates(inputs, passed, seasons)
    spread = RANGE_Z * fit["resid_sd"].to_numpy()[:, None]
    values = {}
    for name, rate in (
        ("mid", rates),
        ("low", rates - spread),
        ("high", rates + spread),
    ):
        worth = np.where(in_contract, minimum + rate * games * model.price, 0.0)
        values[name] = (
            (worth * discount).sum(axis=1) * caps[0],
            fair_values(worth, cash, guaranteed, DISCOUNT_RATE) * caps[0],
        )
    rows = []
    for i, trait in enumerate((PLAYER_VALUE_ROW, TRADE_VALUE_ROW)):
        rows.append(
            pd.DataFrame(
                {
                    "athlete_id": inputs["athlete_id"],
                    "position_group": inputs["position_group"],
                    "trait": trait,
                    "value": values["mid"][i],
                    "uncertainty": (values["high"][i] - values["low"][i]) / 2,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _end_of_season(weekly: pd.DataFrame) -> pd.DataFrame:
    return weekly[weekly["week"].eq(weekly["week"].max())]


def _inputs(
    current: pd.DataFrame, previous: pd.DataFrame | None, season: int
) -> pd.DataFrame:
    """Projection inputs per player: this season to date and last season."""
    out = current.assign(
        rate_0=current["war"] / current["games"], games_0=current["games"]
    )
    if previous is None:
        out = out.assign(rate_1=0.0, games_1=0.0)
    else:
        prior = previous.assign(
            rate_1=previous["war"] / previous["games"], games_1=previous["games"]
        )
        out = out.merge(
            prior[["athlete_id", "rate_1", "games_1"]], on="athlete_id", how="left"
        )
        out[["rate_1", "games_1"]] = out[["rate_1", "games_1"]].fillna(0.0)
    out = out[out["position_group"].isin(VALUED_GROUPS) & out["games_0"].gt(0)]
    return out.assign(age=ages(season, out["athlete_id"])).reset_index(drop=True)


def _with_target(inputs: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    target = target.assign(
        rate_t=target["war"] / target["games"], games_t=target["games"]
    )
    return inputs.merge(target[["athlete_id", "rate_t", "games_t"]], on="athlete_id")


def _weighted_mse(target, prediction, weight) -> float:
    return float(np.sum(weight * (target - prediction) ** 2) / np.sum(weight))


def _fit_group(
    rows: pd.DataFrame, mean: float, with_aging: bool
) -> tuple[np.ndarray, float]:
    """``k, decay`` (and the aging terms when ``with_aging``) minimizing the
    games-weighted squared error, and the residual sd at the fit."""
    r0, g0, r1, g1, age, target, weight = (
        rows[c].to_numpy(float)
        for c in ["rate_0", "games_0", "rate_1", "games_1", "age", "rate_t", "games_t"]
    )

    # Unit-scaled search space: k in tens of games, aging in thousandths of a win.
    units = np.array([10.0, 1.0, 1e-3, 1e-3])[: 4 if with_aging else 2]

    def predict(x):
        k, decay, *aging = x * units
        shrunk = _shrunk(k, decay, mean, r0, g0, r1, g1)
        return shrunk + _aging(*aging, age, 1)[:, 1] if with_aging else shrunk

    scale = _weighted_mse(target, np.full(len(rows), mean), weight)
    result = minimize(
        lambda x: _weighted_mse(target, predict(x), weight) / scale,
        x0=np.array([1.2, 0.5, 0.0, 0.0])[: len(units)],
        method="Powell",
        bounds=[(0.0, 6.0), (0.0, 1.0), (-20.0, 20.0), (-5.0, 5.0)][: len(units)],
    )
    return result.x * units, float(
        np.sqrt(_weighted_mse(target, predict(result.x), weight))
    )


def _evaluate(rows: pd.DataFrame, prediction: np.ndarray, mean: float) -> dict:
    """Holdout error of the model and both baselines, and the bootstrap share of
    player resamples in which the model is worse than the better baseline."""
    target, weight = rows["rate_t"].to_numpy(float), rows["games_t"].to_numpy(float)
    baselines = {
        "carry": rows["rate_0"].to_numpy(float),
        "mean": np.full(len(rows), mean),
    }
    errors = {name: weight * (target - b) ** 2 for name, b in baselines.items()}
    best = min(errors, key=lambda name: errors[name].sum())
    gap = weight * (target - prediction) ** 2 - errors[best]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(rows), (BOOTSTRAP_DRAWS, len(rows)))
    return {
        "n": len(rows),
        "model": _weighted_mse(target, prediction, weight),
        **{name: float(e.sum() / weight.sum()) for name, e in errors.items()},
        "worse_share": float(np.mean(gap[draws].sum(axis=1) > 0)),
    }


def _rest_of_season(
    weekly: pd.DataFrame, previous: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Inputs at the week ``ROS_WEEK`` snapshot, targeted at the rest of the season."""
    early = weekly[
        weekly["week"].eq(weekly.loc[weekly["week"].le(ROS_WEEK), "week"].max())
    ]
    end = _end_of_season(weekly)
    rows = _inputs(early, previous, season).merge(
        end[["athlete_id", "games", "war"]], on="athlete_id", suffixes=("", "_end")
    )
    rows["games_t"] = rows["games_end"] - rows["games"]
    rows = rows[rows["games_t"].gt(0)]
    return rows.assign(rate_t=(rows["war_end"] - rows["war"]) / rows["games_t"])


def _price(ends: dict[int, pd.DataFrame], contracts: pd.DataFrame, seasons) -> float:
    """Cap share per net win: cap spent above the minimum on players with credited
    games at valued groups, over their net WAR, averaged over ``seasons``."""
    caps = season_caps(max(seasons))
    prices = []
    for season in seasons:
        end = ends[season][ends[season]["position_group"].isin(VALUED_GROUPS)]
        paid = end.merge(
            contracts[contracts["season"].eq(season)],
            left_on="athlete_id",
            right_on="gsis_id",
        )
        spend = paid["cap_number"].sum() - len(paid) * league_minimum([season])[0]
        prices.append(spend / paid["war"].sum() / caps[season])
    return float(np.mean(prices))


def _option_salaries(contracts: pd.DataFrame) -> pd.DataFrame:
    """``top, rest`` fifth-year option salary by season and position, set from the
    season before: the mean of the top ``OPTION_TOP_PICKS`` cap numbers for top
    picks and of the 3rd to 25th for the rest of the first round."""
    rows = contracts.assign(position=contracts["position"].replace("FB", "RB"))
    rank = rows.groupby(["season", "position"])["cap_number"].rank(
        method="first", ascending=False
    )
    by = [rows["season"] + 1, rows["position"]]
    return pd.concat(
        {
            "top": rows["cap_number"][rank.le(OPTION_TOP_PICKS)].groupby(by).mean(),
            "rest": rows["cap_number"][rank.between(3, 25)].groupby(by).mean(),
        },
        axis=1,
    )


def _pick_values(
    ends: dict[int, pd.DataFrame], contracts: pd.DataFrame, price: float
) -> list[float]:
    """Cap share by overall pick: a drafted player's discounted worth less his
    rookie-deal cash over his first ``ROOKIE_SEASONS`` seasons, plus for first-round
    picks the fifth-year option's worth less its salary when his first three
    seasons' worth exceeded it. Each part is averaged per pick over every draft
    with its seasons complete and smoothed over neighbouring picks, the option
    within the first round. Picks at positions WAR does not rate are left out."""
    last = max(ends)
    caps = season_caps(last)
    played = pd.concat(
        [end.groupby("athlete_id")["war"].sum().rename(s) for s, end in ends.items()],
        axis=1,
    )
    wins = played.stack().rename("war")
    wins.index.names = ["gsis_id", "season"]
    drafted = contracts[
        contracts["draft_year"].between(min(ends), last - ROOKIE_SEASONS + 1)
        & contracts["draft_overall"].le(DRAFT_PICKS)
        & ~contracts["position"].isin(UNRATED_POSITIONS)
    ]
    rookie = drafted[
        drafted["season"].sub(drafted["draft_year"]).between(0, ROOKIE_SEASONS - 1)
    ].sort_values(["gsis_id", "season"])
    # Cash past the rookie deal's value is an extension's signing bonus.
    paid = rookie.groupby("gsis_id")["cash_paid"].cumsum()
    paid = np.minimum(paid, rookie["rookie_value"].fillna(np.inf))
    rookie = rookie.join(wins, on=["gsis_id", "season"]).assign(
        cost=paid.groupby(rookie["gsis_id"]).diff().fillna(paid),
        year=rookie["season"] - rookie["draft_year"],
    )
    cap = caps.loc[rookie["season"]].to_numpy()
    rookie["worth"] = (
        league_minimum(rookie["season"]) / cap + rookie["war"].fillna(0.0) * price
    )
    rookie["surplus"] = (rookie["worth"] - rookie["cost"] / cap) * (
        1 + DISCOUNT_RATE
    ) ** -rookie["year"]
    per_pick = rookie.groupby(["gsis_id", "draft_overall"])["surplus"].sum()
    per_pick = per_pick.groupby(level="draft_overall").mean()
    per_pick = per_pick.reindex(range(1, DRAFT_PICKS + 1)).interpolate()
    window = 2 * PICK_WINDOW + 1
    values = per_pick.rolling(window, center=True, min_periods=1).mean()

    first = rookie[
        rookie["draft_overall"].le(FIRST_ROUND_PICKS)
        & rookie["draft_year"].le(last - FIFTH_YEAR)
    ]
    option = first.groupby(["gsis_id", "draft_overall", "draft_year", "position"])[
        "worth"
    ].apply(lambda w: w.iloc[:3].mean())
    option = option.rename("early").reset_index()
    option["season"] = option["draft_year"] + FIFTH_YEAR
    salaries = _option_salaries(contracts)
    keys = pd.MultiIndex.from_arrays(
        [option["season"], option["position"].replace("FB", "RB")]
    )
    salary = np.where(
        option["draft_overall"].le(OPTION_TOP_PICKS),
        salaries["top"].reindex(keys).to_numpy(),
        salaries["rest"].reindex(keys).to_numpy(),
    )
    option_cap = caps.loc[option["season"]].to_numpy()
    fifth_war = wins.reindex(
        pd.MultiIndex.from_frame(option[["gsis_id", "season"]])
    ).fillna(0.0)
    fifth = league_minimum(option["season"]) / option_cap + fifth_war.to_numpy() * price
    cost = salary / option_cap
    option["value"] = (
        np.where(option["early"] > cost, fifth - cost, 0.0)
        * (1 + DISCOUNT_RATE) ** -FIFTH_YEAR
    )
    by_pick = option.groupby("draft_overall")["value"].mean()
    by_pick = by_pick.reindex(range(1, FIRST_ROUND_PICKS + 1)).interpolate()
    by_pick = by_pick.rolling(window, center=True, min_periods=1).mean()
    values = values.add(by_pick, fill_value=0.0)
    return [float(v) for v in values]


def fit_value_model(
    dev: tuple[int, ...], holdout: tuple[int, ...], shrink_games: dict[str, float]
) -> ValueModel:
    seasons = [*dev, *holdout]
    weekly = {s: season_war(s, shrink_games) for s in seasons}
    ends = {s: _end_of_season(frame) for s, frame in weekly.items()}
    pairs = {
        s: _with_target(_inputs(ends[s], ends.get(s - 1), s), ends[s + 1])
        for s in seasons[:-1]
    }
    train = pd.concat([pairs[s] for s in dev if s + 1 in dev], ignore_index=True)
    train_ros = pd.concat(
        [_rest_of_season(weekly[s], ends.get(s - 1), s) for s in dev], ignore_index=True
    )
    test = pd.concat([pairs[s - 1] for s in holdout], ignore_index=True)
    ros = pd.concat(
        [_rest_of_season(weekly[s], ends[s - 1], s) for s in holdout], ignore_index=True
    )
    groups, evaluation = {}, {}
    for group in VALUED_GROUPS:
        rows = train[train["position_group"].eq(group)]
        mean = float(np.average(rows["rate_t"], weights=rows["games_t"]))
        (k, decay, age_a, age_b), resid_sd = _fit_group(rows, mean, with_aging=True)
        (k_ros, decay_ros), _ = _fit_group(
            train_ros[train_ros["position_group"].eq(group)], mean, with_aging=False
        )
        games_b, games_a = np.polyfit(rows["games_0"], rows["games_t"], 1)
        fit = GroupFit(
            *(float(v) for v in (k, decay, k_ros, decay_ros, age_a, age_b)),
            mean,
            resid_sd,
            float(games_a),
            float(games_b),
            passes=False,
        )
        next_rows = test[test["position_group"].eq(group)]
        ros_rows = ros[ros["position_group"].eq(group)]
        checks = {
            "next_season": _evaluate(
                next_rows, projected_rates(next_rows, {group: fit}, 2)[:, 1], mean
            ),
            "rest_of_season": _evaluate(
                ros_rows, projected_rates(ros_rows, {group: fit}, 1)[:, 0], mean
            ),
        }
        passes = all(c["worse_share"] < CLEARLY_WORSE for c in checks.values())
        groups[group] = GroupFit(**{**asdict(fit), "passes": passes})
        evaluation[group] = checks
    contracts = read_contract_seasons()
    price = _price(ends, contracts, dev)
    picks = _pick_values(ends, contracts, price)
    return ValueModel(max(dev), price, groups, evaluation, picks)


def value_model_path(last_dev: int) -> Path:
    return MODELS_DIR / f"player_value_{last_dev}.json"


def write_value_model(model: ValueModel) -> Path:
    """The model JSON and a Markdown report beside it."""
    path = value_model_path(model.last_dev_season)
    path.write_text(json.dumps(asdict(model), indent=1, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(_report(model))
    return path


def _report(model: ValueModel) -> str:
    lines = [
        f"# Player value model, development seasons through {model.last_dev_season}",
        "",
        f"Price of a win: {model.price:.4f} of the cap.",
        "",
        "Holdout error relative to the group mean (model / carry forward, share of"
        " bootstrap resamples where the model is worse than the better baseline;"
        f" a group is valued below {CLEARLY_WORSE} in both):",
        "",
        "| Group | k | decay | k_ros | decay_ros | next season | rest of season "
        "| valued |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group, fit in model.groups.items():
        ratios = [
            f"{c['model'] / c['mean']:.3f} / {c['carry'] / c['mean']:.3f}"
            f" ({c['worse_share']:.2f})"
            for c in model.evaluation[group].values()
        ]
        lines.append(
            f"| {group} | {fit.k:.1f} | {fit.decay:.2f} | {fit.k_ros:.2f} | "
            f"{fit.decay_ros:.2f} | {' | '.join(ratios)} | {fit.passes} |"
        )
    cap = season_caps(model.last_dev_season + 5).iloc[-1]
    picks = np.array(model.pick_values) * cap
    lines += [
        "",
        f"Draft pick value, millions of {model.last_dev_season + 5} cap dollars "
        "(first, middle and last of each block of 32 overall picks):",
        "",
        "| Picks | first | middle | last |",
        "|---|---|---|---|",
    ]
    for first in range(0, DRAFT_PICKS, 32):
        lines.append(
            f"| {first + 1}-{first + 32} | {picks[first]:.1f} | "
            f"{picks[first + 15]:.1f} | {picks[first + 31]:.1f} |"
        )
    return "\n".join(lines) + "\n"


def load_value_model(season: int) -> ValueModel:
    """The value model with the latest development season before ``season``."""
    fitted = sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in MODELS_DIR.glob("player_value_*.json")
    )
    earlier = [last for last in fitted if last < season]
    if not earlier:
        raise FileNotFoundError(
            f"no player value model before {season}; run player-value-fit"
        )
    payload = json.loads(value_model_path(earlier[-1]).read_text())
    payload["groups"] = {g: GroupFit(**fit) for g, fit in payload["groups"].items()}
    return ValueModel(**payload)
