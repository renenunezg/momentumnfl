"""Pregame decisions and settlement at the exact recommended line and price."""

import json
from datetime import UTC, datetime, timedelta
from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t as student_t

from backend.model.distributions import student_t_scale
from backend.model.market_blend import cover_push_probabilities
from backend.odds.markets import ODDS_SOURCES, _american_profit

POLICY_VERSION = "nfl-picks-v6"
# Dispersion of actual results around the priced (blended) line, from the
# chronological calibration replay of 2019 through 2025 (1,960 games with a
# closing spread and total). The engine's predictive spread carries rating
# uncertainty the market blend has already removed and overstated every
# underdog's win probability; the residual around the blend is the honest
# width, and it does not vary with the engine's per-game spread.
PRICING_MARGIN_SD = 12.82
PRICING_TOTAL_SD = 13.28
# Weights of the model in the numbers that price picks: the devigged market
# number at decision time moved this far toward the pure margin (spreads and
# moneylines) or the model total (totals). Least-squares slope of the result
# minus the closing line on the model minus the closing line, without an
# intercept as priced, on the 2016 through 2021 development seasons (1,622
# games): margin 0.045 (se 0.11), total -0.15 (se 0.10), held at zero. The
# 2022 through 2025 seasons agree (margin -0.09, total -0.01), and on 2026
# decision-time lines the close moved 0.05 (se 0.04) toward the pure margin
# and -0.05 (se 0.03) toward the model total. No segment (division, rest, QB
# absence, weather, season phase, spread size) survives a held-out check.
# Pricing from the published 0.5 blend instead made every 2-point edge a
# 5-point model disagreement with the market, which the model does not predict.
SIDE_MODEL_WEIGHT = 0.045
TOTAL_MODEL_WEIGHT = 0.0
# Picks per model week. Offers with positive expected value against the
# devigged market are always picks (POSITIVE_EV); when fewer than SLATE_SIZE
# games have a pick, the highest-EV sides fill the slate as RANKED_LEAN, one
# per game. Leans carry their honest win probability, near a coin flip, and
# negative expected value at most prices: a product volume choice, recorded
# under its own reason so the ledger grades the two tiers apart. Totals are
# never leans because the model total carries no weight.
SLATE_SIZE = 5
POSITIVE_EV = "positive_ev"
RANKED_LEAN = "ranked_lean"
LEAN_MARKETS = ("h2h", "spreads")
EDGE_SEARCH_POINTS = 200.0
MAX_OFFER_AGE = timedelta(hours=1)
MAX_FORECAST_AGE = timedelta(days=7)
MARKETS = ("h2h", "spreads", "totals")
RECOMMENDATION_COLUMNS = [
    "game_id",
    "market",
    "season",
    "week",
    "start_date",
    "home_team",
    "away_team",
    "model_version",
    "forecast_as_of",
    "home_missing_input_count",
    "away_missing_input_count",
    "policy_version",
    "decision_at",
    "status",
    "reason",
    "selection",
    "side",
    "point",
    "price",
    "provider",
    "provider_key",
    "market_fetched_at",
    "odds_source",
    "odds_api_event_id",
    "provider_start_date",
    "provider_last_update",
    "match_score",
    "execution_eligibility_verified",
    "win_probability",
    "push_probability",
    "probability_edge",
    "edge_points",
    "expected_value_per_unit",
    "stake_units",
    "model_home_margin",
    "model_total",
    "market_total",
    "margin_sd",
    "total_sd",
    "degrees_of_freedom",
    "source_timestamps",
    "data_flags",
    "pricing_weights",
]
SETTLEMENT_COLUMNS = [
    "game_id",
    "market",
    "decision_at",
    "outcome",
    "home_points",
    "away_points",
    "profit_units",
    "graded_at",
    "settlement_reason",
    "result_source_at",
]


def marginal_cdf(value, sd, df):
    return student_t.cdf(value / student_t_scale(sd, df), df)


def _timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _probabilities(projection, offer, distribution):
    """NFL probabilities, retaining key-number mass and returned ties, from
    the margin and total on the (market-anchored) projection."""
    if offer["market"] in ("spreads", "h2h"):
        point = 0.0 if offer["market"] == "h2h" else offer["point"]
        home, push = cover_push_probabilities(
            projection.home_margin,
            point if offer["side"] == "home" else -point,
            distribution,
            projection.margin_sd,
            projection.degrees_of_freedom,
        )
        win = home if offer["side"] == "home" else 1.0 - home - push
        return win, push, 1.0 - win - push
    threshold = offer["point"]
    mean, sd, df = (
        projection.model_total,
        projection.total_sd,
        projection.degrees_of_freedom,
    )
    win = float(marginal_cdf(mean - floor(threshold) - 0.5, sd, df))
    loss = float(marginal_cdf(ceil(threshold) - 0.5 - mean, sd, df))
    push = max(0.0, 1.0 - win - loss) if threshold == floor(threshold) else 0.0
    loss = 1.0 - win - push
    if offer["side"] == "under":
        win, loss = loss, win
    return win, push, loss


def _market_center(projection, market, candidates, distribution):
    """Median across books of the mean that reproduces each book's devigged
    two-way price, so the vig and price differences inform the market number,
    not only the posted points. NaN without a paired quote."""
    first = "over" if market == "totals" else "home"
    field = "model_total" if market == "totals" else "home_margin"
    means = []
    for offer in candidates:
        if offer["market"] != market or offer["side"] != first:
            continue
        other_point = -offer["point"] if market == "spreads" else offer["point"]
        other = next(
            (
                o
                for o in candidates
                if o["market"] == market
                and o["side"] != first
                and o["provider_key"] == offer["provider_key"]
                and o["point"] == other_point
                and o["provider_last_update"] == offer["provider_last_update"]
            ),
            None,
        )
        if other is None:
            continue
        mine, theirs = (1 / (_american_profit(o["price"]) + 1) for o in (offer, other))
        fair = mine / (mine + theirs)
        base = {"totals": offer["point"], "spreads": -(offer["point"] or 0.0)}.get(
            market, 0.0
        )

        def gap(mean, offer=offer, fair=fair):
            win, _, loss = _probabilities(
                projection._replace(**{field: mean}), offer, distribution
            )
            return win / (win + loss) - fair

        try:
            means.append(brentq(gap, base - 60.0, base + 60.0, xtol=1e-6))
        except ValueError:
            continue
    return float(np.median(means)) if means else float("nan")


def _anchored_projection(projection, market, candidates, distribution):
    """The projection re-anchored on the decision-time market: the devigged
    market number moved SIDE_MODEL_WEIGHT toward the pure margin (sides) or
    TOTAL_MODEL_WEIGHT toward the model total (totals).

    Returns the projection and the market number. The priced field is NaN
    without a paired quote or a model number, so a market is never priced
    from the published blend or the model alone.
    """
    center = _market_center(projection, market, candidates, distribution)
    if market == "totals":
        total = center + TOTAL_MODEL_WEIGHT * (projection.model_total - center)
        return projection._replace(model_total=total), center
    pure = getattr(projection, "pure_home_margin", None)
    pure = np.nan if pure is None or pd.isna(pure) else float(pure)
    margin = center + SIDE_MODEL_WEIGHT * (pure - center)
    return projection._replace(home_margin=margin), center


def _edge_points(projection, offer, distribution, implied):
    """Points the priced line must move against the pick before its no-push
    win probability falls to the price's break-even probability.

    Positive favours the pick. The same shift is required of a favourite and
    an underdog, unlike a probability gap, which is largest where the density
    is highest and is then multiplied by the payout in expected value.
    """
    totals = offer["market"] == "totals"
    field = "model_total" if totals else "home_margin"
    mean = getattr(projection, field)
    direction = 1.0 if offer["side"] in ("home", "over") else -1.0

    def gap(delta):
        shifted = projection._replace(**{field: mean - direction * delta})
        win, push, loss = _probabilities(shifted, offer, distribution)
        return win / (win + loss) - implied

    try:
        return float(brentq(gap, -EDGE_SEARCH_POINTS, EDGE_SEARCH_POINTS))
    except ValueError:
        return float("nan")


def _consensus_total(game_offers):
    """Median posted total across a game's offers, or NaN without one."""
    if game_offers.empty or "market" not in game_offers:
        return float("nan")
    points = pd.to_numeric(
        game_offers.loc[game_offers["market"].eq("totals"), "point"], errors="coerce"
    ).dropna()
    return float(points.median()) if len(points) else float("nan")


def priced_candidates(projection, offers):
    candidates = []
    for offer in offers.to_dict("records"):
        market, side = offer.get("market"), offer.get("selection")
        if market not in MARKETS or side not in (
            ("over", "under") if market == "totals" else ("home", "away")
        ):
            continue
        price, point = offer.get("price"), offer.get("point")
        if pd.isna(price) or not np.isfinite(price) or abs(price) < 100:
            continue
        if market != "h2h" and (
            pd.isna(point)
            or not np.isfinite(point)
            or point * 2 != round(point * 2)
            or (market == "spreads" and abs(point) >= 100)
            or (market == "totals" and point < 0)
        ):
            continue
        offer.update(
            side=side,
            point=None if market == "h2h" else point,
            selection=getattr(projection, f"{side}_team")
            if side in ("home", "away")
            else side.title(),
        )
        # Old archives without exact kickoff provenance fail closed.
        for key in (
            "provider_start_date",
            "provider_last_update",
            "provider_key",
            "provider",
            "market_fetched_at",
            "odds_source",
            "odds_api_event_id",
            "match_score",
        ):
            offer.setdefault(key, None)
        offer["execution_eligibility_verified"] = (
            offer.get("execution_eligibility_verified") is True
        )
        candidates.append(offer)
    return candidates


def _offer_reason(offer, paired, now, start):
    fetched = _timestamp(offer["market_fetched_at"])
    updated = _timestamp(offer["provider_last_update"])
    # NFL requires the explicitly configured bookmaker availability signal.
    if offer.get("execution_eligibility_verified") is not True:
        return "unverified_book_availability"
    event_id = offer["odds_api_event_id"]
    if pd.isna(event_id) or not event_id or offer["odds_source"] not in ODDS_SOURCES:
        return "unverified_source"
    provider_start = _timestamp(offer["provider_start_date"])
    if pd.isna(provider_start) or provider_start != start:
        return "kickoff_mismatch"
    if any(
        pd.isna(offer[key]) or not offer[key] for key in ("provider_key", "provider")
    ):
        return "missing_provider"
    if (
        pd.isna(offer["match_score"])
        or not np.isfinite(offer["match_score"])
        or not 0.95 <= offer["match_score"] <= 1
    ):
        return "uncertain_game_match"
    if pd.isna(fetched) or pd.isna(updated):
        return "missing_price_timestamp"
    if fetched >= start or updated >= start:
        return "in_play_offer"
    if not (updated <= fetched <= now) or now - updated > MAX_OFFER_AGE:
        return "stale_price"
    opposite = {"home": "away", "away": "home", "over": "under", "under": "over"}[
        offer["side"]
    ]
    other_point = -offer["point"] if offer["market"] == "spreads" else offer["point"]
    if not any(
        other["provider_key"] == offer["provider_key"]
        and other["odds_api_event_id"] == offer["odds_api_event_id"]
        and other["provider_start_date"] == offer["provider_start_date"]
        and other["side"] == opposite
        and other["point"] == other_point
        and other["provider_last_update"] == offer["provider_last_update"]
        and other["market_fetched_at"] == offer["market_fetched_at"]
        for other in paired
    ):
        return "unpaired_market"
    return None


def build_recommendations(
    projections, offers, distribution, *, decision_at=None, picked_games=()
):
    """The best eligible offer per game and market, recommended or No Play.

    Every market is priced from the devigged market number at decision time
    moved the evidence-fit model weight toward the model
    (_anchored_projection), not from the published blend, whose model share
    the model has not earned against the market. Each row records the median
    posted total. Everything is priced with the empirical dispersion around
    the priced line, not the engine's wider predictive spread. The best offer
    per market is the one with the highest expected value. Offers with
    positive expected value are picks; the slate is then filled to SLATE_SIZE
    games, counting picked_games (games already holding a frozen pick), with
    the highest-EV sides as ranked leans. Historical calibration is
    diagnostic and never gates forward recommendations or replaces their
    probabilities. Stakes are always one unit, with no compounding.
    """
    now = _timestamp(decision_at or datetime.now(UTC))
    if not offers.empty and "odds_source" not in offers:
        raise ValueError("Market offers predate odds_source; rerun the odds step")
    groups = (
        {game_id: group for game_id, group in offers.groupby("game_id")}
        if not offers.empty
        else {}
    )
    rows, model_sides = [], []
    for projection in projections.itertuples():
        start, forecast = (
            _timestamp(projection.start_date),
            _timestamp(projection.as_of),
        )
        reason = None
        if pd.isna(start) or pd.isna(forecast) or not forecast <= now < start:
            reason = "not_pregame"
        elif now - forecast > MAX_FORECAST_AGE:
            reason = "stale_forecast"
        elif (
            not np.isfinite(
                [
                    projection.home_margin,
                    projection.model_total,
                    projection.margin_sd,
                    projection.total_sd,
                    projection.degrees_of_freedom,
                ]
            ).all()
            or min(projection.margin_sd, projection.total_sd) <= 0
            or projection.degrees_of_freedom <= 2
        ):
            reason = "invalid_model_distribution"
        elif any(
            getattr(projection, f"{side}_missing_input_count", None) != 0
            for side in ("home", "away")
        ):
            reason = "missing_model_inputs"
        else:
            from backend.source_inputs import source_reason

            reason = source_reason(
                getattr(projection, "source_timestamps", None), forecast, now
            )
            flags = getattr(projection, "data_flags", None)
            flags = json.loads(flags) if isinstance(flags, str) else (flags or {})
            for side in ("home", "away"):
                qb_at = _timestamp(flags.get(f"{side}_qb_source_at"))
                if (
                    not flags.get(f"{side}_expected_qb")
                    or pd.isna(qb_at)
                    or not (now - timedelta(hours=48) <= qb_at <= forecast)
                ):
                    reason = reason or "stale_or_missing_expected_qb"
        # Do not create late decisions, including late No Plays.
        if pd.isna(start) or pd.isna(forecast) or not forecast <= now < start:
            continue
        game_offers = groups.get(projection.game_id, offers.iloc[:0])
        candidates = priced_candidates(projection, game_offers)
        market_total = _consensus_total(game_offers)
        priced_projection = projection._replace(
            margin_sd=PRICING_MARGIN_SD, total_sd=PRICING_TOTAL_SD
        )
        for market in MARKETS:
            field = "model_total" if market == "totals" else "home_margin"
            market_projection, center = _anchored_projection(
                priced_projection,
                market,
                [c for c in candidates if c["market"] == market],
                distribution,
            )
            lean = getattr(market_projection, field) - center
            if not np.isfinite(lean):
                market_projection = None
            # The side the model leans to; a ranked lean must take it, never
            # the other side for its price alone.
            model_sides.append(
                ("over" if market == "totals" else "home")
                if lean > 0
                else ("under" if market == "totals" else "away")
                if lean < 0
                else None
            )
            row = {
                key: getattr(projection, key, None)
                for key in (
                    "game_id",
                    "season",
                    "week",
                    "start_date",
                    "home_team",
                    "away_team",
                    "model_version",
                    "home_missing_input_count",
                    "away_missing_input_count",
                    "model_total",
                    "degrees_of_freedom",
                    "source_timestamps",
                    "data_flags",
                )
            }
            row.update(
                market=market,
                forecast_as_of=forecast,
                decision_at=now,
                policy_version=POLICY_VERSION,
                status="no_play",
                reason=reason,
                stake_units=0.0,
                execution_eligibility_verified=False,
                # The margin and total this market was actually priced from.
                model_home_margin=(
                    projection.home_margin
                    if market_projection is None or market == "totals"
                    else market_projection.home_margin
                ),
                model_total=(
                    market_projection.model_total
                    if market_projection is not None and market == "totals"
                    else getattr(projection, "market_informed_total", None)
                ),
                market_total=market_total if np.isfinite(market_total) else None,
                margin_sd=priced_projection.margin_sd,
                total_sd=priced_projection.total_sd,
                pricing_weights=list(map(float, distribution)),
            )
            priced = [c for c in candidates if c["market"] == market]
            empty_reason = reason or "no_valid_price"
            if reason == "invalid_model_distribution":
                priced = []
            if market_projection is None:
                priced = []
                empty_reason = reason or "missing_market_line"
            evaluated = []
            for candidate in priced:
                if candidate["market"] != "h2h" and candidate["point"] * 2 != round(
                    candidate["point"] * 2
                ):
                    continue
                win, push, loss = _probabilities(
                    market_projection, candidate, distribution
                )
                profit = _american_profit(candidate["price"])
                implied = 1 / (profit + 1)
                edge = win / (win + loss) - implied if win + loss > 0 else float("nan")
                ev = win * profit - loss
                block = reason or _offer_reason(candidate, priced, now, start)
                if (
                    not np.isfinite([win, push, loss, edge, ev]).all()
                    or not 0 < win < 1
                    or loss <= 0
                ):
                    block, ev = "invalid_probability", float("-inf")
                evaluated.append((block, implied, candidate, win, push, edge, ev))
            # A blocked offer must never hide an eligible offer with less value.
            if evaluated:
                block, implied, best, win, push, edge, ev = max(
                    evaluated,
                    key=lambda item: (
                        item[0] is None,
                        item[6],
                        str(item[2]["provider_key"]),
                    ),
                )
                points = _edge_points(market_projection, best, distribution, implied)
                if block is None and not np.isfinite(points):
                    block = "invalid_probability"
                for key in (
                    "selection",
                    "side",
                    "point",
                    "price",
                    "provider",
                    "provider_key",
                    "market_fetched_at",
                    "odds_source",
                    "odds_api_event_id",
                    "provider_start_date",
                    "provider_last_update",
                    "match_score",
                    "execution_eligibility_verified",
                ):
                    row[key] = best[key]
                row.update(
                    win_probability=win,
                    push_probability=push,
                    probability_edge=edge,
                    edge_points=points,
                    expected_value_per_unit=ev,
                    reason=block or "outside_slate",
                )
            else:
                row["reason"] = empty_reason
            rows.append(row)
    decisions = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    eligible = decisions["reason"].eq("outside_slate")
    value = eligible & decisions["expected_value_per_unit"].gt(0)
    picked = set(picked_games) | set(decisions.loc[value, "game_id"])
    leans = (
        decisions[
            eligible
            & ~value
            & decisions["market"].isin(LEAN_MARKETS)
            & decisions["side"].eq(pd.Series(model_sides, index=decisions.index))
            & ~decisions["game_id"].isin(picked)
        ]
        .sort_values("expected_value_per_unit", ascending=False)
        .drop_duplicates("game_id")
        .index[: max(SLATE_SIZE - len(picked), 0)]
    )
    for index, label in ((decisions.index[value], POSITIVE_EV), (leans, RANKED_LEAN)):
        decisions.loc[index, ["status", "reason", "stake_units"]] = [
            "recommended",
            label,
            1.0,
        ]
    return decisions


def grade_recommendations(recommendations, games, *, graded_at=None):
    """Settle recorded picks only; never infer a past recommendation from EV."""
    now = _timestamp(graded_at or datetime.now(UTC))
    schedule = games.set_index("game_id")
    rows = []
    for pick in recommendations.itertuples():
        if pick.outcome != "pending" or pick.game_id not in schedule.index:
            continue
        game = schedule.loc[pick.game_id]
        if game["home_team"] != pick.home_team or game["away_team"] != pick.away_team:
            raise ValueError(f"game {pick.game_id}: schedule identity changed")
        current_start = _timestamp(game["start_date"])
        result = None
        state = game.get("game_status", "scheduled")
        if state in ("canceled", "cancelled", "postponed"):
            result = "void"
        elif pd.isna(current_start):
            continue
        # A changed kickoff voids the original price contract. Do not reuse a
        # recommendation on a postponed or rescheduled fixture.
        if result == "void" or current_start != _timestamp(pick.start_date):
            result = "void"
        elif (
            now < _timestamp(pick.start_date)
            or pd.isna(game["completed"])
            or not bool(game["completed"])
            or pd.isna(game["home_points"])
            or pd.isna(game["away_points"])
        ):
            continue
        if pick.status != "recommended":
            if now < _timestamp(pick.start_date):
                continue
            result = "no_play"
        if result is None:
            scores = [game["home_points"], game["away_points"]]
            if any(not np.isfinite(s) or s < 0 or s != int(s) for s in scores):
                raise ValueError(f"game {pick.game_id}: invalid final score")
            margin = float(game["home_points"] - game["away_points"])
            total = float(game["home_points"] + game["away_points"])
            balance = (
                (margin if pick.side == "home" else -margin)
                if pick.market == "h2h"
                else (margin if pick.side == "home" else -margin) + pick.point
                if pick.market == "spreads"
                else (total - pick.point) * (1 if pick.side == "over" else -1)
            )
            result = "win" if balance > 0 else "loss" if balance < 0 else "push"
            if pick.market == "h2h" and balance == 0:
                result = "void"
        profit = (
            pick.stake_units * _american_profit(pick.price)
            if result == "win"
            else -pick.stake_units
            if result == "loss"
            else 0.0
        )
        rows.append(
            dict(
                game_id=pick.game_id,
                market=pick.market,
                decision_at=pick.decision_at,
                outcome=result,
                home_points=game["home_points"],
                away_points=game["away_points"],
                profit_units=profit,
                graded_at=now,
                settlement_reason=(
                    "schedule_change"
                    if result == "void"
                    and (
                        state in ("canceled", "cancelled", "postponed")
                        or current_start != _timestamp(pick.start_date)
                    )
                    else "moneyline_tie"
                    if result == "void"
                    else "confirmed_final"
                ),
                result_source_at=game.get("source_fetched_at"),
            )
        )
    return pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
