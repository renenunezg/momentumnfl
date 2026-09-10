"""Pregame decisions and settlement at the exact recommended line and price."""

import json
from datetime import UTC, datetime, timedelta
from math import ceil, floor

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t as student_t

from backend.model.distributions import student_t_scale
from backend.model.market_blend import blend_margin, cover_push_probabilities
from backend.model.projections import DEFAULT_MARKET_WEIGHT
from backend.odds.markets import _american_profit

POLICY_VERSION = "nfl-picks-v4"
# Minimum points the priced line must sit beyond the offered price's
# break-even line. Measured in margin or total points, the same yardstick for
# favourites and underdogs, so a mispriced tail cannot clear the gate on one
# side only. A versioned starting policy, not a fit to live-season outcomes.
MIN_EDGE_POINTS = 2.0
# Dispersion of actual results around the priced (blended) line, from the
# chronological calibration replay of 2019 through 2025 (1,960 games with a
# closing spread and total). The engine's predictive spread carries rating
# uncertainty the market blend has already removed and overstated every
# underdog's win probability; the residual around the blend is the honest
# width, and it does not vary with the engine's per-game spread.
PRICING_MARGIN_SD = 12.82
PRICING_TOTAL_SD = 13.28
# Weight of the pure model in the margin that prices a moneyline. On the
# 2019 through 2025 replay (1,954 decided games) the pure model carries no
# win/loss information beyond the pre-decision market line (logistic slope
# per 10 points: pure -0.05 with se 0.17, market +1.50 with se 0.16), and
# moneyline picks priced off the 0.5 blend won at the market-fair rate, not
# the priced one (0.300 actual, 0.294 market-fair, 0.401 priced, 217 picks).
# No curve shape beat the key-number Student-t out of sample, so only the
# location changes. 0.2 matches the CFB policy and is the most model opinion
# the evidence tolerates; spreads and totals keep the published blend.
H2H_MODEL_WEIGHT = 0.2
# Minimum recommended picks per decision batch (one model week). When the
# edge gate leaves fewer, the highest-edge positive-EV offers are promoted
# and recorded with FLOOR_REASON so the ledger separates gate picks from
# floor picks. A product volume choice, not a calibration result.
VOLUME_FLOOR = 5
FLOOR_REASON = "volume_floor"
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
    """Blended NFL probabilities, retaining key-number mass and returned ties.

    Sides use the published margin (pure model shrunk toward the pre-decision
    market line). Totals use the model total shrunk toward the median posted
    total across the decision-time offers, supplied on the projection.
    """
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


def _h2h_projection(projection):
    """The projection re-anchored for moneyline pricing: the pre-decision
    market margin moved H2H_MODEL_WEIGHT toward the pure model.

    Returns None without a posted spread or pure margin, so a moneyline is
    never priced from the blend or the pure model alone.
    """
    spread = getattr(projection, "market_home_spread", None)
    pure = getattr(projection, "pure_home_margin", None)
    if any(v is None or pd.isna(v) or not np.isfinite(v) for v in (spread, pure)):
        return None
    market = -float(spread)
    return projection._replace(
        home_margin=market + H2H_MODEL_WEIGHT * (float(pure) - market)
    )


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
    if pd.isna(event_id) or not event_id:
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


def build_recommendations(projections, offers, distribution, *, decision_at=None):
    """One best eligible side per game and market, or an explicit No Play.

    Sides use the blended (published) margin and totals the model total blended
    toward the median posted total, so every edge is measured after shrinking
    toward the market being bet into. Each pick records that market total.
    Moneylines are priced from the pre-decision market margin moved
    H2H_MODEL_WEIGHT toward the pure model. Everything is priced with the
    empirical dispersion around the priced line, not the engine's wider
    predictive spread. An offer qualifies when the priced line sits at least
    MIN_EDGE_POINTS beyond the price's break-even line and expected value is
    positive; the best offer per market is the one with the most points of
    edge, never the largest payout. If fewer than VOLUME_FLOOR picks qualify
    in the batch, the highest-edge positive-EV offers are promoted with
    FLOOR_REASON. Historical calibration is diagnostic and never gates
    forward recommendations or replaces their probabilities. Stakes are
    always one unit, with no compounding.
    """
    now = _timestamp(decision_at or datetime.now(UTC))
    groups = (
        {game_id: group for game_id, group in offers.groupby("game_id")}
        if not offers.empty
        else {}
    )
    rows = []
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
        if np.isfinite(market_total) and np.isfinite(projection.model_total):
            priced_projection = priced_projection._replace(
                model_total=blend_margin(
                    projection.model_total, market_total, DEFAULT_MARKET_WEIGHT
                )
            )
        h2h_projection = _h2h_projection(priced_projection)
        for market in MARKETS:
            market_projection = h2h_projection if market == "h2h" else priced_projection
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
                # The margin this market was actually priced from.
                model_home_margin=(
                    projection.home_margin
                    if market_projection is None
                    else market_projection.home_margin
                ),
                model_total=priced_projection.model_total,
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
                empty_reason = reason or "missing_market_spread"
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
                points = (
                    _edge_points(market_projection, candidate, distribution, implied)
                    if win + loss > 0
                    else float("nan")
                )
                block = reason or _offer_reason(candidate, priced, now, start)
                if (
                    not np.isfinite([win, push, loss, edge, ev, points]).all()
                    or not 0 < win < 1
                    or loss <= 0
                ):
                    block = "invalid_probability"
                if block is None and (points < MIN_EDGE_POINTS or ev <= 0):
                    block = "below_edge_threshold"
                evaluated.append((block, points, candidate, win, push, edge, ev))
            # A blocked offer must never hide a qualifying offer with less edge.
            if evaluated:
                block, points, best, win, push, edge, ev = max(
                    evaluated,
                    key=lambda item: (
                        item[0] is None,
                        item[1],
                        str(item[2]["provider_key"]),
                    ),
                )
                for key in (
                    "selection",
                    "side",
                    "point",
                    "price",
                    "provider",
                    "provider_key",
                    "market_fetched_at",
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
                    status="recommended" if block is None else "no_play",
                    reason=block or "qualifying_edge",
                    stake_units=1.0 if block is None else 0.0,
                )
            else:
                row["reason"] = empty_reason
            rows.append(row)
    decisions = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    shortfall = VOLUME_FLOOR - int(decisions["status"].eq("recommended").sum())
    if shortfall > 0:
        eligible = decisions[
            decisions["reason"].eq("below_edge_threshold")
            & decisions["edge_points"].gt(0)
            & decisions["expected_value_per_unit"].gt(0)
        ]
        promote = eligible.sort_values("edge_points", ascending=False).index[:shortfall]
        decisions.loc[promote, ["status", "reason", "stake_units"]] = [
            "recommended",
            FLOOR_REASON,
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
