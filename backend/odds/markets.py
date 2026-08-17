"""Flatten Odds API events into per-offer rows and price them against model
projections. Spread cover and push probabilities come from the discrete
key-number margin distribution; totals use the Student-t."""

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from backend.model.market_blend import cover_push_probabilities

REVIEW_EDGE_POINTS = 4.0


def _normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]", "", ascii_value.decode().lower())


def _name_score(left: str, right: str) -> float:
    normalized_left = _normalized_name(left)
    normalized_right = _normalized_name(right)
    if normalized_left.startswith(normalized_right) or normalized_right.startswith(
        normalized_left
    ):
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def match_event(
    commence_time, home_team: str, away_team: str, schedule: pd.DataFrame
) -> tuple[str | None, float]:
    """schedule needs game_id, start_date, and full home_team/away_team
    names (Odds API events carry full names like 'Kansas City Chiefs')."""
    commence = pd.to_datetime(commence_time, utc=True)
    time_difference = (
        pd.to_datetime(schedule["start_date"], utc=True) - commence
    ).abs()
    candidates = schedule[time_difference.le(pd.Timedelta(hours=2))].copy()
    if candidates.empty:
        return None, 0.0
    candidates["match_score"] = candidates.apply(
        lambda game: 0.5
        * (
            _name_score(home_team, game.home_team)
            + _name_score(away_team, game.away_team)
        ),
        axis=1,
    )
    best = candidates.sort_values("match_score", ascending=False).iloc[0]
    if float(best.match_score) < 0.65:
        return None, float(best.match_score)
    return str(best.game_id), float(best.match_score)


def flatten_offers(
    events: list[dict],
    schedule: pd.DataFrame,
    fetched_at,
    execution_eligibility_verified: bool,
) -> pd.DataFrame:
    offer_columns = [
        "game_id",
        "odds_api_event_id",
        "provider_key",
        "provider",
        "market",
        "selection",
        "point",
        "price",
        "provider_last_update",
        "event_link",
        "market_link",
        "bet_link",
        "execution_eligibility_verified",
        "market_fetched_at",
        "match_score",
    ]
    offers = []
    for event in events:
        game_id, match_score = match_event(
            event["commence_time"],
            event["home_team"],
            event["away_team"],
            schedule,
        )
        if game_id is None:
            continue
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                if market_key not in {"spreads", "totals"}:
                    continue
                for outcome in market.get("outcomes") or []:
                    if market_key == "spreads":
                        if outcome.get("name") == event["home_team"]:
                            selection = "home"
                        elif outcome.get("name") == event["away_team"]:
                            selection = "away"
                        else:
                            continue
                    else:
                        selection = str(outcome.get("name", "")).lower()
                        if selection not in {"over", "under"}:
                            continue
                    offers.append(
                        {
                            "game_id": game_id,
                            "odds_api_event_id": event.get("id"),
                            "provider_key": bookmaker.get("key"),
                            "provider": bookmaker.get("title"),
                            "market": market_key,
                            "selection": selection,
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "provider_last_update": market.get("last_update")
                            or bookmaker.get("last_update"),
                            "event_link": bookmaker.get("link"),
                            "market_link": market.get("link"),
                            "bet_link": outcome.get("link"),
                            "execution_eligibility_verified": (
                                execution_eligibility_verified
                            ),
                            "market_fetched_at": fetched_at,
                            "match_score": match_score,
                        }
                    )
    frame = pd.DataFrame(offers, columns=offer_columns)
    for column in ("point", "price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _american_profit(price: float) -> float:
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def _fair_american(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    if probability >= 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def compare_priced_offers(
    projections: pd.DataFrame,
    offers: pd.DataFrame,
    margin_distribution: np.ndarray,
) -> pd.DataFrame:
    """margin_distribution is the discrete residual distribution from
    market_blend.fit_margin_residual_distribution (backtest residuals)."""
    rows = []
    for projection in projections.itertuples():
        game_offers = offers[offers["game_id"].eq(projection.game_id)].dropna(
            subset=["point", "price"]
        )
        eligible = game_offers[
            game_offers["execution_eligibility_verified"].astype(bool)
        ]
        executable = not eligible.empty
        candidate_offers = eligible if executable else game_offers
        total_scale = projection.total_sd * np.sqrt(
            (projection.degrees_of_freedom - 2.0)
            / projection.degrees_of_freedom
        )
        candidates = []
        for offer in candidate_offers.itertuples():
            if offer.price == 0:
                continue
            push_probability = 0.0
            if offer.market == "spreads" and offer.selection == "home":
                edge = projection.home_margin + offer.point
                selection = projection.home_team
                probability, push_probability = cover_push_probabilities(
                    projection.home_margin, float(offer.point),
                    margin_distribution,
                )
            elif offer.market == "spreads" and offer.selection == "away":
                edge = -projection.home_margin + offer.point
                selection = projection.away_team
                home_cover, push_probability = cover_push_probabilities(
                    projection.home_margin, -float(offer.point),
                    margin_distribution,
                )
                probability = 1.0 - home_cover - push_probability
            elif offer.market == "totals" and offer.selection == "over":
                edge = projection.model_total - offer.point
                selection = "Over"
                probability = float(student_t.cdf(
                    edge / total_scale, projection.degrees_of_freedom
                ))
            elif offer.market == "totals" and offer.selection == "under":
                edge = offer.point - projection.model_total
                selection = "Under"
                probability = float(student_t.cdf(
                    edge / total_scale, projection.degrees_of_freedom
                ))
            else:
                continue
            loss_probability = 1.0 - probability - push_probability
            expected_value = (
                probability * _american_profit(float(offer.price))
                - loss_probability
            )
            uncertainty = (
                projection.margin_sd
                if offer.market == "spreads"
                else projection.total_sd
            )
            candidates.append(
                {
                    "market": offer.market,
                    "selection": selection,
                    "point": float(offer.point),
                    "price": float(offer.price),
                    "provider": offer.provider,
                    "provider_key": offer.provider_key,
                    "provider_last_update": offer.provider_last_update,
                    "event_link": offer.event_link,
                    "market_link": offer.market_link,
                    "bet_link": offer.bet_link,
                    "edge_points": edge,
                    "edge_standardized": edge / uncertainty,
                    "model_cover_probability": probability,
                    "model_fair_price": _fair_american(
                        probability / max(probability + loss_probability, 1e-9)
                    ),
                    "expected_value_per_unit": expected_value,
                }
            )
        row = {
            "game_id": projection.game_id,
            "start_date": projection.start_date,
            "away_team": projection.away_team,
            "home_team": projection.home_team,
            "model_home_spread": projection.home_spread,
            "model_total": projection.model_total,
            "margin_sd": projection.margin_sd,
            "total_sd": projection.total_sd,
            "model_as_of": projection.as_of,
            "market_available": not game_offers.empty,
            "priced_offer_available": bool(candidates),
            "executable_offer_available": executable,
        }
        if candidates:
            best = max(
                candidates, key=lambda item: item["expected_value_per_unit"]
            )
            row.update(
                {f"best_offer_{key}": value for key, value in best.items()}
            )
            row["review_status"] = (
                "requires_current_source_review"
                if best["edge_points"] >= REVIEW_EDGE_POINTS
                and best["expected_value_per_unit"] > 0
                else "below_material_review_threshold"
            )
        else:
            row["review_status"] = "no_priced_offer"
        row["recommendation_status"] = "not_recommended"
        rows.append(row)
    comparisons = pd.DataFrame(rows)
    if "best_offer_expected_value_per_unit" not in comparisons:
        comparisons["best_offer_expected_value_per_unit"] = np.nan
    return comparisons.sort_values(
        "best_offer_expected_value_per_unit",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )
