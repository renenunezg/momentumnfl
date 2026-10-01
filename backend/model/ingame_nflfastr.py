"""nflfastR's spread-aware win probability, anchored on our pregame margin.

Adopted from nflfastR (MIT): the published xgboost model, its feature
transforms, and its handling of kickoffs and tries. Our published pregame
margin replaces the closing spread the model was trained on. The trees are
evaluated here directly from the exported JSON, so serving needs neither R
nor xgboost.

Two deliberate departures. Extra points use one make probability where
nflfastR varies it by roof. Overtime uses the regulation model on the
overtime clock where nflfastR derives it from its expected points model,
which predates the current overtime rules.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR
from backend.features.ingame import (
    EXTRA_POINT,
    KICKOFF,
    REGULATION_SECONDS,
    TWO_POINT,
)

MODEL_VERSION = "nfl_ingame_nflfastr_v1"
MODEL_PATH = PROCESSED_DIR / "ingame" / "nflfastr_wp_spread.json"
# fastrmodels 1.0.2 wp_model_spread, written by ops/export_nflfastr_wp.R.
MODEL_SHA256 = "b1574b24eed57bb94245c498a7bd275f3f095b13cfbca51bd9abce6bf9b96239"
FEATURES = (
    "receive_2h_ko",
    "spread_time",
    "home",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "Diff_Time_Ratio",
    "score_differential",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
)
# A drive after a kickoff or a try is scored as first and ten at the 25.
DRIVE_START_YARDS_TO_GOAL = 75.0
# nflfastR's field goal model at the extra point distance, outdoors, 2014+.
EXTRA_POINT_MAKE_PROBABILITY = 0.9324
TWO_POINT_MAKE_PROBABILITY = 0.4735
ROW_CHUNK = 20_000


@dataclass(frozen=True, slots=True)
class TreeEnsemble:
    """Trees padded to a common node count; a leaf has left child -1."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    # Split threshold at an internal node, leaf value at a leaf.
    value: np.ndarray
    default_left: np.ndarray
    base_margin: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        trees = np.arange(len(self.left))[:, None]
        out = np.empty(len(features))
        for start in range(0, len(features), ROW_CHUNK):
            chunk = features[start : start + ROW_CHUNK]
            rows = np.arange(len(chunk))[None, :]
            node = np.zeros((len(self.left), len(chunk)), dtype=np.int32)
            while True:
                left = self.left[trees, node]
                internal = left >= 0
                if not internal.any():
                    break
                observed = chunk[rows, self.feature[trees, node]]
                go_left = np.where(
                    np.isnan(observed),
                    self.default_left[trees, node],
                    observed < self.value[trees, node],
                )
                child = np.where(go_left, left, self.right[trees, node])
                node = np.where(internal, child, node)
            margin = self.value[trees, node].sum(axis=0, dtype=np.float64)
            out[start : start + len(chunk)] = margin + self.base_margin
        return 1.0 / (1.0 + np.exp(-out))


def load_model(path=MODEL_PATH) -> TreeEnsemble:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError(f"{path} is not the pinned nflfastR model export")
    learner = json.loads(raw)["learner"]
    if learner["objective"]["name"] != "binary:logistic" or int(
        learner["learner_model_param"]["num_feature"]
    ) != len(FEATURES):
        raise ValueError("Unexpected nflfastR model shape")
    trees = learner["gradient_booster"]["model"]["trees"]
    width = max(len(tree["left_children"]) for tree in trees)

    def padded(key: str, dtype, fill) -> np.ndarray:
        out = np.full((len(trees), width), fill, dtype=dtype)
        for index, tree in enumerate(trees):
            out[index, : len(tree[key])] = tree[key]
        return out

    base_score = float(learner["learner_model_param"]["base_score"])
    return TreeEnsemble(
        left=padded("left_children", np.int32, -1),
        right=padded("right_children", np.int32, -1),
        feature=padded("split_indices", np.int32, 0),
        value=padded("split_conditions", np.float32, 0.0),
        default_left=padded("default_left", bool, False),
        base_margin=float(np.log(base_score / (1.0 - base_score))),
    )


def win_probability(inputs: pd.DataFrame, model: TreeEnsemble) -> np.ndarray:
    """Home win probability at each state from the state and pregame anchor."""
    home = inputs["offense_is_home"].to_numpy(bool)
    sign = np.where(home, 1.0, -1.0)
    kind = inputs["play_kind"].to_numpy(int)
    first_half = inputs["period"].to_numpy(int) <= 2
    game_seconds = inputs["seconds_remaining"].to_numpy(float)
    half_seconds = inputs["half_seconds_remaining"].to_numpy(float)
    decay = np.exp(-4.0 * (REGULATION_SECONDS - game_seconds) / REGULATION_SECONDS)
    home_timeouts = inputs["home_timeouts"].to_numpy(float)
    away_timeouts = inputs["away_timeouts"].to_numpy(float)

    def offense_wins(flip, points, down, distance, yards_to_goal) -> np.ndarray:
        """Score the side with the ball, or its opponent after ``points``."""
        side = np.where(flip, -sign, sign)
        receives = inputs["offense_receives_second_half"].to_numpy(bool)
        receives = np.where(flip & first_half, ~receives, receives)
        score = side * inputs["score_margin"].to_numpy(float) - points
        on_home = side > 0
        columns = {
            "receive_2h_ko": receives,
            "spread_time": side * inputs["pregame_margin"].to_numpy(float) * decay,
            "home": on_home,
            "half_seconds_remaining": half_seconds,
            "game_seconds_remaining": game_seconds,
            "Diff_Time_Ratio": score / decay,
            "score_differential": score,
            "down": down,
            "ydstogo": distance,
            "yardline_100": yards_to_goal,
            "posteam_timeouts_remaining": np.where(
                on_home, home_timeouts, away_timeouts
            ),
            "defteam_timeouts_remaining": np.where(
                on_home, away_timeouts, home_timeouts
            ),
        }
        return model.predict(
            np.column_stack([np.asarray(columns[name], float) for name in FEATURES])
        )

    is_try = np.isin(kind, (EXTRA_POINT, TWO_POINT))
    new_drive = is_try | (kind == KICKOFF)
    down = np.where(new_drive, 1.0, inputs["down"].to_numpy(float))
    distance = np.where(new_drive, 10.0, inputs["distance"].to_numpy(float))
    yards_to_goal = np.where(
        new_drive, DRIVE_START_YARDS_TO_GOAL, inputs["yards_to_goal"].to_numpy(float)
    )
    # After a try the opponent starts a drive, trailing by what the try added.
    opponent_after_miss = offense_wins(is_try, 0.0, down, distance, yards_to_goal)
    offense = opponent_after_miss.copy()
    if is_try.any():
        two_point = kind == TWO_POINT
        opponent_after_make = offense_wins(
            is_try, np.where(two_point, 2.0, 1.0), down, distance, yards_to_goal
        )
        make = np.where(
            two_point, TWO_POINT_MAKE_PROBABILITY, EXTRA_POINT_MAKE_PROBABILITY
        )
        offense[is_try] = (
            1.0
            - (make * opponent_after_make + (1.0 - make) * opponent_after_miss)[is_try]
        )
    return np.where(home, offense, 1.0 - offense)
