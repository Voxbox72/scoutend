"""The model registry: every candidate the harness scores, in one place.

Baselines and real models live side by side on purpose. A model earns its place
by beating ``last_season_npxg``, not by beating a straw man, so the harness
scores all of them on identical rows with identical splits.

Adding a model means adding one function here -- the harness, the CSV output and
the model cards pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

from ..config import MINUTES_FLOOR
from .minutes import ExpectedMinutesModel, MAX_MINUTES, availability_matrix
from .poisson import PoissonRegressor

# For each target: the current-season column that "last season" should use, the
# corresponding expected-metric column, and the shrunk per-90 rate feature.
TARGETS: dict[str, dict[str, str]] = {
    "y_gls_next": {
        "label": "goals",
        "raw": "gls",
        "expected": "npxg",
        "rate_shrunk": "gls_per90_shrunk",
        "rate_raw": "gls_per90",
    },
    "y_ast_next": {
        "label": "assists",
        "raw": "ast",
        "expected": "xag",
        "rate_shrunk": "ast_per90_shrunk",
        "rate_raw": "ast_per90",
    },
}

DEFAULT_TARGETS = ("y_gls_next", "y_ast_next")

# No player has scored or assisted more than 60 in a Big-5 season. Predictions
# are clipped here so a numerical accident cannot dominate an average error.
MAX_SEASON_TOTAL = 60.0

# How many 90s of evidence before the rate model is trusted fully. With 4, a
# player with 4 matches is 50/50 between his own rate and his position's.
RATE_PRIOR_90S = 4.0

# Sanity ceiling on a *modelled* per-90 rate, before blending. The best scoring
# seasons in the Big 5 sit near 1.3 goals per 90; 1.5 leaves headroom without
# letting a 60-minute sample produce a rate no footballer has ever sustained.
RATE_CAPS = {"y_gls_next": 1.5, "y_ast_next": 1.2}


def clip_rate(rate, target: str):
    return np.clip(np.asarray(rate, dtype=float), 0.0, RATE_CAPS[target])


@dataclass(frozen=True)
class Prediction:
    name: str
    family: str  # "baseline" | "model"
    values: np.ndarray


def predict_all(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    columns: list[str],
    train,
    test,
    target: str,
) -> list[Prediction]:
    """Fit every candidate on ``train`` and predict ``test`` for one target."""
    config = TARGETS[target]

    x_train = _impute(matrix.loc[train, columns], matrix.loc[train, columns])
    x_test = _impute(matrix.loc[test, columns], matrix.loc[train, columns])

    y_train = frame.loc[train, target].to_numpy(dtype=float)
    nineties_now = frame.loc[test, "nineties"].fillna(0.0).to_numpy(dtype=float)
    nineties_train = frame.loc[train, "nineties"].fillna(0.0).to_numpy(dtype=float)

    predictions: list[Prediction] = []

    # --- baselines ---------------------------------------------------------
    predictions.append(
        Prediction("season_mean", "baseline", np.full(len(x_test), y_train.mean()))
    )

    position_means = frame.loc[train].groupby("position", observed=True)[target].mean()
    predictions.append(
        Prediction(
            "position_mean",
            "baseline",
            frame.loc[test, "position"].map(position_means).astype(float).fillna(y_train.mean()).to_numpy(),
        )
    )

    raw_name = f"last_season_{config['raw']}"
    predictions.append(
        Prediction(raw_name, "baseline", frame.loc[test, config["raw"]].fillna(0.0).to_numpy())
    )
    expected_name = f"last_season_{config['expected']}"
    predictions.append(
        Prediction(
            expected_name, "baseline", frame.loc[test, config["expected"]].fillna(0.0).to_numpy()
        )
    )

    # Shrunk rate x the minutes he actually played: the cheapest honest
    # expression of "the same player in the same role".
    predictions.append(
        Prediction(
            "rate_x_actual_minutes",
            "baseline",
            nineties_now * frame.loc[test, config["rate_shrunk"]].fillna(0.0).to_numpy(),
        )
    )

    # --- models ------------------------------------------------------------
    ridge = Ridge(alpha=10.0).fit(x_train, y_train)
    predictions.append(Prediction("ridge", "model", ridge.predict(x_test)))

    # Poisson rate model: the per-90 rate is what generalises across roles.
    #
    # Fitted on players with a real role only. A rate estimated from four
    # minutes is not a rate: one goal in four minutes implies 22 goals per 90,
    # and a log-link model will happily fit it. Predictions are still made for
    # everyone -- low-minute players are exactly who a scout is looking at, and
    # the shrunk-rate features keep their projections sane.
    rate_mask = nineties_train * 90.0 >= MINUTES_FLOOR
    if rate_mask.sum() < 500:
        rate_mask = np.ones_like(rate_mask, dtype=bool)
    rate_model = PoissonRegressor(alpha=1.0).fit(
        x_train[rate_mask], y_train[rate_mask], offset=nineties_train[rate_mask]
    )
    rate = clip_rate(rate_model.predict_rate(x_test), target)
    predictions.append(
        Prediction("poisson_rate_x_actual_minutes", "model", rate * np.maximum(nineties_now, 1 / 90))
    )

    # Two-stage: share of minutes next season is its own prediction problem.
    availability = availability_matrix(frame, columns)
    availability_idx = [columns.index(c) for c in availability]
    minutes_target = frame.loc[train, "y_minutes_next"].to_numpy(dtype=float)
    minutes_model = ExpectedMinutesModel().fit(x_train[:, availability_idx], minutes_target)
    expected_nineties = minutes_model.predict(x_test[:, availability_idx]) / 90.0
    predictions.append(Prediction("poisson_two_stage", "model", rate * expected_nineties))

    # Blended rate: trust the model in proportion to how much football the
    # player actually played, and fall back to his position's shrunk rate for
    # everyone else. A 30-minute cameo is not evidence about next season, and
    # without this the model happily projects 60 goals for a 17-year-old.
    shrunk_rate = frame.loc[test, config["rate_shrunk"]].fillna(0.0).to_numpy(dtype=float)
    blended_rate = blend_rate(rate, nineties_now, shrunk_rate)
    predictions.append(
        Prediction("poisson_blended_x_actual_minutes", "model", blended_rate * np.maximum(nineties_now, 1 / 90))
    )
    predictions.append(
        Prediction("poisson_blended_two_stage", "model", blended_rate * expected_nineties)
    )

    # A season total above 60 goals or assists has never happened in the Big 5.
    # Clipping keeps a numerical accident out of a shortlist.
    return [
        Prediction(item.name, item.family, np.clip(item.values, 0.0, MAX_SEASON_TOTAL))
        for item in predictions
    ]


def target_label(target: str) -> str:
    return TARGETS[target]["label"]


def blend_rate(model_rate, nineties, shrunk_rate, prior: float = RATE_PRIOR_90S):
    """Weight the model's rate by how much football the player actually played.

    With ``prior`` 90s of evidence the two are weighted equally; below that the
    position's shrunk rate dominates. A 13-minute cameo is not evidence about
    next season, and an unguarded log-link model will happily turn one goal in
    it into a 60-goal projection.
    """
    nineties = np.asarray(nineties, dtype=float)
    reliability = nineties / (nineties + prior)
    return reliability * np.asarray(model_rate, dtype=float) + (1.0 - reliability) * np.asarray(
        shrunk_rate, dtype=float
    )


def _impute(block: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """Median-impute using *training* medians only."""
    median = reference.median(numeric_only=True)
    return block.fillna(median).fillna(0.0).to_numpy(dtype=float)
