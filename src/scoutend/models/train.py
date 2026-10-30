"""Fit the shipping models on all available data and project the next season.

This is the step that produces the artifact a product would serve: one row per
player with expected minutes, per-90 rates, and projected goals and assists for
the season that has no outcome yet.

Because the dataset currently ends at 2023/24, the projection is for 2024/25 --
which is a genuine forward prediction and *cannot* be scored from this data. The
evaluation in ``eval/harness.py`` is the only evidence of quality; this file
produces the forecast, not the proof.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features as features_module
from ..config import MINUTES_FLOOR, PATHS, Paths
from ..models.card import ModelCard
from ..models.minutes import ExpectedMinutesModel, availability_matrix
from ..models.poisson import PoissonRegressor
from ..models.registry import MAX_SEASON_TOTAL, TARGETS, blend_rate, clip_rate

PROJECTION_COLUMNS = [
    "player_id",
    "player",
    "team",
    "clubs",
    "position",
    "league_code",
    "age",
    "minutes",
    "matches",
    "nineties",
    "expected_minutes_next",
    "gls_per90",
    "gls_per90_shrunk",
    "predicted_goals_next",
    "ast_per90",
    "ast_per90_shrunk",
    "predicted_assists_next",
    "qualified",
]


def train_and_predict(paths: Paths = PATHS) -> pd.DataFrame:
    """Train on every season with a known outcome, project the latest season."""
    from ..eval.harness import load_dataset

    _, targets, features = load_dataset(paths)

    frame = features.merge(
        targets[["player_id", "season", "y_gls_next", "y_ast_next", "y_minutes_next", "has_next_season"]],
        on=["player_id", "season"],
        how="inner",
    )
    matrix, columns = features_module.feature_matrix(frame)

    latest = int(frame["season"].max())
    train = ((frame["season"] < latest) & frame["has_next_season"]).to_numpy()
    predict = (frame["season"] == latest).to_numpy()
    if not train.any() or not predict.any():
        raise RuntimeError("not enough seasons to train and project")

    # Train-only imputation.
    median = matrix.loc[train, columns].median(numeric_only=True)
    x_train = matrix.loc[train, columns].fillna(median).fillna(0.0).to_numpy(dtype=float)
    x_predict = matrix.loc[predict, columns].fillna(median).fillna(0.0).to_numpy(dtype=float)

    nineties_train = frame.loc[train, "nineties"].fillna(0.0).to_numpy(dtype=float)
    qualified = nineties_train * 90.0 >= MINUTES_FLOOR
    if qualified.sum() < 500:
        qualified = np.ones_like(qualified, dtype=bool)

    # --- expected minutes ---------------------------------------------------
    availability = availability_matrix(frame, columns)
    availability_idx = [columns.index(c) for c in availability]
    minutes_model = ExpectedMinutesModel().fit(
        x_train[:, availability_idx],
        frame.loc[train, "y_minutes_next"].to_numpy(dtype=float),
    )
    expected_minutes = minutes_model.predict(x_predict[:, availability_idx])

    output = frame.loc[predict, ["player_id", "player", "team", "clubs", "position",
                                 "league_code", "age", "minutes", "matches", "nineties",
                                 "gls_per90", "gls_per90_shrunk", "ast_per90",
                                 "ast_per90_shrunk", "qualified"]].copy()
    output["expected_minutes_next"] = expected_minutes

    # --- rate models, one per target ---------------------------------------
    # The rate is blended with each player's shrunk rate in proportion to how
    # much football he actually played, exactly as in the evaluation harness.
    shrunk_columns = {"y_gls_next": "gls_per90_shrunk", "y_ast_next": "ast_per90_shrunk"}
    nineties_now = frame.loc[predict, "nineties"].fillna(0.0).to_numpy(dtype=float)

    for target, predicted_column in (
        ("y_gls_next", "predicted_goals_next"),
        ("y_ast_next", "predicted_assists_next"),
    ):
        model = PoissonRegressor(alpha=1.0).fit(
            x_train[qualified],
            frame.loc[train, target].to_numpy(dtype=float)[qualified],
            offset=nineties_train[qualified],
        )
        rate = blend_rate(
            clip_rate(model.predict_rate(x_predict), target),
            nineties_now,
            frame.loc[predict, shrunk_columns[target]].fillna(0.0).to_numpy(dtype=float),
        )
        output[predicted_column] = np.clip(rate * expected_minutes / 90.0, 0.0, MAX_SEASON_TOTAL)

    output = output.sort_values("predicted_goals_next", ascending=False).reset_index(drop=True)
    output.to_csv(
        paths.processed / "predictions_next_season.csv.gz", index=False, compression="gzip"
    )

    _write_projection_card(paths, latest, len(output), len(columns))
    return output


def _write_projection_card(paths: Paths, latest_season: int, rows: int, n_features: int) -> None:
    card = ModelCard(
        name="projection_next_season",
        target="y_gls_next, y_ast_next",
        description=(
            "Poisson per-90 rate models multiplied by an expected-minutes model. "
            "Projects the season after the last one in the dataset."
        ),
        trained_on=list(range(2020, latest_season)),
        evaluated_on=[2021, 2022, 2023],
        features=features_module.FEATURES[:n_features],
        hyperparameters={"alpha_rate": 1.0, "alpha_minutes": 10.0, "shrinkage_prior_90s": 10.0},
        metrics={},
        known_limitations=[
            f"Projects season {latest_season} -> {latest_season + 1}; no outcome data exists yet, "
            "so the projection itself cannot be scored.",
            f"{rows} players projected; rate models are fitted on players with >= {MINUTES_FLOOR} "
            "minutes, so low-minute players are extrapolations guarded by shrinkage.",
            "No league-strength adjustment: a Ligue 1 rate is not corrected toward the Premier League.",
            "No uncertainty interval yet; see the roadmap (bootstrap or Bayesian).",
        ],
    )
    card.write(paths.root / "models")
