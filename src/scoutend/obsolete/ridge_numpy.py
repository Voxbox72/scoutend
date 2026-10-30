"""A dependency-free ridge regression, used as the reference model.

Why not sklearn: the evaluation harness must run anywhere (including a machine
with no network and no wheels), and the reference model needs to be small enough
to reason about. Ridge on the full feature matrix is a deliberately strong
baseline -- if a gradient-boosting ensemble cannot beat it, the ensemble is not
earning its complexity.
"""

from __future__ import annotations

import numpy as np


class RidgeRegressor:
    """Ordinary least squares with L2 penalty and train-only standardisation."""

    def __init__(self, alpha: float = 10.0, clip_min: float = 0.0):
        self.alpha = alpha
        self.clip_min = clip_min
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X, y) -> "RidgeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        Z = (X - self.mean_) / self.scale_
        Z = np.column_stack([Z, np.ones(len(Z))])

        gram = Z.T @ Z
        penalty = self.alpha * np.eye(gram.shape[0])
        penalty[-1, -1] = 0.0  # never penalise the intercept
        self.coef_ = np.linalg.solve(gram + penalty, Z.T @ y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegressor.predict called before fit")
        X = np.asarray(X, dtype=float)
        Z = (X - self.mean_) / self.scale_
        Z = np.column_stack([Z, np.ones(len(Z))])
        predictions = Z @ self.coef_
        return np.clip(predictions, self.clip_min, None)
