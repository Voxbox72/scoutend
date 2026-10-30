"""The models: expected minutes (sklearn), the assist rate (sklearn), and the goal rate.

Two of the three estimators come from scikit-learn. The exception is the Poisson
rate model, and the reason is worth stating plainly:

* A goal *rate* per 90 needs an **exposure offset**:
  ``log(mu) = log(90s) + X @ beta``.
* ``sklearn.linear_model.PoissonRegressor`` has no ``offset`` argument and no
  way to separate the intercept from the exposure. Passing the exposure as
  ``sample_weight`` fits ``mu = exp(b0 + Xb)`` with the *total* as the mean, so
  the intercept absorbs the exposure scale and predictions come out wrong by a
  factor of the exposure -- measured on this data, an intercept of 3.0
  (e^3 = 20 goals) instead of 0.1 (the mean rate).
* Transforming the target to ``y - log(exposure)`` is not an option either: a
  Poisson loss requires non-negative responses.

So the Poisson rate model keeps a small, tested Fisher-scoring implementation
(``scoutend.obsolete.glm_numpy``), which supports an offset exactly. Everything
else uses the library.

Interval censoring: the Poisson model predicts a rate for any exposure, so the
same object gives a per-90 rate *and* a season total at any assumed minutes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import GammaRegressor as _SkGamma

from ..obsolete.glm_numpy import PoissonRegressor

__all__ = ["PoissonRegressor", "GammaRegressor", "MINUTES_ETA_CAP"]

# Minutes run to 3,420, so log(minutes) runs to ~8.1. Exposed for the minutes
# model, which clips its output to this.
MINUTES_ETA_CAP = 9.0


class GammaRegressor:
    """Positive durations with a log link, via scikit-learn.

    Adapter, not a reimplementation: standardise the design (so the L2 penalty
    is comparable across features), fit, and expose ``predict_rate``.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 300, tol: float = 1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self._estimator: _SkGamma | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None:
            return X
        return (X - self.mean_) / self.scale_

    def fit(self, X, y, offset=None) -> "GammaRegressor":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

        estimator = _SkGamma(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        estimator.fit(self._design(X), np.asarray(y, dtype=float))
        self._estimator = estimator
        return self

    def predict_rate(self, X) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("predict_rate called before fit")
        return np.asarray(self._estimator.predict(self._design(X)), dtype=float)

    def predict(self, X, offset=None) -> np.ndarray:
        rate = self.predict_rate(X)
        if offset is None:
            return rate
        return rate * np.asarray(offset, dtype=float)

