"""Log-link GLMs fitted by Fisher scoring (IRLS), in numpy only.

Two models share almost all of their machinery:

* **Poisson** -- for event *counts* (goals, assists) with log(90s) as an offset.
  Variance equals the mean, so the working weights are ``mu``.
* **Gamma** -- for a positive *duration* (minutes played). Variance is
  proportional to ``mu**2``, so the working weights are constant.

Both use the same working response ``z = eta + (y - mu) / mu``; only the weights
differ. That is why this is one base class rather than two implementations.

Why this matters for minutes: fitting ``log(minutes)`` by least squares and then
exponentiating predicts the *median*, and Jensen's inequality makes it
systematically under-predict the mean by a quarter or more. A Gamma GLM with a
log link estimates the conditional mean directly, so the projection is not
biased downward by construction.

Why an offset: a player's goal *total* is rate x opportunity,

    log(E[goals]) = log(90s) + f(features)

so ``f`` is the per-90 rate and ``log(90s)`` is the exposure. Predict the rate
and it can be multiplied by any number of minutes -- which is what makes the
two-stage model (rate x expected minutes) possible.
"""

from __future__ import annotations

import numpy as np

# exp() is clipped to this range inside the loop. Far outside anything football
# produces, but it stops a bad step from producing inf and a NaN gradient.
_ETA_MIN, _ETA_MAX = -12.0, 4.0
# Ridge jitter, so a collinear design matrix cannot make the solve singular.
_SCALE_JITTER = 1e-8


class _LogLinkGLM:
    """Shared Fisher scoring loop. Subclasses define the working weights."""

    # Bounds on the linear predictor. A rate lives near 0 (log goals per 90),
    # but a duration lives near 7 (log minutes), so subclasses must widen them.
    eta_min: float = -12.0
    eta_max: float = 4.0

    def __init__(self, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_iter_: int = 0

    # -- to be provided by subclasses ---------------------------------------
    def _weights(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- shared machinery ----------------------------------------------------
    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.scale_
        return np.column_stack([Z, np.ones(len(Z))])

    def _clip(self, eta: np.ndarray) -> np.ndarray:
        return np.clip(eta, self.eta_min, self.eta_max)

    def fit(self, X, y, offset=None) -> "_LogLinkGLM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        exposure = _as_exposure(offset, len(y))

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < _SCALE_JITTER] = 1.0

        Z = self._design(X)
        log_offset = np.log(exposure)

        penalty = np.eye(Z.shape[1]) * self.alpha
        penalty[-1, -1] = 0.0  # intercept is never penalised

        # Warm start: intercept = mean response per unit exposure. Starting from
        # zero puts mu at 1, which for minutes (mean ~1500) is a terrible place
        # to begin Newton's method.
        beta = np.zeros(Z.shape[1])
        beta[-1] = float(np.clip(np.log(max(np.mean(y / exposure), 1e-6)), self.eta_min, self.eta_max))

        for iteration in range(1, self.max_iter + 1):
            eta = self._clip(Z @ beta + log_offset)
            mu = np.exp(eta)
            weights = self._weights(mu)

            # Working response on the linear-predictor scale (offset removed):
            # both models share z = eta + (y - mu) / mu.
            z = eta - log_offset + (y - mu) / mu
            gradient = Z.T @ (weights * (z - Z @ beta)) - penalty @ beta
            hessian = (Z * weights[:, None]).T @ Z + penalty
            step = np.linalg.solve(hessian + _SCALE_JITTER * np.eye(len(beta)), gradient)
            beta = beta + step
            self.n_iter_ = iteration
            if np.max(np.abs(step)) < self.tol:
                break

        self.coef_ = beta
        return self

    def predict_rate(self, X) -> np.ndarray:
        """Expected response per unit of exposure (per 90 minutes by default)."""
        if self.coef_ is None:
            raise RuntimeError("predict_rate called before fit")
        X = np.asarray(X, dtype=float)
        eta = self._clip(self._design(X) @ self.coef_)
        return np.exp(eta)

    def predict(self, X, offset=None) -> np.ndarray:
        """Expected total for a given exposure (e.g. 34.5 nineties)."""
        exposure = _as_exposure(offset, len(np.asarray(X)))
        return self.predict_rate(X) * exposure


class PoissonRegressor(_LogLinkGLM):
    """For counts with variance equal to the mean."""

    def _weights(self, mu: np.ndarray) -> np.ndarray:
        return mu


class GammaRegressor(_LogLinkGLM):
    """For positive durations; variance proportional to mu squared."""

    # Minutes run to 3,420, so log(minutes) runs to ~8.1.
    eta_min = -8.0
    eta_max = 8.5

    def _weights(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)


def _as_exposure(offset, n: int) -> np.ndarray:
    if offset is None:
        return np.ones(n)
    exposure = np.asarray(offset, dtype=float)
    if exposure.shape != (n,):
        raise ValueError(f"offset must have shape ({n},), got {exposure.shape}")
    # A zero or negative exposure is not a real row; clamp it so log is defined
    # and let the caller's minutes filter exclude the row.
    return np.clip(exposure, 1.0 / 90.0, None)
