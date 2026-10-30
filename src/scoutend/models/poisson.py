"""Backwards-compatible import path.

The GLM implementation moved to :mod:`scoutend.models.glm` when the Gamma
regressor was added. This module re-exports it so older imports keep working.
"""

from .glm import GammaRegressor, PoissonRegressor

__all__ = ["PoissonRegressor", "GammaRegressor"]
