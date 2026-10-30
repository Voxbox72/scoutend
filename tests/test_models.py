"""Tests for the GLM models, the minutes model, and the registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scoutend.clean import build
from scoutend.config import PATHS, Paths
from scoutend.eval import harness
from scoutend.features import build_features
from scoutend.models.glm import GammaRegressor, PoissonRegressor
from scoutend.models.minutes import MAX_MINUTES, ExpectedMinutesModel
from scoutend.models.registry import MAX_SEASON_TOTAL
from scoutend.targets import build_targets


class PoissonTest(unittest.TestCase):
    def test_recovers_a_known_rate_structure(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        x = rng.normal(size=(n, 2))
        exposure = rng.uniform(5, 30, size=n)
        beta = np.array([0.4, -0.25])
        rate = np.exp(beta[0] * x[:, 0] + beta[1] * x[:, 1])
        y = rng.poisson(rate * exposure).astype(float)

        model = PoissonRegressor(alpha=1e-6, max_iter=300).fit(x, y, offset=exposure)
        predicted = model.predict_rate(x)

        # The sign of the relationship is what matters, not the coefficient scale:
        # the adapter standardises internally.
        self.assertGreater(float(np.corrcoef(predicted, x[:, 0])[0, 1]), 0.2)
        self.assertLess(float(np.corrcoef(predicted, x[:, 1])[0, 1]), -0.2)
        self.assertTrue(np.isfinite(predicted).all())

    def test_predictions_are_positive_and_scale_with_exposure(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(500, 3))
        exposure = rng.uniform(1, 30, size=500)
        y = rng.poisson(np.exp(0.2 * x[:, 0]) * exposure).astype(float)
        model = PoissonRegressor(alpha=1.0).fit(x, y, offset=exposure)

        one = model.predict(x, offset=np.full(len(x), 10.0))
        two = model.predict(x, offset=np.full(len(x), 20.0))
        self.assertTrue((one > 0).all())
        np.testing.assert_allclose(two, one * 2, rtol=1e-9)

    def test_zero_exposure_does_not_break_the_fit(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(100, 2))
        y = rng.poisson(1.0, size=100).astype(float)
        model = PoissonRegressor(alpha=1.0).fit(x, y, offset=np.zeros(100))
        self.assertTrue(np.isfinite(model.predict_rate(x)).all())


class GammaTest(unittest.TestCase):
    def test_estimates_the_mean_not_the_median(self) -> None:
        """The whole reason for a Gamma GLM over log-OLS: no downward bias."""
        rng = np.random.default_rng(3)
        n = 4000
        x = rng.normal(size=(n, 1))
        mu = np.exp(7.0 + 0.3 * x[:, 0])
        y = rng.gamma(shape=2.0, scale=mu / 2.0)

        model = GammaRegressor(alpha=0.1, max_iter=200).fit(x, y)
        predicted = model.predict_rate(x)
        bias = predicted.mean() / y.mean() - 1.0
        self.assertLess(abs(bias), 0.05, f"mean bias {bias:.3f} should be near zero")


class MinutesModelTest(unittest.TestCase):
    def test_predictions_are_within_the_season_ceiling(self) -> None:
        rng = np.random.default_rng(4)
        n = 2000
        x = rng.normal(size=(n, 2))
        minutes = np.clip(rng.normal(1500, 700, size=n), 1, MAX_MINUTES)
        model = ExpectedMinutesModel().fit(x, minutes)
        predicted = model.predict(x * 5)  # deliberately out of distribution
        self.assertTrue((predicted >= 0).all())
        self.assertTrue((predicted <= MAX_MINUTES).all())


class RegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        paths = Paths(root=PATHS.root, raw=PATHS.raw, processed=Path(cls._tmp.name))
        paths.ensure()
        players, _ = build(paths)
        cls.targets = build_targets(players)
        cls.features = build_features(players)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_every_prediction_is_a_plausible_season_total(self) -> None:
        results, predictions = harness.evaluate(
            self.targets, self.features, scored_targets=("y_gls_next",), test_seasons=(2023,)
        )
        self.assertFalse(predictions.empty)
        self.assertTrue((predictions["predicted"] >= 0).all())
        self.assertTrue((predictions["predicted"] <= MAX_SEASON_TOTAL).all())
        self.assertTrue(np.isfinite(predictions["predicted"]).all())
        self.assertIn("poisson_blended_two_stage", set(results["model"]))

    def test_models_beat_the_previous_season_baseline_for_assists(self) -> None:
        """Assists are where the rate model has a clear, stable edge."""
        results, _ = harness.evaluate(
            self.targets, self.features, scored_targets=("y_ast_next",), test_seasons=(2022, 2023)
        )
        summary = harness.summarise(results)
        qualified = summary[summary["scope"] == "qualified"].set_index("model")
        self.assertLess(
            qualified.loc["poisson_blended_two_stage", "mae"],
            qualified.loc["last_season_ast", "mae"],
        )


if __name__ == "__main__":
    unittest.main()
