"""Tests for features and the evaluation harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scoutend.clean import build
from scoutend.config import PATHS, Paths
from scoutend.eval import harness
from scoutend.features import FEATURES, build_features, feature_matrix
from scoutend.targets import build_targets


class FeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        paths = Paths(root=PATHS.root, raw=PATHS.raw, processed=Path(cls._tmp.name))
        paths.ensure()
        cls.players, _ = build(paths)
        cls.targets = build_targets(cls.players)
        cls.features = build_features(cls.players)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_no_future_columns(self) -> None:
        forbidden = [c for c in self.features.columns if "next" in c]
        self.assertEqual(forbidden, [])

    def test_per90_is_rate_times_nineties(self) -> None:
        from scoutend.features import PER90_CAPS

        sample = self.features.dropna(subset=["gls_per90"]).head(200)
        expected = (sample["gls"] / (sample["minutes"] / 90.0)).clip(
            upper=PER90_CAPS["gls_per90"]
        )
        np.testing.assert_allclose(sample["gls_per90"], expected, rtol=1e-9)

    def test_shrunk_rate_lies_between_prior_and_raw(self) -> None:
        sample = self.features.dropna(subset=["gls_per90", "gls_per90_shrunk"])
        prior = sample.groupby(["position", "season"])["gls_per90"].transform("mean")
        low = np.minimum(prior, sample["gls_per90"]) - 1e-9
        high = np.maximum(prior, sample["gls_per90"]) + 1e-9
        self.assertTrue(((sample["gls_per90_shrunk"] >= low) & (sample["gls_per90_shrunk"] <= high)).all())

    def test_feature_matrix_matches_declared_features(self) -> None:
        matrix, columns = feature_matrix(self.features)
        self.assertEqual(len(matrix), len(self.features))
        for name in FEATURES:
            self.assertIn(name, columns)
        self.assertTrue(all(np.issubdtype(t, np.number) for t in matrix.dtypes))


class HarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        paths = Paths(root=PATHS.root, raw=PATHS.raw, processed=Path(cls._tmp.name))
        paths.ensure()
        players, _ = build(paths)
        cls.players = players
        cls.targets = build_targets(players)
        cls.features = build_features(players)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_evaluation_is_forward_looking(self) -> None:
        results, predictions = harness.evaluate(
            self.targets, self.features, scored_targets=("y_gls_next",)
        )
        self.assertFalse(results.empty)
        self.assertFalse(predictions.empty)
        # Only seasons that have a following season can be scored.
        self.assertLessEqual(int(results["test_season"].max()), 2023)
        self.assertTrue((results["test_season"] >= 2021).all())

    def test_every_model_is_reported_against_the_mean_baseline(self) -> None:
        results, _ = harness.evaluate(
            self.targets, self.features, scored_targets=("y_gls_next",)
        )
        summary = harness.summarise(results)
        models = set(summary["model"])
        for required in ("season_mean", "position_mean", "ridge"):
            self.assertIn(required, models)

    def test_baseline_beats_the_mean(self) -> None:
        """A sanity check on the data, not on the models: if the previous
        season's npxG does not beat predicting the mean, the join is broken."""
        results, _ = harness.evaluate(
            self.targets, self.features, scored_targets=("y_gls_next",)
        )
        summary = harness.summarise(results)
        qualified = summary[summary["scope"] == "qualified"].set_index("model")
        self.assertLess(qualified.loc["last_season_npxg", "mae"], qualified.loc["season_mean", "mae"])


if __name__ == "__main__":
    unittest.main()
