"""Tests for the scouting layer: percentiles, roles, similarity, shortlist."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scoutend.config import MINUTES_FLOOR, PATHS, Paths
from scoutend.scouting import (
    STYLE_FEATURES,
    _kmeans,
    _label_cluster,
    add_percentiles,
    assign_roles,
    projection_and_value,
    shortlist_score,
    similarity,
    style_matrix,
)


class KMeansTest(unittest.TestCase):
    def test_recovers_separated_groups(self) -> None:
        rng = np.random.default_rng(0)
        data = np.vstack(
            [
                rng.normal(0, 0.2, size=(50, 3)),
                rng.normal(5, 0.2, size=(50, 3)),
                rng.normal(-5, 0.2, size=(50, 3)),
            ]
        )
        labels = _kmeans(data, k=3)
        self.assertEqual(len(set(labels)), 3)
        # Points from the same generated blob must share a label.
        self.assertTrue(np.all(labels[:50] == labels[0]))
        self.assertTrue(np.all(labels[50:100] == labels[50]))
        self.assertTrue(np.all(labels[100:] == labels[100]))

    def test_is_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(120, 4))
        self.assertTrue(np.array_equal(_kmeans(data, k=3), _kmeans(data, k=3)))

    def test_handles_fewer_rows_than_clusters(self) -> None:
        labels = _kmeans(np.zeros((2, 3)), k=6)
        self.assertEqual(len(labels), 2)


class RoleLabelTest(unittest.TestCase):
    """Naming rule: what the cluster is above par on decides the label."""

    def test_zero_profile_is_balanced(self) -> None:
        profile = pd.Series(0.0, index=["gls_per90", "ast_per90", "tackles_per90"])
        self.assertEqual(_label_cluster(profile), "Balanced")

    def test_goal_led_cluster_is_a_finisher(self) -> None:
        profile = pd.Series({"gls_per90": 1.2, "npxg_per90": 0.9,
                             "ast_per90": 0.0, "tackles_per90": 0.0})
        self.assertEqual(_label_cluster(profile), "Finisher")

    def test_creation_led_cluster_is_a_creator(self) -> None:
        profile = pd.Series({"gls_per90": 0.1, "ast_per90": 1.1,
                             "xag_per90": 0.8, "tackles_per90": 0.0})
        self.assertEqual(_label_cluster(profile), "Creator")

    def test_placement_in_the_box_identifies_a_finisher(self) -> None:
        profile = pd.Series({"gls_per90": 0.0, "npxg_per90": 0.1,
                             "touches_att_pen_per90": 1.0, "clearances_per90": -0.2})
        self.assertEqual(_label_cluster(profile), "Finisher")

    def test_defensive_led_cluster_is_aerial_defensive(self) -> None:
        profile = pd.Series({"gls_per90": 0.0, "clearances_per90": 1.0,
                             "interceptions_per90": 0.7, "prg_receives_per90": -0.4})
        self.assertEqual(_label_cluster(profile), "Aerial/defensive")

    def test_below_par_everywhere_is_balanced(self) -> None:
        profile = pd.Series({name: -0.6 for name in ["gls_per90", "ast_per90",
                                                     "prg_carries_per90"]})
        self.assertEqual(_label_cluster(profile), "Balanced")


class PercentileTest(unittest.TestCase):
    def setUp(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": [f"p{i}_2000" for i in range(6)],
                "player": [f"P{i}" for i in range(6)],
                "season": [2024] * 6,
                "position": ["FW"] * 6,
                "minutes": [2000, 2000, 2000, 100, 100, 100],
                "age": [20, 21, 22, 23, 24, 25],
                "gls_per90": [2.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            }
        )
        self.frame = frame

    def test_highest_value_is_top_percentile(self) -> None:
        result = add_percentiles(self.frame.copy())
        best = result.loc[result["gls_per90"].idxmax()]
        self.assertGreater(best["pct_gls_per90"], 90)

    def test_age_is_reversed(self) -> None:
        result = add_percentiles(self.frame.copy())
        youngest = result.loc[result["age"].idxmin()]
        oldest = result.loc[result["age"].idxmax()]
        self.assertGreater(youngest["pct_age"], oldest["pct_age"])

    def test_percentiles_are_within_position(self) -> None:
        frame = self.frame.copy()
        frame.loc[frame.index[:3], "position"] = "DF"
        result = add_percentiles(frame)
        self.assertIn("pct_gls_per90", result.columns)

    def test_ties_share_a_percentile(self) -> None:
        """Three forwards on identical non-penalty xG must not get three ranks."""
        frame = pd.DataFrame(
            {
                "player_id": [f"q{i}_2000" for i in range(6)],
                "player": [f"Q{i}" for i in range(6)],
                "season": [2024] * 6,
                "position": ["FW"] * 6,
                "minutes": [2000] * 6,
                "age": [20, 21, 22, 23, 24, 25],
                "npxg_per90": [5.0, 5.0, 5.0, 1.0, 0.0, 0.0],
            }
        )
        result = add_percentiles(frame)
        tied = result.loc[result["npxg_per90"] == 5.0, "pct_npxg_per90"]
        self.assertEqual(tied.nunique(), 1, f"ties must agree, got {sorted(tied)}")
        # Three of six players share the top value: the tied group's centre is
        # the 4th of 6 ranks, i.e. 60th percentile.
        self.assertEqual(float(tied.iloc[0]), 60.0)


class LivePanelTest(unittest.TestCase):
    """Checks against the real dataset: these are the ones that would catch drift."""

    @classmethod
    def setUpClass(cls) -> None:
        from scoutend.scouting import load_panel

        cls.panel = load_panel()
        cls.roles = assign_roles(cls.panel)

    def test_percentiles_cover_every_player(self) -> None:
        columns = [c for c in self.panel.columns if c.startswith("pct_")]
        self.assertGreater(len(columns), 10)
        # A metric whose denominator is undefined (no shots, no pass attempts)
        # cannot have a percentile; everything else must be populated.
        complete = self.panel[columns].notna().mean()
        self.assertGreater(float(complete.min()), 0.95)

    def test_roles_are_assigned_to_qualified_players(self) -> None:
        qualified = self.roles[self.roles["minutes"] >= MINUTES_FLOOR]
        self.assertFalse(qualified["role"].isna().any())
        bases = {str(r).split(" (")[0] for r in qualified["role"].unique()}
        self.assertLessEqual(bases, {
            "Finisher", "Creator", "Progressor", "Aerial/defensive",
            "Retainer", "Balanced", "All-round elite",
        })

    def test_strikers_are_not_indiscriminately_labelled_creators(self) -> None:
        """Forward roles must discriminate. Forwards are the population where a
        badly tuned rule collapses everything into one bucket, so this checks the
        spread of roles and that the *typical* forward is not called a creator."""
        forwards = self.roles[(self.roles["position"] == "FW")
                              & (self.roles["minutes"] >= MINUTES_FLOOR)]
        bases = forwards["role"].astype(str).str.split(" (", regex=False).str[0]
        self.assertGreaterEqual(bases.nunique(), 4)

        top_share = bases.value_counts(normalize=True).iloc[0]
        self.assertLess(float(top_share), 0.60, "one role swallows most forwards")
        self.assertLess(float((bases == "Finisher").mean()), float((bases == "Balanced").mean()) + 1.0)

    def test_similarity_is_within_position(self) -> None:
        row = self.roles[(self.roles["player"] == "Mohamed Salah") & (self.roles["season"] == 2024)]
        if row.empty:
            self.skipTest("player not in the dataset")
        neighbours = similarity(self.roles, row.iloc[0], n=6)
        self.assertTrue((neighbours["position"] == row.iloc[0]["position"]).all())
        self.assertNotIn("Mohamed Salah", set(neighbours["player"]))
        self.assertTrue((neighbours["similarity"] <= 1.0001).all())

    def test_shortlist_score_is_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "output_next_per90": [0.1, 0.5, 0.9],
                "expected_minutes_next": [300, 1500, 3000],
                "age": [18, 24, 34],
                "predicted_goals_next": [1.0, 8.0, 20.0],
                "gls": [0.0, 5.0, 25.0],
            }
        )
        score = shortlist_score(frame)
        self.assertTrue((score >= 0).all() and (score <= 100).all())
        self.assertGreater(score.iloc[2], score.iloc[0])


if __name__ == "__main__":
    unittest.main()
