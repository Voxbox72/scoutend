"""Tests for the canonical dataset and the next-season targets.

These encode the defects found in the original pipeline (see REVIEW.md) as
executable checks, so they cannot come back:

* F1  labels must never be filled with a default
* F2  a target must come from season + 1, never from the same season
* F4  keeper seasons must match outfield seasons
* F11 no player-season may appear twice after mid-season transfers are merged
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scoutend.clean import build, load_raw_outfield
from scoutend.config import PATHS, Paths
from scoutend.targets import build_targets


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = Paths(root=PATHS.root, raw=PATHS.raw, processed=Path(cls._tmp.name))
        cls.paths.ensure()
        cls.players, cls.keepers = build(cls.paths)
        cls.targets = build_targets(cls.players)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # --- structure ----------------------------------------------------------
    def test_one_row_per_player_season(self) -> None:
        duplicates = self.players.duplicated(subset=["player_id", "season"], keep=False)
        self.assertFalse(duplicates.any(), "mid-season transfers must be merged, not duplicated")

    def test_seasons_are_the_expected_range(self) -> None:
        self.assertEqual(sorted(self.players["season"].unique()), [2020, 2021, 2022, 2023, 2024])

    def test_positions_are_normalised(self) -> None:
        self.assertEqual(set(self.players["position"].unique()), {"DF", "MF", "FW"})
        self.assertEqual(set(self.keepers["position"].unique()), {"GK"})

    def test_goalkeepers_are_out_of_modelling_scope(self) -> None:
        """Keepers are built as a separate table and never enter the models.

        Their value lives in shot-stopping and sweeping, not in goals or assists,
        so mixing them into the outfield feature matrix would be a category error.
        """
        from scoutend.config import KEEPERS_IN_MODEL_SCOPE, MODEL_POSITIONS

        self.assertFalse(KEEPERS_IN_MODEL_SCOPE)
        self.assertNotIn("GK", set(self.players["position"].unique()))
        self.assertEqual(set(MODEL_POSITIONS), {"DF", "MF", "FW"})

        # FBref lists outfielders who deputised in goal in the keeper table too
        # (one case in this dataset). That is real, so the tables are allowed to
        # overlap slightly -- but if it ever became large, something is wrong.
        overlap = set(self.keepers["player_id"]) & set(self.players["player_id"])
        self.assertLessEqual(len(overlap), max(5, int(0.01 * len(self.keepers))))

    def test_minutes_are_positive(self) -> None:
        self.assertTrue((self.players["minutes"] > 0).all())
        self.assertTrue((self.keepers["minutes"] > 0).all())

    def test_no_negative_counts(self) -> None:
        counts = ["gls", "ast", "npxg", "tackles", "passes_completed", "minutes"]
        for column in counts:
            self.assertFalse((self.players[column] < 0).any(), f"{column} must be non-negative")

    # --- F4: keepers ---------------------------------------------------------
    def test_keeper_seasons_align_with_outfield_seasons(self) -> None:
        outfield = set(self.players["season"].unique())
        keeper = set(self.keepers["season"].unique())
        self.assertTrue(keeper <= outfield, f"keeper seasons not joinable: {keeper - outfield}")

    def test_keeper_ratios_are_derived_not_zeros(self) -> None:
        """Save% must not be a literal 0 for keepers who faced no shots."""
        self.assertNotIn("save_pct", self.keepers.columns)
        self.assertIn("saves", self.keepers.columns)
        self.assertIn("shots_on_target_against", self.keepers.columns)

    # --- F2: transfers merged correctly --------------------------------------
    def test_transfer_minutes_are_summed(self) -> None:
        """A player with two clubs must carry the total minutes, not one club's."""
        raw = load_raw_outfield(self.paths)
        raw["player"] = raw["player"].astype(str).str.strip()
        counts = raw.groupby(["player", "season"]).size()
        transferred = counts[counts > 1]
        self.assertGreater(len(transferred), 0, "expected some mid-season transfers in the data")

        # Take the first transferred player-season and verify the minutes add up.
        name, season_code = next(iter(transferred.index))
        rows = raw[(raw["player"] == name) & (raw["season"].astype(str) == str(season_code))]
        expected = float(rows["Playing Time_Min"].fillna(0).sum())

        match = self.players[(self.players["player"] == name) & (self.players["n_clubs"] > 1)]
        self.assertFalse(match.empty, f"{name} should be a multi-club row")
        self.assertAlmostEqual(float(match.iloc[0]["minutes"]), expected, places=3)

    # --- F1/F2: targets ------------------------------------------------------
    def test_targets_only_come_from_next_season(self) -> None:
        self.assertTrue((self.targets["next_season"] == self.targets["season"] + 1).all())

    def test_missing_next_season_means_null_target(self) -> None:
        last_season = self.targets[self.targets["season"] == self.players["season"].max()]
        self.assertTrue(last_season["y_gls_next"].isna().all(), "no 2025 season exists yet")
        self.assertFalse(last_season["has_next_season"].any())

    def test_no_filled_labels(self) -> None:
        """A present target must always be a real football number, never a default."""
        present = self.targets["has_next_season"]
        self.assertTrue(self.targets.loc[present, "y_gls_next"].notna().all())
        self.assertTrue((self.targets.loc[present, "minutes_next"] > 0).all())

        # The old bug wrote 0 into every missing label. Guard the pattern:
        # a player with a next season and minutes must not have a null target.
        null_but_present = self.targets["has_next_season"] & self.targets["y_gls_next"].isna()
        self.assertFalse(null_but_present.any())

    def test_left_dataset_is_flagged_not_zeroed(self) -> None:
        left = self.targets["left_dataset"]
        self.assertTrue(left.any(), "some players do leave the dataset")
        self.assertTrue(self.targets.loc[left, "y_gls_next"].isna().all())

    def test_target_rows_match_player_rows(self) -> None:
        self.assertEqual(len(self.targets), len(self.players))
        self.assertFalse(self.targets.duplicated(subset=["player_id", "season"]).any())

    def test_partial_next_season_is_never_a_target(self) -> None:
        """A mid-season snapshot cannot be a target: 18 goals in 29 games is not
        a full season, and comparing the two is what forced the old pipeline to
        multiply actuals by 1.5."""
        players = pd.DataFrame(
            {
                "player_id": ["a_2000", "b_2000"],
                "season": [2024, 2025],
                "position": ["FW", "FW"],
                "minutes": [2000, 1500],
                "matches": [30, 22],  # 2025 is only 22 rounds old
                "gls": [10, 8],
                "ast": [2, 3],
                "g_plus_a": [12, 11],
                "npxg": [9.0, 7.0],
                "npxg_plus_xag": [11.0, 10.0],
            }
        )
        targets = build_targets(players)
        season_2024 = targets[targets["season"] == 2024].iloc[0]

        self.assertFalse(bool(season_2024["next_season_complete"]))
        self.assertFalse(bool(season_2024["has_next_season"]))
        self.assertTrue(pd.isna(season_2024["y_gls_next"]))
        self.assertFalse(bool(season_2024["left_dataset"]))


if __name__ == "__main__":
    unittest.main()
