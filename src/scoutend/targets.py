"""Next-season targets, built so they cannot point at the wrong season.

The old pipeline used ``groupby('player').shift(-1)``. That breaks in two ways:
a player with two clubs in one season gets his own season as the "next" one, and
a player missing a season silently gets a target from two seasons later.

Here the join is explicit: ``(player_id, season)`` pairs with
``(player_id, season + 1)``. A pair that does not exist produces a null target
and a flag, never a number.

Targets live in their own table. Nothing in ``clean.py`` knows they exist, and
nothing here can write a null target back into the feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import complete_seasons
from .config import MINUTES_FLOOR

TARGET_SOURCE_COLUMNS = [
    "minutes",
    "matches",
    "gls",
    "ast",
    "g_plus_a",
    "npxg",
    "npxg_plus_xag",
]

# Per-90 versions, computed on the *next* season's real football, not on an
# assumed games-played total.
PER90_TARGETS = {
    "gls": "gls_next_per90",
    "ast": "ast_next_per90",
    "g_plus_a": "gp_a_next_per90",
    "npxg": "npxg_next_per90",
    "npxg_plus_xag": "npxg_plus_xag_next_per90",
}


def build_targets(players: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (player_id, season) with next-season outcomes."""
    current = players[["player_id", "season", "position", *TARGET_SOURCE_COLUMNS]].copy()
    current["next_season"] = current["season"] + 1

    future = players[["player_id", "season", *TARGET_SOURCE_COLUMNS]].rename(
        columns={col: f"{col}_next" for col in TARGET_SOURCE_COLUMNS}
    )
    future = future.rename(columns={"season": "next_season"})

    targets = current.merge(future, on=["player_id", "next_season"], how="left")

    # A mid-season snapshot can describe a player, but its totals cannot be a
    # target: "18 goals" over 29 matches is not comparable to a full season.
    complete = complete_seasons(players)
    targets["next_season_complete"] = targets["next_season"].isin(complete)
    incomplete = ~targets["next_season_complete"]
    next_value_columns = [
        column for column in targets.columns
        if column.endswith("_next") and column != "next_season"
    ]
    targets.loc[incomplete, next_value_columns] = np.nan

    targets["has_next_season"] = targets["minutes_next"].notna()

    # A player who was playing this season and has no row next season did not
    # score zero -- he left the dataset (relegated, transferred outside the Big
    # 5, retired, or injured out). That is a different outcome and a separate
    # modelling problem; flag it rather than hiding it.
    targets["left_dataset"] = (
        (~targets["has_next_season"]) & (targets["minutes"] > 0) & targets["next_season_complete"]
    )

    targets["y_gls_next"] = targets["gls_next"]
    targets["y_ast_next"] = targets["ast_next"]
    targets["y_minutes_next"] = targets["minutes_next"]

    nineties_next = targets["minutes_next"] / 90.0
    for source, name in PER90_TARGETS.items():
        targets[name] = targets[f"{source}_next"] / nineties_next.where(nineties_next > 0)

    targets["status_next"] = _status(targets)

    # --- invariants ---------------------------------------------------------
    _assert_no_lookahead(targets)
    _assert_targets_are_real(targets)

    columns = [
        "player_id",
        "season",
        "next_season",
        "next_season_complete",
        "position",
        "has_next_season",
        "left_dataset",
        "y_gls_next",
        "y_ast_next",
        "y_minutes_next",
        "gls_next_per90",
        "ast_next_per90",
        "gp_a_next_per90",
        "npxg_next_per90",
        "npxg_plus_xag_next_per90",
        "status_next",
        "minutes",
        "minutes_next",
    ]
    return targets[columns].sort_values(["season", "player_id"]).reset_index(drop=True)


def _status(targets: pd.DataFrame) -> pd.Series:
    """Breakout / steady / decline -- only for players with a real, sizeable role.

    Thresholds are explicit and configurable rather than buried in a cell:
    a "breakout" is >=25% more goals per 90 next season, on at least
    ``MINUTES_FLOOR`` minutes in *both* seasons, so a two-goal cameo cannot
    register as a breakout.
    """
    enough_now = targets["minutes"] >= MINUTES_FLOOR
    enough_next = targets["minutes_next"] >= MINUTES_FLOOR
    eligible = enough_now & enough_next

    current_rate = targets["gls"] / (targets["minutes"] / 90.0).where(targets["minutes"] > 0)
    next_rate = targets["gls_next"] / (targets["minutes_next"] / 90.0).where(targets["minutes_next"] > 0)
    change = (next_rate - current_rate) / current_rate.where(current_rate > 0)

    status = pd.Series(pd.NA, index=targets.index, dtype="object")
    status[eligible & (change >= 0.25)] = "breakout"
    status[eligible & (change <= -0.25)] = "decline"
    status[eligible & (change.abs() < 0.25)] = "steady"
    # A player who played but did not score: change is undefined, so he is
    # steady by construction unless he starts scoring.
    status[eligible & current_rate.eq(0) & next_rate.gt(0)] = "breakout"
    status[eligible & current_rate.eq(0) & next_rate.eq(0)] = "steady"
    return status


def _assert_no_lookahead(targets: pd.DataFrame) -> None:
    """Every target must come from exactly season + 1."""
    if not (targets["next_season"] == targets["season"] + 1).all():
        raise AssertionError("target join produced a season other than season + 1")

    same_season = targets["has_next_season"] & (targets["next_season"] == targets["season"])
    if same_season.any():
        raise AssertionError(
            f"{int(same_season.sum())} targets resolve to the same season "
            "(mid-season transfer rows not merged?)"
        )


def _assert_targets_are_real(targets: pd.DataFrame) -> None:
    """A null target must mean 'no next season', never 'filled with a default'."""
    missing = targets["y_gls_next"].isna()
    fabricated = missing & targets["has_next_season"]
    if fabricated.any():
        raise AssertionError(
            f"{int(fabricated.sum())} rows have a next season but a null goals target"
        )

    present = targets["y_gls_next"].notna()
    if (targets.loc[present, "minutes_next"] <= 0).any():
        raise AssertionError("a target exists for a player-season with no minutes")
