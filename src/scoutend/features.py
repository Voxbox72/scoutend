"""Model-ready features, deliberately small and leakage-safe.

Design rules:

* **Current season only.** Every feature is computed from the row it belongs to
  (or from an earlier season via an exact ``season - 1`` lag). Nothing reads
  ``season + 1``; that is the target's job.
* **Rates with an explicit denominator.** Per-90 features divide by
  ``minutes / 90`` and are null when minutes are zero, never silently zero.
* **Shrinkage where it matters.** ``*_per90_shrunk`` pulls a player's rate
  toward his position's rate using a minutes-based prior, which is what stops
  one 200-minute cameo from topping a scouting list.

The feature list is intentionally ~25 columns rather than the previous 169.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-90 rate features: canonical count column -> feature name.
PER90_FEATURES = {
    "gls": "gls_per90",
    "npxg": "npxg_per90",
    "xg": "xg_per90",
    "ast": "ast_per90",
    "xag": "xag_per90",
    "xa": "xa_per90",
    "sca": "sca_per90",
    "gca": "gca_per90",
    "shots": "shots_per90",
    "shots_on_target": "sot_per90",
    "passes_completed": "passes_completed_per90",
    "prg_passes": "prg_passes_per90",
    "prg_carries": "prg_carries_per90",
    "prg_receives": "prg_receives_per90",
    "touches_att_pen": "touches_att_pen_per90",
    "take_ons_won": "take_ons_won_per90",
    "tackles": "tackles_per90",
    "interceptions": "interceptions_per90",
    "clearances": "clearances_per90",
    "blocks": "blocks_per90",
}

# Prior strength for the shrinkage prior, in 90s. 10 matches is enough for a
# rotation player to be pulled most of the way to his position's mean.
SHRINKAGE_PRIOR_90S = 10.0

# A per-90 rate from a tiny sample is not a rate, it is an accident: one goal in
# 25 minutes is "3.6 goals per 90". Dividing by actual minutes produces absurd
# inputs (passes_completed_per90 of 315 in this dataset), and a log-link model
# exponentiates them. These caps are generous -- roughly twice an elite season --
# so they never bind for a regular starter but stop a cameo from dominating.
PER90_CAPS = {
    "gls_per90": 3.0,
    "npxg_per90": 3.0,
    "xg_per90": 3.5,
    "ast_per90": 2.5,
    "xag_per90": 2.5,
    "xa_per90": 2.5,
    "sca_per90": 15.0,
    "gca_per90": 4.0,
    "shots_per90": 12.0,
    "sot_per90": 8.0,
    "passes_completed_per90": 130.0,
    "prg_passes_per90": 30.0,
    "prg_carries_per90": 30.0,
    "prg_receives_per90": 35.0,
    "touches_att_pen_per90": 18.0,
    "take_ons_won_per90": 18.0,
    "tackles_per90": 15.0,
    "interceptions_per90": 12.0,
    "clearances_per90": 18.0,
    "blocks_per90": 10.0,
}


def build_features(players: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per (player_id, season)."""
    frame = players.copy()
    nineties = (frame["minutes"] / 90.0).where(frame["minutes"] > 0)

    frame["age"] = frame["season"] - pd.to_numeric(frame["born"], errors="coerce")
    # Age curve: minutes and output rise into the mid-20s and fall after 30.
    # A squared term lets one coefficient describe both sides.
    frame["age_squared"] = frame["age"] ** 2
    frame["nineties"] = nineties
    frame["start_share"] = (frame["starts"] / frame["matches"]).where(frame["matches"] > 0)

    for source, name in PER90_FEATURES.items():
        if source not in frame.columns:
            raise KeyError(f"feature source missing from the canonical table: {source}")
        frame[name] = frame[source] / nineties

    # Winsorise before anything else uses these rates: shrinkage and the models
    # should see a plausible football number, not a 25-minute extrapolation.
    for name, cap in PER90_CAPS.items():
        frame[name] = frame[name].clip(lower=0.0, upper=cap)

    # Ratios (only meaningful with a real denominator).
    frame["shot_accuracy"] = (
        frame["shots_on_target"] / frame["shots"].where(frame["shots"] > 0)
    ).clip(0.0, 1.0)
    frame["pass_completion"] = (
        frame["passes_completed"] / frame["passes_attempted"].where(frame["passes_attempted"] > 0)
    ).clip(0.0, 1.0)
    frame["xag_over_goals_assisted"] = (
        (frame["ast"] - frame["xag"]) / nineties
    ).clip(-3.0, 3.0)  # finishing/passing luck, per 90

    # Shrinkage: empirical-Bayes-style pull toward the position-season mean.
    for source in ("gls", "npxg", "ast"):
        raw = f"{source}_per90"
        prior = frame.groupby(["position", "season"])[raw].transform("mean")
        frame[f"{raw}_shrunk"] = _shrink(frame[raw], prior, nineties, SHRINKAGE_PRIOR_90S)

    # Prior-season form, lagged by exactly one season (no shift over gaps: a
    # missing season yields null, not a two-year-old value).
    lag = frame[["player_id", "season", "gls_per90", "npxg_per90", "minutes"]].copy()
    lag["season"] = lag["season"] + 1
    lag = lag.rename(
        columns={
            "gls_per90": "gls_per90_prev",
            "npxg_per90": "npxg_per90_prev",
            "minutes": "minutes_prev",
        }
    )
    frame = frame.merge(lag, on=["player_id", "season"], how="left")

    frame["position"] = frame["position"].astype("category")
    frame["league_code"] = frame["league_code"].astype("category")

    _assert_current_season_only(frame)
    return frame


def _shrink(
    value: pd.Series, prior: pd.Series, nineties: pd.Series, prior_90s: float
) -> pd.Series:
    """weight = n / (n + k) with n in 90s played."""
    n = nineties.fillna(0.0)
    weight = n / (n + prior_90s)
    return weight * value.fillna(prior) + (1 - weight) * prior


FEATURES = [
    "age",
    "age_squared",
    "minutes",
    "matches",
    "starts",
    "start_share",
    "gls_per90",
    "npxg_per90",
    "xg_per90",
    "ast_per90",
    "xag_per90",
    "xa_per90",
    "sca_per90",
    "gca_per90",
    "shots_per90",
    "sot_per90",
    "passes_completed_per90",
    "prg_passes_per90",
    "prg_carries_per90",
    "prg_receives_per90",
    "touches_att_pen_per90",
    "take_ons_won_per90",
    "tackles_per90",
    "interceptions_per90",
    "clearances_per90",
    "blocks_per90",
    "shot_accuracy",
    "pass_completion",
    "gls_per90_shrunk",
    "npxg_per90_shrunk",
    "ast_per90_shrunk",
    "gls_per90_prev",
    "npxg_per90_prev",
    "minutes_prev",
]


def feature_matrix(
    features: pd.DataFrame,
    *,
    include: list[str] | None = None,
    one_hot: tuple[str, ...] = ("position", "league_code"),
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the numeric design matrix, one-hot encoding categoricals."""
    columns = include or FEATURES
    matrix = features[columns].copy()
    for column in one_hot:
        dummies = pd.get_dummies(features[column].astype(str), prefix=column, dtype=float)
        matrix = pd.concat([matrix, dummies], axis=1)
    return matrix, list(matrix.columns)


def _assert_current_season_only(features: pd.DataFrame) -> None:
    """Guard against anyone adding a column that knows the future."""
    forbidden = [c for c in features.columns if "next" in c]
    if forbidden:
        raise AssertionError(f"features must not contain next-season columns: {forbidden}")
