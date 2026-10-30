"""The scouting layer: percentiles, roles, similarity and shortlist scores.

Everything here answers a question a scout actually asks, not a statistics one:

* **How does he compare?** -> percentiles against players of the same position.
* **What kind of player is he?** -> a role cluster learned from style, not assumed.
* **Who plays like him?** -> nearest neighbours in style space.
* **Should we look at him?** -> a shortlist score combining improvement,
  room to grow, minutes, and how underrated he is by his raw output.

Design notes that matter:

* Percentiles are computed **within position and season**, so a defender is not
  compared to a striker and a 2019/20 centre-back is not compared to a 2023/24 one.
* A player-season with under 450 minutes is excluded from the pool a percentile
  is computed against. Letting a 90-minute cameo into the reference population
  is what produces "99th percentile for carries".
* Style clustering uses **rate or share features only** (never volume totals),
  so a squad player's role is not confused with a starter's.
* Similarity is cosine distance on z-scored style vectors within the same
  position group, which is the only comparison that is meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MINUTES_FLOOR, PATHS, Paths
from .features import build_features

# --- what a scout looks at -------------------------------------------------
# (feature, pretty label, direction) -- direction -1 means lower is better.
OUTFIELD_METRICS: list[tuple[str, str, int]] = [
    ("gls_per90", "Goals /90", 1),
    ("npxg_per90", "npxG /90", 1),
    ("ast_per90", "Assists /90", 1),
    ("xag_per90", "xAG /90", 1),
    ("shots_per90", "Shots /90", 1),
    ("sot_per90", "Shots on target /90", 1),
    ("sca_per90", "Shot-creating actions /90", 1),
    ("gca_per90", "Goal-creating actions /90", 1),
    ("passes_completed_per90", "Passes completed /90", 1),
    ("pass_completion", "Pass completion", 1),
    ("prg_passes_per90", "Progressive passes /90", 1),
    ("prg_carries_per90", "Progressive carries /90", 1),
    ("take_ons_won_per90", "Successful take-ons /90", 1),
    ("prg_receives_per90", "Progressive receptions /90", 1),
    ("touches_att_pen_per90", "Touches in the box /90", 1),
    ("tackles_per90", "Tackles /90", 1),
    ("interceptions_per90", "Interceptions /90", 1),
    ("clearances_per90", "Clearances /90", 1),
    ("blocks_per90", "Blocks /90", 1),
    ("age", "Age", -1),
]

# Style features for clustering and similarity: rates and shares, never totals.
STYLE_FEATURES = [
    "gls_per90",
    "npxg_per90",
    "ast_per90",
    "xag_per90",
    "shots_per90",
    "sca_per90",
    "gca_per90",
    "prg_passes_per90",
    "prg_carries_per90",
    "take_ons_won_per90",
    "prg_receives_per90",
    "touches_att_pen_per90",
    "tackles_per90",
    "interceptions_per90",
    "clearances_per90",
    "blocks_per90",
    "pass_completion",
]

ROLE_LABELS = {
    "Finisher": "Finisher",
    "Creator": "Creator",
    "Progressor": "Ball progressor",
    "Destroyer": "Defensive/aerial",
    "Retainer": "Possession retainer",
}


def load_panel(paths: Paths = PATHS) -> pd.DataFrame:
    """Player-season table with features and percentile ranks attached."""
    players = pd.read_csv(paths.processed / "player_seasons.csv.gz")
    frame = build_features(players)
    frame = frame[frame["position"].isin(["DF", "MF", "FW"])].copy()
    return add_percentiles(frame)


def add_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each metric among qualified players of the same position.

    Computed per season, so a player is judged against his peers in that year.
    ``direction`` flips metrics where lower is better (age).

    Ties share a percentile. Real data has many -- every forward who did not take
    a penalty has identical non-penalty xG per 90 -- and giving them different
    percentiles would invent a ranking the data does not support.
    """
    pool = frame[frame["minutes"] >= MINUTES_FLOOR].copy()
    keys = ["position", "season"]

    for feature, _, direction in OUTFIELD_METRICS:
        if feature not in frame.columns:
            continue
        values = frame[feature]
        grouped = values.groupby([frame[k] for k in keys])
        # Average rank across the tied group, then centred on the group so a
        # value shared by n of the group's players maps to one percentile.
        average_rank = grouped.rank(method="average")
        group_size = grouped.transform("size")
        tied_count = grouped.transform(_tie_counts)
        percentile = ((average_rank - (tied_count + 1) / 2) / (group_size - 1)) * 100
        percentile = percentile.where(group_size > 1, 100.0).round(1)
        if direction < 0:
            percentile = 100 - percentile
        frame[f"pct_{feature}"] = percentile

    empty = pool.groupby(keys).size()
    if empty.empty:
        raise ValueError("no qualified players to build percentiles from")
    return frame


def _tie_counts(series: pd.Series) -> pd.Series:
    """How many values in the group equal each value."""
    counts = series.value_counts()
    return series.map(counts).astype(float)


def style_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Z-scored style vectors, standardised within each position group."""
    columns = [c for c in STYLE_FEATURES if c in frame.columns]
    style = frame[columns].astype(float)

    # Z-score *within position*, so "progressive passes" is not systematically
    # scored higher for midfielders than for centre-backs.
    grouped = style.groupby(frame["position"].astype(str))
    z = (style - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
    z = z.fillna(0.0)
    return z, columns


def _kmeans(matrix: np.ndarray, k: int, iterations: int = 100, seed: int = 42) -> np.ndarray:
    """Plain k-means. Small enough to audit, deterministic with a fixed seed."""
    rng = np.random.default_rng(seed)
    n = len(matrix)
    if n == 0:
        return np.array([], dtype=int)
    k = min(k, n)

    # k-means++ seeding: first centre random, then proportional to distance^2.
    centres = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        distances = np.min(
            ((matrix[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        total = distances.sum()
        weights = distances / total if total > 0 else np.full(n, 1 / n)
        centres.append(matrix[rng.choice(n, p=weights)])
    centres = np.array(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for index in range(k):
            members = matrix[labels == index]
            if len(members):
                centres[index] = members.mean(axis=0)
    return labels


# Style dimensions grouped into the role they indicate. A cluster takes the
# label of the group it is *most above par* on.
FINISHING = {"gls_per90", "npxg_per90", "shots_per90", "touches_att_pen_per90"}
# "Creator" means he makes chances for others. Goal-creating actions include
# his own finishes, so they identify strikers; assists, xAG and SCA do not.
CREATION = {"ast_per90", "xag_per90", "sca_per90"}
PROGRESSION = {"prg_passes_per90", "prg_carries_per90", "take_ons_won_per90"}
DEFENDING = {"tackles_per90", "interceptions_per90", "clearances_per90", "blocks_per90"}
RETENTION = {"prg_receives_per90", "pass_completion"}

ROLE_GROUPS: list[tuple[str, set[str]]] = [
    ("Finisher", FINISHING),
    ("Creator", CREATION),
    ("Progressor", PROGRESSION),
    ("Aerial/defensive", DEFENDING),
    ("Retainer", RETENTION),
]

# How far above the position's average a group must be to earn a role label.
LEAD_THRESHOLD = 0.25
# An "all-round" cluster is above par on most dimensions at once. Chosen from
# the actual clusters: role groups sit at 0-5 dimensions above +0.5 z, the elite
# all-round groups at 8 or more, so the cut goes in the gap between them rather
# than at a plausible-looking round number.
ALL_ROUND_DIMENSIONS = 8


def _label_cluster(profile: pd.Series) -> str:
    """Name a cluster after the role it most exceeds its position's average on.

    The profile is mean z-scores within the position group, so the question is
    "what does this group have *more of* than its peers", never "what is its
    biggest raw number" -- clearances per 90 dwarf passes completed for a
    centre-back, which is why a raw-value rule labels every defender the same.

    One rule, applied in order:

    1. nothing is above ``LEAD_THRESHOLD`` -> **Balanced** (at par everywhere);
    2. above par on most dimensions -> **All-round elite**;
    3. otherwise -> the first role group with a dimension above the threshold,
       ordered so that finishing and creation take precedence over the rest.
    """
    values = profile.dropna()
    if values.empty:
        return "Unclassified"

    above = values[values >= LEAD_THRESHOLD]
    if above.empty:
        return "Balanced"

    # Count *dimensions*, not the share of a short profile: a hand-built test
    # profile with three entries is not evidence of an all-round player.
    if int((values > 0.5).sum()) >= ALL_ROUND_DIMENSIONS:
        return "All-round elite"

    for label, members in ROLE_GROUPS:
        scores = above[above.index.isin(members)]
        if not scores.empty:
            return label

    return "Balanced"


def assign_roles(frame: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Cluster players into style roles, per position group.

    Roles differ by position (a centre-back's "finisher" is not a striker's), so
    each position group is clustered separately with the same k.
    """
    z, columns = style_matrix(frame)
    qualified = (frame["minutes"] >= MINUTES_FLOOR).to_numpy()

    frame = frame.copy()
    frame["role_cluster"] = pd.NA
    frame["role"] = pd.NA

    for position, _ in frame.groupby("position"):
        mask = (frame["position"] == position).to_numpy() & qualified
        indices = np.flatnonzero(mask)
        if len(indices) < k:
            continue

        labels = _kmeans(z.iloc[indices].to_numpy(), k=k)
        cluster_ids = pd.Series([f"{position}-{label}" for label in labels], index=indices)
        frame.loc[frame.index[indices], "role_cluster"] = cluster_ids.to_numpy()

        # Profile each cluster by its *mean z-score* within this position. The
        # positions themselves are position-specific, so no position is ever
        # empty and naming never hits a missing key.
        profiles = {}
        for label in np.unique(labels):
            group = indices[labels == label]
            profiles[int(label)] = z.iloc[group].mean(axis=0)
        naming = {label: _label_cluster(profiles[label]) for label in profiles}

        # A position yields 6 clusters for a handful of roles, so two clusters
        # can earn the same name. A scout choosing between "Progressor" and
        # "Progressor" cannot act on that, so duplicates get an intensity rank
        # by average z: "Progressor" and then "Progressor (core)".
        strength = {label: float(profiles[label].mean()) for label in profiles}
        duplicates: dict[str, list[int]] = {}
        for label, name in naming.items():
            duplicates.setdefault(name, []).append(label)
        for name, members in duplicates.items():
            if len(members) < 2:
                continue
            ordered = sorted(members, key=lambda m: strength[m], reverse=True)
            for rank, label in enumerate(ordered[1:], start=2):
                naming[label] = f"{name} (core)" if rank == 2 else f"{name} (variant {rank})"

        frame.loc[frame.index[indices], "role"] = [naming[int(c)] for c in labels]

    return frame


def similarity(frame: pd.DataFrame, player_row: pd.Series, n: int = 8) -> pd.DataFrame:
    """Players in the same position group with the closest style profile.

    Cosine distance on z-scored style vectors. Same position only: comparing a
    full-back's style to a striker's produces confident nonsense.
    """
    z, columns = style_matrix(frame)
    candidate_mask = (frame["position"] == player_row["position"]).to_numpy()

    matrix = z[candidate_mask].to_numpy()
    names = frame.loc[candidate_mask, "player_id"].to_numpy()

    target = z.loc[[player_row.name]].to_numpy()[0]
    norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
    denom = np.where(norm == 0, 1.0, norm)
    cosine = (matrix @ target) / denom

    result = frame.loc[candidate_mask, ["player_id", "player", "season", "season_label",
                                        "team", "position", "age", "minutes", "role"]].copy()
    result["similarity"] = cosine.round(4)
    result = result[result["player_id"] != player_row["player_id"]]
    return result.nlargest(n, "similarity").reset_index(drop=True)


def projection_and_value(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the next-season projection to the latest season's player rows."""
    latest = frame["season"].max()
    base = frame[frame["season"] == latest].copy()

    merged = base.merge(
        predictions[["player_id", "expected_minutes_next", "predicted_goals_next",
                     "predicted_assists_next"]],
        on="player_id",
        how="left",
    )

    # Minutes-weighted output rate, used as the "what has he actually done"
    # signal. Kept separate from the projection on purpose: one is evidence,
    # the other is a forecast.
    merged["output_next_total"] = merged["predicted_goals_next"] + merged["predicted_assists_next"]
    merged["output_next_per90"] = (
        merged["output_next_total"] / (merged["expected_minutes_next"] / 90.0)
    ).where(merged["expected_minutes_next"] > 0)

    # Opportunity gap: how much of a player's upside is currently blocked.
    merged["minutes_upside"] = merged["expected_minutes_next"] - merged["minutes"]

    merged["shortlist_score"] = shortlist_score(merged)
    return merged.sort_values("shortlist_score", ascending=False).reset_index(drop=True)


def shortlist_score(frame: pd.DataFrame) -> pd.Series:
    """A 0-100 score for "should a scout look at him", and why it is built this way.

    Four components, each on a 0-1 scale:

    * **quality** â€“ projected output per 90, the ability estimate.
    * **role** â€“ expected minutes, the opportunity estimate.
    * **upside** â€“ his age relative to the position's peak: young players with
      production are worth more than the same production from a 31-year-old.
    * **value** â€“ how far his projection exceeds his raw output from last
      season, i.e. the market may be underrating him.

    The weights are a stated opinion, not a fitted quantity. They live in one
    function so a reader can disagree with them in one place.
    """
    def scale(series: pd.Series) -> pd.Series:
        result = series.astype(float)
        span = result.max() - result.min()
        if not np.isfinite(span) or span == 0:
            return pd.Series(0.5, index=result.index)
        return (result - result.min()) / span

    quality = scale(frame["output_next_per90"].fillna(0))
    opportunity = scale(frame["expected_minutes_next"].fillna(0))
    upside = scale(-(frame["age"] - 24).abs())  # peak value near 24
    value = scale((frame["predicted_goals_next"] - frame["gls"]).fillna(0))

    score = 0.40 * quality + 0.25 * opportunity + 0.20 * upside + 0.15 * value
    return (score * 100).round(1)
