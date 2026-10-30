"""Evaluation metrics. No scipy, no sklearn -- rank correlation by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1 - np.sum((y_true - y_pred) ** 2) / denominator)


def spearman(y_true, y_pred) -> float:
    """Rank correlation: does the ordering match, even if the scale is off?

    This is the metric that matters for a scouting list, where the product is a
    ranking, not a point estimate.
    """
    a = pd.Series(np.asarray(y_true, float)).rank()
    b = pd.Series(np.asarray(y_pred, float)).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def top_k_overlap(y_true, y_pred, k: int = 20) -> float:
    """Share of the true top-k that the model also puts in its top-k.

    A blunt but honest measure of "would a scout have found the right players".
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    k = min(k, len(frame))
    if k == 0:
        return float("nan")
    true_top = set(frame.nlargest(k, "y").index)
    pred_top = set(frame.nlargest(k, "p").index)
    return len(true_top & pred_top) / k


def calibration_table(y_true, y_pred, bins: int = 10) -> pd.DataFrame:
    """Predicted vs actual, grouped by prediction decile.

    Aggregate MAE cannot tell you *where* a model is wrong. This can: if the top
    decile is predicted at 12 goals and actually scores 8, the model is
    over-rating its shortlist -- the exact failure a scouting tool must not have.
    """
    frame = pd.DataFrame(
        {"actual": np.asarray(y_true, float), "predicted": np.asarray(y_pred, float)}
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["bin", "n", "mean_predicted", "mean_actual", "bias"]
        )

    try:
        frame["bin"] = pd.qcut(frame["predicted"].rank(method="first"), bins, labels=False)
    except ValueError:  # too few distinct predictions to cut into deciles
        frame["bin"] = 0

    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
    )
    grouped["bias"] = grouped["mean_predicted"] - grouped["mean_actual"]
    return grouped
