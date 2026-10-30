"""The measuring stick.

Every model is scored on a strictly forward-looking split: train on seasons
strictly before the test season, predict the following season. No shuffled
folds, no rows from the test season in training, and baselines are scored on
exactly the same rows as the models.

The output is deliberately unflattering. If a Poisson two-stage model cannot
beat "last season's npxG", the report says so.

Run:  python run.py eval
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features as features_module
from ..config import MINUTES_FLOOR, PATHS, Paths
from ..models.card import ModelCard, card_from_results
from ..models.registry import DEFAULT_TARGETS, TARGETS, predict_all
from .metrics import calibration_table, regression_report, top_k_overlap

DEFAULT_TEST_SEASONS = (2021, 2022, 2023)
TARGET_COLUMNS = ["y_gls_next", "y_ast_next", "y_minutes_next", "has_next_season"]


def load_dataset(paths: Paths = PATHS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed tables, building them first if they are absent."""
    player_path = paths.processed / "player_seasons.csv.gz"
    target_path = paths.processed / "targets.csv.gz"

    if not player_path.exists() or not target_path.exists():
        from ..clean import build
        from ..targets import build_targets

        players, _ = build(paths)
        targets = build_targets(players)
        targets.to_csv(target_path, index=False, compression="gzip")
    else:
        players = pd.read_csv(player_path)
        targets = pd.read_csv(target_path)

    features = features_module.build_features(players)
    return players, targets, features


def evaluate(
    targets: pd.DataFrame,
    features: pd.DataFrame,
    scored_targets: tuple[str, ...] = DEFAULT_TARGETS,
    test_seasons: tuple[int, ...] = DEFAULT_TEST_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every candidate on each forward split.

    Returns ``(metrics, predictions)``. The per-row predictions are kept so the
    report can show calibration -- averages hide whether a model is right for
    the wrong reasons.
    """
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for target in scored_targets:
        frame = features.merge(
            targets[["player_id", "season", *TARGET_COLUMNS]].drop_duplicates(
                subset=["player_id", "season"]
            ),
            on=["player_id", "season"],
            how="inner",
        )
        frame = frame[frame["has_next_season"] & frame[target].notna()].copy()
        matrix, columns = features_module.feature_matrix(frame)

        for test_season in test_seasons:
            train = (frame["season"] < test_season).to_numpy()
            test = (frame["season"] == test_season).to_numpy()

            if not train.any() or not test.any():
                continue
            if set(frame.loc[train, "season"]) & set(frame.loc[test, "season"]):
                raise AssertionError("train/test season overlap")

            predictions = predict_all(frame, matrix, columns, train, test, target)

            truth = frame.loc[test, target].to_numpy(dtype=float)
            qualified = frame.loc[test, "minutes"].fillna(0).to_numpy() >= MINUTES_FLOOR
            trained_on = sorted(int(s) for s in frame.loc[train, "season"].unique())

            for prediction in predictions:
                for scope, mask in (
                    ("all", np.ones(len(truth), dtype=bool)),
                    ("qualified", qualified),
                ):
                    y, p = truth[mask], prediction.values[mask]
                    if len(y) < 20:
                        continue
                    report = regression_report(y, p)
                    report["top20_overlap"] = top_k_overlap(y, p, k=20)
                    metric_rows.append(
                        {
                            "target": target,
                            "target_label": TARGETS[target]["label"],
                            "test_season": test_season,
                            "trained_on": trained_on,
                            "model": prediction.name,
                            "family": prediction.family,
                            "scope": scope,
                            **report,
                            "mean_actual": float(np.mean(y)),
                            "mean_predicted": float(np.mean(p)),
                        }
                    )

                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "test_season": test_season,
                            "player_id": frame.loc[test, "player_id"].to_numpy(),
                            "model": prediction.name,
                            "actual": truth,
                            "predicted": prediction.values,
                            "qualified": qualified,
                        }
                    )
                )

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Pool folds: weight the fold metrics by fold size."""

    def weighted(group: pd.DataFrame) -> pd.Series:
        weight = group["n"] / group["n"].sum()
        return pd.Series(
            {
                "n": int(round(group["n"].sum())),
                "family": group["family"].iloc[0],
                "mae": float((group["mae"] * weight).sum()),
                "rmse": float((group["rmse"] * weight).sum()),
                "r2": float(group["r2"].mean()),
                "spearman": float((group["spearman"] * weight).sum()),
                "top20_overlap": float((group["top20_overlap"] * weight).sum()),
                "bias": float((group["mean_predicted"] - group["mean_actual"]).mean()),
            }
        )

    summary = (
        results.groupby(["target", "scope", "model"], observed=True)
        .apply(weighted, include_groups=False)
        .reset_index()
        .sort_values(["target", "scope", "mae"])
    )
    summary["n"] = summary["n"].astype(int)
    return summary


def write_report(summary: pd.DataFrame, path) -> None:
    """A small markdown report, so the numbers are readable outside a terminal."""
    lines = ["# Evaluation report", "", "Next-season goals and assists, forward splits only.", ""]
    for target in summary["target"].unique():
        label = TARGETS[target]["label"]
        for scope in ("qualified", "all"):
            block = summary[(summary["target"] == target) & (summary["scope"] == scope)]
            if block.empty:
                continue
            title = "players with >= 450 minutes" if scope == "qualified" else "all players"
            lines += [f"## {label.title()} — {title}", ""]
            lines.append("| model | family | n | MAE | RMSE | R² | Spearman | top-20 | bias |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for _, row in block.iterrows():
                lines.append(
                    f"| `{row['model']}` | {row['family']} | {row['n']} | "
                    f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} | "
                    f"{row['spearman']:.3f} | {row['top20_overlap']:.3f} | {row['bias']:+.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


CARD_DESCRIPTIONS = {
    "poisson_blended_two_stage": (
        "Ship candidate. Poisson per-90 rate blended toward the position's shrunk rate by "
        "minutes played, multiplied by a Gamma expected-minutes model."
    ),
    "poisson_blended_x_actual_minutes": (
        "Blended Poisson rate with the player's current minutes as exposure. "
        "Separates the rate model from the minutes model."
    ),
    "poisson_two_stage": (
        "Unblended Poisson rate multiplied by expected minutes. Kept for comparison: "
        "it is what the blend is protecting against."
    ),
    "poisson_rate_x_actual_minutes": (
        "Unblended Poisson per-90 rate with current minutes as exposure."
    ),
    "ridge": "L2 linear regression on the full feature matrix. The reference model.",
}


def build_cards(results: pd.DataFrame, summary: pd.DataFrame, directory) -> list[ModelCard]:
    """Write a card for each model that is a candidate for shipping."""
    cards: list[ModelCard] = []

    for target in results["target"].unique():
        label = TARGETS[target]["label"]
        target_summary = summary[summary["target"] == target]
        target_results = results[results["target"] == target]
        trained_on = sorted(
            {
                int(season)
                for seasons in target_results["trained_on"]
                for season in seasons
            }
        )
        evaluated_on = sorted(int(s) for s in target_results["test_season"].unique())

        for model, description in CARD_DESCRIPTIONS.items():
            if model not in set(target_results["model"]):
                continue
            card = card_from_results(
                name=f"{model}_{label}",
                model_key=model,
                target=target,
                description=description,
                results=target_summary,
                features=features_module.FEATURES,
                trained_on=trained_on,
                evaluated_on=evaluated_on,
                hyperparameters={"alpha": 1.0} if "poisson" in model else {"alpha": 10.0},
                limitations=[
                    "Big-5 only; no cross-league strength adjustment yet.",
                    "Training window ends at season 2023; 2024/25 and 2025/26 are not ingested.",
                    "Totals ignore penalties taken and opposition quality.",
                ],
            )
            card.write(directory)
            cards.append(card)
    return cards


def run(paths: Paths = PATHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, targets, features = load_dataset(paths)
    results, predictions = evaluate(targets, features)
    summary = summarise(results)

    results.to_csv(paths.processed / "eval_by_fold.csv", index=False)
    summary.to_csv(paths.processed / "eval_summary.csv", index=False)
    if _has_parquet():
        predictions.to_parquet(paths.processed / "eval_predictions.parquet", index=False)
    else:
        predictions.to_csv(
            paths.processed / "eval_predictions.csv.gz", index=False, compression="gzip"
        )
    write_report(summary, paths.processed / "eval_report.md")

    calibration_frames = []
    for (target, model), group in predictions.groupby(["target", "model"], observed=True):
        table = calibration_table(group["actual"], group["predicted"], bins=10)
        table["target"] = target
        table["model"] = model
        calibration_frames.append(table)
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        paths.processed / "eval_calibration.csv", index=False
    )

    build_cards(results, summary, paths.root / "models")
    return results, summary


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False
