"""Command line interface: ``python run.py build|eval|train|all``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .clean import build
from .config import PATHS
from .eval.harness import run as run_evaluation
from .models.registry import TARGETS
from .models.train import train_and_predict
from .targets import build_targets


def _build() -> None:
    from .clean import complete_seasons

    players, keepers = build(PATHS)
    targets = build_targets(players)
    targets.to_csv(PATHS.processed / "targets.csv.gz", index=False, compression="gzip")

    seasons = sorted(players["season"].unique())
    complete = complete_seasons(players)
    partial = [s for s in seasons if s not in complete]

    print(f"player_seasons       {players.shape[0]:>6} rows x {players.shape[1]:>3} cols")
    print(f"goalkeeper_seasons   {keepers.shape[0]:>6} rows x {keepers.shape[1]:>3} cols")
    print(f"targets              {targets.shape[0]:>6} rows "
          f"({int(targets['has_next_season'].sum())} with a real next season)")
    print(f"players              {players['player_id'].nunique():>6} distinct ids")
    print(f"seasons              {seasons}")
    print(f"complete seasons     {sorted(complete)}")
    if partial:
        print(f"PARTIAL seasons      {partial}  (features only, never a target)")
    print(f"multi-club rows      {int(players['multi_club'].sum()):>6}")
    print(f"written to           {PATHS.processed}")


def _eval() -> None:
    results, summary = run_evaluation(PATHS)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    for target in summary["target"].unique():
        label = TARGETS[target]["label"]
        for scope in ("qualified", "all"):
            block = summary[(summary["target"] == target) & (summary["scope"] == scope)]
            if block.empty:
                continue
            title = ">= 450 minutes" if scope == "qualified" else "all players"
            print(f"\n=== next-season {label}, {title} ===")
            print(
                block[["model", "family", "n", "mae", "rmse", "r2", "spearman", "top20_overlap", "bias"]]
                .to_string(index=False)
            )

    print(f"\nreport    {PATHS.processed / 'eval_report.md'}")
    print(f"summary   {PATHS.processed / 'eval_summary.csv'}")
    print(f"cards     {PATHS.root / 'models'}")


def _train() -> None:
    predictions = train_and_predict(PATHS)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    top = predictions.nlargest(12, "predicted_goals_next")
    print("\n=== top projected goals for 2025 (trained through 2023/24) ===")
    print(
        top[["player", "team", "position", "age", "minutes", "expected_minutes_next",
             "predicted_goals_next", "predicted_assists_next"]].to_string(index=False)
    )
    print(f"\nwritten to {PATHS.processed / 'predictions_next_season.csv.gz'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scoutend", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="raw files -> canonical dataset + targets")
    sub.add_parser("eval", help="score every model against the baselines")
    sub.add_parser("train", help="fit on all data and project the next season")
    sub.add_parser("all", help="build, eval and train")

    args = parser.parse_args(argv)

    if args.command in ("build", "all"):
        _build()
    if args.command in ("eval", "all"):
        _eval()
    if args.command in ("train", "all"):
        _train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
