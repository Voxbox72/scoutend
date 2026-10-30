"""Model cards: the numbers that let someone else decide whether to trust a model.

Every model that ships gets a JSON card (machine-readable) and a Markdown card
(human-readable) recording what it predicts, on which data, with which
baselines it was compared, and how well it did. A model without a card does not
go in the repository.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class ModelCard:
    name: str
    target: str
    description: str
    trained_on: list[int]
    evaluated_on: list[int]
    features: list[str]
    hyperparameters: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    baselines_beaten: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    python: str = field(default_factory=lambda: platform.python_version())
    version: str = "0.3.0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{self.name}.json"
        markdown_path = directory / f"{self.name}.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, markdown_path

    def to_markdown(self) -> str:
        lines = [
            f"# Model card: {self.name}",
            "",
            self.description,
            "",
            f"- **Predicts:** `{self.target}`",
            f"- **Trained on seasons:** {', '.join(map(str, self.trained_on))}",
            f"- **Evaluated on seasons:** {', '.join(map(str, self.evaluated_on)) or 'not yet scored'}",
            f"- **Features:** {len(self.features)}",
            f"- **Python:** {self.python} · **card version:** {self.version}",
            f"- **Created:** {self.created_at}",
            "",
            "## Hyperparameters",
            "",
        ]
        for key, value in self.hyperparameters.items():
            lines.append(f"- `{key}`: {value}")

        lines += ["", "## Metrics", "", "| metric | value |", "|---|---|"]
        for key, value in self.metrics.items():
            lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")

        if self.baselines_beaten:
            lines += ["", "## Baselines beaten", ""]
            lines += [f"- `{name}`" for name in self.baselines_beaten]

        if self.known_limitations:
            lines += ["", "## Known limitations", ""]
            lines += [f"- {item}" for item in self.known_limitations]

        return "\n".join(lines) + "\n"


def card_from_results(
    name: str,
    target: str,
    description: str,
    results: pd.DataFrame,
    features: list[str],
    trained_on: list[int],
    evaluated_on: list[int],
    model_key: str | None = None,
    hyperparameters: dict[str, object] | None = None,
    limitations: list[str] | None = None,
) -> ModelCard:
    """Build a card from the harness's pooled summary for one model.

    ``name`` is the file name (model + target); ``model_key`` is the name the
    registry used inside ``results``.
    """
    key = model_key or name
    rows = results[results["model"] == key]
    qualified = rows[rows["scope"] == "qualified"]
    chosen = qualified if not qualified.empty else rows

    metrics: dict[str, float] = {}
    for metric in ("mae", "rmse", "r2", "spearman", "top20_overlap"):
        if metric in chosen.columns and not chosen[metric].empty:
            metrics[metric] = float(chosen[metric].iloc[0])

    beaten = []
    if not chosen.empty:
        scope = chosen["scope"].iloc[0]
        mine = float(chosen["mae"].iloc[0])
        for other in results.loc[results["scope"] == scope, "model"].unique():
            if other == key:
                continue
            other_mae = results[(results["model"] == other) & (results["scope"] == scope)]["mae"]
            if not other_mae.empty and float(other_mae.iloc[0]) > mine:
                beaten.append(other)

    return ModelCard(
        name=name,
        target=target,
        description=description,
        trained_on=trained_on,
        evaluated_on=evaluated_on,
        features=features,
        hyperparameters=hyperparameters or {},
        metrics=metrics,
        baselines_beaten=beaten,
        known_limitations=limitations or [],
    )
