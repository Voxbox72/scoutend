"""Filesystem layout and tunable constants.

All paths are resolved from the repository root (the directory containing
``pyproject.toml``), so nothing depends on the current working directory or on
a hard-coded ``C:\\Users\\...`` path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Season convention -------------------------------------------------------
# FBref identifies a season by its start year ("2324" = 2023/24). Throughout
# this project a season is an *integer end year*: 2324 -> 2024, meaning the
# season that finished in 2024. `season_label` gives the human form ("2023/24").
FIRST_SEASON = 2020
LAST_SEASON = 2024

# A player-season needs this many minutes before rate metrics are trustworthy.
# 450' = 5 full matches. Rows below it are kept (they are real) but flagged.
MINUTES_FLOOR = 450

# A complete Big-5 season is 38 rounds. If the busiest player in a season has
# fewer than this many matches, the snapshot was taken mid-season -- it can be
# used as *features* but never as a *target*, because a half-season total is not
# comparable to a full-season one. (The original project did compare them, and
# then multiplied the actuals by 1.5 to compensate.)
PARTIAL_SEASON_MAX_MATCHES = 30

# --- Modelling scope ---------------------------------------------------------
# Outfield only. Goalkeepers are a different problem: their value lives in
# shot-stopping (PSxG +/-, save%) and sweeping, their per-90 volumes are not
# comparable to outfielders, and scoring them with a goals model would be
# nonsense. The keeper table is built and kept (`goalkeeper_seasons.csv.gz`)
# but deliberately excluded from features, models and evaluation until a
# dedicated keeper phase. See docs/METHODOLOGY.md, "Scope and exclusions".
MODEL_POSITIONS = ("DF", "MF", "FW")
KEEPERS_IN_MODEL_SCOPE = False

# Which FBref league strings we ingest, and their stable short codes.
LEAGUE_CODES = {
    "ENG-Premier League": "ENG1",
    "ESP-La Liga": "ESP1",
    "FRA-Ligue 1": "FRA1",
    "GER-Bundesliga": "GER1",
    "ITA-Serie A": "ITA1",
}


def project_root() -> Path:
    """Walk up from this file until we find the repository root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate the project root (pyproject.toml not found)")


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path

    @classmethod
    def default(cls) -> "Paths":
        root = project_root()
        raw = Path(os.environ.get("SCOUTEND_RAW_DIR", root / "Project Data"))
        processed = Path(os.environ.get("SCOUTEND_PROCESSED_DIR", root / "data" / "processed"))
        return cls(root=root, raw=raw, processed=processed)

    def outfield_source(self, season: int) -> Path:
        return self.raw / "Post_clean" / f"f{season}.csv"

    def keeper_source(self, season: int) -> Path:
        return self.raw / "Pre_clean" / f"{season}_keepers.csv"

    def ensure(self) -> None:
        self.processed.mkdir(parents=True, exist_ok=True)


PATHS = Paths.default()
