"""Build the canonical player-season tables.

One row per (``player_id``, ``season``). A player who turns out for two clubs is
merged into a single row by summing everything that is a count and taking club
context from the club where he played most.

Run via ``python run.py build`` (or ``python -m scoutend.cli build``).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from . import schema
from .config import LEAGUE_CODES, MINUTES_FLOOR, PATHS, Paths

# --- helpers ----------------------------------------------------------------


def _slug(text: str) -> str:
    """'Milan Đurić' -> 'milan-djuric' (stable, ascii, url-safe)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def make_player_id(names: pd.Series, born: pd.Series) -> pd.Series:
    """A readable, reproducible key: name slug + birth year.

    Death-of-complexity note: a surrogate integer id would be cleaner, but this
    one survives re-ingestion (a new season does not renumber anyone) and is
    debuggable by eye. Birth year disambiguates the common name collisions
    ("Danny Ward" the keeper vs the winger) far better than name alone.
    """
    years = pd.to_numeric(born, errors="coerce").astype("Int64").astype(str)
    years = years.where(years != "<NA>", "na")
    return names.map(_slug) + "_" + years


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce FBref's sometimes-stringly numeric columns."""
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


# --- loading ----------------------------------------------------------------


def available_outfield_seasons(paths: Paths = PATHS) -> list[int]:
    """Seasons present on disk, so a new ingested file is picked up automatically."""
    directory = paths.raw / "Post_clean"
    seasons = [
        int(path.stem[1:])
        for path in directory.glob("f*.csv")
        if path.stem[1:].isdigit()
    ]
    return sorted(set(seasons))


def load_raw_outfield(paths: Paths = PATHS, seasons: list[int] | None = None) -> pd.DataFrame:
    """Concatenate the per-season merged stat tables, tagging the read season."""
    seasons = seasons or available_outfield_seasons(paths)
    if not seasons:
        raise FileNotFoundError(
            f"no season files found in {paths.raw / 'Post_clean'} "
            "(expected f<season>.csv, e.g. f2024.csv)"
        )
    frames = []
    for season in seasons:
        path = paths.outfield_source(season)
        if not path.exists():
            raise FileNotFoundError(f"missing outfield source for season {season}: {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame["__source_season"] = season
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return raw


def load_raw_keepers(paths: Paths = PATHS, seasons: list[int] | None = None) -> pd.DataFrame:
    """Keepers are optional: seasons without a keeper export are skipped.

    They are out of modelling scope (see ``config.KEEPERS_IN_MODEL_SCOPE``), so a
    missing keeper file must never block a build.
    """
    seasons = seasons or available_outfield_seasons(paths)
    frames = []
    for season in seasons:
        path = paths.keeper_source(season)
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["__source_season"] = season
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- cleaning ---------------------------------------------------------------


def _assert_known_columns(frame: pd.DataFrame, mapping: dict[str, str], kind: str) -> None:
    """Fail loudly if FBref renames a column we depend on."""
    missing = [col for col in mapping if col not in frame.columns]
    if missing:
        raise KeyError(
            f"{kind}: expected columns are missing from the source "
            f"({len(missing)} of {len(mapping)}): {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}. "
            "FBref probably renamed a column; update src/scoutend/schema.py."
        )


def clean_outfield(raw: pd.DataFrame) -> pd.DataFrame:
    _assert_known_columns(raw, schema.OUTFIELD_SUM_COLUMNS, "outfield")
    _assert_known_columns(raw, schema.OUTFIELD_TEAM_COLUMNS, "outfield (team)")
    _assert_known_columns(raw, {c: c for c in schema.OUTFIELD_CONTEXT_COLUMNS}, "outfield (context)")

    frame = raw.copy()

    # League filter + code.
    unknown = sorted(set(frame["league"]) - set(LEAGUE_CODES))
    frame = frame[frame["league"].isin(LEAGUE_CODES)].copy()
    frame["league_code"] = frame["league"].map(LEAGUE_CODES)

    # Season: prefer the values in the file (they are the FBref codes), fall
    # back to the file's own season if a future export is already normalised.
    raw_season = frame["season"].astype(str).str.replace(r"\.0$", "", regex=True)
    mapped = raw_season.map(schema.FBREF_SEASON_TO_END_YEAR)
    frame["season"] = mapped.fillna(frame["__source_season"]).astype("int64")

    # Identity.
    frame["player"] = frame["player"].astype(str).str.strip()
    frame["born"] = pd.to_numeric(frame["born"], errors="coerce")
    frame["player_id"] = make_player_id(frame["player"], frame["born"])

    # Numeric coercion.
    sum_cols = list(schema.OUTFIELD_SUM_COLUMNS)
    team_cols = list(schema.OUTFIELD_TEAM_COLUMNS)
    frame = _numeric(frame, sum_cols + team_cols + ["Playing Time_Min"])

    # Drop rows with no playing time at all: they cannot inform a rate and
    # they are usually squad members who never appeared.
    frame["Playing Time_Min"] = frame["Playing Time_Min"].fillna(0)
    zero_minutes = int((frame["Playing Time_Min"] <= 0).sum())
    frame = frame[frame["Playing Time_Min"] > 0].copy()

    # The playing-time table also lists squad members who never appeared, so a
    # row can exist without any performance record. Those cannot inform a rate.
    no_record = int(frame["Performance_Gls"].isna().sum())
    frame = frame[frame["Performance_Gls"].notna()].copy()

    # Primary club = most minutes.
    order = frame.sort_values(["player_id", "season", "Playing Time_Min"], ascending=[True, True, False])
    primary = order.drop_duplicates(subset=["player_id", "season"], keep="first")
    primary = primary[
        ["player_id", "season", "team", "league", "league_code", "nation", "pos"]
        + team_cols
    ].rename(columns=schema.OUTFIELD_TEAM_COLUMNS)

    # Sum the counts.
    sums = frame.groupby(["player_id", "season"], as_index=False)[sum_cols].sum(min_count=1)
    sums = sums.rename(columns=schema.OUTFIELD_SUM_COLUMNS)

    # Clubs per player-season.
    clubs = (
        frame.groupby(["player_id", "season"])["team"]
        .apply(lambda s: "|".join(sorted(s.unique())))
        .rename("clubs")
        .reset_index()
    )
    n_clubs = (
        frame.groupby(["player_id", "season"])["team"].nunique().rename("n_clubs").reset_index()
    )

    out = sums.merge(primary, on=["player_id", "season"], how="left")
    out = out.merge(clubs, on=["player_id", "season"], how="left")
    out = out.merge(n_clubs, on=["player_id", "season"], how="left")

    # Names / context.
    names = frame.groupby(["player_id", "season"])["player"].first().rename("player").reset_index()
    born = frame.groupby("player_id")["born"].first().rename("born").reset_index()
    out = out.merge(names, on=["player_id", "season"], how="left")
    out = out.merge(born, on="player_id", how="left")

    out["position"] = out["pos"].map(_normalise_position)
    out["season_label"] = out["season"].map(schema.season_label)
    out["is_big5"] = True
    out["qualified"] = out["minutes"].fillna(0) >= MINUTES_FLOOR
    out["multi_club"] = out["n_clubs"].fillna(1) > 1

    out = _finalise(out, dropped_zero_minutes=zero_minutes, unknown_leagues=unknown)
    out.attrs["dropped_no_record"] = no_record
    return out


def _normalise_position(value: object) -> str:
    """FBref gives 'DF', 'MF', 'FW' (and compounds like 'DF,FW' in raw pulls)."""
    text = str(value)
    if "GK" in text:
        return "GK"
    for token, name in (("FW", "FW"), ("MF", "MF"), ("DF", "DF")):
        if token in text:
            return name
    return "UNK"


def clean_keepers(raw: pd.DataFrame) -> pd.DataFrame:
    _assert_known_columns(raw, schema.KEEPER_SUM_COLUMNS, "keepers")
    frame = raw.copy()

    frame = frame[frame["league"].isin(LEAGUE_CODES)].copy()
    frame["league_code"] = frame["league"].map(LEAGUE_CODES)

    raw_season = frame["season"].astype(str).str.replace(r"\.0$", "", regex=True)
    mapped = raw_season.map(schema.FBREF_SEASON_TO_END_YEAR)
    frame["season"] = mapped.fillna(frame["__source_season"]).astype("int64")

    frame["player"] = frame["player"].astype(str).str.strip()
    frame["born"] = pd.to_numeric(frame["born"], errors="coerce")
    frame["player_id"] = make_player_id(frame["player"], frame["born"])

    sum_cols = list(schema.KEEPER_SUM_COLUMNS)
    weighted_cols = list(schema.KEEPER_WEIGHTED_COLUMNS)
    frame = _numeric(frame, sum_cols + weighted_cols + ["Playing Time_Min"])
    frame["Playing Time_Min"] = frame["Playing Time_Min"].fillna(0)

    # NOTE: the old pipeline did fillna(0) here, which turned "undefined" into
    # "the worst possible value" -- a keeper with no shots faced showed Save%=0.
    # Keepers with no minutes are dropped instead; undefined ratios stay NaN.

    keepers = frame[frame["Playing Time_Min"] > 0].copy()

    primary = (
        keepers.sort_values(["player_id", "season", "Playing Time_Min"], ascending=[True, True, False])
        .drop_duplicates(subset=["player_id", "season"], keep="first")[
            ["player_id", "season", "team", "league", "league_code", "nation"]
        ]
    )

    sums = keepers.groupby(["player_id", "season"], as_index=False)[sum_cols].sum(min_count=1)
    sums = sums.rename(columns=schema.KEEPER_SUM_COLUMNS)

    weighted = (
        keepers.groupby(["player_id", "season"])
        .apply(
            lambda g: pd.Series(
                {
                    target: _weighted_average(g[source], g["Playing Time_Min"])
                    for source, target in schema.KEEPER_WEIGHTED_COLUMNS.items()
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    out = sums.merge(primary, on=["player_id", "season"], how="left")
    out = out.merge(weighted, on=["player_id", "season"], how="left")

    names = keepers.groupby(["player_id", "season"])["player"].first().rename("player").reset_index()
    born = keepers.groupby("player_id")["born"].first().rename("born").reset_index()
    out = out.merge(names, on=["player_id", "season"], how="left")
    out = out.merge(born, on="player_id", how="left")

    out["position"] = "GK"
    out["season_label"] = out["season"].map(schema.season_label)
    out["is_big5"] = True
    out["qualified"] = out["minutes"].fillna(0) >= MINUTES_FLOOR

    out = _finalise(out, dropped_zero_minutes=int((frame["Playing Time_Min"] <= 0).sum()),
                    unknown_leagues=[])
    return out


def complete_seasons(players: pd.DataFrame, threshold: int | None = None) -> set[int]:
    """Seasons whose snapshot covers a full campaign.

    Judged by the busiest player: a finished Big-5 season has somebody on 35+
    appearances, a mid-season snapshot does not. Used by ``targets`` so a
    half-season is never scored against a full-season model.
    """
    from .config import PARTIAL_SEASON_MAX_MATCHES

    limit = threshold if threshold is not None else PARTIAL_SEASON_MAX_MATCHES
    busiest = players.groupby("season")["matches"].max()
    return {int(season) for season, top in busiest.items() if top >= limit}


def _finalise(frame: pd.DataFrame, *, dropped_zero_minutes: int, unknown_leagues: list[str]) -> pd.DataFrame:
    """Order columns, derive per-90 helpers, and run the hard invariants."""
    front = [
        "player_id", "player", "season", "season_label", "league", "league_code",
        "team", "clubs", "n_clubs", "multi_club", "position", "nation", "born",
        "minutes", "matches", "starts", "qualified", "is_big5",
    ]
    front = [c for c in front if c in frame.columns]
    rest = [c for c in frame.columns if c not in front and c not in ("pos",)]
    out = frame[front + rest].copy()

    out["nineties"] = out["minutes"] / 90.0

    # --- invariants ---------------------------------------------------------
    duplicates = out.duplicated(subset=["player_id", "season"], keep=False)
    if duplicates.any():
        sample = out.loc[duplicates, ["player_id", "player", "season", "team"]].head(5)
        raise ValueError(f"duplicate (player_id, season) rows after cleaning:\n{sample}")

    if out["minutes"].isna().any() or (out["minutes"] < 0).any():
        raise ValueError("minutes must be present and non-negative")



    # Counts only: team context legitimately contains negatives (+/- and On-Off).
    count_cols = [
        c
        for c in list(schema.OUTFIELD_SUM_COLUMNS.values()) + list(schema.KEEPER_SUM_COLUMNS.values())
        if c in out.columns and c not in schema.SIGNED_COLUMNS
    ]
    negative = [c for c in count_cols if (out[c].dropna() < 0).any()]
    if negative:
        raise ValueError(f"negative values in count columns: {negative[:8]}")

    if out["season"].isna().any():
        raise ValueError("unmapped season values")

    out.attrs["dropped_zero_minutes"] = dropped_zero_minutes
    out.attrs["unknown_leagues"] = unknown_leagues

    out = out.sort_values(["season", "player_id"]).reset_index(drop=True)
    # pandas drops .attrs on some operations; re-attach after the final reshape.
    out.attrs["dropped_zero_minutes"] = dropped_zero_minutes
    out.attrs["unknown_leagues"] = unknown_leagues
    return out


# --- top level --------------------------------------------------------------


def build(paths: Paths = PATHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build both canonical tables and write them to ``data/processed``.

    The keeper table is written for completeness but is out of modelling scope:
    see ``config.KEEPERS_IN_MODEL_SCOPE`` and docs/METHODOLOGY.md.
    """
    paths.ensure()

    players = clean_outfield(load_raw_outfield(paths))
    raw_keepers = load_raw_keepers(paths)
    keepers = (
        clean_keepers(raw_keepers)
        if len(raw_keepers)
        else pd.DataFrame({"player_id": pd.Series(dtype=str), "season": pd.Series(dtype=int)})
    )

    players.to_csv(paths.processed / "player_seasons.csv.gz", index=False, compression="gzip")
    keepers.to_csv(paths.processed / "goalkeeper_seasons.csv.gz", index=False, compression="gzip")

    return players, keepers
