"""ScoutEnd — a scouting interface over the trained models.

Run:  streamlit run app.py
Requires: pip install -e ".[app]"

Two tabs:

* **Find players** — filter to what a club needs and rank by a scoring rule whose
  weights you control.
* **Compare** — one or two players side by side: projected output, a percentile
  radar, per-metric comparison, similar players, and history.

Everything shown is a percentile against the player's own position in that
season, so a defender is never judged against a striker. The model evidence lives
in `data/processed/eval_report.md` and the model cards, not in a tab nobody
reads.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from scoutend.config import PATHS  # noqa: E402
from scoutend.scouting import (  # noqa: E402
    OUTFIELD_METRICS,
    assign_roles,
    load_panel,
    projection_and_value,
    similarity,
)

st.set_page_config(page_title="ScoutEnd", page_icon="⚽", layout="wide")

METRIC_LABELS = {feature: label for feature, label, _ in OUTFIELD_METRICS}
POSITION_NAMES = {"DF": "Defender", "MF": "Midfielder", "FW": "Forward"}

# Groups that appear in the comparison table, in reading order.
ATTACK = ["gls_per90", "npxg_per90", "shots_per90", "sot_per90", "touches_att_pen_per90"]
CREATION = ["ast_per90", "xag_per90", "sca_per90", "gca_per90"]
POSSESSION = ["passes_completed_per90", "pass_completion", "prg_passes_per90",
              "prg_carries_per90", "take_ons_won_per90"]
DEFENDING = ["tackles_per90", "interceptions_per90", "clearances_per90", "blocks_per90"]
RADAR_METRICS = ATTACK + CREATION + ["passes_completed_per90", "prg_passes_per90",
                                     "prg_carries_per90", "take_ons_won_per90"] + DEFENDING


def _stamp(path: pathlib.Path) -> float:
    """Modification time, used as a cache key.

    ``@st.cache_data`` on a no-argument function caches forever, so re-running
    ``python run.py eval`` outside the app had no effect until the server was
    restarted. Keying on the file's mtime invalidates the cache automatically.
    """
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_panel(stamp: float) -> pd.DataFrame:
    panel = assign_roles(load_panel())
    predictions = pd.read_csv(PATHS.processed / "predictions_next_season.csv.gz")
    return projection_and_value(panel, predictions)


def get_panel() -> pd.DataFrame:
    return _load_panel(
        max(
            _stamp(PATHS.processed / "player_seasons.csv.gz"),
            _stamp(PATHS.processed / "predictions_next_season.csv.gz"),
        )
    )


def _safe(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if number != number else number  # NaN check
    except (TypeError, ValueError):
        return default


def comparison_radar(rows: list[tuple[str, pd.Series]], metrics: list[str]):
    """Interactive radar comparing up to two players on percentile scores."""
    import plotly.graph_objects as go

    names = [METRIC_LABELS.get(m, m) for m in metrics]
    figure = go.Figure()
    for label, row in rows:
        values = [_safe(row.get(f"pct_{m}")) for m in metrics]
        figure.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=names + names[:1],
                fill="toself",
                name=label,
                hovertemplate="%{theta}<br>%{r:.0f} percentile<extra></extra>",
            )
        )
    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=540,
        margin=dict(l=70, r=70, t=30, b=30),
    )
    return figure


def comparison_table(rows: list[tuple[str, pd.Series]], metrics: list[str]) -> pd.DataFrame:
    """One row per metric, one column per player, plus the gap."""
    data = {"Metric": [METRIC_LABELS.get(m, m) for m in metrics]}
    for label, row in rows:
        data[label] = [
            f"{_safe(row.get(f'pct_{m}')):.0f}" for m in metrics
        ]
    table = pd.DataFrame(data).set_index("Metric")
    if len(rows) == 2:
        left = [_safe(rows[0][1].get(f"pct_{m}")) for m in metrics]
        right = [_safe(rows[1][1].get(f"pct_{m}")) for m in metrics]
        table["Edge"] = [
            (rows[0][0] if a > b else rows[1][0] if b > a else "level")
            for a, b in zip(left, right)
        ]
    return table


def projection_table(rows: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    """The business end: what each player is expected to produce next season."""
    records = []
    for label, row in rows:
        records.append(
            {
                "Player": label,
                "Position": POSITION_NAMES.get(row["position"], row["position"]),
                "Age": _safe(row.get("age")),
                "Minutes": _safe(row.get("minutes")),
                "Role": row.get("role") or "—",
                "Projected goals": round(_safe(row.get("predicted_goals_next")), 1),
                "Projected assists": round(_safe(row.get("predicted_assists_next")), 1),
                "Projected minutes": round(_safe(row.get("expected_minutes_next"))),
                "Output /90": round(_safe(row.get("output_next_per90")), 2),
                "Score": round(_safe(row.get("shortlist_score")), 1),
            }
        )
    return pd.DataFrame(records)


def render_compare(panel: pd.DataFrame, current: pd.DataFrame) -> None:
    st.subheader("Compare players")

    c1, c2 = st.columns(2)
    positions = ["DF", "MF", "FW"]
    position = c1.selectbox(
        "Position", positions, format_func=lambda p: POSITION_NAMES[p]
    )
    pool = current[current["position"] == position]
    # Keep the two candidate lists independent so a user can compare across
    # seasons of the same player, not only across players.
    candidates = pool.assign(
        label=pool["player"] + " · " + pool["season_label"] + " · " + pool["team"]
    ).sort_values("minutes", ascending=False)

    labels = candidates["label"].tolist()
    if len(labels) < 1:
        st.info("No players for that position in the latest season.")
        return

    first = c1.selectbox("Player A", labels, index=0)
    second = c2.selectbox(
        "Player B", ["— none —"] + labels, index=min(1, len(labels))
    )

    rows: list[tuple[str, pd.Series]] = []
    for label in (first, second):
        if label == "— none —":
            continue
        match = candidates[candidates["label"] == label]
        if not match.empty:
            rows.append((match.iloc[0]["player"], match.iloc[0]))

    st.dataframe(projection_table(rows), hide_index=True, use_container_width=True)

    st.plotly_chart(comparison_radar(rows, RADAR_METRICS), use_container_width=True)
    st.caption(
        "Percentiles compare each player against others of the same position in "
        "the same season (90 = better than 90% of them). "
        + (
            "The table below shows the percentile behind every bar."
            if len(rows) == 2
            else "Pick a Player B above to compare directly."
        )
    )

    with st.expander("Metric-by-metric comparison", expanded=len(rows) == 2):
        st.dataframe(
            comparison_table(rows, ATTACK + CREATION + POSSESSION + DEFENDING),
            use_container_width=True,
        )

    for label, row in rows:
        with st.expander(f"{label} — most similar players"):
            similar = similarity(panel, row, n=8)
            st.dataframe(
                similar[["player", "season_label", "team", "age", "minutes",
                         "role", "similarity"]],
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("#### Season history")
    history_columns = ["season_label", "team", "position", "age", "minutes",
                       "gls", "gls_per90", "npxg_per90", "ast_per90", "xag_per90"]
    for label, row in rows:
        track = (
            panel[panel["player_id"] == row["player_id"]]
            .sort_values("season")
            [history_columns]
            .round(2)
        )
        st.markdown(f"**{label}**")
        st.dataframe(track, hide_index=True, use_container_width=True)


def render_find(panel: pd.DataFrame, current: pd.DataFrame) -> None:
    st.subheader("Find players")
    st.caption(
        "Rank candidates on a scoring rule you control: projected output per 90 "
        "for ability, expected minutes for opportunity, the age curve, and how "
        "far the projection exceeds last season's output."
    )

    with st.expander("Scoring weights", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        w_quality = c1.slider("Quality", 0.0, 1.0, 0.40, 0.05)
        w_role = c2.slider("Opportunity", 0.0, 1.0, 0.25, 0.05)
        w_age = c3.slider("Age curve", 0.0, 1.0, 0.20, 0.05)
        w_value = c4.slider("Improvement", 0.0, 1.0, 0.15, 0.05)

    f1, f2, f3, f4 = st.columns(4)
    positions = f1.multiselect(
        "Position", ["DF", "MF", "FW"], default=["DF", "MF", "FW"],
        format_func=lambda p: POSITION_NAMES[p],
    )
    min_minutes = f2.slider("Minimum minutes", 0, 2500, 900, 100)
    max_age = f3.slider("Maximum age", 17, 40, 27)
    max_value = f4.number_input("Rows shown", 5, 50, 20, 5)

    total = max(1e-9, w_quality + w_role + w_age + w_value)
    candidates = current[
        current["position"].isin(positions)
        & (current["minutes"] >= min_minutes)
        & (current["age"] <= max_age)
    ].copy()

    if candidates.empty:
        st.info("No players match those filters.")
        return

    candidates["shortlist_score"] = (
        w_quality / total * candidates["output_next_per90"].rank(pct=True)
        + w_role / total * candidates["expected_minutes_next"].rank(pct=True)
        + w_age / total * (-(candidates["age"] - 24).abs()).rank(pct=True)
        + w_value / total
        * (candidates["predicted_goals_next"] - candidates["gls"]).rank(pct=True)
    ) * 100

    ranked = candidates.sort_values("shortlist_score", ascending=False)
    shown = ranked.head(int(max_value)).copy()
    shown.insert(0, "Rank", range(1, len(shown) + 1))

    st.markdown(f"**{len(candidates)} players match.** Showing the top {len(shown)}.")

    st.dataframe(
        shown[["Rank", "player", "team", "position", "age", "role", "minutes",
               "predicted_goals_next", "predicted_assists_next",
               "expected_minutes_next", "output_next_per90", "shortlist_score"]]
        .rename(columns={
            "player": "Player", "team": "Team", "position": "Pos", "age": "Age",
            "role": "Role", "minutes": "Minutes", "predicted_goals_next": "Proj. goals",
            "predicted_assists_next": "Proj. assists",
            "expected_minutes_next": "Proj. minutes",
            "output_next_per90": "Output /90", "shortlist_score": "Score",
        })
        .round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download full shortlist (CSV)",
        ranked[[
            "player", "team", "position", "age", "role", "minutes",
            "predicted_goals_next", "predicted_assists_next", "expected_minutes_next",
            "output_next_per90", "shortlist_score",
        ]].to_csv(index=False).encode("utf-8"),
        file_name="scoutend_shortlist.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("ScoutEnd")
    st.caption(
        "Big 5 European leagues · 2019/20 to 2023/24 · outfield players · "
        "next-season goals and assists projected from each player's last season."
    )

    if not (PATHS.processed / "predictions_next_season.csv.gz").exists():
        st.error("No artifacts found. Run `python run.py all` first.")
        return

    panel = get_panel()
    latest = int(panel["season"].max())
    current = panel[panel["season"] == latest].copy()

    tab_find, tab_compare = st.tabs(["Find players", "Compare"])
    with tab_find:
        render_find(panel, current)
    with tab_compare:
        render_compare(panel, current)

    st.divider()
    st.caption(
        "Projections are point estimates from a two-stage model (rate × expected "
        "minutes) and have no uncertainty interval. Data covers the Big 5 only and "
        "ends at 2023/24 — this is a retrospective scouting tool, not a live feed."
    )


if __name__ == "__main__":
    main()
