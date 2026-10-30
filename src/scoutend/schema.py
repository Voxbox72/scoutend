"""Canonical column definitions: what we keep from FBref, and how.

Two rules drive this module:

1. **Only true counts are summed across clubs.** A player who moves in January
   has one row per club. Goals, minutes, passes and tackles add up; per-90
   rates, percentages and averages do not. Summing a percentage is meaningless,
   and taking the first row silently discards half a season.
2. **Rates are derived downstream, never stored.** The dataset stores
   ``gls`` and ``minutes``, not ``gls_per90``. One place computes rates
   (``features.py``), so the definition can never drift between tables.

The result is ~90 columns instead of 166: the columns that carry scouting
signal, without the algebraic duplicates (``Ast``/``Performance_Ast``,
``xAG``/``Expected_xAG``, ``PrgP``/``Progression_PrgP``) or the near-empty
pass-type splits.
"""

from __future__ import annotations

# --- Outfield ---------------------------------------------------------------
# fbref column -> canonical name. Every entry here is a *count* or a match
# total, so it is summed when a player appears for two clubs in one season.
OUTFIELD_SUM_COLUMNS: dict[str, str] = {
    # availability / opportunity
    "Playing Time_MP": "matches",
    "Playing Time_Starts": "starts",
    "Playing Time_Min": "minutes",
    # output
    "Performance_Gls": "gls",
    "Performance_Ast": "ast",
    "Performance_G+A": "g_plus_a",
    "Performance_G-PK": "gls_non_pen",
    "Performance_PK": "pk_scored",
    "Performance_PKatt": "pk_att",
    "Performance_CrdY": "yellow",
    "Performance_CrdR": "red",
    # expected output
    "Expected_xG": "xg",
    "Expected_npxG": "npxg",
    "Expected_xAG": "xag",
    "Expected_npxG+xAG": "npxg_plus_xag",
    # progression
    "Progression_PrgC": "prg_carries",
    "Progression_PrgP": "prg_passes",
    "Progression_PrgR": "prg_receives",
    # shooting
    "Standard_Sh": "shots",
    "Standard_SoT": "shots_on_target",
    "Standard_FK": "fk_shots",
    # passing
    "Total_Cmp": "passes_completed",
    "Total_Att": "passes_attempted",
    "Total_TotDist": "pass_distance",
    "Total_PrgDist": "pass_prog_distance",
    "Short_Cmp": "short_passes_completed",
    "Short_Att": "short_passes_attempted",
    "Medium_Cmp": "medium_passes_completed",
    "Medium_Att": "medium_passes_attempted",
    "Long_Cmp": "long_passes_completed",
    "Long_Att": "long_passes_attempted",
    "Expected_xA": "xa",
    "KP": "key_passes",
    "1/3": "passes_into_final_third",
    "PPA": "passes_into_box",
    "CrsPA": "crosses_into_box",
    "Pass Types_Live": "pass_live",
    "Pass Types_Dead": "pass_dead",
    "Pass Types_FK": "pass_fk",
    "Pass Types_TB": "pass_through_ball",
    "Pass Types_Sw": "pass_switch",
    "Pass Types_Crs": "pass_cross",
    "Pass Types_TI": "pass_throw_in",
    "Pass Types_CK": "pass_corner",
    "Corner Kicks_In": "corners_in",
    "Corner Kicks_Out": "corners_out",
    "Corner Kicks_Str": "corners_straight",
    "Outcomes_Cmp": "pass_outcome_complete",
    "Outcomes_Off": "pass_outcome_offside",
    "Outcomes_Blocks": "pass_outcome_blocked",
    # shot-creating / goal-creating actions
    "SCA_SCA": "sca",
    "SCA Types_PassLive": "sca_pass_live",
    "SCA Types_PassDead": "sca_pass_dead",
    "SCA Types_TO": "sca_take_on",
    "SCA Types_Sh": "sca_shot",
    "SCA Types_Fld": "sca_foul_drawn",
    "SCA Types_Def": "sca_def",
    "GCA_GCA": "gca",
    "GCA Types_PassLive": "gca_pass_live",
    "GCA Types_PassDead": "gca_pass_dead",
    "GCA Types_TO": "gca_take_on",
    "GCA Types_Sh": "gca_shot",
    "GCA Types_Fld": "gca_foul_drawn",
    "GCA Types_Def": "gca_def",
    # defending
    "Tackles_Tkl": "tackles",
    "Tackles_TklW": "tackles_won",
    "Tackles_Def 3rd": "tackles_def_third",
    "Tackles_Mid 3rd": "tackles_mid_third",
    "Tackles_Att 3rd": "tackles_att_third",
    "Challenges_Tkl": "challenge_tackles",
    "Challenges_Att": "challenges",
    "Challenges_Lost": "challenges_lost",
    "Blocks_Blocks": "blocks",
    "Blocks_Sh": "blocks_shot",
    "Blocks_Pass": "blocks_pass",
    "Int": "interceptions",
    "Clr": "clearances",
    "Err": "errors",
    # possession / carrying / receiving
    "Touches_Touches": "touches",
    "Touches_Def Pen": "touches_def_pen",
    "Touches_Def 3rd": "touches_def_third",
    "Touches_Mid 3rd": "touches_mid_third",
    "Touches_Att 3rd": "touches_att_third",
    "Touches_Att Pen": "touches_att_pen",
    "Touches_Live": "touches_live",
    "Take-Ons_Att": "take_ons_att",
    "Take-Ons_Succ": "take_ons_won",
    "Take-Ons_Tkld": "take_ons_tackled",
    "Carries_Carries": "carries",
    "Carries_TotDist": "carry_distance",
    "Carries_PrgDist": "carry_prog_distance",
    "Carries_PrgC": "carry_progressive",
    "Carries_1/3": "carries_final_third",
    "Carries_CPA": "carries_into_box",
    "Carries_Mis": "carries_miscontrolled",
    "Carries_Dis": "carries_dispossessed",
    "Receiving_Rec": "receives",
}

# Counts that may legitimately be negative. Excluded from the data-quality
# assertion that rejects negative counts.
SIGNED_COLUMNS = {"psxg_plus_minus"}

# Team-context columns. These describe the *club*, not the player, so they are
# taken from the club where he played the most minutes and never summed.
OUTFIELD_TEAM_COLUMNS: dict[str, str] = {
    "Team Success_PPM": "team_ppm",
    "Team Success_onG": "team_goals_for_on_pitch",
    "Team Success_onGA": "team_goals_against_on_pitch",
    "Team Success_+/-": "team_plus_minus",
    "Team Success_+/-90": "team_plus_minus_per90",
    "Team Success_On-Off": "team_on_off",
    "Team Success (xG)_onxG": "team_xg_for_on_pitch",
    "Team Success (xG)_onxGA": "team_xg_against_on_pitch",
    "Team Success (xG)_xG+/-": "team_xg_plus_minus",
    "Team Success (xG)_On-Off": "team_xg_on_off",
}

# Context columns copied from the highest-minutes row.
OUTFIELD_CONTEXT_COLUMNS = ("nation", "pos", "born")

# Dropped on purpose, with the reason. Keeping this list explicit is what stops
# the column explosion from creeping back.
OUTFIELD_DROPPED = {
    "Unnamed: 0": "index artefact of the original CSV export",
    "age": "string '24-123' in some seasons; age is recomputed from `born`",
    "alter_position": "duplicate of `pos` after normalisation",
    "90s": "derived: minutes / 90",
    "Standard_Dist": "average shot distance; not summable, low signal, dropped",
    "Standard_Gls": "duplicate of Performance_Gls",
    "Standard_PK": "duplicate of Performance_PK",
    "Standard_PKatt": "duplicate of Performance_PKatt",
    "Ast": "duplicate of Performance_Ast",
    "xAG": "duplicate of Expected_xAG",
    "PrgP": "duplicate of Progression_PrgP",
    "Att": "duplicate of Total_Att",
    "Tkl+Int": "derived: tackles + interceptions",
    "Challenges_Tkl%": "derived from challenge_tackles / challenges",
    "Take-Ons_Succ%": "derived from take_ons_won / take_ons_att",
    "Take-Ons_Tkld%": "derived from take_ons_tackled / take_ons_att",
    "Total_Cmp%": "derived from passes_completed / passes_attempted",
    "Short_Cmp%": "derived",
    "Medium_Cmp%": "derived",
    "Long_Cmp%": "derived",
    "Standard_SoT%": "derived from shots_on_target / shots",
    "Standard_Sh/90": "derived per-90",
    "Standard_SoT/90": "derived per-90",
    "Standard_G/Sh": "derived ratio",
    "Standard_G/SoT": "derived ratio",
    "Expected_npxG/Sh": "derived ratio",
    "Expected_G-xG": "derived: gls - xg",
    "Expected_np:G-xG": "derived: gls_non_pen - npxg",
    "Expected_A-xAG": "derived: ast - xag",
    "SCA_SCA90": "derived per-90",
    "GCA_GCA90": "derived per-90",
    "Playing Time_Mn/MP": "derived",
    "Playing Time_Min%": "derived",
    "Starts_Starts": "duplicate of starts",
    "Starts_Mn/Start": "derived",
    "Starts_Compl": "derived",
    "Subs_Subs": "derived: matches - starts",
    "Subs_Mn/Sub": "derived",
    "Subs_unSub": "derived",
}

# --- Goalkeepers ------------------------------------------------------------
KEEPER_SUM_COLUMNS: dict[str, str] = {
    "Playing Time_MP": "matches",
    "Playing Time_Starts": "starts",
    "Playing Time_Min": "minutes",
    "Performance_GA": "goals_against",
    "Performance_SoTA": "shots_on_target_against",
    "Performance_Saves": "saves",
    "Performance_W": "wins",
    "Performance_D": "draws",
    "Performance_L": "losses",
    "Performance_CS": "clean_sheets",
    "Penalty Kicks_PKatt": "pk_faced",
    "Penalty Kicks_PKA": "pk_conceded",
    "Penalty Kicks_PKsv": "pk_saved",
    "Penalty Kicks_PKm": "pk_missed",
    "Goals_OG": "own_goals",
    "Expected_PSxG": "psxg",
    "Expected_PSxG+/-": "psxg_plus_minus",
    "Launched_Cmp": "launched_completed",
    "Launched_Att": "launched_attempted",
    "Passes_Att (GK)": "gk_passes_attempted",
    "Passes_Thr": "gk_passes_thrown",
    "Goal Kicks_Att": "goal_kicks",
    "Crosses_Opp": "crosses_faced",
    "Crosses_Stp": "crosses_stopped",
    "Sweeper_#OPA": "defensive_actions_outside_box",
}

# Averaged across clubs, weighted by minutes.
KEEPER_WEIGHTED_COLUMNS: dict[str, str] = {
    "Passes_AvgLen": "pass_avg_length",
    "Goal Kicks_AvgLen": "goal_kick_avg_length",
    "Sweeper_AvgDist": "sweeper_avg_distance",
}

KEEPER_DROPPED = {
    "90s": "derived: minutes / 90",
    "Performance_GA90": "derived per-90",
    "Performance_Save%": "derived: saves / shots_on_target_against",
    "Performance_CS%": "derived: clean_sheets / matches",
    "Penalty Kicks_Save%": "derived",
    "Goals_GA": "duplicate of Performance_GA",
    "Goals_PKA": "duplicate of Penalty Kicks_PKA",
    "Goals_FK": "goals conceded from free kicks; too sparse to be useful",
    "Goals_CK": "goals conceded from corners; too sparse to be useful",
    "Expected_PSxG/SoT": "derived ratio",
    "Expected_/90": "derived per-90",
    "Launched_Cmp%": "derived",
    "Passes_Launch%": "derived",
    "Goal Kicks_Launch%": "derived",
    "Crosses_Stp%": "derived",
    "Sweeper_#OPA/90": "derived per-90",
}

# --- Season normalisation ---------------------------------------------------
FBREF_SEASON_TO_END_YEAR = {
    "1920": 2020,
    "2021": 2021,
    "2122": 2022,
    "2223": 2023,
    "2324": 2024,
    "2425": 2025,
    "2526": 2026,
    "2019": 2020,  # soccerdata's ambiguous "2019" -> 2019/20
}


def season_label(season_end_year: int) -> str:
    """2020 -> '2019/20'."""
    start = season_end_year - 1
    return f"{start}/{str(season_end_year)[-2:]}"
