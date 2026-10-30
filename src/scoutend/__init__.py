"""ScoutEnd: football scouting and next-season projection.

Modules
-------
config    filesystem layout, season convention, tunable constants
schema    canonical column definitions and what we drop (and why)
clean     raw FBref tables -> one row per player-season
targets   next-season outcomes, built so they cannot look ahead
features  leakage-safe model features with minutes shrinkage
eval      baselines, forward-looking splits, metrics
"""

__version__ = "1.0.0"
