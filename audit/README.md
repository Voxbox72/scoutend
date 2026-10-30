# Audit scripts

These reproduce every number quoted in `../REVIEW.md`. They need only `pandas` and `numpy`
(no scikit-learn, no network). Run from the repository root:

    python audit/00_extract_notebooks.py      # dumps notebook code+outputs to audit/notebooks/
    python audit/01_schema_and_targets.py     # shape, seasons, duplicate player-seasons, lag-target audit
    python audit/02_label_contamination.py    # same-season / skipped-season labels, minutes floor
    python audit/03_coverage_and_joins.py     # league x season matrix, keeper season codes, 2025 join rate
    python audit/04_baselines.py              # mean / previous-season / npxG baselines on the exact split
    python audit/05_ridge_and_leak_scan.py    # ridge reference model + per-feature |corr| with the target
    python audit/06_2025_label_corruption.py  # proves fillna(0) turns 64.9% of weighted labels into 0
    python audit/07_reported_mae_artefact.py  # proves the 1.21 MAE comes from fabricated test rows

`00_extract_notebooks.py` writes plain-text dumps into `audit/notebooks/` (git-ignored);
the notebooks themselves remain the source of truth.
