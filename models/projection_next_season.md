# Model card: projection_next_season

Poisson per-90 rate models multiplied by an expected-minutes model. Projects the season after the last one in the dataset.

- **Predicts:** `y_gls_next, y_ast_next`
- **Trained on seasons:** 2020, 2021, 2022, 2023
- **Evaluated on seasons:** 2021, 2022, 2023
- **Features:** 34
- **Python:** 3.11.1 · **card version:** 0.3.0
- **Created:** 2026-10-30T22:23:23+00:00

## Hyperparameters

- `alpha_rate`: 1.0
- `alpha_minutes`: 10.0
- `shrinkage_prior_90s`: 10.0

## Metrics

| metric | value |
|---|---|

## Known limitations

- Projects season 2024 -> 2025; no outcome data exists yet, so the projection itself cannot be scored.
- 2508 players projected; rate models are fitted on players with >= 450 minutes, so low-minute players are extrapolations guarded by shrinkage.
- No league-strength adjustment: a Ligue 1 rate is not corrected toward the Premier League.
- No uncertainty interval yet; see the roadmap (bootstrap or Bayesian).
