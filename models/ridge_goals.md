# Model card: ridge_goals

L2 linear regression on the full feature matrix. The reference model.

- **Predicts:** `y_gls_next`
- **Trained on seasons:** 2020, 2021, 2022
- **Evaluated on seasons:** 2021, 2022, 2023
- **Features:** 34
- **Python:** 3.11.1 · **card version:** 0.3.0
- **Created:** 2026-10-30T21:28:57+00:00

## Hyperparameters

- `alpha`: 10.0

## Metrics

| metric | value |
|---|---|
| mae | 1.8627 |
| rmse | 2.8319 |
| r2 | 0.4637 |
| spearman | 0.6009 |
| top20_overlap | 0.4330 |

## Baselines beaten

- `rate_x_actual_minutes`
- `last_season_gls`
- `position_mean`
- `season_mean`

## Known limitations

- Big-5 only; no cross-league strength adjustment yet.
- Training window ends at season 2023; 2024/25 and 2025/26 are not ingested.
- Totals ignore penalties taken and opposition quality.
