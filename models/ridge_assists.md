# Model card: ridge_assists

L2 linear regression on the full feature matrix. The reference model.

- **Predicts:** `y_ast_next`
- **Trained on seasons:** 2020, 2021, 2022
- **Evaluated on seasons:** 2021, 2022, 2023
- **Features:** 34
- **Python:** 3.11.1 · **card version:** 0.3.0
- **Created:** 2026-10-30T21:28:58+00:00

## Hyperparameters

- `alpha`: 10.0

## Metrics

| metric | value |
|---|---|
| mae | 1.3426 |
| rmse | 1.8876 |
| r2 | 0.3264 |
| spearman | 0.5377 |
| top20_overlap | 0.1837 |

## Baselines beaten

- `poisson_two_stage`
- `last_season_xag`
- `rate_x_actual_minutes`
- `position_mean`
- `last_season_ast`
- `season_mean`

## Known limitations

- Big-5 only; no cross-league strength adjustment yet.
- Training window ends at season 2023; 2024/25 and 2025/26 are not ingested.
- Totals ignore penalties taken and opposition quality.
