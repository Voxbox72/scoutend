# Model card: poisson_blended_x_actual_minutes_assists

Blended Poisson rate with the player's current minutes as exposure. Separates the rate model from the minutes model.

- **Predicts:** `y_ast_next`
- **Trained on seasons:** 2020, 2021, 2022
- **Evaluated on seasons:** 2021, 2022, 2023
- **Features:** 34
- **Python:** 3.11.1 · **card version:** 0.3.0
- **Created:** 2026-10-30T21:28:57+00:00

## Hyperparameters

- `alpha`: 1.0

## Metrics

| metric | value |
|---|---|
| mae | 1.3284 |
| rmse | 1.8870 |
| r2 | 0.3253 |
| spearman | 0.5436 |
| top20_overlap | 0.2507 |

## Baselines beaten

- `poisson_rate_x_actual_minutes`
- `ridge`
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
