# Model card: poisson_two_stage_assists

Unblended Poisson rate multiplied by expected minutes. Kept for comparison: it is what the blend is protecting against.

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
| mae | 1.3473 |
| rmse | 1.9333 |
| r2 | 0.2937 |
| spearman | 0.5190 |
| top20_overlap | 0.2667 |

## Baselines beaten

- `last_season_xag`
- `rate_x_actual_minutes`
- `position_mean`
- `last_season_ast`
- `season_mean`

## Known limitations

- Big-5 only; no cross-league strength adjustment yet.
- Training window ends at season 2023; 2024/25 and 2025/26 are not ingested.
- Totals ignore penalties taken and opposition quality.
