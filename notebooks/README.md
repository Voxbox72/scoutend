# Original graduation-project notebooks

These four notebooks are the project **as submitted in 2025**, kept unmodified as
the historical record. They are not part of the pipeline and nothing in `src/`
reads them.

| Notebook | What it did |
|---|---|
| `Data_collection.ipynb` | Fetched nine FBref player table types per season with `soccerdata`, merged them on `(league, season, team, player)`, wrote one wide CSV per season. |
| `Data_Finalization.ipynb` | Renamed season codes, split by position, concatenated the keeper tables, filled nulls. |
| `Model.ipynb` | Four experiments (goals/assists × 2024/2025): Random Forest + XGBoost + LightGBM on 169 features, a fixed 0.4/0.3/0.3 ensemble, and post-hoc multiplier boosts. |
| `Model_Playground.ipynb` | 78 cells of exploration: correlation pruning, lagged averages, breakout classification, hyperparameter and ensemble-weight searches. |

## Why they are here

Reading them against the data is how the current pipeline was designed. `REVIEW.md`
documents what the audit found; the most consequential items:

- the reported **MAE 1.21** was produced by 820 test rows whose target had been
  fabricated by a `fillna(median)` that ran *after* the target column existed;
- the 2025 model trained on **64.9% synthetic-zero labels**, because `fillna(0)`
  ran after the target was created and the affected season was weighted 4×;
- mid-season transfers gave **8.5% of labels** the wrong season, via
  `groupby('player').shift(-1)`;
- the keeper table was never season-mapped, so it could not be joined to anything.

Every one of those is now an executable test in `tests/`, and every number in the
review is reproducible with the scripts in `audit/`.

## Running them

They will not run as-is: they use absolute `C:\Users\daboo\...` paths and the
Anaconda environment they were written against. The pipeline that replaced them
runs with `python run.py all` — see the repository README.
