# ScoutEnd

A football scouting system and next-season performance model for the Big 5
European leagues: it projects how many goals and assists each outfield player
will produce next season, explains that projection as **rate Ã— expected
minutes**, and turns it into a scouting shortlist and player profiles.

**Status:** `v1.0.0` â€” data layer, evaluation harness, models, scouting layer,
interface, tests and documentation complete. A retrospective analysis on a fixed
archive; see the limitations at the bottom.

| Read this | For |
|---|---|
| [docs/CASE_STUDY.md](docs/CASE_STUDY.md) | The story: what was wrong with the original project, what replaced it, the results |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The mathematics: shrinkage, Poisson/Gamma GLMs, IRLS, evaluation design |
| [REVIEW.md](REVIEW.md) | The original audit: 27 findings with reproducible evidence |
| [notebooks/](notebooks/) | The graduation project as submitted, kept unmodified |
| [models/](models/) | Model cards (JSON + Markdown) for every shipping model |

---

## Quickstart

Requires Python 3.11+, `pandas`, `numpy` and `scikit-learn`. Nothing else is needed
for the pipeline or the tests.

```bash
python run.py build     # Project Data/ -> data/processed (dataset, targets)
python run.py eval      # baselines vs candidate models, model cards, calibration
python run.py train     # fit on all data, project the next season
python run.py all       # all three

python -m unittest discover -s tests -t .   # 46 tests, ~40 s
```

The interface:

```bash
pip install -e ".[app]"
streamlit run app.py
```

## Why this repository exists

The graduation-project notebooks are still here â€” historical record, and
`Model_Playground.ipynb` holds useful exploration â€” but they are no longer the
pipeline. Their headline metric, *MAE 1.21 on next-season goals*, was an
artefact: 820 of the 2,681 test rows had a target fabricated by a
`fillna(median)` that ran after the target column was created, and a three-model
GBM ensemble scored essentially the same as plain ridge regression on the same
features. `REVIEW.md` documents 27 findings; `audit/` reproduces every number.

## The data

12,515 player-season rows, 5,044 players, five leagues (EPL, La Liga, Ligue 1,
Bundesliga, Serie A), seasons 2019/20 â†’ 2023/24, 123 canonical columns.

**Scope:** outfield players only. Goalkeepers are cleaned and stored as a separate
table but excluded from features, models and evaluation â€” their value is
shot-stopping and sweeping, not goals and assists, and a goals model applied to
them is a category error. They are a separate phase (see the roadmap).

| table | rows | grain |
|---|---|---|
| `player_seasons.csv.gz` | 12,515 | one row per outfield player-season, mid-season transfers merged |
| `goalkeeper_seasons.csv.gz` | 1,027 | one row per keeper-season â€” data only, out of scope |
| `targets.csv.gz` | 12,515 | next-season outcomes, null where none exist |

Two rules the pipeline enforces rather than hopes for:

- **A target comes from `season + 1` or it is null.** The join is asserted; a
  player with no next season is flagged `left_dataset`, never scored as zero.
- **A partial season is never a target.** If the busiest player has fewer than
  30 appearances the season is a mid-season snapshot: usable as features,
  excluded from targets.

## Results

Forward splits (train strictly before the test season), next-season goals,
players with â‰¥450 minutes:

| model | MAE | RÂ² | Spearman | top-20 |
|---|---|---|---|---|
| **poisson_blended_two_stage** | **1.806** | 0.458 | 0.599 | 0.433 |
| poisson_blended_x_actual_minutes | 1.826 | 0.457 | 0.599 | 0.450 |
| last season's npxG *(baseline)* | 1.847 | 0.440 | 0.592 | 0.432 |
| ridge | 1.850 | 0.466 | 0.595 | 0.433 |
| last season's goals *(baseline)* | 2.055 | 0.279 | 0.546 | 0.467 |
| season mean *(baseline)* | 2.547 | âˆ’0.002 | 0.000 | 0.000 |

Assists: **1.324** against 1.422 for last season's xAG (âˆ’6.9%); goals âˆ’2.2%.
A real but modest edge over a strong naive baseline, plus a decomposition a
single number cannot give.

## How it is built

```
src/scoutend/
  config.py        paths, season convention, minutes floor, modelling scope
  schema.py        the canonical columns; and an explicit list of what we drop
  clean.py         raw FBref tables -> one row per player-season (+ keepers)
  targets.py       next-season outcomes (join on season + 1, never shift)
  features.py      34 leakage-safe features with minutes shrinkage
  models/
    glm.py         rate and duration models: Poisson (offset) + Gamma, via sklearn
    minutes.py     expected minutes (Gamma GLM)
    registry.py    every candidate, scored on identical rows and splits
    train.py       fit on all data, project the next season
    card.py        model cards
  eval/
    metrics.py     MAE / RMSE / RÂ² / Spearman / top-k / calibration
    harness.py     forward splits, scoring, report, cards
  scouting.py      percentiles, role clusters, similarity, shortlist scoring
  obsolete/        the original numpy estimators, kept as reference
app.py             Streamlit interface
tests/             46 tests, each encoding a failure that was actually found
audit/             scripts that reproduce every number in REVIEW.md
docs/              case study + methodology
```

**Engineering controls:** tests encode the original defects (duplicate
player-seasons, wrong-season targets, filled labels, unaligned keeper seasons);
every shipping model writes a model card; CI builds the data, runs the tests,
evaluates and uploads the report, so the README numbers cannot silently drift.

## Limitations

A retrospective analysis on a fixed archive. **The data stops at 2023/24** â€”
FBref now sits behind a challenge that refuses programmatic requests, so there is
no live feed. The pipeline is source-agnostic (a new source needs a `schema.py`
mapping, not a rewrite), but that work is not done. Also: Big 5 only; ~1,700 rows
per fold, so small differences are within noise; players who leave the Big 5 are
dropped; no uncertainty intervals; and next-season goals are a proxy for value
that ignores defensive contribution and price.

Full list: [docs/CASE_STUDY.md](docs/CASE_STUDY.md) Â§6.

## Roadmap

- [x] canonical dataset, targets, test suite
- [x] evaluation harness: baselines, rank metrics, calibration, model cards
- [x] expected-minutes model, Poisson/Gamma rate models
- [x] scouting layer: percentiles, role clustering, similarity, shortlists
- [x] Streamlit interface
- [ ] live data source (Understat or a licensed feed) and fresh seasons
- [ ] uncertainty intervals on projections
- [ ] selling-league coverage + cross-league strength adjustment
- [ ] goalkeepers, as their own modelling problem

## Data

Source: [FBref](https://fbref.com) via [`soccerdata`](https://github.com/probberechts/soccerdata),
collected for the graduation project. FBref data is subject to FBref's terms of
use; this project is for personal/educational use.

