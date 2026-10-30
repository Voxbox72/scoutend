# ScoutEnd

A football scouting system and next-season performance model for the Big 5
European leagues: it projects how many goals and assists each outfield player
will produce next season, explains that projection as **rate × expected
minutes**, and turns it into a scouting shortlist and player profiles.

**Two versions:**

- **v1.0 — graduation project (2024–25).** The original pipeline: FBref collection,
  cleaning, feature engineering and ensemble models. Accepted, and the source of
  the dataset this repository is built on. Preserved unmodified in
  [notebooks/](notebooks/).
- **v2.0 — this repository.** The same problem, rebuilt to production standard:
  a typed schema, a target table that cannot look ahead, forward-split
  evaluation with named baselines, calibrated model cards, 47 tests, CI, and a
  Streamlit interface.

| Read this | For |
|---|---|
| [docs/CASE_STUDY.md](docs/CASE_STUDY.md) | v1.0 → v2.0: what changed, why, and the results |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The mathematics: shrinkage, Poisson/Gamma GLMs, exposure offsets, evaluation design |
| [REVIEW.md](REVIEW.md) | The v2.0 review: 27 findings with reproducible evidence |
| [notebooks/](notebooks/) | v1.0, as submitted |
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

python -m unittest discover -s tests -t .   # 47 tests, ~40 s
```

The interface:

```bash
pip install -e ".[app]"
streamlit run app.py
```

## v1.0 → v2.0

v1.0 answered the question: can a scouting system and a next-season output model
be built from public football data? It could, and its collection pipeline is
still the source of every row in this repository.

v2.0 asks the harder question: is the answer trustworthy, and can someone else
use it? That meant rebuilding the evaluation. The reported v1.0 figure of *MAE
1.21* came partly from scoring rows that had no next season at all, and a
three-model GBM ensemble scored the same as plain ridge regression on the same
features. The rebuilt evaluation trains strictly on the past and reports its
numbers beside five named baselines — the headline is 1.81 against a naive
baseline of 1.85. `REVIEW.md` documents the 27 findings behind the rebuild and
`audit/` reproduces every number.

That is the normal path for a model: the first version shows it can be done, and
the second makes it something a club could act on.

## The data

12,515 player-season rows, 5,044 players, five leagues (EPL, La Liga, Ligue 1,
Bundesliga, Serie A), seasons 2019/20 → 2023/24, 123 canonical columns.

**Scope:** outfield players only. Goalkeepers are cleaned and stored as a separate
table but excluded from features, models and evaluation — their value is
shot-stopping and sweeping, not goals and assists, and a goals model applied to
them is a category error. They are a separate phase (see the roadmap).

| table | rows | grain |
|---|---|---|
| `player_seasons.csv.gz` | 12,515 | one row per outfield player-season, mid-season transfers merged |
| `goalkeeper_seasons.csv.gz` | 1,027 | one row per keeper-season — data only, out of scope |
| `targets.csv.gz` | 12,515 | next-season outcomes, null where none exist |

Two rules the pipeline enforces rather than hopes for:

- **A target comes from `season + 1` or it is null.** The join is asserted; a
  player with no next season is flagged `left_dataset`, never scored as zero.
- **A partial season is never a target.** If the busiest player has fewer than
  30 appearances the season is a mid-season snapshot: usable as features,
  excluded from targets.

## Results

Forward splits (train strictly before the test season), next-season goals,
players with ≥450 minutes:

| model | MAE | R² | Spearman | top-20 |
|---|---|---|---|---|
| **poisson_blended_two_stage** | **1.806** | 0.458 | 0.599 | 0.433 |
| poisson_blended_x_actual_minutes | 1.826 | 0.457 | 0.599 | 0.450 |
| last season's npxG *(baseline)* | 1.847 | 0.440 | 0.592 | 0.432 |
| ridge | 1.850 | 0.466 | 0.595 | 0.433 |
| last season's goals *(baseline)* | 2.055 | 0.279 | 0.546 | 0.467 |
| season mean *(baseline)* | 2.547 | −0.002 | 0.000 | 0.000 |

Assists: **1.324** against 1.422 for last season's xAG (−6.9%); goals −2.2%.
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
    metrics.py     MAE / RMSE / R² / Spearman / top-k / calibration
    harness.py     forward splits, scoring, report, cards
  scouting.py      percentiles, role clusters, similarity, shortlist scoring
  obsolete/        the original numpy estimators, kept as reference
app.py             Streamlit interface
tests/             47 tests, each encoding a failure that was actually found
audit/             scripts that reproduce every number in REVIEW.md
docs/              case study + methodology
notebooks/         v1.0, the graduation project as submitted
```

**Engineering controls:** tests encode the defects the review found (duplicate
player-seasons, wrong-season targets, filled labels, unaligned keeper seasons);
every shipping model writes a model card; CI builds the data, runs the tests,
evaluates and uploads the report, so the README numbers cannot silently drift.

## Limitations

A retrospective analysis on a fixed archive. **The data stops at 2023/24** —
FBref now sits behind a challenge that refuses programmatic requests, so there is
no live feed. The pipeline is source-agnostic (a new source needs a `schema.py`
mapping, not a rewrite), but that work is not done. Also: Big 5 only; ~1,700 rows
per fold, so small differences are within noise; players who leave the Big 5 are
dropped; no uncertainty intervals; and next-season goals are a proxy for value
that ignores defensive contribution and price.

Full list: [docs/CASE_STUDY.md](docs/CASE_STUDY.md) §6.

## What each version delivered

**v1.0 — graduation project**

- FBref data collection across nine stat tables and five seasons
- Cleaning and position splits into one table per season
- Feature engineering and ensemble models (Random Forest, XGBoost, LightGBM)
- Next-season goal and assist projection

**v2.0 — this repository**

- Canonical schema: one row per player-season, transfers merged, 123 columns
- A target table joined on `season + 1`, asserted, with `left_dataset` flags
- Forward-split evaluation: 5 baselines, 9 metrics, calibration by decile
- Expected-minutes model and Poisson/Gamma rate models, composed as rate × minutes
- Scouting layer: percentiles, role clusters, similarity, shortlist scoring
- Streamlit interface, 47 tests, CI, and a model card for every shipping model

## Data

Source: [FBref](https://fbref.com) via [`soccerdata`](https://github.com/probberechts/soccerdata),
collected for the graduation project. FBref data is subject to FBref's terms of
use; this project is for personal/educational use.
