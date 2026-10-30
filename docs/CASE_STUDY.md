# ScoutEnd — Case Study

Two versions of the same project, five years apart:

| | **v1.0** — graduation project (2025) | **v2.0** — this repository |
|---|---|---|
| Goal | Can scouting reports and next-season output be modelled from public data? | Make that model trustworthy, reproducible and usable |
| Scope | Big 5 leagues, outfield players, goals and assists | Same data, rebuilt end to end |
| Stack | pandas, scikit-learn, XGBoost, LightGBM, Jupyter | Python package, typed schema, test suite, CI, Streamlit |
| Evaluation | K-fold on a player panel | Forward splits, explicit baselines, calibration |
| Status | Accepted — provided the entire dataset used here | Production-quality rebuild |

v1.0 is preserved unmodified in [`notebooks/`](../notebooks/): the full pipeline
that collected the data, cleaned it, engineered features and trained the
ensembles. **It produced the dataset this repository is built on**, and it is why
v2.0 starts from five seasons of clean, structured Big 5 data rather than from
scratch.

## The short version

| | v1.0 (as submitted) | v2.0 (this repo) |
|---|---|---|
| Headline metric | MAE **1.21** on next-season goals | MAE **1.81**, reported next to named baselines |
| Metrics shown | model only | 5 baselines, 9 metrics, calibration by decile |
| Validation | K-fold on a player panel | train strictly before test season |
| Duplicate player-seasons | merged implicitly | merged explicitly, with a test |
| Reproducibility | notebooks + absolute paths | `python run.py all`, one command |
| Tests | — | 47 |
| Model documentation | — | model cards (JSON + Markdown) |
| Product surface | — | Streamlit app: find players, compare, profiles |
| CI | — | build → test → evaluate → train on every push |

---

## 1. What the v2.0 review found

Coming back to v1.0 means re-deriving every number from the raw data. Four places
where the evaluation was measuring slightly less than it claimed, and how each is
now handled:

### 1.1 Test rows without a next season were being scored

`Model.ipynb` created the target column, then later ran
`df.fillna(df.median(numeric_only=True))` inside a feature-engineering loop. The
final season's rows have no next season, so their target became the median goals
value — and those rows stayed in the test set.

Re-deriving the same split with a plain ridge regression:

| evaluation | MAE |
|---|---|
| All season-2023 rows (v1.0's method) | **1.229** |
| Rows with a real next-season target only | **1.420** |
| The rows with no next season, alone | 0.796 |

So roughly a fifth of v1.0's reported accuracy came from rows that had no
outcome to predict, and the ensemble matched a linear model on the same features.

**v2.0:** targets live in a separate table, built by joining `(player_id,
season)` to `(player_id, season + 1)`. A missing join produces null and a
`left_dataset` flag — never a number. An assertion fails the build if the join
ever resolves elsewhere.

### 1.2 The future-season model's labels were dominated by defaults

In the 2025 cells, `fillna(0)` ran after the target was created, so the most
recent season received a target of **0 goals** — and was then weighted 4×:

```
rows in the training frame            13,175
rows whose label is a synthetic 0     64.9%
target mean of the 4x-weighted block  0.219   (vs 1.71 for real seasons)
```

The hand-written multiplier boosts existed to correct the resulting
under-prediction.

**v2.0:** no fill on a target column anywhere; the invariant is a test.

### 1.3 Mid-season transfers blurred the target

`groupby('player').shift(-1)` assumes one row per player-season. 693
player-seasons have two rows (January transfers), so "next season" sometimes
resolved to the other club in the *same* season.

**v2.0:** transfers are merged into one row by summing counts, and the target is
an explicit join on `season + 1`. Both are covered by tests.

### 1.4 Scope creep in the keeper table

The keeper table was never season-mapped, so it carried raw FBref codes and
could not be joined to anything.

**v2.0:** keepers are cleaned and season-mapped, then deliberately excluded from
the outfield models — their value is shot-stopping, not goals, and they get their
own phase.

---

## 2. What v2.0 is

A five-layer pipeline where each layer depends only on the one below:

```
Project Data/ (raw FBref CSVs, committed)
   │  clean.py      merge transfer rows into one row per player-season
   ▼
player_seasons   12,515 rows · targets.py → next-season outcomes on season+1
   │  features.py  34 features, winsorised, shrunk toward position means
   ▼
models/glm.py    Poisson rate model · Gamma minutes model · empirical-Bayes blend
   │  eval/        forward splits, 5 baselines, calibration, model cards
   ▼
scouting.py      percentiles · role clusters · similarity · shortlist
   │  app.py       Streamlit: find players, compare, profiles
```

### 2.1 The decision that mattered

An early version of the projection model gave **one player 60 goals from 62
minutes played**. Three guards came out of that:

1. winsorise per-90 features at domain caps (twice an elite season);
2. fit rates only on players with ≥450 minutes;
3. blend the modelled rate with the player's position-shrunk rate, weighted by
   minutes played — an empirical-Bayes blend with a 4-nineties prior.

### 2.2 Two models, composed

Goals are counts, so the rate model is a Poisson GLM with the log exposure as an
offset:

$$\log \mathbb{E}[y_i] = \log(\text{90s}_i) + \mathbf{x}_i^\top\beta$$

The offset makes $\mathbf{x}^\top\beta$ the **per-90 rate**; multiplying by an
expected-minutes model turns it into a season total. Minutes are a positive
duration, so the minutes model is a Gamma GLM with a log link — fitting
`log(minutes)` by least squares instead under-predicted mean minutes by
**17.5%**, and every projection would have inherited that shrink.

```
projected goals = goals per 90 × expected minutes / 90
                  └ ability ┘   └──── opportunity ───┘
```

---

## 3. Results

Forward splits: train on seasons strictly before the test season, predict the
next one, for 2021, 2022 and 2023. Next-season goals, players with ≥450 minutes:

| model | MAE | R² | Spearman | top-20 overlap |
|---|---|---|---|---|
| **poisson_blended_two_stage** | **1.806** | 0.458 | 0.599 | 0.433 |
| poisson_blended_x_actual_minutes | 1.826 | 0.457 | 0.599 | 0.450 |
| poisson_two_stage (unblended) | 1.834 | 0.442 | 0.592 | 0.417 |
| last season's npxG *(baseline)* | 1.847 | 0.440 | 0.592 | 0.432 |
| ridge | 1.850 | 0.466 | 0.595 | 0.433 |
| last season's goals *(baseline)* | 2.055 | 0.279 | 0.546 | 0.467 |
| position mean *(baseline)* | 2.214 | 0.170 | 0.419 | 0.066 |

Next-season assists, same scope:

| model | MAE | R² | Spearman |
|---|---|---|---|
| **poisson_blended_two_stage** | **1.324** | 0.319 | 0.535 |
| last season's xAG *(baseline)* | 1.422 | 0.232 | 0.518 |
| last season's assists *(baseline)* | 1.586 | −0.038 | 0.437 |

Calibration, goals, top decile: predicted 7.75, actual 8.28 — 5% conservative,
and no decile off by more than 0.64 goals.

**What this does and does not claim.** The model beats the strongest naive
baseline by 2.2% on goals and 6.9% on assists, with better ranking in both. On
4,449 rows those differences are modest. What the model adds beyond a point
estimate is *decomposition*: a scout can see whether a player is rated for
ability or for opportunity, which one number cannot express.

---

## 4. The product

`streamlit run app.py`, two tabs:

- **Find players** — filter by position, minutes and age, then rank by a scoring
  rule whose four weights are exposed as sliders: 40% projected output per 90,
  25% expected minutes, 20% age curve, 15% projected improvement.
- **Compare** — two players side by side: projected output, an interactive
  percentile radar, a metric-by-metric table with an edge column, their most
  similar players, and full season history.

The role clusters are learned, not assumed:

| cluster | defining dimensions (mean z within position) |
|---|---|
| Finisher | goals +1.31, npxG +1.01, box touches +0.70 |
| Creator | progressive passes +0.79, SCA +0.66 |
| Progressor | take-ons +0.43, carries +0.38 |
| All-round elite | goal-creating actions +1.25, SCA +1.08, assists +1.05 |
| Aerial/defensive | interceptions +0.55, tackles +0.37 |

---

## 5. What went wrong during the rebuild, and what fixed it

Every one of these was caught by the harness, not by inspection — which is the
argument for building the harness first.

| failure | how it surfaced | fix |
|---|---|---|
| 60 goals from a 62-minute cameo | per-row predictions inspected, not just the mean | winsorise rates, minutes floor, blend |
| Minutes under-predicted 17.5% | mean prediction vs mean actual, printed every run | Gamma GLM instead of log-OLS |
| Minutes capped at 54 | a bound correct for rates inherited by a duration model | per-model linear-predictor bounds |
| Penalty decile inflated | calibration by decile, not just MAE | blend + rate caps |
| Role labels collapsed into one bucket | labels checked against known players | label from z-scores within position, threshold set from the real distribution |
| Equal stats, unequal percentiles | visible in the comparison view | tied values share a rank |

---

## 6. Honest limitations

1. **Big 5 only**; the players a club can sign cheaply are mostly outside it.
2. **The data stops at 2023/24.** FBref moved behind a challenge that refuses
   programmatic requests, so this is a retrospective analysis on a fixed
   archive, not a live feed. The pipeline is source-agnostic — a new source
   needs a `schema.py` mapping, not a rewrite — but that work is not done.
3. **~1,700 rows per fold.** Differences under ~3% MAE are not distinguishable
   from noise and no confidence intervals are computed for model differences.
4. **Survivorship.** Players who leave the Big 5 are dropped, not modelled, so
   the surviving population is more successful than the input population.
5. **The minutes model is thin.** No contracts, injuries, transfers or
   managerial change.
6. **No uncertainty intervals.** A 19-year-old with 600 minutes should carry a
   much wider interval than a 28-year-old with 2,800.
7. **Prediction ≠ value.** Goals next season is a proxy. It ignores defensive
   contribution, pressing, availability and price.

---

## 7. What I would say in an interview

> "v1.0 was my graduation project: a full scouting and projection pipeline on
> five seasons of Big 5 data, and it is still what the dataset in v2.0 is built
> from. When I picked it back up, I wanted to know which parts held up, so I
> re-derived every number from the raw data. Two of the evaluation choices —
> scoring rows that had no next season, and a default fill running after the
> target column existed — were flattering the reported error. v2.0 rebuilds the
> evaluation with forward splits, named baselines and calibration checks, and the
> honest number is 1.81 against a naive baseline of 1.85. The model earns its
> place for the decomposition — rate × expected minutes — not for beating the
> baseline by a lot. The lesson I took was to build the measuring stick before
> the model."

Working with data in production is exactly this: the first version answers the
question, and the second makes the answer trustworthy.
