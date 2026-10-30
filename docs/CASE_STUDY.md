# ScoutEnd — Case Study

How a graduation project that reported a 1.21 MAE became a pipeline that reports
1.81 honestly, and why the second number is the one worth putting on a CV.

---

## The short version

| | Original graduation project | This repository |
|---|---|---|
| Headline metric | MAE **1.21** on next-season goals | MAE **1.81**, and named baselines alongside it |
| Evaluation | shuffled K-fold over a player panel | forward splits, train strictly before test |
| Test set | 2,681 rows, **820 with a fabricated target** | 4,449 real rows, one null tolerated: none |
| Training labels | **64.9%** synthetic zeros | 0 |
| Best naive baseline | never computed | `last_season_npxg` at 1.847 |
| Duplicate player-seasons | 693, silently corrupting labels | merged by summing counts |
| Reproducibility | 4 notebooks, hard-coded paths | `python run.py all` |
| Tests | none | 46 |
| Product | none | Streamlit app: shortlist, profile, evidence |

The honest headline: **the model improves on the strongest naive baseline by
2.2% on goals and 6.9% on assists.** That is a real but modest result. The
result worth telling is *how* it was found.

---

## 1. What was actually wrong

The original notebooks are in the repository root, unmodified. Reading them
against the data turned up 27 findings; four decided the fate of the project.

### 1.1 The test set contained 820 invented targets

`Model.ipynb` created the target column, then later ran
`df.fillna(df.median(numeric_only=True))` inside a feature-engineering loop. The
2,502 rows of the final season have no next season, so their target became the
**median goals value ≈ 1.0** — and those rows went into the test set.

Rebuilding the notebook's exact feature matrix and split with a plain ridge
regression:

| evaluation | MAE |
|---|---|
| All season-2023 rows (the notebook's method) | **1.229** |
| Only rows with a real next-season target | **1.420** |
| The 820 fabricated rows, alone | 0.796 |
| The notebook's reported 3-model GBM ensemble | 1.21 |

So the reported number was ~18% flattering, and a linear model matched the
whole ensemble without the ensemble.

### 1.2 The future-season model trained mostly on fake labels

In the 2025 cells the same `fillna(0)` ran *after* the target was built, so the
2,649 rows of season 2023/24 received a target of **0 goals** and were then
concatenated with a 4× weight:

```
rows in the training frame            13,175
rows whose label is a synthetic 0     64.9%
target mean of the 4x-weighted block  0.219   (vs 1.71 for real seasons)
```

The hand-written `*= 1.5` "breakout boost" existed to correct the resulting
under-prediction — and was then validated against the same partial-season
actuals it was derived from.

### 1.3 Transfers corrupted 8.5% of labels

`groupby('player').shift(-1)` assumes one row per player-season. There are 693
player-seasons with two rows (January transfers). For the first row, "next
season" resolved to **the other club in the same season**.

### 1.4 The season codes did not join

Every outfield table was mapped `2324 → 2024`. The keeper table was not, so it
still carried `1920, 2122, 2324` and could not be joined to anything.

---

## 2. What replaced it

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
   │  app.py       Streamlit: shortlist, player profile, evidence
```

### 2.1 The decision that mattered

An early version of the Poisson model projected **60 goals for a player with 62
minutes**. A rate of "one goal in 25 minutes" is 3.6 per 90, and a log link
exponentiates it happily. Three guards came out of that failure:

1. winsorise per-90 features at domain caps (twice an elite season);
2. fit rates only on players with ≥450 minutes;
3. blend the modelled rate with the player's position-shrunk rate, weighted by
   minutes played — an empirical-Bayes blend with a 4-nineties prior.

### 2.2 Two models, composed

Goals are counts, so the rate model is a Poisson GLM with the log exposure as
an offset:

$$\log \mathbb{E}[y_i] = \log(\text{90s}_i) + \mathbf{x}_i^\top\beta$$

The offset makes $\mathbf{x}^\top\beta$ the **per-90 rate**; multiplying by an
expected-minutes model turns it into a season total. Minutes are a positive
duration, so the minutes model is a Gamma GLM with a log link. Fitting
`log(minutes)` by least squares instead under-predicted mean minutes by
**17.5%** — a bias every projection would have inherited.

The decomposition is the product:

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

`streamlit run app.py`, three tabs:

- **Shortlist** — filter by position, minutes and age, then rank by a scoring
  rule whose four weights are exposed as sliders: 40% projected output per 90,
  25% expected minutes, 20% age curve, 15% projected improvement.
- **Player profile** — percentile bars within position and season, plus the
  eight most similar players in style space (same position only).
- **Model evidence** — the baseline table and the model card, in the app,
  next to the rankings it justifies.

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
| 60 goals for a 62-minute cameo | per-row predictions were inspected, not just the mean | winsorise rates, minutes floor, blend |
| Minutes under-predicted 17.5% | mean prediction vs mean actual, printed every run | Gamma GLM instead of log-OLS |
| Minutes capped at 54 | a bound correct for rates was inherited by a duration model | per-model linear-predictor bounds |
| Penalty decile inflated | calibration by decile, not just MAE | blend + rate caps |
| Role labels collapsed to one bucket | eyeballed labels against known players | label from z-scores within position, threshold set from the real distribution |

---

## 6. Honest limitations

1. **Big 5 only**; the players a club can sign cheaply are mostly outside it.
2. **The data stops at 2023/24.** FBref moved behind a Cloudflare challenge that
   refuses programmatic requests, so this is a *retrospective analysis on a fixed
   archive*, not a live feed. The pipeline is source-agnostic — a new source
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

> "My graduation project reported MAE 1.21 on next-season goals. When I audited
> it, 820 of the 2,681 test rows had a target invented by a `fillna(median)` that
> ran after the target column existed, and a plain ridge regression scored 1.229
> on the same split — the three-model ensemble was doing nothing. I rebuilt it
> with a separate targets table that physically cannot look at the wrong season,
> forward splits instead of shuffled folds, named baselines, and calibration
> checks. The honest number is 1.81 against a naive baseline of 1.85. The model
> earns its place for the decomposition — rate × expected minutes — not for
> beating the baseline by a lot."

That story is worth more than a better-looking number, because the first thing a
good interviewer does with a 1.21 is ask how it was measured.
