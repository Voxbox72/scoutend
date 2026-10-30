# ScoutEnd  Methodology & Code Guide

Everything this repository does, why it does it that way, and the mathematics
behind it. It is written to be read end to end, but §1§3 are enough to
understand the system, and §6 is where the modelling happens.

> **Audience.** You, six months from now, and anyone interviewing you. If a
> section cannot be explained aloud in two minutes, it is too long  tell me and
> I will cut it.

---

## Contents

1. [What the system does](#1-what-the-system-does)
2. [Scope and exclusions](#2-scope-and-exclusions)
3. [The prediction problem, formally](#3-the-prediction-problem-formally)
4. [The data pipeline](#4-the-data-pipeline)
5. [Features and shrinkage](#5-features-and-shrinkage)
6. [The models](#6-the-models)
7. [Baselines](#7-baselines)
8. [Evaluation methodology](#8-evaluation-methodology)
9. [Results](#9-results)
10. [Software architecture and engineering controls](#10-software-architecture-and-engineering-controls)
11. [Limitations and threats to validity](#11-limitations-and-threats-to-validity)
12. [Roadmap](#12-roadmap)
13. [Glossary](#13-glossary)

---

## 1. What the system does

ScoutEnd estimates, for every outfield player in the Big 5 leagues, **how many
goals and assists he will produce next season**, and explains that estimate as:

```
projected goals  =  goals per 90    expected minutes / 90
                     ability      opportunity 
```

That decomposition is the point. A scout's real question is never "how many
goals did he score", it is "how good is he, and will he get to show it". Two
players with eight goals are not the same asset if one did it in 900 minutes and
the other in 2,700.

The numbers are not the product. The product is a **ranking you can defend**:
the system is scored on whether it puts the right players at the top, whether
its confidence is honest, and whether every claim can be traced to a measurement.

## 2. Scope and exclusions

In scope: outfield players (`DF`, `MF`, `FW`), Big 5 leagues, seasons 2019/20 
2023/24, next-season goals and assists.

**Goalkeepers are out of scope.** Their value is dominated by shot-stopping
(`PSxG ±`, save percentage) and sweeping actions, their per-90 volumes are not
comparable to outfielders, and a goals/assists model applied to them is a
category error. The pipeline still builds a clean `goalkeeper_seasons` table
because it is nearly free and the cleaning is already tested, but no feature,
model or evaluation touches it (`config.KEEPERS_IN_MODEL_SCOPE = False`, and a
test enforces it). A dedicated keeper phase  with keeper-specific targets and
metrics  is on the roadmap, not in the model.

Also deliberately excluded for now, each for a stated reason:

| Excluded | Why |
|---|---|
| Defensive actions as *targets* | Tackles and interceptions are largely a function of team style and possession; predicting them well needs team-context models, not player rates. |
| Set-piece and penalty value | Penalty goals are close to a constant per attempt; separating them adds noise before it adds signal. |
| Opposition strength | Requires match-level data we do not ingest yet. Acknowledged as a limitation (§11). |
| Players who leave the Big 5 | Their outcome is missing *and* not missing-at-random. Flagged (`left_dataset`), never imputed as zero. |

## 3. The prediction problem, formally

For player $i$ in season $t$ we observe a feature vector $\mathbf{x}_{i,t}$
describing his current season, and we want

$$
y_{i,t+1} \;=\; \text{goals scored in season } t+1 .
$$

**Why next season and not this one.** Predicting the current season is
interpolation: the target is partly inside the features and any model will look
excellent. Next season is the only horizon where the prediction is a genuine
claim about the future, and it is the horizon a club actually cares about.

**The two quantities.** Let $m_{i,t+1}$ be minutes played next season. Then

$$
y_{i,t+1} \;=\; \underbrace{\frac{y_{i,t+1}}{m_{i,t+1}/90}}_{\text{rate}}
\;\times\; \underbrace{\frac{m_{i,t+1}}{90}}_{\text{opportunity}} ,
$$

and the two factors are modelled separately because they have different drivers:
the rate is a function of skill and role, the minutes are a function of
availability, age, squad depth and the manager. Multiplying two unbiased
estimates gives a product whose error is smaller than the error of modelling the
total directly, because the two errors are largely independent.

**Notation.**

| Symbol | Meaning |
|---|---|
| $y_i$ | goals (or assists) in the target season |
| $n_i$ | exposure, in 90-minute units (`minutes / 90`) |
| $\mathbf{x}_i$ | 34 features (§5) |
| $\lambda_i$ | per-90 rate, $\mathbb{E}[y_i]/n_i$ |
| $\mu_i$ | $\mathbb{E}[y_i] = \lambda_i n_i$ |
| $\beta$ | model coefficients |
| $\alpha$ | L2 penalty strength |

## 4. The data pipeline

```
FBref (via soccerdata, collected for the graduation project)
    9 stat tables per season: standard, shooting, passing, passing_types,
    goal_shot_creation, defense, possession, playing_time, misc
  
Project Data/Post_clean/f2020f2024.csv         committed inputs
    clean.py
      rename 120 source columns to canonical names (schema.py)
      map FBref season codes 19202020  23242024
      sum counts across clubs for mid-season transfers
      drop rows with zero minutes
  
data/processed/player_seasons.csv.gz           12,515 rows  123 cols
    targets.py      join (player_id, season) to (player_id, season + 1)
  
data/processed/targets.csv.gz                  6,942 rows with a real outcome
```

### 4.1 The grain, and why it matters

One row per `(player_id, season)`. The source has 693 player-seasons with **two
rows**  a January transfer gives a player a row per club. The old pipeline used
`groupby('player').shift(-1)` for the target, which for those rows produced a
"next season" that was the *other club in the same season*: 8.5% of training
labels pointed at the wrong season.

The fix is not to drop the duplicates but to merge them correctly: **sum
everything that is a count** (goals, minutes, passes, tackles), **take team
context from the club where he played most**, and keep `n_clubs` so the merge is
auditable. 657 rows are multi-club. A test verifies minutes are summed.

One artefact worth knowing about: 24 players in the source carry more than one
birth date, which splits 33 player-seasons into two `player_id`s. Some of those
are genuinely different people with the same name (the id's birth year is doing
its job); some are a data-entry inconsistency. Distinguishing them needs an
external ID crosswalk, which is a roadmap item. The effect on the models is 33
rows out of 12,515.

### 4.2 The target is a join, not a shift

```python
current[["player_id", "season", "next_season"]]  # next_season = season + 1
    .merge(future, on=["player_id", "next_season"], how="left")
```

A player with no row next season gets a **null** target plus `left_dataset = True`.
The invariant is asserted, not hoped for:

```python
assert (targets["next_season"] == targets["season"] + 1).all()
assert not (targets["y_gls_next"].isna() & targets["has_next_season"]).any()
```

That second line is the specific fix for the defect that made the original
report look 30% better than it was (a `fillna(median)` wrote a fake target into
rows that had no next season).

### 4.3 Season convention

`season` is the **end year**: `2024` means 2023/24. `season_label` carries
`"2023/24"` for humans. FBref's `2324` maps to `2024` once, in `schema.py`, for
outfield players and keepers alike  the original pipeline forgot the keepers,
which left their table unjoinable to everything else.

### 4.4 What is dropped, and why

The source has 166 columns; the canonical table keeps 123. The rest are
algebraic duplicates (`Ast` = `Performance_Ast`), per-90 or percentage columns
that must be recomputed from counts (`Standard_SoT%`), or averages that cannot
be summed (`Standard_Dist`). Each is listed with a reason in
`schema.OUTFIELD_DROPPED`  deletion with a note, not deletion by accident.

## 5. Features and shrinkage

Thirty-four features (`features.FEATURES`), all computable from the current
season alone or from an exact `season  1` lag. No column may contain the string
`next`; a test enforces it.

Groups: availability (minutes, matches, starts, start share, prior minutes),
age and age², per-90 volume (shots, SCA, GCA, passes, carries, tackles),
per-90 output (goals, npxG, xG, assists, xAG, xA), efficiency (shot accuracy,
pass completion), and three **shrunk** rates.

### 5.1 Per-90 rates are not rates at small samples

One goal in 25 minutes is "3.6 goals per 90". Left alone, this is the single
fastest way to break a model: a log link exponentiates it, and the first
Poisson run in this repository projected **60 goals for a 62-minute cameo**.

Two guards:

1. **Winsorisation at domain caps** (`features.PER90_CAPS`). Caps are set at
   roughly twice an elite season (3.0 goals/90, 130 passes/90), so they never
   bind for a regular starter. They are domain knowledge, not fitted
   quantities, so they cannot leak from the test set.
2. **Shrinkage toward the position-season mean** (below).

### 5.2 The shrinkage estimator, derived

Treat a player's per-90 rate as a Poisson process with an unknown $\lambda$, and
put a $\text{Gamma}(a, b)$ prior on it with prior mean $m = a/b$. After observing
$y$ events in $n$ nineties, the posterior is

$$
\lambda \mid y, n \;\sim\; \text{Gamma}(a + y,\; b + n),
$$

whose mean is the Bayes estimator under squared error:

$$
\hat\lambda_{\text{post}} \;=\; \frac{a + y}{b + n}
\;=\; \underbrace{\frac{n}{n+b}}_{\text{weight}}\cdot\frac{y}{n}
\;+\; \underbrace{\frac{b}{n+b}}_{\text{1} - \text{weight}}\cdot m .
$$

That is exactly the form in `features._shrink`:

```python
weight = n / (n + prior_90s)           # prior_90s = b = 10
return weight * rate + (1 - weight) * position_season_mean
```

So `prior_90s = 10` is not a magic number: it says **the prior is worth ten
matches of evidence**. A player with 10 nineties is at 50/50; with 30 nineties
he is 75% his own rate; with 2 nineties he is 17% his own rate. Restating it in
those terms is what makes it defensible in a review.

The prior mean is the position  season mean, so a forward is shrunk toward
forwards, not toward the league.

### 5.3 The age curve and the lag

`age` and `age²` let one pair of coefficients describe the rise to the mid-20s
and the decline after 30 without a hand-drawn breakpoint. The lag features
(`gls_per90_prev`, `npxg_per90_prev`, `minutes_prev`) are joined on an exact
`season  1`: a player missing a season gets null, never a two-year-old value.

## 6. The models

Three pieces. Gamma and Ridge come from scikit-learn; the Poisson rate model is
implemented here, because sklearn's Poisson regression has no exposure offset (see
§6.1). The original numpy versions of all three are kept in
`src/scoutend/obsolete/` as reference.

### 6.1 Poisson GLM for the rate (goals, assists)

Goals are counts. Their variance far exceeds their mean (next-season goals:
mean 2.2, variance 12.9), which is the signature of a count process, not a normal
one. Squared-error regression on raw counts  what the graduation notebooks did
 fits the handful of 30-goal seasons and distorts everyone else.

Model, with a **log exposure offset**:

$$
y_i \sim \text{Poisson}(\mu_i), \qquad
\log \mu_i = \log n_i + \mathbf{x}_i^\top\beta .
$$

The offset carries the exposure; $\mathbf{x}_i^\top\beta$ is therefore the
per-90 **rate** on a log scale:

$$
\lambda_i = \mathbb{E}[y_i]/n_i = e^{\mathbf{x}_i^\top\beta}.
$$

Log-likelihood (dropping the $\log y_i!$ term):

$$
\ell(\beta) = \sum_i \left[ y_i\,(\log n_i + \mathbf{x}_i^\top\beta) - n_i e^{\mathbf{x}_i^\top\beta} \right]
\;-\; \frac{\alpha}{2}\lVert\beta\rVert^2 .
$$

**Score and Fisher information.** With $\mu_i = n_i e^{\mathbf{x}_i^\top\beta}$:

$$
\nabla_\beta \ell = \mathbf{X}^\top(\mathbf{y} - \boldsymbol\mu) - \alpha\beta,
\qquad
\mathcal{I}(\beta) = \mathbf{X}^\top \mathbf{W} \mathbf{X} + \alpha \mathbf{I},
\quad \mathbf{W} = \operatorname{diag}(\boldsymbol\mu).
$$

**Fisher scoring (IRLS).** Newton's method on the penalised likelihood, where
each step is a weighted least-squares solve. Using the working response
$z_i = \eta_i + (y_i - \mu_i)/\mu_i$:

$$
\beta \leftarrow \beta + \left(\mathbf{X}^\top\mathbf{W}\mathbf{X} + \alpha\mathbf{I}\right)^{-1}
\left[\mathbf{X}^\top\mathbf{W}(\mathbf{z} - \mathbf{X}\beta) - \alpha\beta\right].
$$

Note $\mathbf{W}(\mathbf{z} - \mathbf{X}\beta) = \mathbf{y} - \boldsymbol\mu$, so
this is exactly the Newton step.

**Why not scikit-learn here.** `sklearn.linear_model.PoissonRegressor` has no
`offset` argument. Passing the exposure as `sample_weight` fits
$\mu = e^{b_0 + \mathbf{x}^\top\beta}$ with the *total* as its mean, so $b_0$
absorbs the exposure: measured on the 2023 fold, the intercept comes out at 3.0
($e^3 \approx 20$ goals) instead of 0.1 ($\approx$ the mean per-90 rate), and every
prediction is wrong by a factor of the exposure. Transforming the target to
$y - \log(\text{exposure})$ is not available either  a Poisson loss requires
non-negative responses. The Gamma and Ridge models, which need no offset, use the
library.

The implementation:

```python
eta    = clip(Z @ beta + log_offset, -12, 4)
mu     = exp(eta)
w      = mu                                  # Poisson variance function
z      = eta - log_offset + (y - mu) / mu    # working response
grad   = Z.T @ (w * (z - Z @ beta)) - penalty @ beta
hess   = (Z * w[:, None]).T @ Z + penalty
beta  += solve(hess, grad)
```

Details that matter:

- **Standardisation** inside `fit`, using training statistics only, so the L2
  penalty is applied on a comparable scale and the closed-form solve is stable.
- **Warm start** at the intercept that reproduces the mean rate. Starting from
  $\beta = 0$ puts $\mu$ at 1, which for minutes (mean  1,500) is a terrible
  place to begin Newton's method.
- **Clipping the linear predictor** to $[-12, 4]$ keeps a bad step from
  producing `inf` and a `NaN` gradient. A rate of $e^4 \approx 54$ goals per 90
  is far outside football, so the clip never binds on real data.
- **The intercept is never penalised.**

**Training population.** The rate model is fitted only on players with at least
450 minutes. A rate estimated from four minutes is not a rate, and a log-link
model will happily fit one goal in four minutes as 22 goals per 90. Predictions
are still made for *everyone*  low-minute players are exactly who a scout is
looking at  with the blend below keeping them sane.

### 6.2 Gamma GLM for expected minutes

A player's minutes are a positive, right-skewed duration. The obvious approach,
least squares on $\log(\text{minutes})$, is subtly wrong: it estimates the
**median**, and exponentiating a shrunk prediction under-predicts the **mean**
(Jensen's inequality). Measured on the 2023 fold:

| model | mean prediction | mean actual | bias |
|---|---|---|---|
| Ridge on `log(minutes)` | 1,211 | 1,468 | **17.5%** |
| Gamma GLM, log link | 1,493 | 1,468 | +1.7% |

Every goal projection is multiplied by this number, so a 17.5% shrink silently
deflates the entire product. The Gamma GLM estimates the mean directly:

$$
y_i \sim \text{Gamma}(\phi, \mu_i/\phi), \qquad \operatorname{Var}(y_i) = \mu_i^2/\phi,
\qquad \log\mu_i = \mathbf{x}_i^\top\beta .
$$

With $\partial\mu/\partial\eta = \mu$, the working weights become
$w_i = \mu_i^2 / \operatorname{Var}(y_i) = \phi$, i.e. **constant**. So the same
IRLS loop runs with `w = 1` and the same working response
$z_i = \eta_i + (y_i - \mu_i)/\mu_i$. That is why `PoissonRegressor` and
`GammaRegressor` are two subclasses of one `_LogLinkGLM`: they differ only in
`_weights`.

Features are **availability only** (`availability_matrix`): age, age², minutes,
matches, starts, start share, prior minutes, position. Output is excluded on
purpose, so the minutes model can be multiplied by any rate model without
double-counting.

### 6.3 Two-stage composition

$$
\widehat{y}_{i,t+1} \;=\; \hat\lambda_i \;\times\; \frac{\hat m_{i,t+1}}{90},
$$

with $\hat\lambda$ from the Poisson model and $\hat m$ from the Gamma model. The
two are trained independently and combined multiplicatively.

### 6.4 Blending the rate with the prior

Even after winsorisation, a 13-minute cameo can produce a feature vector the
model has rarely seen, and it will extrapolate confidently. The final rate is
therefore an **empirical-Bayes blend**, the same idea as §5.2 but applied to the
model's output:

$$
\hat\lambda^{\text{blend}}_i \;=\;
\underbrace{\frac{n_i}{n_i + k}}_{\text{trust in the model}}\hat\lambda^{\text{model}}_i
\;+\;
\underbrace{\frac{k}{n_i + k}}_{\text{trust in the prior}}\hat\lambda^{\text{shrunk}}_i ,
\qquad k = 4 .
$$

With 4 nineties of evidence the two are weighted equally; a player with 30
nineties is 88% the model's estimate. Two additional rails, both consequences of
things the harness caught:

- the modelled rate is capped at 1.5 goals per 90 (1.2 assists), above the best
  sustained seasons in the data;
- the final season total is clipped to 60, above any Big-5 season on record.

These are not cosmetic. Without them the projection table was headed by players
with 562 minutes and 60 projected goals; a scouting tool that does that is
worse than no tool.

### 6.5 Why not a gradient-boosted ensemble?

The graduation project used Random Forest + XGBoost + LightGBM on 169 features.
On the same split and feature matrix, a plain ridge regression scored **1.229**
against the ensemble's reported **1.21**  i.e. the ensemble added nothing
measurable, while adding three dependencies, hyperparameters and a 2 MB model.

This repository's position is: **a model earns complexity by beating the simpler
one on the harness.** Ridge is kept as the reference model precisely to police
that. If a GBM beats `poisson_blended_two_stage` on repeated folds by a margin
that survives inspection, it ships  with the same card and the same tests. The
current evidence does not support it, and the count likelihood is the more
appropriate tool for the target anyway.

## 7. Baselines

A model that is not compared to something is a claim, not a result. Five
baselines, all scored on exactly the same rows:

| baseline | what it says |
|---|---|
| `season_mean` | "everyone scores the average"  the floor |
| `position_mean` | "everyone scores the average for his position" |
| `last_season_gls` (`ast`) | "he'll do what he did last year"  what a fan would say |
| `last_season_npxg` (`xag`) | "he'll score what his chances deserved"  **the one to beat** |
| `rate_x_actual_minutes` | "same rate, same minutes" |

`last_season_npxg` is the honest opponent. Expected goals are already a
stabilised estimate of finishing *quality*, so it is a strong predictor; beating
it requires the model to add something beyond "regress last season's xG".

## 8. Evaluation methodology

### 8.1 Splits: forward only

For each test season $t \in \{2021, 2022, 2023\}$, train on every season
$< t$ and predict $t$. The invariant is asserted:

```python
if set(frame.loc[train, "season"]) & set(frame.loc[test, "season"]):
    raise AssertionError("train/test season overlap")
```

**Why not `KFold(shuffle=True)`.** The original code did exactly that on a panel
where each player appears up to five times. It leaks in two ways:

| leakage | mechanism | how it is prevented here |
|---|---|---|
| Temporal | a shuffled fold puts 2021 in training and 2022 in validation, so the model sees the future | train seasons strictly < test season |
| Group | the same player appears in train and validation, so the model memorises him | split by season, not by row |
| Preprocessing | median imputation computed over the full frame leaks test statistics | medians computed on the training block only |
| Target | a target written into the feature table, or filled with a default | targets live in a separate table; null means null |

### 8.2 Metrics, and what each one is for

With $e_i = y_i - \hat y_i$:

$$
\text{MAE} = \frac{1}{n}\sum|e_i|, \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum e_i^2}, \qquad
R^2 = 1 - \frac{\sum e_i^2}{\sum (y_i - \bar y)^2}.
$$

- **MAE** is the headline: it is in the units a scout thinks in ("off by 1.8 goals").
- **RMSE** punishes big misses; the gap between MAE and RMSE is the fat tail.
- **R²** compares against predicting the mean; negative means worse than the mean.
- **Spearman** $\rho$ is the Pearson correlation of the **ranks**. This is the
  metric that matters for a shortlist, because a scouting list is a ranking, not
  a point estimate. A model can have mediocre MAE and an excellent ranking.
- **Top-20 overlap**: of the 20 players who actually scored most next season,
  what fraction does the model also put in its top 20? Blunt, product-shaped,
  and hard to game.
- **Bias** $= \overline{\hat y} - \overline{y}$: does the model systematically
  over- or under-predict? A model can win on MAE and still be biased.
- **Calibration by decile**: mean prediction vs mean outcome within each decile
  of predictions. Aggregate MAE cannot tell you *where* a model is wrong; this
  can. If the top decile is predicted 12 and scores 8, the shortlist is inflated
   the one failure a scouting tool must not have.

### 8.3 Two scopes

Every metric is reported for **all players** and for **players with 450
minutes**. The second is the honest one for scouting (rates from 90 minutes are
noise), while the first catches models that behave badly on fringe players 
which is how the 60-goal cameo was found.

### 8.4 How to read a result

1. Compare against `last_season_npxg`, not against `season_mean`.
2. Require improvement on **more than one metric** (MAE **and** rank, ideally).
3. Check `bias` and calibration before trusting the headline.
4. Remember the sample: ~1,700 rows per fold, ~4,500 pooled. A 2% MAE
   difference is within noise; four models all beating the baseline by a
   consistent 27% is not.

## 9. Results

Next-season **goals**, 450 minutes, pooled over the three folds ($n = 4{,}449$):

| model | MAE | RMSE | R² | Spearman | top-20 | bias |
|---|---|---|---|---|---|---|
| **poisson_blended_two_stage** | **1.806** | 2.848 | 0.458 | **0.599** | 0.433 | 0.171 |
| poisson_blended_x_actual_minutes | 1.826 | 2.849 | 0.457 | 0.599 | 0.450 | +0.070 |
| poisson_two_stage | 1.834 | 2.889 | 0.442 | 0.592 | 0.417 | 0.139 |
| poisson_rate_x_actual_minutes | 1.837 | 2.873 | 0.447 | 0.598 | 0.450 | +0.082 |
| last season's npxG *(baseline)* | 1.847 | 2.893 | 0.440 | 0.592 | 0.432 | 0.000 |
| ridge | 1.850 | 2.825 | 0.466 | 0.595 | 0.433 | +0.102 |
| rate  actual minutes *(baseline)* | 1.936 | 2.979 | 0.406 | 0.557 | 0.483 | +0.146 |
| last season's goals *(baseline)* | 2.055 | 3.281 | 0.279 | 0.546 | 0.467 | +0.337 |
| position mean *(baseline)* | 2.214 | 3.524 | 0.170 | 0.419 | 0.066 | 0.247 |
| season mean *(baseline)* | 2.547 | 3.872 | 0.002 | 0.000 | 0.000 | 0.175 |

Next-season **assists**, same scope:

| model | MAE | R² | Spearman | top-20 |
|---|---|---|---|---|
| **poisson_blended_two_stage** | **1.324** | 0.319 | 0.535 | 0.234 |
| poisson_blended_x_actual_minutes | 1.328 | 0.325 | 0.544 | 0.251 |
| ridge | 1.334 | 0.334 | 0.539 | 0.183 |
| last season's xAG *(baseline)* | 1.422 | 0.232 | 0.518 | 0.233 |
| last season's assists *(baseline)* | 1.586 | 0.038 | 0.437 | 0.167 |

**Calibration**, goals, blended two-stage  the top decile is predicted 7.75 and
actually scores 8.28 (5% conservative), and no decile is off by more than 0.64
goals:

| decile | mean predicted | mean actual | bias |
|---|---|---|---|
| 0 | 0.45 | 0.48 | 0.04 |
| 3 | 1.04 | 0.94 | +0.09 |
| 6 | 1.96 | 1.97 | 0.01 |
| 8 | 4.07 | 4.71 | 0.64 |
| 9 | 7.75 | 8.28 | 0.53 |

**Interpretation.** Goals improve by 2.2% over the strongest baseline and assists
by 6.9%; ranking improves in both. That is a real but modest margin on 4,449
rows  treat the goal result as "not worse, and better explained", and the
assist result as the more convincing one. The value that is *not* marginal is
structural: every projection decomposes into rate and opportunity, which is a
sentence a Head of Recruitment can act on and a raw MAE cannot.

## 10. Software architecture and engineering controls

```
config.py   paths, season convention, minutes floor, scope          (no imports)
schema.py   column definitions, what is dropped and why             (no imports)
clean.py    raw tables -> canonical grain
targets.py  next-season outcomes, join-based, asserted
features.py 34 features + shrinkage
models/
  glm.py        Poisson (offset, numpy) + Gamma (sklearn)
  minutes.py    expected minutes
  registry.py   baselines + candidates, one list
  train.py      fit on everything, project the unplayed season
  card.py       model cards
eval/
  metrics.py    MAE/RMSE/R/Spearman/top-k/calibration
  harness.py    forward splits, scoring, report, cards
obsolete/       the original numpy estimators, kept as reference
```

Four controls that keep this honest, and none of them are optional:

1. **The registry.** Baselines and models are scored through the same function
   on the same rows. A model cannot be measured on a friendlier subset.
2. **Tests as encoded defects.** Every failure found in the audit became a test:
   duplicate player-seasons, targets from the wrong season, filled labels,
   keeper seasons not aligned, negative counts, target columns in the feature
   matrix. They fail loudly if the bug returns.
3. **Model cards.** Every shipping model writes a JSON + Markdown card
   (`models/`) with its target, training window, metrics, the baselines it
   beats, and its known limitations. A model without a card does not ship.
4. **CI.** `.github/workflows/ci.yml` builds the data, runs 28 tests, evaluates,
   trains the projection and uploads the report  so the numbers in the README
   cannot silently drift from the code.

## 11. Limitations and threats to validity

Stated plainly, because a model's limitations are part of its specification.

1. **Coverage.** Big 5 only. The players a club can actually sign cheaply are
   mostly outside it. No cross-league strength adjustment exists yet, so a
   Ligue 1 rate is treated as equivalent to a Premier League rate  it is not.
2. **Two seasons missing.** The data stops at 2023/24, so the 2025 projection
   cannot be scored. Everything reported here is retrospective.
3. **Statistical power.** ~1,700 rows per fold. Differences under ~3% MAE are
   not distinguishable from noise, and no confidence intervals are computed for
   the model differences. Treat the ordering of the top four models as a tie.
4. **Censoring.** Players who leave the Big 5 are dropped, not modelled. Teams
   that get relegated leave the dataset. The surviving population is therefore
   systematically more successful than the input population  a survivorship
   bias that flatters every model.
5. **The minutes model is thin.** It knows age, role and recent minutes. It does
   not know contract length, transfer rumours, squad arrivals, injuries or
   managerial change  which is most of what actually determines minutes.
6. **Identity is name + birth year.** 24 players in this dataset have more than
   one recorded birth date, so 33 player-seasons split into two ids. A proper
   cross-source ID is needed before a second data source is added.
7. **No uncertainty.** Projections are point estimates. A 19-year-old with 600
   minutes of good football should carry a much wider interval than a 28-year-old
   with 2,800, and the current output does not say so.
8. **Team context is static.** `team_strength` and `team_ppm` come from the
   current season; a transfer to a better team is invisible until it happens.
9. **Opposition quality** is not modelled at all (no match-level data ingested).
10. **Construct validity.** "Goals next season" is a proxy for "value to a club".
    It ignores defensive contribution, pressing, build-up, availability and price.
    A striker who scores 12 and presses like a maniac may be worth more than one
    who scores 15 and walks.

## 12. Roadmap

| Step | What | Why it is next |
|---|---|---|
| 1 | Ingest 2024/25 and 2025/26 | The projection is unscoreable and the product is two seasons stale. |
| 2 | Selling leagues (Eredivisie, Primeira Liga, Championship, Belgium) + a cross-league strength coefficient | Coverage is what makes a scouting tool useful. |
| 3 | Scouting layer: role clustering, percentile radars, similarity search, shortlists | Turns projections into a product surface. |
| 4 | Uncertainty: bootstrap or a Bayesian version of the GLMs | A point estimate without a range is not decision-grade. |
| 5 | FastAPI service, then the interface | The serving boundary; deliberately last. |
| 6 | Goalkeeper phase | Different targets (PSxG ±, save%), different metrics, its own evaluation. |
| 7 | Revisit gradient boosting with proper repeated CV | Only if it beats the current model on the harness. |

## 13. Glossary

| Term | Meaning here |
|---|---|
| **npxG** | non-penalty expected goals: the quality of the chances a player took |
| **xAG** | expected assisted goals: the quality of the chances a player created |
| **per 90** | rate per 90 minutes played |
| **90s / nineties** | number of 90-minute units played (`minutes / 90`) |
| **exposure** | the denominator of a rate; here, nineties |
| **offset** | a predictor whose coefficient is fixed at 1, used to expose a rate from a count |
| **shrinkage** | pulling an estimate toward a prior when evidence is thin |
| **empirical Bayes** | estimating the prior from the data (here: position-season means) |
| **GLM** | generalised linear model: a distribution family + a link function |
| **IRLS / Fisher scoring** | iterative weighted least squares; how the GLMs are fitted |
| **deviance / score** | how the likelihood changes with the parameters |
| **MAE / RMSE / R²** | error in original units / error with a tail penalty / variance explained |
| **Spearman ρ** | correlation of ranks; the metric for a shortlist |
| **calibration** | whether predicted magnitudes match observed magnitudes |
| **forward split** | train on the past, test on the future |
| **leakage** | information in training that would not be available at prediction time |
| **model card** | the document that states what a model is, does, and cannot do |

## References

- FBref (Sports Reference)  the data source.
- Nelder & Wedderburn (1972), *Generalized Linear Models*  the GLM framework and IRLS.
- Efron & Morris (1975), *Data Analysis Using Stein's Estimator and its Generalisations*  shrinkage and empirical Bayes.
- Duan (1983), *Smearing Estimate*  the log-transform bias problem.
- Robertson, Lyle, Hughes (2018), *Rethinking the Value of Grit*  a readable argument for why expected metrics beat raw output in football analytics.

