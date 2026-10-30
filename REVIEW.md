# ScoutEnd v2.0 — Review of v1.0 and the Rebuild Plan

**Reviewed:** v1.0 (`notebooks/`): `Data_collection.ipynb`, `Data_Finalization.ipynb`, `Model.ipynb`, `Model_Playground.ipynb`, plus `Project Data/**` (21 CSVs, ~50 MB)
**Date:** 28 Oct 2026
**Method:** full read of all four notebooks (including every stored output), plus independent re-computation of the data lineage, labels, splits and headline metrics with pandas/numpy.

> Environment note: this machine has no working `scikit-learn` / `lightgbm` / `xgboost` and no outbound network, and the Anaconda install the notebooks were written against is broken (`numpy: DLL load failed`). So the notebooks were **not re-run**. Their logic was re-implemented exactly and their numbers re-derived from the CSVs — that is what the evidence below is based on (scripts in `audit/`).

---

## 1. Verdict

v1.0 answers its question well: a Big-5 scouting and next-season output model is
buildable from public data, and the pipeline that collected and structured five
seasons of it still supplies every row in v2.0. The data engineering is sound
(FBref, multi-table merge, per-90 normalisation, position splits).

What v1.0 does not do is show that its own numbers can be trusted, and that is
what v2.0 adds. Three classes of gap:

1. **The evaluation measures less than it reports.** The headline *MAE 1.21 on
   next-season goals* includes 820 test rows with no next season, whose target
   came from a `fillna(median)` that ran after the target column was created. A
   plain **ridge regression** on the same feature matrix and split scores
   **1.229**, so the ensemble is not adding measurable accuracy over a linear
   model. This is the classic gap between a working prototype and a validated
   one.
2. **The future-season cells inherit default labels.** `fillna(0)` runs after the
   target is built, so 64.9% of the weighted training rows carry a literal "0
   goals next season", including the whole 2024 block weighted 4×. The
   "breakout boosts" downstream are compensating for it.
3. **Nothing outside the notebooks can use the result.** No package, no pinned
   environment, no tests, no model artifacts, no interface. The shipped
   `attackers_with_predictions.csv` holds 24 predictions for 24 low-scoring
   forwards.

None of this makes the underlying work wrong: the signal is real, and once
measured properly the same feature set beats every naive baseline by ~18%
(§5). What is missing is the measuring stick, the packaging, and the product —
exactly the work v2.0 does.

---

## 2. How the pipeline actually works today

```
soccerdata/FBref  ──►  9 stat tables per season (standard, shooting, passing,
                        passing_types, goal_shot_creation, defense, possession,
                        playing_time, misc)
        │
        └─ outer merge on (league, season, team, player)     ──► 2020..2024.csv
                                                                  (166 cols)
                                                                        │
Data_Finalization: rename seasons 1920→2020 … 2324→2024, coerce to int,
                   split by pos into defenders / midfielders / attackers,
                   concat keepers separately, fillna(0)                  │
                                                                        ▼
                                                       FBREF.csv  (13,175 × 166)
                                                       + test2025.csv (2,546, season 2425)
                                                                        │
Model.ipynb: 4 near-duplicate experiments (Goals 2024, Goals 2025,
             Assists 2024, Assists 2025), each: lag target by player,
             hand-built features, RF+XGB+LGBM on a StandardScaler,
             fixed 0.4/0.3/0.3 ensemble, KFold(3, shuffle=True),
             post-hoc multiplier boosts, MAE on partial 2025 actuals.
```

**Data coverage:** 5 leagues (EPL, La Liga, Ligue 1, Bundesliga, Serie A), seasons 2019/20 → 2023/24, 13,175 player-season rows, 5,044 distinct players, 166 columns. Keepers in a separate 52-column table.

**This is two seasons stale.** We are in October 2026: 2024/25 is complete and 2025/26 is ~10 matchdays in. `test2025.csv` is a *partial* 2024/25 snapshot (mean 14.3 matches played, cut around March 2025).

---

## 3. What is genuinely good

| Area | Why it matters |
|---|---|
| Source choice | FBref is the best free source for this: xG, xAG, progressive carries/passes, SCA/GCA, per-90 and percentile-ready stats. That's the correct foundation for scouting. |
| Multi-table merge | 9 stat types merged into one wide player-season table is exactly the right shape for feature work. |
| Position segmentation | Splitting DF/MF/FW and treating keepers separately is the right product decision — a scouting tool is position-relative by nature. |
| xG thinking | Centring the model on `Expected_npxG` rather than raw goals is the professional choice; overperformance (`G-xG`) as a feature is a legitimate scout signal. |
| Minutes normalisation | Using `90s` and per-90 rates rather than raw counts is correct and often missed. |
| Exploration breadth | `Model_Playground.ipynb` (78 cells) shows 40+ iterations: correlation pruning, lag-2 averages, breakout classification, ensemble-weight search. The instinct to iterate and compare is the right one — it just needs a harness so the comparisons mean something. |

---

## 4. Findings

Severity: **S1** = materially affects the reported result · **S2** = materially degrades quality · **S3** = hygiene/maintainability.

### 4.1 Labels and targets

**F1 · S1 · The 2025 training set is 65% fake labels.**
`Model.ipynb` cell 1 creates `Performance_Gls_next` (line 29) and only later, inside the feature-engineering loop, runs `df.fillna(0, inplace=True)` (line 51). The 2,502 rows of season 2024 have no next season, so their target becomes **0**, and those rows are then concatenated with a 4× weight. Measured on the notebook's own code path:

```
rows in the training frame             13,175
rows whose label is a synthetic 0       64.9%
target mean of the 4x-weighted 2024 block  0.219  (vs 1.71–1.78 for real seasons)
```

The `dropna(subset=['Performance_Gls_next'])` that was supposed to remove them is a no-op, because there are no NaNs left. This is the single most damaging defect in the project: the model is explicitly taught that the most heavily weighted season is a season of zero goals, and the hand-written `*= 1.5` breakout boosts exist to patch the resulting under-prediction.

**F2 · S1 · 693 training labels resolve to the same season.**
`groupby('player').shift(-1)` assumes one row per player-season. There are **693 player-seasons with two rows** (mid-season transfers: Cagliari and Crotone for Ounas, Barcelona and Betis for Abde Ezzalzouli, …). For the first row, "next season" resolves to the *other club in the same season*. 8.5% of the 8,131 real training labels are wrong for this reason. A further **504 labels (6.2%)** silently jump a season because the player has no row in the following season.

**F3 · S2 · The target is never checked for sanity.**
No assertion that `shift(-1)` landed on `season + 1`; no handling of players leaving the Big 5, retiring, or dropping to a lower league (target = missing, not zero — those are different events and should be modelled differently).

**F4 · S2 · Keepers were filled with zeros and never season-mapped.**
`keepers_all.fillna(0)` writes `Save% = 0` for keepers who faced no shots (31 rows) and `PSxG/SoT = 0` where it is undefined — both are *undefined*, not zero, and 0 is the worst possible looking value for a scouting UI. Separately, `goalkeepers.csv` still carries FBref codes `1920, 2021, 2122, 2223, 2324` while every outfield file was mapped to `2020…2024`: **the keeper table cannot be joined to the outfield table at all.**

### 4.2 Evaluation

**F5 · S1 · The reported MAE includes rows with no outcome.** In cell 6 the same `fillna(median)` runs after the target exists, so the 820 season-2023 rows with no next season get `Performance_Gls_next = median(≈1.0)`. Those rows are in the test set and are ~free to predict. Re-running the notebook's exact feature matrix and split with a **ridge regression** (no GBMs):

| Evaluation | MAE |
|---|---|
| All 2,681 season-2023 rows (the notebook's method) | **1.229** |
| Only the 1,861 rows with a **real** next-season target | **1.420** |
| The 820 rows with no next season, on their own | 0.796 |
| Notebook's reported 3-model GBM ensemble | 1.21 |

So the ensemble's apparent edge over a linear model is ~0.02 MAE — noise. The reported figure is dominated by the fake rows; the real-target number is 1.42.

**F6 · S1 · The 2025 validation is circular.** `actual_2025` (partial-season goals) is used to *boost the prediction* (`if partial_goals > row['Expected_npxG']: predicted = max(predicted, partial_goals*1.5)`), and then the same `actual_2025` is used to *score* it. This cannot fail to look good. The assists model is worse: it multiplies the **actuals** by `scaling_factor = 1.5` before computing MAE, to convert a partial season into a full one — an invented constant, not a measurement.

**F7 · S2 · Cross-validation leaks across players and time.** `KFold(n_splits=3, shuffle=True)` on a panel where the same player appears up to 5 times, and where seasons are currency. Validation rows share a player with training rows, so the CV MAE is optimistic and tells you nothing about next-season performance.

**F8 · S2 · Recency weighting is done by duplicating rows, badly.** `data_full[df['season']==2022] * 2` is not a sample weight: it multiplies every column, including strings and the season code (`2022 * 2 = 4044`). It inflates memory, breaks any downstream join on `season`, and interacts badly with the KFold grouping. `sample_weight=` exists for exactly this.

**F9 · S2 · No baseline was ever computed.** The project never compares against "last season's goals", "last season's npxG", or "position-season mean". Without a baseline, 1.21 versus 1.42 is invisible, and you cannot tell whether 40 playground iterations improved anything.

**F10 · S2 · Aggregate metrics hide the product**. A single MAE over 2,681 rows dominated by fringe players with <450 minutes (29% of the dataset, 3,322 of them with 0 goals) says little about the ~300 players a club actually scouts. There is no rank correlation, no calibration check, no top-N hit rate, no MAE restricted to established players.

### 4.3 Data quality

**F11 · S2 · No player identity.** Joins are on the `player` string. The same human appears as different strings across sources and seasons, and 5,044 raw names normalise to 5,043 — so the collision/alias rate is low *within* FBref, but name-only keys will break the moment you add a second source (which any real scouting tool needs: transfer values, contracts, positions, injuries). There is no `player_id`, no `team_id`, no cross-source mapping table.

**F12 · S2 · Only 5 leagues.** For scouting, the interesting players are usually in the Eredivisie, Primeira Liga, Belgian Pro League, Championship, Süper Lig, Brazil, Argentina. Big-5-only means the tool can only recommend players who are already expensive.

**F13 · S2 · No league-strength adjustment.** `Team_Strength` is `Team Success_onG / league mean`, which normalises *within* a league but not *across* leagues. A 12-goal season in Ligue 1 and in the EPL are treated as comparable. Real scouting tools apply a cross-league coefficient.

**F14 · S3 · Season labelling is ambiguous and undocumented.** `seasons=1920` (2019/20) was mapped to `2020`; `seasons=2021` triggers soccerdata's "ambiguous: interpreting as 20-21" warning. The result is internally consistent but off-by-one from how anyone reads "season 2020", and the keeper table was never mapped (F4).

**F15 · S3 · Hardcoded absolute paths.** Every notebook reads `C:\Users\daboo\...`. Nothing runs on another machine, and `Data_Finalization` reads `all_data.csv` from the user root while writing into the current directory.

**F16 · S3 · No schema validation anywhere.** No column set assertion, no dtype assertion, no row-count expectation between `Pre_clean` → `Post_clean` → `Ready_data`. A silent FBref column rename would corrupt everything downstream without an error.

### 4.4 Modelling

**F17 · S2 · Squared error on a count target.** Goals are counts with overdispersion (variance ≈ 12.9 vs mean 2.2). Squared error on raw counts plus a 169-feature GBM is the least suitable option available; a Poisson/Negative-Binomial or a rate model (goals per 90, minutes-weighted) is both better calibrated and directly interpretable. Right now the model has no notion of "how many minutes will he play", so it cannot separate *ability* from *opportunity* — which is precisely the scout's question.

**F18 · S2 · Feature set is unfiltered.** 169 features including near-constant pass-type splits, `Standard_FK`, `Corner Kicks_Out`, and several algebraic duplicates of the target family. ~25 features account for essentially all importance (`Recent_xG_Form` 0.21, `Age` 0.10, `Medium_Cmp%` 0.05 …). No collinearity pruning, no missing-indicator, no minutes floor.

**F19 · S2 · Hand-tuned magic constants.** `*= 2.0` for the latest season, `*= 10.0` for Breakout_Signal, `*(1 + 0.3*(Gls>20))`, `Age_Decline` slope `-0.1`, ensemble weights 0.4/0.3/0.3, `clip(upper=40)`. Each is an unjustified number fitted by eye against the same validation data, and `Breakout_Signal` divides by nothing — it is an unbounded product of two percentage changes and team goals, so it is dominated by tiny denominators.

**F20 · S2 · `Adjusted_Goal_Rate` and similar features double-count.** Adding +30% to a player's rate because he scored >20 goals bakes an assumption into a feature and then lets the model re-learn it.

**F21 · S3 · No artifacts.** Models are never serialised. There is no version, no training date, no feature list, no metrics file. `attackers_with_predictions.csv` is a hand-exported table with 24 rows.

**F22 · S3 · Position dummies would double-count if positions were multi-valued.** In the current data `pos` is single-valued, so this is latent rather than active — but the raw collection produces `FW,MF` strings, so the moment collection changes, every multi-position player will land in two position-MAE buckets.

### 4.5 Product / engineering

**F23 · S1 · No application.** No API, no UI, no notebooks-for-demo, no schema, no deployment. The scouting report and dashboard that this was "fed to" do not exist in this workspace.

**F24 · S1 · No repository, no environment, no tests, no docs.** Not under version control; no `requirements.txt`/lockfile (the Anaconda env is irreproducible and currently broken); zero tests; no README. A hiring manager opening this folder sees four notebooks and a `Project Data` dump.

**F25 · S2 · Stale, mixed cell outputs.** `Model.ipynb` contains an `ImportError` alongside results, and the LightGBM logs (8,783 training rows) do not match the code as written (26,507 rows after the weighting described in F8) — the saved outputs are from several different runs. Anyone reading the notebook cannot reproduce a single number from it.

**F26 · S2 · One notebook per experiment, four near-duplicates.** Goals/assists × 2024/2025 repeat ~80 lines each, so every fix has to be made four times — and F1 was fixed only in the 2024 cells. This is a direct cause of the inconsistency.

**F27 · S3 · Modern-Python incompatibility.** Under pandas 3.1 (installed here) `inplace=` on `fillna`/`drop` is deprecated and the notebooks' Anaconda-era behaviour is gone. Nothing in the project is pinned, so "works on my machine" has already expired.

---

## 5. How good is it, really?

Reconstructed honestly from the CSVs (season 2023 → 2024, all positions, the notebook's own feature matrix):

| Predictor | MAE | R² |
|---|---|---|
| Season mean | 2.320 | 0.00 |
| Position-season mean | 2.131 | — |
| Last season's goals | 1.893 | — |
| Last season's npxG | 1.733 | — |
| 2-season average goals | 1.800 | — |
| **Ridge on the project's 155 features** | **1.673** | 0.38 |
| v1.0 GBM ensemble, *all rows* (as reported) | 1.21 | — |
| Project's GBM ensemble, *honest real-target rows* | 1.42 (ridge-equivalent) | — |

Read this carefully, because it is the good news: **the feature set carries real signal** — ~18–20% better than the best naive baseline, and R² ≈ 0.38–0.42 for next-season goals, which is in the normal range for this problem. The project's weakness was never the idea or the features. It was that the scoreboard was rigged, so nobody (including you) could see what was actually working.

---

## 6. What "modernize / operationalize" should mean

Three products, and the third is what makes it a portfolio piece rather than a notebook:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PLANE                                                            │
│    ingestion (FBref + a 2nd source) → raw lake (parquet) → schema tests   │
│    → canonical star schema (player, team, league, season, match)          │
│    → feature store (versioned feature tables)                             │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. MODELLING PLANE                                                       │
│    minutes model → rate model (Poisson/NB, minutes-weighted) → calibration│
│    + role clustering + similarity + league strength coefficients          │
│    + eval harness (baselines, time-split CV, rank metrics) → model cards  │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. PRODUCT PLANE                                                         │
│    FastAPI service (search, compare, shortlist, report)                  │
│    → UI (radar, percentile bars, shortlist builder)                      │
│    → scheduled refresh + monitoring + tests + docs                       │
└──────────────────────────────────────────────────────────────────────────┘
```

Concretely, the model v2 I would build:

1. **Two-stage, as scouts actually think.** Stage A: expected minutes next season (this is half of the product and you currently have none of it). Stage B: goals/assists **per 90** given minutes, with `minutes` as an offset in a Poisson GLM. Predicted totals = rate × expected minutes, which decomposes cleanly in the UI ("elite rate, limited role").
2. **Shrinkage instead of magic constants.** Empirical-Bayes/ridge shrinkage of per-90 rates toward position×league×age priors, weighted by minutes. This replaces `Adjusted_Goal_Rate`, the `*2`/`*10` multipliers and `Breakout_Signal` with one defensible estimator.
3. **Cross-league strength coefficients** estimated from players who move between leagues (a real, publishable technique) — or, cheaply, from a market-value regression as a proxy.
4. **Scouting layer:** role clustering (k-means/GMM on per-90 style vectors, position-relative), percentile radars, k-NN similarity ("find me the next X"), and a shortlist scorer that combines ability, age curve, value, and fit to a club's playing style.
5. **Evaluation that can fail:** fixed time-split folds, naive + ridge + Poisson baselines, MAE/RMSE/R², Spearman on per-90 rates, calibration plot, top-20 hit rate, and a minutes-restricted variant. Every model ships a JSON model card with these numbers.

---

## 7. Roadmap to March 2027 (7-on / 7-off)

Assumption: on a service week you can give this ~5–7 h total (1 h on weeknights), on an off week ~25–30 h. That is roughly **30 h per cycle, ~17 cycles → ~500 h**. Enough for everything below if the scope is disciplined.

| Cycle | Phase | Deliverable (something that runs and is committed) |
|---|---|---|
| 0 | **Foundation** (now) | Git repo, `pyproject`/lockfile, `src/` layout, config-driven paths, `audit/` findings encoded as failing tests, README |
| 1 | Data v2 | Canonical schema with surrogate `player_id`/`team_id`; dedupe mid-season transfers; fix keeper seasons; **correct lag target (`season+1` or null)**; minutes floor flag; parquet lake |
| 2 | Eval harness | Baselines (mean/previous/npxG/ridge) + time-split CV + rank metrics; reproduces §5 table in CI; model cards |
| 3 | Model v1.1 | Goals + assists as minutes-weighted Poisson/NB rate models; expected-minutes model; compare against the harness from cycle 2 |
| 4 | Ingestion 2024/25 + 2025/26 | live FBref pull, incremental + cached, schema tests, backfill both missing seasons |
| 5 | Second data source | Transfermarkt (values, contract, DOB) + ID mapping; league-strength coefficient v0 |
| 6 | Scouting layer | role clustering, percentile radars, similarity search, shortlists (notebook + figures first) |
| 7 | API | FastAPI: `/players`, `/players/{id}`, `/similar`, `/shortlist`, OpenAPI docs, tests |
| 8 | UI v1 | Player profile (radar + percentiles + projection with uncertainty), search, shortlist |
| 9 | Deployment | containerised, scheduled refresh, monitoring, public URL |
| 10 | Write-up 1 | "How I caught my own model cheating" — the F1/F5 story is genuinely strong content |
| 11 | Broader coverage | 3–4 additional leagues + cross-league adjustment |
| 12 | Polish/interfaces | squad-needs analysis, comparison view, exportable PDF scouting report |
| 13–16 | Buffer + interview prep | case study, 2 blog posts, CV, system-design drills, behavioural stories; **March 2027 = done** |

**Interview value by phase:** 0–2 → data engineering + evaluation rigour (rare and highly valued); 3–5 → modelling and MLOps; 6–9 → full-stack product; 10 → communication. Phases 0–2 are also the cheapest to do well, which is why they come first.

**Two constraints to hold yourself to:** (a) every cycle ends with something committed and runnable, never "explored a bit"; (b) once the harness exists, no model enters the repo without beating a stated baseline on it.

---

## 8. What has been built (Phases 0–3)

**Phase 0–1 (data layer).** Committed, tested.

1. **Repository** — `git init`, `.gitignore`, `pyproject.toml` with a core tier (`pandas`, `numpy`) and an optional `ml` tier that the data layer does not depend on.
2. **`src/` layout** — `config`, `schema`, `clean`, `targets`, `features`, `models`, `eval`, `cli`, plus `run.py` so it works without installing.
3. **`player_seasons`** — 12,515 rows × 123 columns, one row per player-season. 693 duplicate player-seasons from mid-season transfers are merged by summing counts (657 multi-club rows); 33 player-seasons split into two ids because the source records two birth years for the same name (documented as a limitation). Keeper seasons mapped, undefined ratios left null instead of zero.
4. **`targets`** — 6,942 rows with a real next season, joined on `season + 1` with an `AssertionError` if the join ever resolves otherwise, plus `left_dataset` for players who disappear from the dataset rather than scoring zero.

**Phase 2 (evaluation infrastructure).** A model registry, so baselines and candidates are scored on identical rows and identical splits, plus calibration by decile, rank metrics, and a JSON + Markdown model card for every shipping model. 27 tests. A GitHub Actions workflow builds the data, runs the tests, evaluates and uploads the report, so the numbers cannot drift silently.

**Phase 3 (models).** Poisson rate models with a log-exposure offset, a Gamma expected-minutes model, and an empirical-Bayes blend between the model rate and the player's position-shrunk rate.

Results on the forward folds (next-season goals, ≥450 minutes): the two-stage model reaches **MAE 1.806 and Spearman 0.599**, against **1.847 / 0.592** for last season's npxG and 1.850 / 0.595 for ridge. Assists improve more: **1.324 vs 1.422** for last season's xAG (−6.9%).

What that means, stated plainly: a 2.2% MAE gain on goals is real but small, and this project's own harness is what makes that visible. The durable value is the decomposition — every projection is a rate × expected minutes, so a scout can tell whether a player is rated for ability or for opportunity — and the guards the harness forced: winsorised per-90 features, rate caps, and a minutes-based blend without which the model projected 60 goals for a player with 62 minutes.

Reproduce: `python run.py build`, `python run.py eval`, `python run.py train`, `python -m unittest discover -s tests -t .`

**Next cycle:** ingest 2024/25 and 2025/26 (the data is two seasons stale, and the current projection for 2025 cannot be scored), then selling-league coverage with a cross-league strength coefficient, then the scouting layer (role clustering, percentile radars, similarity search), then the API.

---

## 9. Decisions taken, and what is still open

Answered by you:

1. **Target job** — **ML engineer**. So the portfolio weight goes to pipelines, testing, evaluation rigour, then serving — not to notebooks.
2. **Interface** — deferred to a later phase, then likely Streamlit if the cost is genuinely lower, FastAPI + React if it is close. The service boundary is designed now so either can be attached later.
3. **Scope** — **Big 5 + 3–4 selling leagues** (Eredivisie, Primeira Liga, Championship, Belgian Pro League). Adding them is a data-layer change, which is why `league_code` and the league filter are already in `schema.py`.
4. **Original interface** — lost. Rebuilding from the new data model.

Still open, to settle before the scouting layer:

- Which selling leagues first, given each one costs ingestion + name disambiguation + a strength coefficient.
- Whether the first product surface is a shortlist generator (rank players by fit to a role) or a player profile (explain one player). The shortlist is the more useful tool and the harder demo.
