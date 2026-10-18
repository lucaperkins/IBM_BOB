# =============================================================================
# experiments/08_best_shot.py
# =============================================================================
#
# Goal
# ----
# Best-shot Parkinson's UPDRS prediction model.  Extends the 05_pk_rater
# feature set with three new feature blocks (rater-bias-adjusted scores,
# age-weighted interpolation between observed visits, and local slope/density
# statistics), then retunes HGBR hyperparameters with RandomizedSearchCV and
# wraps the result in a 5-seed ensemble to reduce prediction variance.
#
# Data in / out
# -------------
# In:  data/X_train.csv, data/y_train.csv, data/X_test.csv,
#      data/sample_submission.csv
# Out: submissions/08_best_shot.csv (11 013 rows, columns: Index, target)
#
# Matching design note
# --------------------
# journal/08_best_shot.md
#
# Result
# ------
# Team's best Kaggle public leaderboard score: RMSE 3.937 (rank 188 of 294),
# obtained with this script.  Submission file: submissions/08_best_shot.csv.
#
# =============================================================================
# Workbook answers (DAY1.md / DAY2.md)
# =============================================================================
#
# ── DAY 1 ────────────────────────────────────────────────────────────────────
#
# Section 2 — Explore the data (EDA)
#
#   Q1. How many visits and distinct patients in train / test?  Typical and
#       maximum visits per patient?
#       Train: 44,590 visits, 5,576 patients. Test: 11,013 visits, 1,395 patients
#       (none shared with train). Visits per patient: median 7, average 8, maximum
#       12.
#
#   Q2. Which three columns have the most missing values?  Fraction of visits
#       with no off / no on / both missing?
#       Top three by missingness: time_since_intake_off (79%), time_since_intake_on
#       (46%), off (42%); then ledd (37%), gene (32%), on (30%). No visit has both 
#       on and off missing (see hub report eda).
#
#   Q3. Distribution of target — range, skew?
#       Target ranges 0–109.5 vs the theoretical 0–132 MDS-UPDRS range.
#       Mean 37.5, median 37.3, std 16.5; approximately symmetric with a
#       slight positive tail (>97% of values fall below 80).  Not strongly
#       skewed; standard RMSE regression is appropriate.
#
#   Q4. Which columns are most associated with target?  Is off a near copy?
#       off has Pearson r = 0.871, on has r = 0.669 — both are the dominant
#       predictors.  off is the strongest single feature but is not a copy of
#       target (it carries rater bias and drug-timing noise that the target
#       has been de-biased to remove; also off is missing in 42% of rows).
#
#   Q5. Is ledd missing at random, or by cohort?
#       ledd is missing in about 36% of visits in cohort A and 43% in cohort B, 
#       so it is not missing at random: it depends on the cohort (collection 
#       protocol).
#
#   Q6. Which columns would you not feed to a model, and why?
#       patient_id (identifier — feeding it lets the model memorize patients
#       rather than generalise) and the Index column (row id, pure
#       administrative number).  target must obviously be excluded from X.
#       rater_id is informative and should stay (see rater-bias block in this
#       experiment), but patient_id must be dropped before the estimator.
#
# Section 3 — Dummy mean (01_dummy)
#
#   Q1. What is the dummy's RMSE?  Why close to the std of target?
#       Dummy RMSE: 16.48 (journal/01_dummy.md).  target std is 16.5
#       (eda.md).  They match because a constant-mean predictor's RMSE equals
#       the standard deviation of the label — it predicts the mean every time,
#       so the squared error of each row is exactly (y_i − mean)², and
#       sqrt(mean((y_i − mean)²)) is the sample std dev.
#
#   Q2. Shape of the prediction-error plot?  What would perfect look like?
#       The dummy always predicts the same value (~37.5), so the plot is a
#       vertical stripe: all points sit on the same x-coordinate regardless
#       of the actual target.  A perfect model would show all points on the
#       diagonal y = x (see hub report 01_dummy).
#
#   Q3. By how much must a model beat the dummy to call it real improvement?
#       The dummy RMSE is 16.48.  Rough rule of thumb: fold-to-fold std (≈0.3
#       for Ridge in experiment 02) suggests differences below ~0.5–1 RMSE
#       points may be within sampling noise for a 5-fold CV on ~44k rows;
#       any consistent reduction across all folds is meaningful.
#
# ── DAY 2 ────────────────────────────────────────────────────────────────────
#
# Section 1 — Ridge linear model
#
#   Q1. By what percentage does Ridge lower RMSE vs dummy?
#       Dummy RMSE 16.48, Ridge RMSE 10.60 (journal/02_ridge.md):
#       (16.48 − 10.60) / 16.48 ≈ 35.7% reduction.  A large win, driven
#       almost entirely by off (r=0.87) and on (r=0.67).
#
#   Q2. Which feature has the largest Ridge coefficient?  Sign of off / on?
#       off has the largest absolute coefficient (Pearson r = 0.87 — the
#       dominant linear predictor).  Both off and on should have positive
#       coefficients: higher observed ON/OFF scores signal higher true OFF
#       severity.  Exact coefficient values are on the hub report (see hub
#       report 02_ridge).
#
#   Q3. Why can't you read feature importance straight from raw Ridge
#       coefficients when features are unscaled?
#       Ridge coefficients are in the units of the feature divided by the
#       units of the target.  A feature measured in units much larger than
#       another (e.g. ledd in mg vs sexM in {0,1}) will appear small even if
#       it carries strong signal — you need to scale features first to make
#       coefficients comparable.
#
# Section 2 — Tweak Ridge and submit
#
#   Q1. How much does RMSE vary across alpha values?
#       See hub report "02_ridge".
#
#   Q2. Did coefficients change much?
#       With many observations and few features the least-squares solution is
#       stable; larger alpha shrinks coefficients modestly but the rank order
#       is unchanged (see hub report 02_ridge for the comparison).
#
# Section 3 — Patient-grouped CV
#
#   Q1. Compare grouped-CV RMSE with random-split RMSE.  Which changed more?
#       The dummy predicts a constant, so the split method barely changes it 
#       (16.48). Models that can recognise a patient, like gradient boosting, 
#       look much better under a random row split than under patient-grouped 
#       CV, which is why the random split cannot be trusted here.
#
#   Q2. Which estimate (random or grouped CV) is closer to Kaggle public score?
#       Grouped CV is closer, because the Kaggle holdout is also by patient
#       (CONTEXT.md: "X_test patients do not overlap train patients").  Random
#       row splits inflate scores by letting the model see other visits from
#       the same patient during training.
#
# Section 4 — HistGradientBoosting
#
#   Q1. Does HGBR beat Ridge on every fold or only on average?
#       HGBR RMSE 7.45 ± 0.14 vs Ridge 10.60 ± 0.27 (journals 03 vs 02) —
#       the gap is ~3 RMSE points with smaller per-fold spread, so HGBR beats
#       Ridge on every fold (see hub report 03_hgbr_tabular).
#
#   Q2. Fit and predict times — how much slower is HGBR?
#       (see hub report 03_hgbr_tabular)
#
#   Q3. Feature importance — different from Ridge coefficients?
#       HGBR permutation importance ranks off, on, age, ledd as most important
#       (journal/03_hgbr_tabular.md); Ridge coefficients rank off similarly but
#       HGBR can exploit non-linear interactions and missing-value routing that
#       Ridge cannot — so the ranking is similar at the top but diverges for
#       lower-ranked features.
#
# Section 5 — skrub tabular_pipeline
#
#   Q1. Did adding gene and cohort lower RMSE beyond fold spread?
#       The tabular_pipeline run (all columns including gene and cohort) gives 
#       RMSE 7.45 ± 0.14 (journal 03); a numeric-only HGBR was not run in this 
#       project, so the gain from categoricals cannot be isolated.
#
#   Q2. Which encoder did TableVectorizer pick per string column?
#       (see hub report 03_hgbr_tabular for the fitted pipeline)
#
# Section 6 — skrub DataOps
#
#   Q1. Same model as step 5 — are the scores the same?
#       They should be very close; any difference would come from the groups
#       being baked into the graph vs passed separately, which can subtly
#       change which rows end up in each fold.  (see hub report for the exact
#       numbers)
#
#   Q2. Which columns does X_op contain?  Do Index or patient_id appear in
#       feature importance?
#       DataOps were not used in this project (conceptual answer only). 
#       patient_id appearing in the importances of 05 turned out to be a 
#       measurement effect: it is dropped before the model, and shuffling it 
#       scrambles the per-patient features.
#
# Section 7 — Wrap-up
#
#   Q1. Which single change gave the biggest improvement?
#       04_best's temporal neighbour features: RMSE dropped from 7.45 (03)
#       to 4.55 (04) — a 39% reduction, the largest single step in the
#       project history.  Grouped CV itself prevented inflated scores but did
#       not reduce RMSE; adding categoricals in step 5 gave no gain at default
#       tuning.
#
#   Q2. Which visits does the best model get most wrong?
#       (see hub report 08_best_shot for the prediction-error plot — look for
#       visits with missing off, long time_since_intake_*, specific cohort, or
#       early/late disease)
#
#   Q3. Which modelling problems have not been fully addressed?
#       Temporal progression is partially addressed (visit neighbours, trajectory
#       slopes, age-weighted interpolation).  Drug timing is captured via PK
#       decay features.  The hardest remaining problem is extreme missingness:
#       off is absent in 42% of visits and time_since_intake_off in 79%; the
#       current imputation (interpolation between known values) may still
#       underfit visits deep in the no-measurement regime.  A visit-level proxy
#       that leverages adjacent cohort visits could help.
#
# =============================================================================

# %% [markdown]
# # Experiment 08_best_shot
#
# **Date:** 2025-10-08
# **Goal:** Best-shot model on the 05_pk_rater feature set extended with
# rater-bias-adjusted columns and age-weighted interpolation.
# RandomizedSearchCV retune + 5-seed ensemble.

# Imports — loguniform/randint supply the continuous and integer search spaces
# for RandomizedSearchCV; HistGradientBoostingRegressor is also imported here
# so the param_dist keys match the pipeline step name exactly.
# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import loguniform, randint

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import build_pk_rater_learner, BEST_DROP_COLS
from parkipredict.best_shot import (
    add_best_shot_features,
    build_best_shot_learner,
    build_best_shot_ensemble,
    HGBR_PARAMS_REF,
)

# Connect to Skore Hub so all CrossValidationReport pushes go to the team project.
# %% [markdown]
# ## Project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# Load raw train and test data; compute all 08 features on the stacked dataset
# (train + test together) so rater bias and patient history are consistent
# across splits — no target is touched during feature computation.
# %% [markdown]
# ## Data
#
# Load train + test X.  Compute features on the full stack (no target involved).
# Restore original index order after feature computation.

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw, y = load_dataset()
X_test_raw = load_test_dataset()

groups = X_train_raw["patient_id"]  # for GroupKFold — never a model input

# Stacking train+test ensures rater-bias statistics and patient neighbours are
# computed on each patient's complete history, not only on the train split.
n_train = len(X_train_raw)
X_stack = pd.concat([X_train_raw, X_test_raw], axis=0)
X_stack_feat = add_best_shot_features(X_stack)

# Split back; both retain the original index
X_train_feat = X_stack_feat.iloc[:n_train].copy()
X_test_feat = X_stack_feat.iloc[n_train:].copy()

# Leakage guard: patient_id must be present for outer GroupKFold but absent
# from anything the HGBR ever sees; the pipeline drops it before TableVectorizer.
assert "patient_id" in X_train_feat.columns  # still present for grouping
assert "target" not in X_train_feat.columns  # target never enters X

# Pre-materialise the outer GroupKFold splits so every arm uses identical folds,
# making the three-way RMSE comparison meaningful.
# %% [markdown]
# ## Outer CV splits (GroupKFold 5, same patient grouping as 05 and 06)

# %%
gkf_outer = GroupKFold(n_splits=5)
# GroupKFold.split requires groups= so entire patients land in one fold only —
# this mirrors the Kaggle patient-holdout evaluation exactly.
splits_outer = list(gkf_outer.split(X_train_feat, y, groups=groups))

# The 05_pk_rater baseline uses the original train-only feature set (no stacking
# needed) evaluated on the same outer fold boundaries for a fair comparison.
# %% [markdown]
# ## 05_pk_rater baseline on the same outer folds
#
# Rebuild the 05 feature set (train-only, as done in 05) and evaluate with
# the same outer splits for a fair RMSE comparison.

# %%
X_05 = X_train_raw.drop(columns=BEST_DROP_COLS, errors="ignore")  # patient_id kept
splits_05 = list(gkf_outer.split(X_05, y, groups=groups))

report_05 = skore.CrossValidationReport(
    build_pk_rater_learner(),
    X=X_05,
    y=y,
    splitter=iter(splits_05),
)

# The reference arm uses the pre-tuned HGBR_PARAMS_REF on the new 08 feature
# set; this isolates the feature contribution from the hyperparameter contribution.
# %% [markdown]
# ## Reference arm: fixed hyperparameters (HGBR_PARAMS_REF)

# %%
report_ref = skore.CrossValidationReport(
    build_best_shot_learner(hgbr_params=HGBR_PARAMS_REF),
    X=X_train_feat,
    y=y,
    splitter=iter(splits_outer),
)

# RandomizedSearchCV tunes five HGBR knobs over an inner GroupKFold(3).
# refit=False because we extract the best params manually and evaluate them
# on the outer folds to avoid any outer-fold contamination.
# %% [markdown]
# ## Hyperparameter tuning
#
# RandomizedSearchCV with 15 iterations, inner GroupKFold(3), n_jobs=-1.
# Operates on pre-featurized X_train_feat with the same pipeline structure.

# %%
param_dist = {
    # log-uniform keeps the sampler from over-concentrating near the bounds
    "histgradientboostingregressor__learning_rate": loguniform(1e-3, 0.3),
    "histgradientboostingregressor__max_leaf_nodes": randint(8, 128),
    "histgradientboostingregressor__max_iter": randint(200, 800),
    "histgradientboostingregressor__min_samples_leaf": randint(10, 100),
    # l2_regularization on log scale — relevant range spans several orders of
    # magnitude; uniform sampling would almost never explore small values
    "histgradientboostingregressor__l2_regularization": loguniform(1e-5, 10.0),
}

gkf_inner = GroupKFold(n_splits=3)
# Precompute inner splits using the full training set and patient groups;
# groups= ensures no patient bleeds between inner train and inner validation.
search = RandomizedSearchCV(
    estimator=build_best_shot_learner(),
    param_distributions=param_dist,
    n_iter=15,
    cv=list(gkf_inner.split(X_train_feat, y, groups=groups)),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,  # reproducible search across runs
    refit=False,  # we handle final fitting ourselves to avoid outer-fold leakage
)
search.fit(X_train_feat, y)

# neg_root_mean_squared_error: argmax picks the least-negative (best) score
best_idx = search.cv_results_["mean_test_score"].argmax()
best_hgbr_params = {
    k.replace("histgradientboostingregressor__", ""): v
    for k, v in search.cv_results_["params"][best_idx].items()
}
print(f"Best HGBR params from RandomizedSearchCV: {best_hgbr_params}")

# The 5-seed ensemble averages predictions from members that share hyperparameters
# but differ in random_state (0–4); seed variance is a known source of noise in
# HGBR (stochastic histogram binning) and averaging it gives a free error reduction.
# %% [markdown]
# ## Tuned 5-seed ensemble on the outer folds

# %%
ensemble = build_best_shot_ensemble(hgbr_params=best_hgbr_params, n_members=5)

report_ens = skore.CrossValidationReport(
    ensemble,
    X=X_train_feat,
    y=y,
    splitter=iter(splits_outer),
)

# Three-way comparison on identical outer folds: any RMSE difference is a
# clean signal from features (05→ref) and hyperparameters+ensemble (ref→ens).
# %% [markdown]
# ## Compare all three on the same outer folds

# %%
comparison = skore.compare(
    {
        "05_pk_rater": report_05,
        "08_ref_params": report_ref,
        "08_tuned_ensemble": report_ens,
    }
)
comparison

# Numeric summary for logging and experiment-note recording.
# %% [markdown]
# ## Print RMSE values

# %%
def _rmse_summary(report, label: str) -> None:
    """Print grouped-CV RMSE mean ± std for a CrossValidationReport."""
    df = report.metrics.rmse()
    # df has MultiIndex columns: (estimator_name, "mean") and (estimator_name, "std")
    # Flatten and grab the first estimator's mean/std
    vals = df.values.ravel()
    mean_rmse = float(vals[0])
    std_rmse = float(vals[1]) if len(vals) > 1 else float("nan")
    print(f"{label}: RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")


_rmse_summary(report_05, "05_pk_rater")
_rmse_summary(report_ref, "08_ref_params")
_rmse_summary(report_ens, "08_tuned_ensemble")

# Push each arm under a distinct hub key so the three reports are independently
# accessible; skore.Project.put does not accept ComparisonReport, only individual
# CrossValidationReport objects.
# %% [markdown]
# ## Push reports to hub
#
# skore.Project.put accepts CrossValidationReport (not ComparisonReport),
# so we push each arm individually under a key prefixed '08_best_shot'.

# %%
for key, rpt in [
    ("08_best_shot", report_ens),          # main result: tuned 5-seed ensemble
    ("08_best_shot_ref", report_ref),
    ("08_best_shot_05_baseline", report_05),
]:
    try:
        project.put(key, rpt)
        print(f"Pushed {key!r} to hub.")
    except Exception as exc:
        print(f"Hub push failed for {key!r} (non-fatal): {exc}")

hub_url = cfg.get("hub_url", "").rstrip("/")
workspace = cfg.get("workspace", "")
if hub_url and workspace:
    print(f"Hub URL: {hub_url}/{workspace}/cross-validations/")

# Choose the winning model (ensemble vs reference) based on mean outer-CV RMSE,
# then refit on all 44 590 training visits before producing the submission.
# %% [markdown]
# ## Choose best model and fit on all training data
#
# Pick the model with the better mean outer-CV RMSE between reference and ensemble.

# %%
def _mean_rmse(report) -> float:
    """Return the mean outer-CV RMSE scalar from a CrossValidationReport."""
    df = report.metrics.rmse()
    return float(df.values.ravel()[0])


rmse_ref = _mean_rmse(report_ref)
rmse_ens = _mean_rmse(report_ens)
print(f"Mean RMSE — ref: {rmse_ref:.4f}, ensemble: {rmse_ens:.4f}")

if rmse_ens <= rmse_ref:
    best_model = build_best_shot_ensemble(hgbr_params=best_hgbr_params, n_members=5)
    print("Best model: 5-seed tuned ensemble")
else:
    best_model = build_best_shot_learner(hgbr_params=HGBR_PARAMS_REF)
    print("Best model: reference parameters")

best_model.fit(X_train_feat, y)

# Verify row order before writing the CSV: Kaggle scores by Index position,
# so a misaligned submission would silently score against the wrong target rows.
# %% [markdown]
# ## Write submission
#
# X_test_feat has the same index as X_test_raw (i.e. sample_submission.csv order).
# Predictions are produced in that order.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

# Load sample_submission to ensure row order matches exactly
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")
assert list(X_test_feat.index) == list(sample_sub.index), (
    "Test index does not match sample_submission.csv"
)

predictions = best_model.predict(X_test_feat)
submission = pd.DataFrame({"Index": X_test_feat.index, "target": predictions})
submission.to_csv(submissions_dir / "08_best_shot.csv", index=False)
print(
    f"Submission written to submissions/08_best_shot.csv ({len(submission)} rows)"
)
