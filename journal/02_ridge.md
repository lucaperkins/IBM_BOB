# 02_ridge

<!-- Design note for experiments/02_ridge.py -->

## Question / hypothesis

Does a Ridge linear model with median imputation, evaluated under
patient-grouped 5-fold CV, beat the dummy floor and reveal how much
of the target variance is explained by the available features?

## Motivation

- **Sourcing strategy:** user (direct request this session)
- **Source(s):**
  - User asked: "Run a Ridge linear model (median imputation) and
    compare it to the dummy under patient-grouped 5-fold CV
    (GroupKFold on patient_id). Push the report to the hub as 02_ridge."
  - From EDA (`data/eda.md`): `on` and `off` (UPDRS sub-scores)
    have Pearson r = 0.87 / 0.67 with target; `ledd` / `age` add
    moderate signal. Ridge can exploit these linear correlations.
  - From `01_dummy` audit (SKD009): skore's internal HGB baseline
    achieves RMSE 7.16, R² 0.811 — the realistic ceiling for a linear
    model without feature engineering.
- **Why this matters:** Ridge is the simplest real model that uses all
  features. Comparing it to `01_dummy` under the same GroupKFold
  splitter tells us how much of the target is linearly predictable
  from the visit features, and whether the heavy missingness
  (handled here by median imputation) is a bottleneck.

## Method

- **Files touched:**
  - `src/parkipredict/pipeline.py` — add `build_ridge_learner()`
    (Option 2: new function; `build_learner` from `01_dummy` unchanged)
  - `src/parkipredict/evaluate.py` — add `splitter_grouped`
    (`GroupKFold(n_splits=5)` with `patient_id` groups)
  - `experiments/02_ridge.py` — new script using the above +
    `ComparisonReport` to compare against `01_dummy`
- **Change versus 01_dummy:**
  - Estimator: `Pipeline([SimpleImputer(strategy="median"), Ridge()])`
    applied to numeric columns; categorical columns
    (`patient_id`, `cohort`, `gene`, `rater_id`) are dropped (Ridge
    cannot handle strings).
  - Splitter: `GroupKFold(n_splits=5)` with `groups=X["patient_id"]`
    — no same-patient leakage across folds.
  - Comparison: `ComparisonReport` against the stored `01_dummy`
    EstimatorReport to visualise the lift.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id`.
  Patient-level holdout: the same patient never appears in both train
  and test within a fold.
- **Out of scope for this experiment:** categorical encoding,
  feature engineering, hyperparameter tuning, non-linear models.

## Risks / things that could invalidate the result

- Dropping categorical columns (`patient_id`, `cohort`, `gene`,
  `rater_id`) discards some signal (EDA shows `cohort` Cramér V = 0.09
  with target). Encoding them is a follow-up.
- Median imputation on `off` (42% missing) and `on` (30% missing)
  attenuates those features' signal; a more careful imputation
  (e.g., using correlated columns) could improve results.
- The `ComparisonReport` compares a CV report (`02_ridge`) to a
  holdout report (`01_dummy`); the numbers are not directly
  comparable (different split strategies). The comparison is
  qualitative.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 10.60 ± 0.27, MAE 8.53 ± 0.18, R² 0.587 ± 0.016 (GroupKFold 5-fold, patient holdout). Beats the dummy (RMSE 16.5) by 36%. Still 3.4 RMSE points above skore's HGB baseline (RMSE 7.16).
- **Implication for next iteration:** Linear model with only 8 numeric features gets to R² 0.59 — the `on`/`off` sub-scores carry most of the linear signal. SKD008 (highly correlated features: age ↔ age_at_diagnosis r=0.94) and SKD009 (worse than HGB) suggest non-linear encoding and/or the categorical features (`gene`, `cohort`, `rater_id`) are the next levers. HistGradientBoosting handles NaNs natively and can take all 12 features.
