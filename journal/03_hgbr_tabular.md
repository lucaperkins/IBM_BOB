# 03_hgbr_tabular

<!-- Design note for experiments/03_hgbr_tabular.py -->

## Question / hypothesis

Does a skrub `tabular_pipeline("regressor")` with all available
features (including `time_since_diagnosis = age − age_at_diagnosis`)
and patient-grouped 5-fold CV beat the Ridge baseline, and how close
does it get to skore's internal HGB benchmark (RMSE 7.16)?

## Motivation

- **Sourcing strategy:** user (direct request this session)
- **Source(s):**
  - User asked: "Build a skrub tabular_pipeline('regressor') using
    all columns except Index, patient_id and target. Add a
    time_since_diagnosis = age - age_at_diagnosis feature. Evaluate
    with patient-grouped 5-fold CV (GroupKFold on patient_id) and
    compare against 02_ridge and the dummy. Push to the hub as
    03_hgbr_tabular."
  - From `02_ridge` audit (SKD008): `age` ↔ `age_at_diagnosis`
    near-collinear (r=0.94); collapsing into disease duration removes
    redundancy.
  - From `02_ridge` audit (SKD009): HGB baseline RMSE 7.16 vs Ridge
    RMSE 10.60 — the gap is explained by Ridge dropping 4 categorical
    features and not handling NaNs natively.
- **Why this matters:** `tabular_pipeline` wraps
  `HistGradientBoostingRegressor` (which handles NaN natively) inside
  a `TableVectorizer` that encodes strings and dates automatically.
  Using all 12 input columns + the derived `time_since_diagnosis`
  feature closes the categorical gap that hurt Ridge.

## Method

- **Files touched:**
  - `src/parkipredict/pipeline.py` — add `add_time_since_diagnosis()`
    (stateless feature function) and `build_hgbr_tabular_learner()`
    (new function, Option 2; `build_learner` and `build_ridge_learner`
    unchanged)
  - `experiments/03_hgbr_tabular.py` — new script
- **Change versus 02_ridge:**
  - Estimator: `skrub.tabular_pipeline("regressor")` —
    `TableVectorizer` for all columns + `HistGradientBoostingRegressor`
    (default).
  - Features: all 10 non-target, non-Index columns (drop `patient_id`
    after extracting `groups`) plus `time_since_diagnosis` derived
    before fitting.
  - Missing data: handled natively by HGBR — no manual imputation.
  - Splitter: same `GroupKFold(n_splits=5)` on `patient_id`.
  - Comparison: `skore.compare` with stored `02_ridge` report and a
    fresh `01_dummy` CV report, all on the same GroupKFold folds.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id`.
- **Out of scope:** hyperparameter tuning, target encoding, temporal
  features, patient-level aggregation.

## Risks / things that could invalidate the result

- `tabular_pipeline` uses default HGBR hyperparameters; performance
  may be below a tuned model.
- `time_since_diagnosis` is a linear combination of `age` and
  `age_at_diagnosis`; it replaces the near-collinear pair — if HGBR
  already handles collinearity gracefully the gain may be small.
- `TableVectorizer` applies `MinHashEncoder` to high-cardinality
  strings like `patient_id` and `rater_id`; since `patient_id` is
  excluded, only `rater_id`, `cohort`, `gene` are encoded — low risk.
- The comparison includes `01_dummy` re-evaluated on GroupKFold folds
  (different split from the original holdout in `01_dummy`), so the
  dummy RMSE in the comparison will differ slightly from the stored
  report.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 7.45 ± 0.14, MAE 5.85 ± 0.10, R² 0.796 ± 0.003 (GroupKFold 5-fold, patient holdout). Beats Ridge by 30% on RMSE. Virtually matches skore's HGB internal benchmark (RMSE 7.16 vs our 7.45 — gap of 0.29).
- **Implication for next iteration:** Adding all features + time_since_diagnosis closes most of the gap to the HGB ceiling. Remaining checks: SKD009 still slightly below the internal benchmark (tuning HGB hyperparameters), SKD012 some features with near-zero permutation importance, SKD016 not tuned. Next lever: hyperparameter search (n_estimators, max_leaf_nodes, learning_rate) or feature selection.
