# Experiment 04_best

## Question / hypothesis

Can temporal neighbour features (previous/next visit values, deltas, age gaps,
visit number) plus per-patient off aggregates, combined with RandomizedSearchCV
tuning of the HGBR hyperparameters and patient-grouped inner CV, push RMSE below
03_hgbr_tabular (7.45)?

## Motivation

03_hgbr_tabular treats every row as independent. Parkinson's disease progression
is longitudinal: the same patient's previous off/on score and the trend since
the last visit should be strong predictors of the current target. Adding
within-patient temporal context (shift-1 and shift+1 neighbours, deltas, age
gaps, visit number) gives the model information it currently cannot access.

`off` is also often missing (79% in EDA); the nearest and last observed off are
proxy aggregates that recover that signal even when the current visit's off is
NaN. A missingness indicator for off/on/ledd lets the HGBR distinguish
"measurement not taken" from a true zero.

RandomizedSearchCV (20 iterations, GroupKFold inner CV) tunes
learning_rate, max_leaf_nodes, max_iter, min_samples_leaf, l2_regularization
without leaking patient identities between inner folds.

## Method

Feature engineering (stateless, FunctionTransformer):
1. Sort each patient's visits by age.
2. Within patient: shift on/off by ±1 for previous/next visit values.
3. Within patient: delta_on = on − on_prev, delta_off = off − off_prev.
4. Within patient: age_gap_prev = age − age_prev, age_gap_next = age_next − age.
5. Within patient: visit_number (0-based rank by age).
6. Per-patient aggregates: mean_on, mean_off, last_off, nearest_off
   (off value at the visit with the smallest |age − current age| among visits
   that have a non-NaN off, excluding self when possible).
7. Missingness flags: off_missing, on_missing, ledd_missing (0/1).

Drop columns: Index, patient_id, target (as in 03), plus time_since_diagnosis
is kept from the add_time_since_diagnosis step already present.

Pipeline: FunctionTransformer(add_best_features) → skrub.tabular_pipeline("regressor")
with the HGBR replaced by a RandomizedSearchCV-tuned HGBR using GroupKFold(5)
inner CV.

Outer CV: GroupKFold(5), patient groups — same as 03_hgbr_tabular.

Comparison: report_dummy (01), report_ridge (02), report_hgbr (03),
report_best (04) in one skore.compare.

## Risks

- Neighbour features from the test set also need to be computed relative to the
  same patient's rows — the feature function must process train+test consistently
  per patient (no target leakage since we only touch X).
- near_off aggregate requires identifying observations with non-NaN off; with
  heavy missingness this may just replicate the current off.
- Tuning 20 iterations × 5-fold inner CV on ~44k rows could take several
  minutes; mitigated by small max_iter and n_jobs=-1.
- visit_number and shift features are computed on the sorted order of X, which
  must match across fit / predict calls — the FunctionTransformer handles this
  correctly because it operates per-patient on whatever rows are passed.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 4.55 ± 0.08 (GroupKFold 5-fold outer CV); inner CV tuning best RMSE 4.54; vs 03_hgbr_tabular 7.45 ± 0.14
- **Implication for next iteration:** Temporal context and per-patient off aggregates drove a 39% RMSE reduction over 03. The model is now strongly patient-ID-aware; next steps could explore target encoding or more visit-level context (e.g. cumulative UPDRS trajectory).
