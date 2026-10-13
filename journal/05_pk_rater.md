# Experiment 05_pk_rater

## Question / hypothesis

Can three blocks of X-only features — pharmacokinetic (PK) decay curves,
per-rater bias correctors, and within-patient linear UPDRS trajectories —
improve on 04_best (RMSE 4.55 ± 0.08) without any hyperparameter retuning?

## Motivation

04_best captured temporal context via neighbour shifts and per-patient
aggregates. Three unexploited signals remain:

1. **Medication timing**: `time_since_intake_on/off` tells us where the patient
   is on the levodopa dose-response curve. A single exponential decay
   exp(−t/τ) for τ ∈ {1, 2, 4} hours encodes that curve in a form the HGBR can
   use directly; interactions with the `on`/`off` score weight the decay by the
   observed drug effect.

2. **Rater bias**: `rater_id` is already categorical and handled by
   TableVectorizer, but the model has no explicit signal about whether a given
   rater scores systematically higher or lower than the patient's typical level.
   Computing `rater_bias_off/on = mean rater deviation from patient mean` gives
   the HGBR a correctable bias term.

3. **Disease trajectory**: a per-patient OLS slope of off/on vs age encodes the
   rate of disease progression.  The predicted value at the current visit age
   acts as a trajectory-extrapolated baseline.

## Method

Feature engineering (all computed on X, never on target):

1. For τ in {1, 2, 4} h: `decay_on_τh = exp(−time_since_intake_on/τ)`,
   `decay_off_τh`, `interact_on_τh = on × decay_on_τh`, `interact_off_τh`;
   `decay_ratio_on_off = decay_on_1h / decay_off_1h`.

2. Per-patient means of on/off → deviation per row → per-rater mean deviation
   (`rater_bias_off`, `rater_bias_on`).

3. Per-patient OLS of off/on ~ age: slope, intercept, predicted value at visit
   age (6 columns: `traj_off_slope`, `traj_off_intercept`, `traj_off_pred`,
   `traj_on_slope`, `traj_on_intercept`, `traj_on_pred`).

All 04_best features (temporal shifts, per-patient aggregates, missingness
flags, time_since_diagnosis) are applied first; the new features extend them.

Hyperparameters: reused from 04_best RandomizedSearchCV (hardcoded in
`HGBR_PARAMS_04`), no new search.

CV: GroupKFold(5), same patient groups as 04_best.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 4.43 ± 0.05 (GroupKFold 5-fold outer CV); vs 04_best 4.55 ± 0.08
- **Gain:** −0.12 RMSE (−2.6%), std tighter (0.05 vs 0.08)
- **Top 5 permutation importance:** off, patient_id, on, age, age_at_diagnosis
- **Hub URL:** https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/70976
- **Submission:** submissions/05_pk_rater.csv (11 013 rows)
- **Implication for next iteration:** PK/rater/trajectory features yield a
  modest but consistent gain. The top importances (off, patient_id, on, age)
  are unchanged — the new features are not yet displacing core signals. Possible
  next steps: hyperparameter retune on the expanded feature set, or target
  encoding for rater_id / patient-level effects.
