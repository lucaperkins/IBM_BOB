# Experiment 07_patient_stack

## Question / hypothesis

Can a two-stage patient-context stacker improve on the best prior model
(06B_ensemble, RMSE 4.40 ± 0.06) by exploiting the fact that a patient's
true OFF score evolves slowly while individual visit-level predictions are
noisy?

## Motivation

Stage 1 (05_pk_rater pipeline) makes an independent prediction for every
visit.  But visits from the *same patient* share a slowly-drifting disease
state — if the model overpredicts visit A and underpredicts visit B for a
patient, that is pure noise.  A second stage that sees all other visits'
predictions for the same patient before correcting the current visit can
partially cancel that noise.

The stage-2 signal:
- `ctx_mean` / `ctx_std` of the other visits' predictions → patient-level
  anchor and spread.
- `ctx_prev_pred` / `ctx_next_pred` → local temporal smoothness.
- `ctx_loo_at_age` → leave-one-out per-patient OLS of predictions vs age,
  evaluated at the current age → disease-trajectory-corrected baseline.
- `ctx_n_visits` → reliability weight (more visits → tighter context).
- Own stage-1 prediction, `off`, `on`, `age`, missingness flags → the
  stage-2 HGBR can blend the context signal with the original strong
  features.

## Method

Implemented as a single sklearn estimator `PatientStackLearner`
(fit / predict) so skore can cross-validate it identically to any other
model.

**Stage 1:** 05_pk_rater pipeline (all PK/rater/trajectory features +
`_drop_patient_id_05` + TableVectorizer + HGBR with `HGBR_PARAMS_04`,
single seed, no ensemble).

**Fit:**
1. Inner `GroupKFold(3)` on `patient_id` → OOF stage-1 predictions (no
   target leakage).
2. Refit stage 1 on all training rows.
3. Build stage-2 features from OOF predictions (see motivation above).
4. Fit a small HGBR (`max_iter=300`, `max_leaf_nodes=31`,
   `learning_rate=0.05`, `min_samples_leaf=20`) on the stage-2 features
   against the target.

**Predict:** stage-1 predictions from the refitted model → same stage-2
features computed within each test patient → stage 2.

Constraints enforced: target never used as a feature; stage 2 only trained
on OOF stage-1 predictions; `patient_id` never a model input.

**Comparison report:** `07_patient_stack` key on the hub, comparing
05_pk_rater, 06B_ensemble, and 07_patient_stack on the same
`GroupKFold(5)` outer splits.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 4.27 ± 0.12 (GroupKFold 5-fold outer CV); vs 06B_ensemble 4.40 ± 0.06
- **Gain:** −0.13 RMSE vs 06B (−3.0%); −0.17 vs 05 (−3.8%)
- **Winner of three-way comparison:** 07_patient_stack (best RMSE)
- **Hub URL:** https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/84317
- **Submission:** submissions/07_patient_stack.csv (11 013 rows)
- **Implication for next iteration:** The patient-context stacker yields a clear
  improvement over both 05 and 06B.  The std is wider (0.12 vs 0.06), which
  may reflect higher sensitivity to fold composition when the second stage
  relies on within-patient context.  Possible next steps: tune stage-2
  hyperparameters, increase the inner CV splits to tighten OOF quality, or
  investigate whether additional patient-level context features (e.g. visit
  trend over time, rate of change) further help stage 2.
