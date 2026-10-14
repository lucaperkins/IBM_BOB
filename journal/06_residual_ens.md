# Experiment 06_residual_ens

## Question / hypothesis

Do two ensemble/residual strategies improve on 05_pk_rater (RMSE 4.43 ± 0.05)?

- **Variant A (residual):** train HGBR on `target − off_proxy`, predict residual
  and add back `off_proxy` at inference.  Hypothesis: removing the dominant
  signal `off` from the target reduces variance and helps the HGBR focus on the
  correction term.
- **Variant B (ensemble):** average the predictions of 5 independent HGBR models
  that share all hyperparameters but use `random_state` 0–4.  Hypothesis: the
  prediction variance from stochastic split/init in HGBR is non-negligible, and
  simple averaging reduces it without extra data.

## Motivation

05_pk_rater showed `off` is the top feature (permutation importance). Using it
as an additive proxy removes most of the target signal before the HGBR has to
learn it, leaving only the hard residual to predict. Variant B tests whether
the HGBR's stochastic components introduce enough variance that averaging five
seeds gives a free gain (cost: 5× fit time).

## Method

Both variants reuse the full `add_pk_rater_features` + `_drop_patient_id_05`
feature pipeline from 05.  `patient_id` is explicitly verified absent from the
HGBR input (assertion in experiment script).

**Variant A:**
1. Compute `off_proxy`: `off` where available, then `traj_off_pred`, then
   `nearest_off`, then global mean (train-set mean stored for leakage-free
   prediction).
2. Fit HGBR on `y − off_proxy` (residual).
3. Predict: residual prediction + `off_proxy`.

**Variant B:**
- 5 full pipelines (feature step → TableVectorizer → HGBR), seeds 0–4.
- Final prediction = mean of the five predictions.

Baseline: `05_pk_rater` rerun on the same `GroupKFold(5)` splits.
HGBR hyperparameters: `HGBR_PARAMS_04` (unchanged from 04/05).

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Results (GroupKFold 5-fold outer CV):**
  - 05_pk_rater (baseline): RMSE **4.44 ± 0.06**
  - Variant A (residual): RMSE **5.91 ± 1.29** ← worse (high variance per fold)
  - Variant B (ensemble 5×): RMSE **4.40 ± 0.06** ← winner
- **Winner:** 06B_ensemble (RMSE 4.40 vs 4.44 baseline, −0.04, −0.8%)
- **Hub URL:** https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/76897
- **Submission:** submissions/06_residual_ens.csv (11 013 rows)
- **Implication for next iteration:** Variant A fails because `off_proxy` built
  from within-fold features leaks residual structure that differs across folds,
  inflating variance.  Variant B shows a small consistent gain from seed
  averaging.  The margin is narrow; a richer ensemble (different feature subsets
  or stacking) or a full hyperparameter retune on 05's feature set is the most
  promising next step.
