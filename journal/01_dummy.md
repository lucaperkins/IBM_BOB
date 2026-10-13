# 01_dummy

<!-- Design note for experiments/01_dummy.py -->

## Question / hypothesis

Does a constant mean-prediction baseline give a useful lower bound
on RMSE for the true-OFF UPDRS target?

## Motivation

- **Sourcing strategy:** user (direct request this session)
- **Source(s):**
  - User asked: "Run a dummy regressor from sklearn to act as my
    baseline model. Push to the hub with the name '01_dummy' as the
    report name to the project 'bobathon-esilv'."
- **Why this matters:** Before evaluating any real model we need a
  floor — a number that a trivially correct answer (always predict
  the mean) achieves. Any model whose RMSE is worse than the dummy
  is not worth deploying. The lab guide explicitly requires this
  as the first experiment.

## Method

- **Files touched:** `src/parkipredict/data.py`,
  `src/parkipredict/pipeline.py`, `src/parkipredict/evaluate.py`,
  `experiments/01_dummy.py`
- **Change versus baseline (or previous experiment):** This is the
  first experiment. The learner is `sklearn.dummy.DummyRegressor`
  with `strategy="mean"` — it ignores features and always predicts
  the mean of the training labels. No feature engineering.
- **Cross-validation:** `splitter=0.2` (a random 80/20 row holdout)
  per the lab guide for the first two experiments. The patient-group
  leakage is intentional at this stage — that is the point the guide
  makes. Grouped CV is introduced in a later experiment.
- **Out of scope for this experiment:** feature engineering, any
  real estimator, grouped cross-validation.

## Risks / things that could invalidate the result

- The 80/20 split is at the visit level, so the same patient may
  appear in both train and test. The RMSE will be slightly optimistic
  compared to a patient-level holdout. This is expected and
  intentional at this stage.
- `DummyRegressor(strategy="mean")` always predicts the same value
  for every row; on the test split it is measuring how spread out
  the target is, not how well any features predict it.

## Status

- **State:** done
- **Approved by user on:** 2025-10-07
- **Headline result:** RMSE 16.48, MAE 13.48, R² ≈ 0 (80/20 visit holdout). HGB baseline (skore check SKD009) achieves RMSE 7.16, MAE 5.63, R² 0.811 — that is the real target to beat.
- **Implication for next iteration:** The dummy floor (RMSE ≈ 16.5) is now set. The `on` / `off` features (Pearson r = 0.87 / 0.67 with target from EDA) are the obvious next lever; a Ridge or HistGradientBoosting with those features should approach the skore HGB baseline (RMSE ≈ 7).
