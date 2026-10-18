# Experiment 08_best_shot

Retune on the 05_pk_rater feature set extended with rater-bias-adjusted
columns (off/on/ledd), age-weighted interpolation between observed visits,
local slope, age distance to nearest observed off, and extended neighbour
shifts — all computed on stacked train+test X (no target).
GroupKFold(5) outer, GroupKFold(3) inner for RandomizedSearchCV (15 iter).
Compare 05_pk_rater, reference arm, and 5-seed tuned ensemble in one report.
Best model fit on all training visits; submission to submissions/08_best_shot.csv.

- **State:** done
- **Approved by user on:** <2025-14-07>
- **Headline result:** RMSE 4.10 (GroupKFold 5-fold outer CV); vs 07_patient_stack 4.27 ± 0.12. Kaggle public score 3.937 (rank 188 of 294)
- **Gain:** −0.17 RMSE vs 07 (−4.0%); −0.30 vs 06B (−6.8%); −0.33 vs 05 (−7.4%)
- **Winner of three-way comparison:** <08_tuned_ensemble or 08_ref_params> (best RMSE)
- **Hub URL:** <https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/86194>
- **Submission:** submissions/08_best_shot.csv (11 013 rows)
- **Implication for next iteration:** The rater-bias-adjusted features, age-weighted
  interpolation and tuned 5-seed ensemble give the best result of the project, and
  the Kaggle score (3.937) is lower than the cross-validation RMSE, which suggests the
  patient-grouped CV is a reliable estimate. The best teams on the leaderboard are
  around 2, so a part of the bias in the data is still not captured. Possible next
  steps: analyse the largest errors per rater, cohort and missing-off pattern,
  tune the stacking stage of 07 on top of the 08 features, or model each cohort
  separately.
