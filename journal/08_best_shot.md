# Experiment 08_best_shot

Retune on the 05_pk_rater feature set extended with rater-bias-adjusted
columns (off/on/ledd), age-weighted interpolation between observed visits,
local slope, age distance to nearest observed off, and extended neighbour
shifts — all computed on stacked train+test X (no target).
GroupKFold(5) outer, GroupKFold(3) inner for RandomizedSearchCV (15 iter).
Compare 05_pk_rater, reference arm, and 5-seed tuned ensemble in one report.
Best model fit on all training visits; submission to submissions/08_best_shot.csv.

- **State:** pending
