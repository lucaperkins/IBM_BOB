"""CV strategies for skore.evaluate.

``splitter`` — 80/20 random visit-level holdout used by 01_dummy.
``splitter_grouped`` — GroupKFold(5) on patient_id, introduced in
02_ridge to eliminate same-patient leakage across folds.

Metric overrides are absent; skore picks task-appropriate defaults
(RMSE, MAE, R² for regression).
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# 80/20 random visit-level holdout: 80% train, 20% test.
# Same-patient rows can appear on both sides of the split — that is
# intentional for 01_dummy (the point is to show the naive result).
splitter: float = 0.2

# Patient-grouped 5-fold CV: no same-patient leakage across folds.
# Pass groups=X["patient_id"] to skore.evaluate when using this splitter.
splitter_grouped = GroupKFold(n_splits=5)
