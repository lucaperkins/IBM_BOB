# %% [markdown]
# # Experiment 02_ridge: Ridge + median imputation, patient-grouped 5-fold CV
#
# **Date:** 2025-10-07
# **Goal:** Does Ridge with median imputation beat the dummy under
# patient-grouped 5-fold CV? Uses GroupKFold(n_splits=5) on patient_id
# so the same patient never appears in both train and test within a fold.
# A ComparisonReport is produced via skore.compare to see both models
# side-by-side.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

import pandas as pd
from sklearn.model_selection import GroupKFold

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import NUMERIC_COLS, build_learner, build_ridge_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data
#
# Ridge cannot handle string columns, so we select only the numeric
# features defined in NUMERIC_COLS. The target is unchanged.

# %%
X_full, y = load_dataset()
X = X_full[NUMERIC_COLS]
groups = X_full["patient_id"]

# %% [markdown]
# ## Evaluate
#
# GroupKFold(n_splits=5): the same patient never appears in both train
# and test within a fold. We pre-generate the split indices once and
# reuse them for both models so the fold geometry is identical.
#
# Each model gets its own CrossValidationReport; skore.compare
# combines them into a single ComparisonReport.

# %%
gkf = GroupKFold(n_splits=5)
splits_dummy = list(gkf.split(X, y, groups=groups))
splits_ridge = list(gkf.split(X, y, groups=groups))

report_dummy = skore.CrossValidationReport(
    build_learner(),
    X=X,
    y=y,
    splitter=iter(splits_dummy),
)
report_ridge = skore.CrossValidationReport(
    build_ridge_learner(),
    X=X,
    y=y,
    splitter=iter(splits_ridge),
)

comparison = skore.compare({"01_dummy": report_dummy, "02_ridge": report_ridge})
comparison

# %% [markdown]
# ## Persist
#
# project.put only accepts EstimatorReport or CrossValidationReport.
# We persist the Ridge CV report under key `02_ridge`; the comparison
# is surfaced in-script above for review.

# %%
project.put("02_ridge", report_ridge)

# %% [markdown]
# ## Submit
#
# Refit on all training rows, predict on the test set.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test = X_test_full[NUMERIC_COLS]

final = build_ridge_learner()
final.fit(X, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"Index": X_test_full.index, "target": predictions})
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print(f"Submission written to submissions/02_ridge.csv ({len(submission)} rows)")
