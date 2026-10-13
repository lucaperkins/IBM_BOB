# %% [markdown]
# # Experiment 03_hgbr_tabular: skrub tabular_pipeline, all features + time_since_diagnosis
#
# **Date:** 2025-10-07
# **Goal:** Does a skrub tabular_pipeline("regressor") on all features
# (including the derived time_since_diagnosis) under patient-grouped
# 5-fold CV beat the Ridge baseline? How close to skore's HGB benchmark?
# **Result:** filled in after the run.

# %%
import skore
from skore import login

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import (
    HGBR_DROP_COLS,
    NUMERIC_COLS,
    build_hgbr_tabular_learner,
    build_learner,
    build_ridge_learner,
)

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
# X for the HGBR pipeline: all columns except patient_id and target.
# patient_id is retained separately for GroupKFold groups.
# X_ridge uses only the 8 numeric cols (as in 02_ridge).
# X_dummy matches X_ridge (DummyRegressor ignores features anyway).

# %%
X_full, y = load_dataset()
groups = X_full["patient_id"]

# HGBR: drop patient_id (used as group key only) and target
X_hgbr = X_full.drop(columns=HGBR_DROP_COLS, errors="ignore")

# Ridge and dummy: numeric columns only (as in 02_ridge)
X_ridge = X_full[NUMERIC_COLS]

# %% [markdown]
# ## Evaluate
#
# All three models are evaluated on the same GroupKFold(5) patient folds.
# Each gets its own pre-generated split list so the iterator is not
# exhausted after the first model.

# %%
gkf = GroupKFold(n_splits=5)

splits_dummy = list(gkf.split(X_ridge, y, groups=groups))
splits_ridge = list(gkf.split(X_ridge, y, groups=groups))
splits_hgbr  = list(gkf.split(X_hgbr, y, groups=groups))

report_dummy = skore.CrossValidationReport(
    build_learner(),
    X=X_ridge,
    y=y,
    splitter=iter(splits_dummy),
)
report_ridge = skore.CrossValidationReport(
    build_ridge_learner(),
    X=X_ridge,
    y=y,
    splitter=iter(splits_ridge),
)
report_hgbr = skore.CrossValidationReport(
    build_hgbr_tabular_learner(),
    X=X_hgbr,
    y=y,
    splitter=iter(splits_hgbr),
)

comparison = skore.compare(
    {"01_dummy": report_dummy, "02_ridge": report_ridge, "03_hgbr_tabular": report_hgbr}
)
comparison

# %% [markdown]
# ## Persist
#
# project.put accepts EstimatorReport or CrossValidationReport.
# Push the HGBR CV report under key 03_hgbr_tabular.

# %%
project.put("03_hgbr_tabular", report_hgbr)

# %% [markdown]
# ## Submit
#
# Refit the HGBR on all training visits, predict test visits, write CSV.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test_hgbr = X_test_full.drop(columns=HGBR_DROP_COLS, errors="ignore")

final = build_hgbr_tabular_learner()
final.fit(X_hgbr, y)
predictions = final.predict(X_test_hgbr)

submission = pd.DataFrame({"Index": X_test_full.index, "target": predictions})
submission.to_csv(submissions_dir / "03_hgbr_tabular.csv", index=False)
print(f"Submission written to submissions/03_hgbr_tabular.csv ({len(submission)} rows)")
