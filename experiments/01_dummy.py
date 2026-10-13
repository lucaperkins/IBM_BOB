# %% [markdown]
# # Experiment 01_dummy: DummyRegressor mean baseline
#
# **Date:** 2025-10-07
# **Goal:** Establish the RMSE floor — always predicting the
# training-set mean. Every real model must beat this score.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.evaluate import splitter
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import build_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project
#
# Open the hub project. Credentials come from the gitignored `.skore`
# file written by `python scripts/install_skore.py`. The report key
# is the file stem (`01_dummy`).

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner

# %%
X, y = load_dataset()
learner = build_learner()

# %% [markdown]
# ## Evaluate
#
# `splitter=0.2` is an 80/20 random visit-level holdout.
# skore picks regression metrics automatically (RMSE, MAE, R²).

# %%
report = skore.evaluate(learner, X, y, splitter=splitter)
report

# %% [markdown]
# ## Persist
#
# Key = file stem `01_dummy`.

# %%
project.put("01_dummy", report)

# %% [markdown]
# ## Submit
#
# Refit on all training rows, predict the test visits, write the
# Kaggle submission to `submissions/01_dummy.csv`.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset()

final = build_learner()
final.fit(X, y)
predictions = final.predict(X_test)

import pandas as pd  # noqa: E402

submission = pd.DataFrame({"Index": X_test.index, "target": predictions})
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
print(f"Submission written to submissions/01_dummy.csv ({len(submission)} rows)")
