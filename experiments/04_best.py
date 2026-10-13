# %% [markdown]
# # Experiment 04_best: temporal neighbours + per-patient aggregates + tuned HGBR
#
# **Date:** 2025-10-07
# **Goal:** Beat 03_hgbr_tabular (RMSE 7.45) by adding within-patient temporal
# context (previous/next visit values, deltas, age gaps, visit number),
# per-patient off aggregates (mean, last, nearest observed), and
# missingness flags, combined with RandomizedSearchCV-tuned HGBR hyperparameters
# using patient-grouped inner CV to prevent leakage.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

import pandas as pd
from scipy.stats import loguniform
from scipy.stats import randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

import skrub

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import (
    BEST_DROP_COLS,
    HGBR_DROP_COLS,
    add_best_features,
    add_time_since_diagnosis,
    build_best_learner,
    build_hgbr_tabular_learner,
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
# X_best keeps patient_id so that add_best_features can sort visits per patient
# and compute within-patient temporal features.  patient_id is dropped inside
# the pipeline (drop_patient_id step) before TableVectorizer sees it.

# %%
X_full, y = load_dataset()
groups = X_full["patient_id"]

# 04_best: keep patient_id for temporal features; drop only target
X_best = X_full.drop(columns=BEST_DROP_COLS, errors="ignore")

# 03 for comparison
X_hgbr = X_full.drop(columns=HGBR_DROP_COLS, errors="ignore")

# %% [markdown]
# ## Tune 04_best hyperparameters
#
# Apply feature engineering once on the full training set, vectorize once,
# then run RandomizedSearchCV (20 iterations, GroupKFold(5) inner CV).
# Doing this outside the final pipeline avoids repeating feature engineering
# 100 times (20 candidates x 5 inner folds).

# %%
X_best_featurized = add_time_since_diagnosis(add_best_features(X_best))
X_best_featurized = X_best_featurized.drop(columns=["patient_id"], errors="ignore")

tv = skrub.tabular_pipeline("regressor").named_steps["tablevectorizer"]
X_vec = tv.fit_transform(X_best_featurized)

param_dist = {
    "learning_rate": loguniform(0.01, 0.3),
    "max_leaf_nodes": randint(15, 255),
    "max_iter": randint(100, 500),
    "min_samples_leaf": randint(10, 100),
    "l2_regularization": loguniform(1e-6, 10.0),
}

inner_cv = GroupKFold(n_splits=5)
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=0),
    param_distributions=param_dist,
    n_iter=20,
    cv=inner_cv,
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=-1,
    random_state=0,
    verbose=1,
)
search.fit(X_vec, y, groups=groups)
best_params = search.best_estimator_.get_params()
tuned_keys = list(param_dist.keys())
hgbr_params_best = {k: best_params[k] for k in tuned_keys}
hgbr_params_best["random_state"] = 0
print("Best params:", hgbr_params_best)
print("Inner CV best RMSE:", -search.best_score_)

# %% [markdown]
# ## Evaluate
#
# Outer GroupKFold(5) over patient groups.

# %%
gkf = GroupKFold(n_splits=5)

splits_hgbr = list(gkf.split(X_hgbr, y, groups=groups))
splits_best = list(gkf.split(X_best, y, groups=groups))

report_hgbr = skore.CrossValidationReport(
    build_hgbr_tabular_learner(),
    X=X_hgbr,
    y=y,
    splitter=iter(splits_hgbr),
)

report_best = skore.CrossValidationReport(
    build_best_learner(hgbr_params=hgbr_params_best),
    X=X_best,
    y=y,
    splitter=iter(splits_best),
)

comparison = skore.compare(
    {
        "03_hgbr_tabular": report_hgbr,
        "04_best": report_best,
    }
)
comparison

# %% [markdown]
# ## Persist
#
# Push the 04_best CV report to the skore hub under key 04_best.

# %%
try:
    project.put("04_best", report_best)
    print("Pushed 04_best to hub.")
except Exception as exc:
    print(f"Hub push failed (non-fatal): {exc}")

# %% [markdown]
# ## Submit
#
# Refit on all training visits, predict test visits, write CSV.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test_best = X_test_full.drop(columns=BEST_DROP_COLS, errors="ignore")

final = build_best_learner(hgbr_params=hgbr_params_best)
final.fit(X_best, y)
predictions = final.predict(X_test_best)

submission = pd.DataFrame({"Index": X_test_full.index, "target": predictions})
submission.to_csv(submissions_dir / "04_best.csv", index=False)
print(f"Submission written to submissions/04_best.csv ({len(submission)} rows)")
