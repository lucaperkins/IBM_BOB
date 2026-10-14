# %% [markdown]
# # Experiment 08_best_shot
#
# **Date:** 2025-10-08
# **Goal:** Best-shot model on the 05_pk_rater feature set extended with
# rater-bias-adjusted columns and age-weighted interpolation.
# RandomizedSearchCV retune + 5-seed ensemble.

# %%
import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import loguniform, randint

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import build_pk_rater_learner, BEST_DROP_COLS
from parkipredict.best_shot import (
    add_best_shot_features,
    build_best_shot_learner,
    build_best_shot_ensemble,
    HGBR_PARAMS_REF,
)

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
# Load train + test X.  Compute features on the full stack (no target involved).
# Restore original index order after feature computation.

# %%
DATA_DIR = PROJECT_ROOT / "data"

X_train_raw, y = load_dataset()
X_test_raw = load_test_dataset()

groups = X_train_raw["patient_id"]  # for GroupKFold

# Stack train and test for feature computation (no target used)
n_train = len(X_train_raw)
X_stack = pd.concat([X_train_raw, X_test_raw], axis=0)
X_stack_feat = add_best_shot_features(X_stack)

# Split back; both retain the original index
X_train_feat = X_stack_feat.iloc[:n_train].copy()
X_test_feat = X_stack_feat.iloc[n_train:].copy()

# patient_id is kept in X_train_feat for GroupKFold grouping reference,
# but it is NEVER a model input — the pipeline drops it before TV.
assert "patient_id" in X_train_feat.columns  # still present for grouping
assert "target" not in X_train_feat.columns

# %% [markdown]
# ## Outer CV splits (GroupKFold 5, same patient grouping as 05 and 06)

# %%
gkf_outer = GroupKFold(n_splits=5)
splits_outer = list(gkf_outer.split(X_train_feat, y, groups=groups))

# %% [markdown]
# ## 05_pk_rater baseline on the same outer folds
#
# Rebuild the 05 feature set (train-only, as done in 05) and evaluate with
# the same outer splits for a fair RMSE comparison.

# %%
X_05 = X_train_raw.drop(columns=BEST_DROP_COLS, errors="ignore")  # patient_id kept
splits_05 = list(gkf_outer.split(X_05, y, groups=groups))

report_05 = skore.CrossValidationReport(
    build_pk_rater_learner(),
    X=X_05,
    y=y,
    splitter=iter(splits_05),
)

# %% [markdown]
# ## Reference arm: fixed hyperparameters (HGBR_PARAMS_REF)

# %%
report_ref = skore.CrossValidationReport(
    build_best_shot_learner(hgbr_params=HGBR_PARAMS_REF),
    X=X_train_feat,
    y=y,
    splitter=iter(splits_outer),
)

# %% [markdown]
# ## Hyperparameter tuning
#
# RandomizedSearchCV with 15 iterations, inner GroupKFold(3), n_jobs=-1.
# Operates on pre-featurized X_train_feat with the same pipeline structure.

# %%
param_dist = {
    "histgradientboostingregressor__learning_rate": loguniform(1e-3, 0.3),
    "histgradientboostingregressor__max_leaf_nodes": randint(8, 128),
    "histgradientboostingregressor__max_iter": randint(200, 800),
    "histgradientboostingregressor__min_samples_leaf": randint(10, 100),
    "histgradientboostingregressor__l2_regularization": loguniform(1e-5, 10.0),
}

gkf_inner = GroupKFold(n_splits=3)
# For inner CV we need the train portion only; groups aligns with X_train_feat
search = RandomizedSearchCV(
    estimator=build_best_shot_learner(),
    param_distributions=param_dist,
    n_iter=15,
    cv=list(gkf_inner.split(X_train_feat, y, groups=groups)),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    refit=False,
)
search.fit(X_train_feat, y)

# Extract best params
best_idx = search.cv_results_["mean_test_score"].argmax()
best_hgbr_params = {
    k.replace("histgradientboostingregressor__", ""): v
    for k, v in search.cv_results_["params"][best_idx].items()
}
print(f"Best HGBR params from RandomizedSearchCV: {best_hgbr_params}")

# %% [markdown]
# ## Tuned 5-seed ensemble on the outer folds

# %%
ensemble = build_best_shot_ensemble(hgbr_params=best_hgbr_params, n_members=5)

report_ens = skore.CrossValidationReport(
    ensemble,
    X=X_train_feat,
    y=y,
    splitter=iter(splits_outer),
)

# %% [markdown]
# ## Compare all three on the same outer folds

# %%
comparison = skore.compare(
    {
        "05_pk_rater": report_05,
        "08_ref_params": report_ref,
        "08_tuned_ensemble": report_ens,
    }
)
comparison

# %% [markdown]
# ## Print RMSE values

# %%
def _rmse_summary(report, label: str) -> None:
    """Print grouped-CV RMSE mean ± std for a CrossValidationReport."""
    df = report.metrics.rmse()
    # df has MultiIndex columns: (estimator_name, "mean") and (estimator_name, "std")
    # Flatten and grab the first estimator's mean/std
    vals = df.values.ravel()
    mean_rmse = float(vals[0])
    std_rmse = float(vals[1]) if len(vals) > 1 else float("nan")
    print(f"{label}: RMSE {mean_rmse:.4f} ± {std_rmse:.4f}")


_rmse_summary(report_05, "05_pk_rater")
_rmse_summary(report_ref, "08_ref_params")
_rmse_summary(report_ens, "08_tuned_ensemble")

# %% [markdown]
# ## Push reports to hub
#
# skore.Project.put accepts CrossValidationReport (not ComparisonReport),
# so we push each arm individually under a key prefixed '08_best_shot'.

# %%
for key, rpt in [
    ("08_best_shot", report_ens),          # main result: tuned 5-seed ensemble
    ("08_best_shot_ref", report_ref),
    ("08_best_shot_05_baseline", report_05),
]:
    try:
        project.put(key, rpt)
        print(f"Pushed {key!r} to hub.")
    except Exception as exc:
        print(f"Hub push failed for {key!r} (non-fatal): {exc}")

hub_url = cfg.get("hub_url", "").rstrip("/")
workspace = cfg.get("workspace", "")
if hub_url and workspace:
    print(f"Hub URL: {hub_url}/{workspace}/cross-validations/")

# %% [markdown]
# ## Choose best model and fit on all training data
#
# Pick the model with the better mean outer-CV RMSE between reference and ensemble.

# %%
def _mean_rmse(report) -> float:
    df = report.metrics.rmse()
    return float(df.values.ravel()[0])


rmse_ref = _mean_rmse(report_ref)
rmse_ens = _mean_rmse(report_ens)
print(f"Mean RMSE — ref: {rmse_ref:.4f}, ensemble: {rmse_ens:.4f}")

if rmse_ens <= rmse_ref:
    best_model = build_best_shot_ensemble(hgbr_params=best_hgbr_params, n_members=5)
    print("Best model: 5-seed tuned ensemble")
else:
    best_model = build_best_shot_learner(hgbr_params=HGBR_PARAMS_REF)
    print("Best model: reference parameters")

best_model.fit(X_train_feat, y)

# %% [markdown]
# ## Write submission
#
# X_test_feat has the same index as X_test_raw (i.e. sample_submission.csv order).
# Predictions are produced in that order.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

# Load sample_submission to ensure row order matches exactly
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")
assert list(X_test_feat.index) == list(sample_sub.index), (
    "Test index does not match sample_submission.csv"
)

predictions = best_model.predict(X_test_feat)
submission = pd.DataFrame({"Index": X_test_feat.index, "target": predictions})
submission.to_csv(submissions_dir / "08_best_shot.csv", index=False)
print(
    f"Submission written to submissions/08_best_shot.csv ({len(submission)} rows)"
)
