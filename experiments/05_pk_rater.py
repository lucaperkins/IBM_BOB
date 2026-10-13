# %% [markdown]
# # Experiment 05_pk_rater: PK decay + rater corrector + trajectory features
#
# **Date:** 2025-10-07
# **Goal:** Extend 04_best with three X-only feature blocks —
# pharmacokinetic decay curves, per-rater bias correctors, and within-patient
# linear UPDRS trajectories — keeping the same tuned HGBR hyperparameters.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import (
    BEST_DROP_COLS,
    HGBR_DROP_COLS,
    build_best_learner,
    build_hgbr_tabular_learner,
    build_pk_rater_learner,
    HGBR_PARAMS_04,
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
# Same split strategy as 04_best: keep patient_id in X for within-patient
# feature engineering; drop only target.

# %%
X_full, y = load_dataset()
groups = X_full["patient_id"]

X_pk = X_full.drop(columns=BEST_DROP_COLS, errors="ignore")   # patient_id kept
X_best = X_full.drop(columns=BEST_DROP_COLS, errors="ignore")  # 04 baseline
X_hgbr = X_full.drop(columns=HGBR_DROP_COLS, errors="ignore")  # 03 comparison

# %% [markdown]
# ## Evaluate
#
# Outer GroupKFold(5) over patient groups — same folds as 04_best.

# %%
gkf = GroupKFold(n_splits=5)

splits_hgbr = list(gkf.split(X_hgbr, y, groups=groups))
splits_best = list(gkf.split(X_best, y, groups=groups))
splits_pk = list(gkf.split(X_pk, y, groups=groups))

report_hgbr = skore.CrossValidationReport(
    build_hgbr_tabular_learner(),
    X=X_hgbr,
    y=y,
    splitter=iter(splits_hgbr),
)

report_best = skore.CrossValidationReport(
    build_best_learner(hgbr_params=HGBR_PARAMS_04),
    X=X_best,
    y=y,
    splitter=iter(splits_best),
)

report_pk = skore.CrossValidationReport(
    build_pk_rater_learner(),
    X=X_pk,
    y=y,
    splitter=iter(splits_pk),
)

comparison = skore.compare(
    {
        "03_hgbr_tabular": report_hgbr,
        "04_best": report_best,
        "05_pk_rater": report_pk,
    }
)
comparison

# %% [markdown]
# ## Permutation importance
#
# Fit once on the full training set, compute permutation importance
# on a 20% hold-out (first fold test set used as a proxy).

# %%
train_idx, test_idx = splits_pk[0]
learner_for_imp = build_pk_rater_learner()
learner_for_imp.fit(X_pk.iloc[train_idx], y.iloc[train_idx])

result_imp = permutation_importance(
    learner_for_imp,
    X_pk.iloc[test_idx],
    y.iloc[test_idx],
    n_repeats=10,
    random_state=0,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)
imp_df = pd.DataFrame(
    {
        "feature": X_pk.columns.tolist(),
        "importance_mean": result_imp.importances_mean,
        "importance_std": result_imp.importances_std,
    }
).sort_values("importance_mean", ascending=False)
print("Top 10 permutation importance:")
print(imp_df.head(10).to_string(index=False))

# %% [markdown]
# ## Persist
#
# Push the 05_pk_rater CV report to the skore hub under key 05_pk_rater.

# %%
try:
    project.put("05_pk_rater", report_pk)
    print("Pushed 05_pk_rater to hub.")
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
X_test_pk = X_test_full.drop(columns=BEST_DROP_COLS, errors="ignore")

final = build_pk_rater_learner()
final.fit(X_pk, y)
predictions = final.predict(X_test_pk)

submission = pd.DataFrame({"Index": X_test_full.index, "target": predictions})
submission.to_csv(submissions_dir / "05_pk_rater.csv", index=False)
print(f"Submission written to submissions/05_pk_rater.csv ({len(submission)} rows)")
