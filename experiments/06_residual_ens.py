# %% [markdown]
# # Experiment 06_residual_ens: residual target + HGBR ensemble
#
# **Date:** 2025-10-07
# **Goal:** Test two variants on top of 05_pk_rater features:
#   Variant A — HGBR predicts (target − off_proxy), adds back off_proxy.
#   Variant B — average of 5 HGBR with different random seeds.
# Keep the best variant; same GroupKFold(5) outer CV as 05.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset, load_test_dataset
from parkipredict.hub import load_skore_credentials
from parkipredict.pipeline import (
    BEST_DROP_COLS,
    HGBR_PARAMS_04,
    build_ensemble_learner,
    build_pk_rater_learner,
    build_residual_learner,
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
# patient_id is kept in X so feature functions can compute within-patient
# aggregates; it is dropped inside each learner before TableVectorizer sees it.

# %%
X_full, y = load_dataset()
groups = X_full["patient_id"]

# All three variants use the same X (patient_id kept, target dropped).
X = X_full.drop(columns=BEST_DROP_COLS, errors="ignore")

# %% [markdown]
# ## Verify patient_id is not seen by the HGBR
#
# Confirm that after feature engineering and the drop step, patient_id is absent.

# %%
from parkipredict.pipeline import _add_all_features_05, _drop_patient_id_05  # noqa: E402

_X_check = _drop_patient_id_05(_add_all_features_05(X.head(10)))
assert "patient_id" not in _X_check.columns, "patient_id leaked into HGBR input!"
print("patient_id correctly absent from HGBR input columns.")

# %% [markdown]
# ## Evaluate all three variants
#
# Outer GroupKFold(5) over patient groups — same folds as 05_pk_rater.

# %%
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups=groups))

print("Running 05_pk_rater report...")
report_05 = skore.CrossValidationReport(
    build_pk_rater_learner(),
    X=X,
    y=y,
    splitter=iter(splits),
)

print("Running Variant A (residual) report...")
report_A = skore.CrossValidationReport(
    build_residual_learner(),
    X=X,
    y=y,
    splitter=iter(splits),
)

print("Running Variant B (ensemble) report...")
report_B = skore.CrossValidationReport(
    build_ensemble_learner(),
    X=X,
    y=y,
    splitter=iter(splits),
)

# %% [markdown]
# ## Compare

# %%
rmse_05 = report_05.metrics.rmse(aggregate=["mean", "std"])
rmse_A  = report_A.metrics.rmse(aggregate=["mean", "std"])
rmse_B  = report_B.metrics.rmse(aggregate=["mean", "std"])

print("05_pk_rater RMSE:\n", rmse_05)
print("Variant A (residual) RMSE:\n", rmse_A)
print("Variant B (ensemble) RMSE:\n", rmse_B)

comparison = skore.compare(
    {
        "05_pk_rater": report_05,
        "06A_residual": report_A,
        "06B_ensemble": report_B,
    }
)
comparison

# %% [markdown]
# ## Select winner and persist to hub

# %%
# rmse_XX is a DataFrame with MultiIndex columns (Estimator, Aggregate).
# Extract "mean" column from any estimator name position.
def _rmse_mean(df: "pd.DataFrame") -> float:
    col = [c for c in df.columns if c[1] == "mean"][0]
    return float(df.iloc[0][col])

rmse_05_val = _rmse_mean(rmse_05)
rmse_A_val  = _rmse_mean(rmse_A)
rmse_B_val  = _rmse_mean(rmse_B)

best_rmse = min(rmse_05_val, rmse_A_val, rmse_B_val)

if best_rmse == rmse_B_val:
    winner_name = "06B_ensemble"
    winner_report = report_B
    winner_builder = build_ensemble_learner
elif best_rmse == rmse_A_val:
    winner_name = "06A_residual"
    winner_report = report_A
    winner_builder = build_residual_learner
else:
    winner_name = "05_pk_rater (unchanged)"
    winner_report = report_05
    winner_builder = build_pk_rater_learner

print(f"Winner: {winner_name}  RMSE={best_rmse:.4f}")

try:
    project.put("06_residual_ens", winner_report)
    print("Pushed 06_residual_ens to hub.")
except Exception as exc:
    print(f"Hub push failed (non-fatal): {exc}")

# %% [markdown]
# ## Submit
#
# Refit winner on all training visits, predict test visits, write CSV.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test = X_test_full.drop(columns=BEST_DROP_COLS, errors="ignore")

final = winner_builder()
final.fit(X, y)
predictions = final.predict(X_test)

submission = pd.DataFrame({"Index": X_test_full.index, "target": predictions})
submission.to_csv(submissions_dir / "06_residual_ens.csv", index=False)
print(f"Submission written to submissions/06_residual_ens.csv ({len(submission)} rows)")
