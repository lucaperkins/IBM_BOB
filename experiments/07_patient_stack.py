# %% [markdown]
# # Experiment 07_patient_stack: two-stage patient-context stacker
#
# **Date:** 2025-10-07
# **Goal:** Add a second stage that corrects each visit's prediction using the
# stage-1 predictions of the other visits of the same patient.
# Compare against 05_pk_rater and 06B_ensemble on the same GroupKFold(5) splits.
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
    build_ensemble_learner,
    build_patient_stack_learner,
    build_pk_rater_learner,
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
# patient_id kept in X so within-patient feature engineering and grouping work.

# %%
X_full, y = load_dataset()
groups = X_full["patient_id"]

# All three learners receive the same X (target dropped, patient_id kept).
X = X_full.drop(columns=BEST_DROP_COLS, errors="ignore")

# %% [markdown]
# ## Evaluate
#
# Same GroupKFold(5) splits on patient_id as 05 and 06.

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

print("Running 06B_ensemble report...")
report_06B = skore.CrossValidationReport(
    build_ensemble_learner(),
    X=X,
    y=y,
    splitter=iter(splits),
)

print("Running 07_patient_stack report...")
report_07 = skore.CrossValidationReport(
    build_patient_stack_learner(),
    X=X,
    y=y,
    splitter=iter(splits),
)

# %% [markdown]
# ## Compare

# %%
rmse_05  = report_05.metrics.rmse(aggregate=["mean", "std"])
rmse_06B = report_06B.metrics.rmse(aggregate=["mean", "std"])
rmse_07  = report_07.metrics.rmse(aggregate=["mean", "std"])

print("05_pk_rater RMSE:\n", rmse_05)
print("06B_ensemble RMSE:\n", rmse_06B)
print("07_patient_stack RMSE:\n", rmse_07)

comparison = skore.compare(
    {
        "05_pk_rater":      report_05,
        "06B_ensemble":     report_06B,
        "07_patient_stack": report_07,
    }
)
comparison

# %% [markdown]
# ## Select winner and persist to hub

# %%
def _rmse_mean(df: pd.DataFrame) -> float:
    col = [c for c in df.columns if c[1] == "mean"][0]
    return float(df.iloc[0][col])

rmse_05_val  = _rmse_mean(rmse_05)
rmse_06B_val = _rmse_mean(rmse_06B)
rmse_07_val  = _rmse_mean(rmse_07)

print(f"RMSE summary — 05: {rmse_05_val:.4f}  06B: {rmse_06B_val:.4f}  07: {rmse_07_val:.4f}")

best_rmse = min(rmse_05_val, rmse_06B_val, rmse_07_val)
if best_rmse == rmse_07_val:
    winner_name = "07_patient_stack"
    winner_builder = build_patient_stack_learner
elif best_rmse == rmse_06B_val:
    winner_name = "06B_ensemble"
    winner_builder = build_ensemble_learner
else:
    winner_name = "05_pk_rater"
    winner_builder = build_pk_rater_learner

print(f"Winner: {winner_name}  RMSE={best_rmse:.4f}")

try:
    project.put("07_patient_stack", report_07)
    print("Pushed 07_patient_stack to hub.")
except Exception as exc:
    print(f"Hub push failed (non-fatal): {exc}")

# %% [markdown]
# ## Retrieve hub URL

# %%
try:
    run_id = report_07.estimator_reports_[0].run_metadata["run_id"]
    hub_url = (
        f"https://skore.probabl.ai/{cfg['workspace']}/bobathon-esilv"
        f"/cross-validations/{run_id}"
    )
    print(f"Hub URL: {hub_url}")
except Exception:
    print("Hub URL not available from metadata — check the hub UI.")

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
submission.to_csv(submissions_dir / "07_patient_stack.csv", index=False)
print(f"Submission written to submissions/07_patient_stack.csv ({len(submission)} rows)")
