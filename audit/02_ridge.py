# %% [markdown]
# # Audit — 02_ridge: Ridge + median imputation, patient-grouped 5-fold CV
#
# Read-only review of the stored CrossValidationReport: its checks and metrics.

# %%
import skore
from skore import login

from parkipredict.hub import load_skore_credentials

# %% [markdown]
# ## Open the project
#
# Same Project the experiment wrote to (hub mode, project bobathon-esilv).

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL from put() stdout:
#   https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/66415
# Type segment is `cross-validations` (plural) → id type is `cross-validation`.

# %%
REPORT_ID = "skore:report:cross-validation:66415"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
