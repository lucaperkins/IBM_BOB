# %% [markdown]
# # Audit — 03_hgbr_tabular: skrub tabular_pipeline, all features + time_since_diagnosis
#
# Read-only review of the stored CrossValidationReport: checks and metrics.

# %%
import skore
from skore import login

from parkipredict.hub import load_skore_credentials

# %% [markdown]
# ## Open the project

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
#   https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/66676
# Type segment `cross-validations` → id type `cross-validation`.

# %%
REPORT_ID = "skore:report:cross-validation:66676"

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
