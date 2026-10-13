# IBM_BOB: Parkinson's true-OFF score prediction

Team project for the ESILV Machine Learning "bobathon" ([Kaggle: bobathon-esilv](https://www.kaggle.com/competitions/bobathon-esilv/)). For each patient visit, we predict the unbiased ("true") OFF MDS-UPDRS motor score (`target`). The metric is RMSE, and lower is better. Test patients never appear in train, so we validate with patient-grouped CV.

- **Team:** <names of the 4 teammates>
- **Kaggle team:** <exact team name>
- **Skore Hub workspace:** `parkipredict`, project `bobathon-esilv`
- **Background and data description:** [docs/CONTEXT.md](docs/CONTEXT.md)
- **Original lab guides:** [SETUP](docs/SETUP.md), [DAY1](docs/DAY1.md), [DAY2](docs/DAY2.md)

## Current status

All scores below are RMSE, and every experiment is tracked in [journal/JOURNAL.md](journal/JOURNAL.md).

| Key | Model | Validation | RMSE | Kaggle public | Hub report |
|---|---|---|---|---|---|
| `01_dummy` | Mean predictor | 80/20 random visit split | 16.48 | <score> | <url> |
| `02_ridge` | Ridge, median imputation, 8 numeric features | GroupKFold(5) on `patient_id` | 10.60 ± 0.27 | <score> | [report](https://skore.probabl.ai/parkipredict/bobathon-esilv/cross-validations/66415) |
| `03_...` | | | | | |

**Next:** HistGradientBoosting / skrub `tabular_pipeline` with categorical columns and `time_since_diagnosis = age - age_at_diagnosis`.

## Getting started (each teammate, on their own machine)

1. Accept the GitHub invite, then clone:
```bash
   git clone https://github.com/lucaperkins/IBM_BOB.git
   cd IBM_BOB
```
2. Join the Skore Hub workspace `parkipredict` (you need an invite) and join our Kaggle team.
3. Follow [docs/SETUP.md](docs/SETUP.md) step 5 to create the venv, install requirements, install the Bob skills, sign in to Skore, and run the checks. Then install the project package:
```bash
   pip install -e .
   python -m pytest   # must end with "all passed"
```
   `.skore` (your personal API key), `.venv/` and `.bob/skills/` are per machine and are not in git.
4. In Bob, tick **Skill** in the Permissions menu for each new chat.

## Workflow

- **Before you start:** `git pull`, then read `journal/JOURNAL.md` to see what has been tried.
- **Run an experiment:** prompt Bob. All code must be written and run in Bob with evaluation through skore.
- **Use patient-grouped CV** (`GroupKFold` on `patient_id`) for every model, never a random row split.
- **When Bob finishes:** make sure `journal/JOURNAL.md` is updated, then run:
```bash
  git add -A
  git commit -m "<key>: <one-line result>"
  git push
```
- **Submit to Kaggle:** upload `submissions/<key>.csv` (header `Index,target`, one row per test visit) with the Skore Hub report URL in the Submission Description. `submissions/` is gitignored, so regenerate the CSV from `experiments/<key>.py` if you need it.

## Rules

- Each Kaggle submission needs a **new** Hub report key. Never reuse a key, and never use the reserved key `eda`.
- Claim the next key below before you start, so two people don't take the same number.

| Key | Owner | Idea |
|---|---|---|
| `03` | <name> | <idea> |
| `04` | <name> | <idea> |

## Repository layout

| Path | What it is |
|---|---|
| `data/` | Competition CSVs and the EDA outputs |
| `experiments/` | One script per experiment (`01_dummy.py`, ...) |
| `journal/` | `JOURNAL.md` index and one design note per experiment |
| `src/parkipredict/` | Shared project code |
| `audit/`, `tests/` | Pipeline audits and setup checks |
| `docs/` | Original lab guides and background |
