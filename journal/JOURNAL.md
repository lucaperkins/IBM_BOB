# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's Disease UPDRS score prediction (patient visit tabular data)
- **Goal:** Predict a patient's composite UPDRS score (`target`) from clinical visit features (age, medication, prior UPDRS sub-scores)
- **Last experiment:** 07_patient_stack - done
- **Last result:** RMSE 4.27 ± 0.12 (07_patient_stack, GroupKFold 5-fold); vs 06B_ensemble 4.40 ± 0.06

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-10-07
  - env manager: pip+venv - recorded: 2025-10-07
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: none - recorded: 2025-10-07
  - package name (`src/<pkg>/`): parkipredict - recorded: 2025-10-07
  - skore mode: hub - recorded: 2025-10-07
  - skore hub workspace: parkipredict - recorded: 2025-10-07
  - skore mlflow tracking uri: n/a - recorded: 2025-10-07
  - student prior: some-sklearn - recorded: 2025-10-07
  - CV splitter family: GroupKFold (for patient-grouped experiments; splitter=0.2 for 01_dummy/02_ridge) - recorded: 2025-10-07

## Data understanding (EDA)

- **Status:** done - 2025-10-07
- **Summary:** 44,590 rows × 13 columns (12 features + `target`). 5,576 unique patients with multiple visits each — `GroupKFold(patient_id)` is required for CV. `off` and `on` (UPDRS sub-scores) are the dominant predictors (Pearson r = 0.87 and 0.67). Heavy missingness in `ledd` (37%), `time_since_intake_on` (46%), `time_since_intake_off` (79%) — an estimator handling NaNs natively (e.g., `HistGradientBoostingRegressor`) is indicated.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — visit-level 80/20 holdout floor | done | RMSE 16.48, MAE 13.48 | [design note](01_dummy.md) |
| `02_ridge` | Ridge + median imputation, GroupKFold(5) patient-grouped CV | done | RMSE 10.60 ± 0.27, R² 0.587 | [design note](02_ridge.md) |
| `03_hgbr_tabular` | skrub tabular_pipeline all features + time_since_diagnosis, GroupKFold(5) | done | RMSE 7.45 ± 0.14, R² 0.796 | [design note](03_hgbr_tabular.md) |
| `04_best` | temporal neighbours + per-patient off aggregates + missingness flags + tuned HGBR (RandomizedSearchCV GroupKFold inner CV), GroupKFold(5) outer | done | RMSE 4.55 ± 0.08 | [design note](04_best.md) |
| `05_pk_rater` | PK decay curves (τ=1/2/4 h) + rater-bias correctors + within-patient OLS trajectory, reused 04 HGBR params, GroupKFold(5) | done | RMSE 4.43 ± 0.05 | [design note](05_pk_rater.md) |
| `06_residual_ens` | Variant A: residual HGBR (target − off_proxy); Variant B: 5-seed HGBR ensemble; winner=06B_ensemble | done | RMSE 4.40 ± 0.06 (B wins; A=5.91 high-var) | [design note](06_residual_ens.md) |
| `07_patient_stack` | Two-stage stacker: stage 1 = 05_pk_rater; stage 2 = HGBR on patient-context features of OOF stage-1 preds | done | RMSE 4.27 ± 0.12 (winner vs 05 and 06B) | [design note](07_patient_stack.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| <!-- B1 --> | <!-- "investigate target-bin>0.95 residual bias via target transform" --> | <!-- `skore:01_baseline` --> |
