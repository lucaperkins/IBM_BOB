# EDA: Parkinson's Disease UPDRS Score Prediction

**Generated:** 2025-10-07  
**Data:** `data/X_train.csv` + `data/y_train.csv` (merged on `Index`)  
**Interactive report:** [data/eda_train.html](eda_train.html)

---

## Dataset at a glance

| Property | Value |
|---|---|
| Rows | 44,590 |
| Columns (features + target) | 13 |
| Target | `target` (continuous regression score) |
| Train split | `X_train.csv` / `y_train.csv` |
| Test split | `X_test.csv` (11,013 rows, no target) |

Each row is a **patient visit**. The `patient_id` column appears 5,576 unique times across 44,590 rows — meaning patients have multiple visits (avg ~8 visits per patient).

---

## Per-column findings

| Column | Dtype | Missing % | Unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5,576 | Group key — same patient across many rows |
| `cohort` | string | 0 % | 2 | Binary cohort label (A/B) |
| `sexM` | int | 0 % | 2 | Binary sex flag |
| `gene` | string | **32.4 %** | 4 | High missingness; 4 categories including `LRRK2+` |
| `age_at_diagnosis` | float | **5.2 %** | 563 | Moderate missingness |
| `age` | float | 0 % | 1,150 | No missing values |
| `ledd` | float | **36.6 %** | 1,320 | Levodopa equivalent daily dose — heavy missingness |
| `time_since_intake_on` | float | **46.4 %** | 64 | Nearly half missing |
| `time_since_intake_off` | float | **78.8 %** | 178 | Very high missingness — majority of rows lack this |
| `rater_id` | string | 0 % | 50 | 50 unique raters |
| `on` | float | **29.6 %** | 83 | UPDRS score in ON state |
| `off` | float | **42.4 %** | 101 | UPDRS score in OFF state |
| `target` | float | 0 % | 867 | **Regression target** — no missing values |

**Missingness summary:** Several key clinical columns have substantial missing rates — `time_since_intake_off` (~79%), `time_since_intake_on` (~46%), `ledd` (~37%). The pipeline will need robust imputation or an estimator that handles NaNs natively (e.g., `HistGradientBoostingRegressor`).

---

## Target

`target` is a continuous regression score (UPDRS-like Parkinson's severity scale).

| Statistic | Value |
|---|---|
| Min | 0.0 |
| Q1 (25th pct) | 25.6 |
| Median | 37.3 |
| Mean | 37.5 |
| Q3 (75th pct) | 49.3 |
| Max | 109.5 |
| Std dev | 16.5 |
| IQR | 23.7 |

The distribution is **approximately symmetric around the median (~37)**, with a slight positive tail (max 109.5, but >97% of values fall below 80). No missing target values in training. A standard regression metric (RMSE, MAE, or R²) is appropriate. No stratification is needed for CV — the continuous target is well-spread with no extreme skew.

---

## Structure signals

**Datetime columns:** None detected by dtype. No explicit timestamp column exists in the data.

**Group / ID columns (high unique ratio):**

| Column | Unique ratio |
|---|---|
| `patient_id` | 12.5% (5,576 unique patients across 44,590 rows) |
| `ledd` | 3.0% |
| `age` | 2.6% |

`patient_id` is the critical signal: the same patient appears in multiple rows. If the train/test split is done at the row level without accounting for patients, information about a patient in training can leak into the test set (the model has "seen" that patient's other visits). This points strongly to **`GroupKFold` with `patient_id` as the group key** for cross-validation.

`rater_id` has 50 distinct values with zero missingness — it is a known source of variability and may warrant attention (rater effects on UPDRS scoring).

---

## Associations

**Strongest feature → target correlations (Pearson, from skrub `column_associations`):**

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` (UPDRS OFF state) | **0.871** | 0.406 |
| `on` (UPDRS ON state) | **0.669** | 0.368 |
| `age` | 0.310 | 0.114 |
| `ledd` | 0.298 | 0.236 |
| `age_at_diagnosis` | 0.133 | 0.051 |

**Notable feature–feature associations:**

- `age_at_diagnosis` ↔ `age`: Pearson r = **0.942** — these two are nearly collinear. Including both may add noise; one could be dropped or the difference (disease duration) derived.
- `on` ↔ `off`: Pearson r = **0.872** — strongly correlated with each other *and* both with `target`. The `off` score is the single strongest predictor (r = 0.87).
- `on` and `off` are themselves partial UPDRS subscores, which explains their high correlation with the global target. This is not leakage — they are collected at the same visit — but the model will lean heavily on them.

**Leakage check:** The `on` / `off` columns are UPDRS sub-scores measured at the same clinical visit, so their correlation with `target` is structural and expected. `time_since_intake_*` captures medication timing, which can causally affect the scores. No implausibly perfect correlations were found.

---

## Modelling implications

| Finding | Implication |
|---|---|
| `patient_id` repeats across rows | Use **`GroupKFold(groups=patient_id)`** — never split at the row level without accounting for patient identity |
| Continuous, approximately symmetric target | Standard regression; **RMSE or R²** as the primary metric |
| Heavy missingness in `ledd`, `time_since_intake_*`, `gene` | Use an estimator that handles NaN natively (e.g., `HistGradientBoostingRegressor`) **or** impute explicitly in the pipeline |
| `on` / `off` are the dominant predictors | Include both; also worth testing a model without them to understand the value of the other features |
| `age_at_diagnosis` ↔ `age` near-collinear (r = 0.94) | Consider deriving disease duration (`age − age_at_diagnosis`) and dropping one |
| `gene` 32% missing, 4 categories | Impute as a separate category ("unknown") or use a missing indicator; test impact |
| `rater_id` with 50 categories | High-cardinality categorical — needs encoding (e.g., target encoding or an embedding-capable estimator); may capture rater-specific scoring bias |
| No datetime column | `TimeSeriesSplit` is not indicated; `GroupKFold` is the primary CV candidate |

---

## Open questions

1. **What is the exact definition of `target`?** It appears to be a composite UPDRS (Unified Parkinson's Disease Rating Scale) score, but the precise subscale is not documented in the data.
2. **Are `on` and `off` always available at prediction time?** They are missing in 30–42% of training rows. If they are missing at inference, the model must degrade gracefully. This affects whether they can be first-class features.
3. **Is the test set a patient-level holdout or row-level?** If `patient_id` in `X_test.csv` overlaps with training patients, the task is interpolation (a different difficulty than extrapolation to new patients).
4. **Is `rater_id` available at prediction time?** If not, it should be excluded from the pipeline.
