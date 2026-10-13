"""Learner declarations for Parkinson's UPDRS prediction experiments.

``build_learner``              — ``DummyRegressor(mean)`` (01_dummy).
``build_ridge_learner``        — ``SimpleImputer(median) + Ridge`` (02_ridge).
``add_time_since_diagnosis``   — stateless feature: age − age_at_diagnosis.
``build_hgbr_tabular_learner`` — ``skrub.tabular_pipeline("regressor")``
                                 on all features (03_hgbr_tabular).
``add_best_features``          — temporal neighbours, per-patient aggregates,
                                 and missingness flags for 04_best.
``build_best_learner``         — tuned HGBR via RandomizedSearchCV with
                                 patient-grouped inner CV (04_best).

Each builder is independent so experiments import only what they need.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import skrub
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def build_learner(
    data_dir_preview: str | Path | None = None,  # noqa: ARG001
) -> DummyRegressor:
    """Return an unfit DummyRegressor that always predicts the mean.

    Parameters
    ----------
    data_dir_preview :
        Unused for this estimator (no DataOps graph). Accepted for API
        consistency with later experiments.

    Returns
    -------
    DummyRegressor
        sklearn estimator with ``strategy="mean"``.
    """
    return DummyRegressor(strategy="mean")


# ---------------------------------------------------------------------------
# 02_ridge
# ---------------------------------------------------------------------------

#: Numeric feature columns used by the Ridge pipeline.
#: Categorical columns (patient_id, cohort, gene, rater_id) are excluded
#: because Ridge cannot handle string inputs.
NUMERIC_COLS: list[str] = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def build_ridge_learner(
    data_dir_preview: str | Path | None = None,  # noqa: ARG001
) -> Pipeline:
    """Return an unfit Ridge pipeline with median imputation.

    Selects ``NUMERIC_COLS`` (drops categorical columns that Ridge
    cannot encode), imputes missing values with each column's median,
    then fits a Ridge linear regressor (default alpha=1.0).

    Parameters
    ----------
    data_dir_preview :
        Unused for this estimator. Accepted for API consistency.

    Returns
    -------
    Pipeline
        sklearn Pipeline: SimpleImputer(strategy="median") → Ridge().
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge()),
        ]
    )


# ---------------------------------------------------------------------------
# 03_hgbr_tabular
# ---------------------------------------------------------------------------

#: Columns excluded from the feature matrix for 03_hgbr_tabular.
#: ``patient_id`` is used only as the GroupKFold group key;
#: ``target`` is the label.  ``Index`` is the DataFrame index, not a column.
HGBR_DROP_COLS: list[str] = ["patient_id", "target"]


def add_time_since_diagnosis(X: pd.DataFrame) -> pd.DataFrame:
    """Add ``time_since_diagnosis = age − age_at_diagnosis`` to ``X``.

    Replaces the near-collinear (age, age_at_diagnosis) pair (Pearson
    r = 0.94 in EDA) with a single derived feature that captures disease
    duration in years.  The original columns are kept so the pipeline
    can still use their individual values if informative.

    Parameters
    ----------
    X :
        Feature DataFrame with at least ``age`` and ``age_at_diagnosis``
        columns.  Missing values in either column propagate to the new
        column (HGBR handles NaN natively).

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with an additional ``time_since_diagnosis`` column.
    """
    X = X.copy()
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    return X


def build_hgbr_tabular_learner(
    data_dir_preview: str | Path | None = None,  # noqa: ARG001
) -> Pipeline:
    """Return an unfit skrub tabular_pipeline("regressor").

    Uses all input columns except ``patient_id`` and ``target``
    (``patient_id`` is kept in the DataFrame passed to the experiment
    for GroupKFold grouping but dropped before this learner sees it).
    Prepends ``add_time_since_diagnosis`` as a stateless step.

    ``tabular_pipeline("regressor")`` wraps:
    - ``TableVectorizer``: handles strings, dates, and NaN natively;
    - ``HistGradientBoostingRegressor``: handles NaN natively, no
      explicit imputation needed.

    Parameters
    ----------
    data_dir_preview :
        Unused. Accepted for API consistency with other builders.

    Returns
    -------
    Pipeline
        sklearn Pipeline: add_time_since_diagnosis → tabular_pipeline.
    """
    from sklearn.preprocessing import FunctionTransformer

    feature_step = FunctionTransformer(add_time_since_diagnosis)
    tabular = skrub.tabular_pipeline("regressor")
    return Pipeline(
        steps=[
            ("add_features", feature_step),
            *tabular.steps,
        ]
    )


# ---------------------------------------------------------------------------
# 04_best
# ---------------------------------------------------------------------------

#: Columns excluded from the raw feature matrix for 04_best.
#: ``patient_id`` is KEPT in X so that ``add_best_features`` can use it
#: for within-patient temporal features.  It is dropped inside the pipeline
#: by a ``FunctionTransformer`` step before ``TableVectorizer`` sees it.
#: ``target`` is the label and never in X.
BEST_DROP_COLS: list[str] = ["target"]


def add_best_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add temporal neighbour features, per-patient aggregates, and missingness flags.

    All operations are performed within each patient's sorted visit sequence
    using only columns already present in X — never the target.

    Features added
    --------------
    Neighbours (shift within patient sorted by age):
      on_prev, off_prev, on_next, off_next
      delta_on  (on − on_prev), delta_off (off − off_prev)
      age_gap_prev (age − age_prev), age_gap_next (age_next − age)
      visit_number (0-based rank by age within patient)

    Per-patient aggregates:
      mean_on, mean_off
      last_off  (off at the latest visit with a non-NaN off value)
      nearest_off (off at the visit with the smallest |age gap| among visits
                   with non-NaN off, excluding the current row when possible)

    Missingness flags:
      off_missing, on_missing, ledd_missing  (1 = NaN in original column)

    Parameters
    ----------
    X :
        Feature DataFrame with columns including ``patient_id``, ``age``,
        ``on``, ``off``, and ``ledd``.  The ``target`` column must NOT be
        present (the function reads only feature columns).

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with additional columns; original index is preserved.
    """
    X = X.copy()

    # Missingness flags first (before any transforms)
    X["off_missing"] = X["off"].isna().astype("int8")
    X["on_missing"] = X["on"].isna().astype("int8")
    X["ledd_missing"] = X["ledd"].isna().astype("int8")

    # Sort within patient by age; keep original index to restore order later
    X = X.sort_values(["patient_id", "age"])

    grp = X.groupby("patient_id", sort=False)

    # Shifted neighbour values
    X["on_prev"] = grp["on"].shift(1)
    X["off_prev"] = grp["off"].shift(1)
    X["on_next"] = grp["on"].shift(-1)
    X["off_next"] = grp["off"].shift(-1)

    # Deltas vs previous visit
    X["delta_on"] = X["on"] - X["on_prev"]
    X["delta_off"] = X["off"] - X["off_prev"]

    # Age gaps
    age_prev = grp["age"].shift(1)
    age_next = grp["age"].shift(-1)
    X["age_gap_prev"] = X["age"] - age_prev
    X["age_gap_next"] = age_next - X["age"]

    # Visit number (0-based rank by age within patient)
    X["visit_number"] = grp.cumcount()

    # Per-patient aggregates: mean on/off
    X["mean_on"] = grp["on"].transform("mean")
    X["mean_off"] = grp["off"].transform("mean")

    # last_off: off value at the last visit with non-NaN off, broadcast to all rows
    # Compute per-patient: take the last non-NaN off value, map back via patient_id.
    last_off_map = (
        X[X["off"].notna()]
        .groupby("patient_id")["off"]
        .last()
    )
    X["last_off"] = X["patient_id"].map(last_off_map)

    # nearest_off: off at the closest-age visit (excluding self) that has non-NaN off.
    # Computed row-by-row via iterrows would be too slow; use a per-patient loop.
    nearest_off_vals: dict[int, float] = {}
    for pid, group in grp:
        ages = group["age"].values
        offs = group["off"].values
        idx = group.index.tolist()
        for i in range(len(ages)):
            mask = ~pd.isna(offs).copy()
            mask_excl = mask.copy()
            mask_excl[i] = False  # exclude self when possible
            if not mask_excl.any():
                nearest_off_vals[idx[i]] = offs[i]
            else:
                dists = abs(ages - ages[i]).astype(float)
                dists[~mask_excl] = float("inf")
                nearest_off_vals[idx[i]] = offs[dists.argmin()]
    X["nearest_off"] = pd.Series(nearest_off_vals)

    # Restore original index order (sort_values changed it)
    X = X.sort_index()
    return X



def _add_all_features_04(X: pd.DataFrame) -> pd.DataFrame:
    """Apply add_best_features then add_time_since_diagnosis (module-level for pickle)."""
    return add_time_since_diagnosis(add_best_features(X))


def _drop_patient_id_04(X: pd.DataFrame) -> pd.DataFrame:
    """Drop patient_id column so TableVectorizer never sees it (module-level for pickle)."""
    return X.drop(columns=["patient_id"], errors="ignore")


def build_best_learner(
    hgbr_params: "dict | None" = None,
    data_dir_preview: "str | Path | None" = None,  # noqa: ARG001
) -> "Pipeline":
    """Return an unfit feature-engineering + HGBR pipeline for 04_best.

    Pipeline layout::

        add_best_features (FunctionTransformer)
          -> add_time_since_diagnosis (FunctionTransformer)
          -> drop_patient_id (FunctionTransformer)
          -> TableVectorizer  (from skrub.tabular_pipeline)
          -> HistGradientBoostingRegressor

    ``patient_id`` is kept in X through the feature steps (required by
    ``add_best_features`` for within-patient temporal shifts) and dropped
    just before ``TableVectorizer``.

    Hyperparameter tuning is performed in the experiment script via
    ``RandomizedSearchCV`` on the full training set before the outer CV
    evaluation.  The best params are then passed here via ``hgbr_params``
    so the outer ``skore.CrossValidationReport`` evaluates a fixed-param
    pipeline without any leakage risk.

    Parameters
    ----------
    hgbr_params :
        Dict of HGBR hyperparameters (e.g.
        ``{"learning_rate": 0.05, "max_leaf_nodes": 63}``).
        If ``None``, the HGBR uses its sklearn defaults.
    data_dir_preview :
        Unused.  Accepted for API consistency.

    Returns
    -------
    Pipeline
        sklearn Pipeline: feature engineering -> TableVectorizer -> HGBR.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import FunctionTransformer

    feature_step = FunctionTransformer(_add_all_features_04)
    drop_step = FunctionTransformer(_drop_patient_id_04)
    tabular = skrub.tabular_pipeline("regressor")

    hgbr = HistGradientBoostingRegressor(**(hgbr_params or {}))

    tabular_steps_no_last = tabular.steps[:-1]
    return Pipeline(
        steps=[
            ("add_features", feature_step),
            ("drop_patient_id", drop_step),
            *tabular_steps_no_last,
            ("histgradientboostingregressor", hgbr),
        ]
    )