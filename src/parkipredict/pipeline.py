"""Learner declarations for Parkinson's UPDRS prediction experiments.

``build_learner``              — ``DummyRegressor(mean)`` (01_dummy).
``build_ridge_learner``        — ``SimpleImputer(median) + Ridge`` (02_ridge).
``add_time_since_diagnosis``   — stateless feature: age − age_at_diagnosis.
``build_hgbr_tabular_learner`` — ``skrub.tabular_pipeline("regressor")``
                                 on all features (03_hgbr_tabular).

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
