"""Learner declarations for Parkinson's UPDRS prediction experiments.

``build_learner`` — ``DummyRegressor(mean)`` floor baseline (01_dummy).
``build_ridge_learner`` — ``SimpleImputer(median) + Ridge`` (02_ridge).

Each function is independent so a later experiment can import either
without pulling in the other's dependencies.
"""

from __future__ import annotations

from pathlib import Path

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
