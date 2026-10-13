"""Smoke test for ``experiments/02_ridge.py``.

IID flat-table pipeline with ``SimpleImputer(median) + Ridge``.
Fits on a training subset and predicts on a disjoint held-out subset;
asserts prediction count matches the held-out row count and that
predictions are finite.

The soft-assertion baseline will be filled in from the design note
headline after the first run; for now a loose bound of 3 × dummy floor
(3 × 16.5 = 49.5 MAE) is used.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset
from parkipredict.pipeline import NUMERIC_COLS, build_ridge_learner

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: DummyRegressor floor MAE ≈ 16.5 (from 01_dummy run).
# Ridge should beat it substantially; allow 3× headroom for the small smoke window.
_DUMMY_MAE_FLOOR = 16.5


@pytest.fixture(scope="module")
def smoke_split():
    """Return (X_train, y_train, X_smoke, y_smoke) from the first 500 rows.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        ``(X_train, y_train, X_smoke, y_smoke)`` using only numeric columns.
        First 400 rows for fitting, next 100 for prediction.
    """
    X_full, y = load_dataset()
    X = X_full[NUMERIC_COLS]
    X_small, y_small = X.iloc[:500], y.iloc[:500]
    n_train = 400
    X_train, y_train = X_small.iloc[:n_train], y_small.iloc[:n_train]
    X_smoke, y_smoke = X_small.iloc[n_train:], y_small.iloc[n_train:]
    return X_train, y_train, X_smoke, y_smoke


def test_02_ridge_row_count(smoke_split):
    """Prediction count must match the number of predict-grid rows.

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert len(predictions) == len(X_smoke), (
        f"got {len(predictions)} predictions for {len(X_smoke)} rows"
    )


def test_02_ridge_predictions_not_nan(smoke_split):
    """Predictions must be finite (no NaN or Inf from imputation failure).

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert np.all(np.isfinite(predictions)), "predictions contain NaN or Inf"


def test_02_ridge_mae_below_dummy_floor(smoke_split):
    """Ridge MAE must be below 3 × dummy floor (3 × 16.5 = 49.5).

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, y_smoke = smoke_split
    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    smoke_mae = mean_absolute_error(y_smoke, predictions)
    assert smoke_mae < 3 * _DUMMY_MAE_FLOOR, (
        f"smoke MAE {smoke_mae:.2f} > 3 × dummy floor "
        f"({3 * _DUMMY_MAE_FLOOR:.2f})"
    )
