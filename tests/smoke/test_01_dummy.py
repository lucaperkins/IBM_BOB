"""Smoke test for ``experiments/01_dummy.py``.

IID flat-table pipeline with ``DummyRegressor(strategy="mean")``.
Fits on a training subset and predicts on a disjoint held-out
subset; asserts that the prediction count equals the held-out
row count and that the smoke MAE is within a factor of 3 of the
expected RMSE floor (~16.5, roughly equal to the target std from EDA).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset
from parkipredict.pipeline import build_learner

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: DummyRegressor(mean) MAE ≈ target std ≈ 16.5
# (from data/eda.md; updated when the design note Status.headline fills in).
# We allow 3× headroom for the smaller smoke window.
_EXPECTED_DUMMY_MAE = 16.5


@pytest.fixture(scope="module")
def smoke_split():
    """Return (X_train, y_train, X_smoke, y_smoke) from the first 500 rows.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        ``(X_train, y_train, X_smoke, y_smoke)`` where the first
        400 rows are used for fitting and the next 100 for prediction.
    """
    X, y = load_dataset()
    # Use the first 500 rows to keep the smoke test fast.
    X_small, y_small = X.iloc[:500], y.iloc[:500]
    n_train = 400
    X_train, y_train = X_small.iloc[:n_train], y_small.iloc[:n_train]
    X_smoke, y_smoke = X_small.iloc[n_train:], y_small.iloc[n_train:]
    return X_train, y_train, X_smoke, y_smoke


def test_01_dummy_row_count(smoke_split):
    """Prediction count must match the number of predict-grid rows.

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert len(predictions) == len(X_smoke), (
        f"got {len(predictions)} predictions for {len(X_smoke)} rows"
    )


def test_01_dummy_predictions_not_nan(smoke_split):
    """Predictions must be finite (no NaN-poisoning).

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert np.all(np.isfinite(predictions)), "predictions contain NaN or Inf"


def test_01_dummy_mae_bound(smoke_split):
    """Smoke MAE must be within 3 × expected DummyRegressor floor.

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, y_smoke = smoke_split
    learner = build_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    smoke_mae = mean_absolute_error(y_smoke, predictions)
    assert smoke_mae < 3 * _EXPECTED_DUMMY_MAE, (
        f"smoke MAE {smoke_mae:.2f} > 3 × expected floor "
        f"({3 * _EXPECTED_DUMMY_MAE:.2f})"
    )
