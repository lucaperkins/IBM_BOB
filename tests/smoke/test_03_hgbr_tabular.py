"""Smoke test for ``experiments/03_hgbr_tabular.py``.

skrub ``tabular_pipeline("regressor")`` with all features and a derived
``time_since_diagnosis`` column. Fits on 400 rows, predicts on 100
disjoint rows; asserts row-count equality and finite predictions.

Soft-assertion baseline: Ridge MAE ≈ 8.53 (from 02_ridge run).
HGBR should beat or match it; allow 3× headroom for the small window.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from parkipredict import PROJECT_ROOT
from parkipredict.data import load_dataset
from parkipredict.pipeline import HGBR_DROP_COLS, build_hgbr_tabular_learner

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: Ridge MAE 8.53 from 02_ridge GroupKFold run.
# HGBR should be below; allow 3× for the small smoke window.
_RIDGE_MAE = 8.53


@pytest.fixture(scope="module")
def smoke_split():
    """Return (X_train, y_train, X_smoke, y_smoke) on HGBR feature set.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        ``(X_train, y_train, X_smoke, y_smoke)`` using all features
        except patient_id and target.  First 400 rows for fitting,
        next 100 for prediction.
    """
    X_full, y = load_dataset()
    X = X_full.drop(columns=HGBR_DROP_COLS, errors="ignore")
    X_small, y_small = X.iloc[:500], y.iloc[:500]
    n_train = 400
    X_train, y_train = X_small.iloc[:n_train], y_small.iloc[:n_train]
    X_smoke, y_smoke = X_small.iloc[n_train:], y_small.iloc[n_train:]
    return X_train, y_train, X_smoke, y_smoke


def test_03_hgbr_tabular_row_count(smoke_split):
    """Prediction count must equal the number of predict-grid rows.

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_hgbr_tabular_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert len(predictions) == len(X_smoke), (
        f"got {len(predictions)} predictions for {len(X_smoke)} rows"
    )


def test_03_hgbr_tabular_predictions_not_nan(smoke_split):
    """Predictions must be finite (no NaN / Inf from the pipeline).

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, _ = smoke_split
    learner = build_hgbr_tabular_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    assert np.all(np.isfinite(predictions)), "predictions contain NaN or Inf"


def test_03_hgbr_tabular_mae_below_3x_ridge(smoke_split):
    """Smoke MAE must be below 3 × Ridge MAE (3 × 8.53 = 25.6).

    Parameters
    ----------
    smoke_split :
        Pytest fixture returning (X_train, y_train, X_smoke, y_smoke).
    """
    X_train, y_train, X_smoke, y_smoke = smoke_split
    learner = build_hgbr_tabular_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_smoke)

    smoke_mae = mean_absolute_error(y_smoke, predictions)
    assert smoke_mae < 3 * _RIDGE_MAE, (
        f"smoke MAE {smoke_mae:.2f} > 3 × Ridge baseline "
        f"({3 * _RIDGE_MAE:.2f})"
    )
