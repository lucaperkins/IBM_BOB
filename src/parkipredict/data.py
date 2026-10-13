"""Data loading for Parkinson's UPDRS prediction experiments.

Loads ``X_train`` / ``y_train`` from the project's ``data/`` directory
and returns them as a merged DataFrame plus the target series. Also
provides ``load_test_dataset`` for producing the Kaggle submission.

The data folder is user-owned and never modified by this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkipredict import PROJECT_ROOT

TARGET_COL = "target"
DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from the train split.

    Parameters
    ----------
    data_dir :
        Directory that contains ``X_train.csv`` and ``y_train.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` (all columns except the target) and
        target series ``y``, both indexed by ``Index``.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    X = pd.read_csv(root / "X_train.csv", index_col="Index")
    y = pd.read_csv(root / "y_train.csv", index_col="Index")[TARGET_COL]
    return X, y


def load_test_dataset(
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return ``X_test`` for the Kaggle submission.

    Parameters
    ----------
    data_dir :
        Directory that contains ``X_test.csv``.
        Defaults to ``<project_root>/data``.

    Returns
    -------
    pd.DataFrame
        Test feature matrix indexed by ``Index``.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    return pd.read_csv(root / "X_test.csv", index_col="Index")
