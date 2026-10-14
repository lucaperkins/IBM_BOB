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
``add_pk_rater_features``      — PK decay features, rater-bias corrector, and
                                 within-patient linear trajectory for 05_pk_rater.
``build_pk_rater_learner``     — same tuned HGBR as 04_best plus new features.
``PatientStackLearner``        — two-stage stacker: stage 1 = 05_pk_rater
                                 pipeline; stage 2 = HGBR on per-patient
                                 prediction context features (07_patient_stack).
``build_patient_stack_learner`` — builder for PatientStackLearner.

Each builder is independent so experiments import only what they need.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
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


# ---------------------------------------------------------------------------
# 05_pk_rater
# ---------------------------------------------------------------------------

#: Hardened best HGBR hyperparameters from 04_best RandomizedSearchCV.
HGBR_PARAMS_04: dict = {
    "learning_rate": 0.017171860465888467,
    "max_leaf_nodes": 132,
    "max_iter": 497,
    "min_samples_leaf": 95,
    "l2_regularization": 5.1401896641681114e-05,
    "random_state": 0,
}


def add_pk_rater_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add PK-decay, rater-bias, and within-patient trajectory features.

    All features are computed on X only — never on the target.

    PK timing (columns ``time_since_intake_on``, ``time_since_intake_off``):
      For tau in {1, 2, 4} hours:
        decay_on_<tau>h  = exp(−time_since_intake_on / tau)
        decay_off_<tau>h = exp(−time_since_intake_off / tau)
        interact_on_<tau>h  = on  * decay_on_<tau>h   (0 when on is NaN)
        interact_off_<tau>h = off * decay_off_<tau>h  (0 when off is NaN)
      decay_ratio_on_off = decay_on_1h / (decay_off_1h + 1e-9) when both exist.

    Rater corrector (column ``rater_id``):
      rater_id is kept as-is for TableVectorizer categorical encoding.
      rater_bias_off = per-rater mean of (off − patient_mean_off), broadcast.
      rater_bias_on  = per-rater mean of (on  − patient_mean_on),  broadcast.

    Trajectory (linear regression of off ~ age and on ~ age within patient):
      traj_off_slope, traj_off_intercept
      traj_on_slope,  traj_on_intercept
      traj_off_pred   = traj_off_intercept + traj_off_slope * age
      traj_on_pred    = traj_on_intercept  + traj_on_slope  * age

    Parameters
    ----------
    X :
        Feature DataFrame with at minimum ``patient_id``, ``age``, ``on``,
        ``off``, ``time_since_intake_on``, ``time_since_intake_off``,
        ``rater_id``.  Must NOT contain ``target``.

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with additional feature columns; original index preserved.
    """
    import numpy as np

    X = X.copy()

    # ------------------------------------------------------------------
    # 1. PK decay features
    # ------------------------------------------------------------------
    tsi_on = X["time_since_intake_on"]   # hours, may be NaN
    tsi_off = X["time_since_intake_off"]  # hours, may be NaN

    for tau in (1, 2, 4):
        d_on = np.exp(-tsi_on / tau)
        d_off = np.exp(-tsi_off / tau)
        X[f"decay_on_{tau}h"] = d_on
        X[f"decay_off_{tau}h"] = d_off
        # interaction: 0 where on/off is NaN (fill_value=0 to avoid NaN × NaN)
        X[f"interact_on_{tau}h"] = X["on"].fillna(0) * d_on.fillna(0)
        X[f"interact_off_{tau}h"] = X["off"].fillna(0) * d_off.fillna(0)

    # Ratio only where both decay values are available
    d_on1 = X["decay_on_1h"]
    d_off1 = X["decay_off_1h"]
    both_exist = d_on1.notna() & d_off1.notna()
    X["decay_ratio_on_off"] = np.where(both_exist, d_on1 / (d_off1 + 1e-9), np.nan)

    # ------------------------------------------------------------------
    # 2. Rater-bias corrector
    # ------------------------------------------------------------------
    # patient-level means (broadcast, NaN-safe)
    pat_grp = X.groupby("patient_id", sort=False)
    patient_mean_off = pat_grp["off"].transform("mean")
    patient_mean_on = pat_grp["on"].transform("mean")

    # deviation from patient mean (NaN where off/on is NaN)
    dev_off = X["off"] - patient_mean_off
    dev_on = X["on"] - patient_mean_on

    # per-rater mean deviation, then broadcast back to rows
    tmp = X[["rater_id"]].copy()
    tmp["dev_off"] = dev_off
    tmp["dev_on"] = dev_on
    rater_bias_off = tmp.groupby("rater_id")["dev_off"].transform("mean")
    rater_bias_on = tmp.groupby("rater_id")["dev_on"].transform("mean")
    X["rater_bias_off"] = rater_bias_off
    X["rater_bias_on"] = rater_bias_on

    # ------------------------------------------------------------------
    # 3. Within-patient linear trajectory (off ~ age, on ~ age)
    # ------------------------------------------------------------------
    traj_cols = {
        "traj_off_slope": np.nan,
        "traj_off_intercept": np.nan,
        "traj_off_pred": np.nan,
        "traj_on_slope": np.nan,
        "traj_on_intercept": np.nan,
        "traj_on_pred": np.nan,
    }
    for col, default in traj_cols.items():
        X[col] = default

    for pid, grp_df in pat_grp:
        ages = grp_df["age"].values.astype(float)
        idx = grp_df.index

        # off trajectory
        off_vals = grp_df["off"].values.astype(float)
        mask_off = ~np.isnan(off_vals) & ~np.isnan(ages)
        if mask_off.sum() >= 2:
            slope_off, intercept_off = np.polyfit(ages[mask_off], off_vals[mask_off], 1)
        else:
            slope_off, intercept_off = 0.0, np.nanmean(off_vals) if mask_off.any() else np.nan
        X.loc[idx, "traj_off_slope"] = slope_off
        X.loc[idx, "traj_off_intercept"] = intercept_off
        X.loc[idx, "traj_off_pred"] = intercept_off + slope_off * ages

        # on trajectory
        on_vals = grp_df["on"].values.astype(float)
        mask_on = ~np.isnan(on_vals) & ~np.isnan(ages)
        if mask_on.sum() >= 2:
            slope_on, intercept_on = np.polyfit(ages[mask_on], on_vals[mask_on], 1)
        else:
            slope_on, intercept_on = 0.0, np.nanmean(on_vals) if mask_on.any() else np.nan
        X.loc[idx, "traj_on_slope"] = slope_on
        X.loc[idx, "traj_on_intercept"] = intercept_on
        X.loc[idx, "traj_on_pred"] = intercept_on + slope_on * ages

    return X


def _add_all_features_05(X: pd.DataFrame) -> pd.DataFrame:
    """Apply 04_best features, time_since_diagnosis, and 05 PK/rater/trajectory features."""
    return add_pk_rater_features(add_time_since_diagnosis(add_best_features(X)))


def _drop_patient_id_05(X: pd.DataFrame) -> pd.DataFrame:
    """Drop patient_id before TableVectorizer (module-level for pickle)."""
    return X.drop(columns=["patient_id"], errors="ignore")


def build_pk_rater_learner(
    hgbr_params: "dict | None" = None,
    data_dir_preview: "str | Path | None" = None,  # noqa: ARG001
) -> "Pipeline":
    """Return an unfit pipeline for 05_pk_rater.

    Extends 04_best with PK-decay, rater-bias, and trajectory features.
    Uses ``HGBR_PARAMS_04`` by default (the tuned params from 04_best);
    pass a different dict to override.

    Parameters
    ----------
    hgbr_params :
        HGBR hyperparameter dict.  Defaults to ``HGBR_PARAMS_04``.
    data_dir_preview :
        Unused.  Accepted for API consistency.

    Returns
    -------
    Pipeline
        sklearn Pipeline: feature engineering → TableVectorizer → HGBR.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import FunctionTransformer

    params = hgbr_params if hgbr_params is not None else HGBR_PARAMS_04
    feature_step = FunctionTransformer(_add_all_features_05)
    drop_step = FunctionTransformer(_drop_patient_id_05)
    tabular = skrub.tabular_pipeline("regressor")
    hgbr = HistGradientBoostingRegressor(**params)
    tabular_steps_no_last = tabular.steps[:-1]
    return Pipeline(
        steps=[
            ("add_features", feature_step),
            ("drop_patient_id", drop_step),
            *tabular_steps_no_last,
            ("histgradientboostingregressor", hgbr),
        ]
    )


# ---------------------------------------------------------------------------
# 06_residual_ens  (Variant A: residual learner; Variant B: HGBR ensemble)
# ---------------------------------------------------------------------------


def compute_off_proxy(
    X: pd.DataFrame,
    fallback_mean: "float | None" = None,
) -> pd.Series:
    """Compute off_proxy: off if present, else traj_off_pred, else nearest_off.

    Requires that ``add_pk_rater_features`` has already been called so that
    ``traj_off_pred`` and ``nearest_off`` columns exist.

    Parameters
    ----------
    X :
        Featurized DataFrame (post add_pk_rater_features).
    fallback_mean :
        Value used to fill any remaining NaN after off, traj_off_pred, and
        nearest_off are exhausted.  If ``None``, uses ``X["off"].mean()``.

    Returns
    -------
    pd.Series
        off_proxy values aligned on X.index; no NaN values remain.
    """
    proxy = X["off"].copy()
    missing = proxy.isna()
    proxy[missing] = X.loc[missing, "traj_off_pred"]
    still_missing = proxy.isna()
    proxy[still_missing] = X.loc[still_missing, "nearest_off"]
    # Final fallback: use provided mean or compute from available values
    if proxy.isna().any():
        fill = fallback_mean if fallback_mean is not None else float(proxy.mean())
        proxy = proxy.fillna(fill)
    return proxy


class _ResidualLearner(BaseEstimator, RegressorMixin):
    """Variant A: fit HGBR on (target − off_proxy), predict residual + off_proxy.

    Parameters
    ----------
    hgbr_params :
        HGBR hyperparameter dict passed to HistGradientBoostingRegressor.
    """

    def __init__(self, hgbr_params: dict) -> None:
        self.hgbr_params = hgbr_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ResidualLearner":
        from sklearn.ensemble import HistGradientBoostingRegressor

        # Build all features (keep patient_id for feature engineering)
        X_feat = _add_all_features_05(X)
        # Store train-time off mean for consistent NaN fallback at predict time
        self._off_global_mean = float(X_feat["off"].mean())
        off_proxy = compute_off_proxy(X_feat, fallback_mean=self._off_global_mean)

        # Drop patient_id before vectorizer
        X_drop = _drop_patient_id_05(X_feat)

        tv = skrub.tabular_pipeline("regressor").named_steps["tablevectorizer"]
        self._tv = tv
        X_vec = tv.fit_transform(X_drop)

        self._hgbr = HistGradientBoostingRegressor(**self.hgbr_params)
        residual = y.values - off_proxy.values
        self._hgbr.fit(X_vec, residual)
        return self

    def predict(self, X: pd.DataFrame) -> "pd.ndarray":
        X_feat = _add_all_features_05(X)
        off_proxy = compute_off_proxy(X_feat, fallback_mean=self._off_global_mean)
        X_drop = _drop_patient_id_05(X_feat)
        X_vec = self._tv.transform(X_drop)
        residual_pred = self._hgbr.predict(X_vec)
        return residual_pred + off_proxy.values

    # sklearn-compatible duck-typing
    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {"hgbr_params": self.hgbr_params}

    def set_params(self, **params: object) -> "_ResidualLearner":
        if "hgbr_params" in params:
            self.hgbr_params = params["hgbr_params"]
        return self


def build_residual_learner(
    hgbr_params: "dict | None" = None,
    data_dir_preview: "str | Path | None" = None,  # noqa: ARG001
) -> "_ResidualLearner":
    """Return an unfit residual learner for Variant A of 06_residual_ens.

    Predicts (target − off_proxy) then adds back off_proxy at inference.

    Parameters
    ----------
    hgbr_params :
        HGBR hyperparameter dict.  Defaults to ``HGBR_PARAMS_04``.
    data_dir_preview :
        Unused.  Accepted for API consistency.

    Returns
    -------
    _ResidualLearner
        Sklearn-duck-typed estimator.
    """
    return _ResidualLearner(hgbr_params=hgbr_params or HGBR_PARAMS_04)


class _EnsembleLearner(BaseEstimator, RegressorMixin):
    """Variant B: average predictions of 5 HGBR models with different random seeds.

    Wraps the full 05_pk_rater pipeline (features + TV + HGBR) five times,
    differing only in ``random_state``.

    Parameters
    ----------
    hgbr_params :
        Base HGBR hyperparameter dict.  ``random_state`` is overridden per member.
    n_members :
        Number of ensemble members (default 5).
    """

    def __init__(self, hgbr_params: dict, n_members: int = 5) -> None:
        self.hgbr_params = hgbr_params
        self.n_members = n_members

    def _make_member(self, seed: int) -> Pipeline:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import FunctionTransformer

        params = {**self.hgbr_params, "random_state": seed}
        feature_step = FunctionTransformer(_add_all_features_05)
        drop_step = FunctionTransformer(_drop_patient_id_05)
        tabular = skrub.tabular_pipeline("regressor")
        hgbr = HistGradientBoostingRegressor(**params)
        tabular_steps_no_last = tabular.steps[:-1]
        return Pipeline(
            steps=[
                ("add_features", feature_step),
                ("drop_patient_id", drop_step),
                *tabular_steps_no_last,
                ("histgradientboostingregressor", hgbr),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_EnsembleLearner":
        self._members = []
        for seed in range(self.n_members):
            m = self._make_member(seed)
            m.fit(X, y)
            self._members.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> "pd.ndarray":
        import numpy as np

        preds = np.stack([m.predict(X) for m in self._members], axis=0)
        return preds.mean(axis=0)

    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {"hgbr_params": self.hgbr_params, "n_members": self.n_members}

    def set_params(self, **params: object) -> "_EnsembleLearner":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_ensemble_learner(
    hgbr_params: "dict | None" = None,
    n_members: int = 5,
    data_dir_preview: "str | Path | None" = None,  # noqa: ARG001
) -> "_EnsembleLearner":
    """Return an unfit 5-member HGBR ensemble for Variant B of 06_residual_ens.

    Parameters
    ----------
    hgbr_params :
        Base HGBR hyperparameter dict.  Defaults to ``HGBR_PARAMS_04``.
    n_members :
        Number of ensemble members (default 5).
    data_dir_preview :
        Unused.  Accepted for API consistency.

    Returns
    -------
    _EnsembleLearner
        Sklearn-duck-typed estimator.
    """
    return _EnsembleLearner(hgbr_params=hgbr_params or HGBR_PARAMS_04, n_members=n_members)


# ---------------------------------------------------------------------------
# 07_patient_stack  (two-stage patient-context stacker)
# ---------------------------------------------------------------------------


def _build_stage2_features(
    patient_ids: "pd.Series",
    ages: "pd.Series",
    stage1_preds: "np.ndarray",
    X_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build stage-2 features from stage-1 predictions within each patient.

    For every visit the following are computed using *only* the other visits of
    the same patient (leave-one-out context), so the visit's own target is never
    used:

    * ``ctx_mean``      — leave-one-out mean of the other visits' stage-1 preds.
    * ``ctx_std``       — leave-one-out std (NaN for singletons).
    * ``ctx_prev_pred`` — stage-1 prediction of the immediately preceding visit
                          ordered by age (NaN if first visit).
    * ``ctx_next_pred`` — stage-1 prediction of the immediately following visit
                          ordered by age (NaN if last visit).
    * ``ctx_loo_at_age``— leave-one-out per-patient OLS of predictions vs age
                          evaluated at the current visit age (NaN for singletons).
    * ``ctx_n_visits``  — total number of visits for the patient.
    * ``stage1_pred``   — the visit's own stage-1 prediction (passed through).

    Plus pass-through features from ``X_raw``: ``off``, ``on``, ``age``, and
    the missingness flags ``off_missing``, ``on_missing``, ``ledd_missing``.

    Uses a fully vectorised pandas implementation; the per-visit inner loop is
    avoided via LOO identities: LOO_mean_i = (total_sum − p_i) / (n − 1).

    Parameters
    ----------
    patient_ids :
        Series of patient identifiers aligned with ``stage1_preds``.
    ages :
        Series of visit ages aligned with ``stage1_preds``.
    stage1_preds :
        1-D array of stage-1 predictions; one value per visit.
    X_raw :
        Original (pre-feature-engineering) DataFrame for this split, used to
        extract ``off``, ``on``, ``age``, and missingness flags.

    Returns
    -------
    pd.DataFrame
        Stage-2 feature matrix; index matches ``patient_ids.index``.
    """
    import numpy as np

    idx = patient_ids.index

    # ----------------------------------------------------------------
    # Working frame sorted by (patient_id, age) for shifted features
    # ----------------------------------------------------------------
    work = pd.DataFrame(
        {
            "pid": patient_ids.values,
            "age": ages.values,
            "pred": stage1_preds,
        },
        index=idx,
    ).sort_values(["pid", "age"])

    grp = work.groupby("pid", sort=False)

    # --- ctx_n_visits ---
    work["ctx_n_visits"] = grp["pred"].transform("count").astype(float)

    # --- LOO mean: (sum − self) / (n − 1) ---
    pat_sum = grp["pred"].transform("sum")
    n = work["ctx_n_visits"]
    work["ctx_mean"] = (pat_sum - work["pred"]) / (n - 1)
    work.loc[n == 1, "ctx_mean"] = np.nan

    # --- LOO std (sample std of the n-1 other visits) ---
    # Unbiased: var = [sum_sq_loo - (sum_loo)²/(n-1)] / (n-2)
    pat_sumsq = grp["pred"].transform(lambda x: (x**2).sum())
    loo_sum_sq = pat_sumsq - work["pred"] ** 2          # sum of squares of the n-1 others
    loo_sum = pat_sum - work["pred"]                     # sum of the n-1 others
    loo_var = (loo_sum_sq - (loo_sum ** 2) / (n - 1)) / (n - 2)
    loo_var = loo_var.clip(lower=0)                      # guard fp noise
    work["ctx_std"] = np.where(n > 2, np.sqrt(loo_var), np.nan)
    work.loc[n <= 2, "ctx_std"] = np.nan

    # --- prev / next prediction (temporal neighbours within patient) ---
    work["ctx_prev_pred"] = grp["pred"].shift(1)
    work["ctx_next_pred"] = grp["pred"].shift(-1)

    # --- LOO OLS slope/intercept via sum identities ---
    # Full-patient sums for OLS
    pat_n = n
    pat_sum_a = grp["age"].transform("sum")
    pat_sum_p = pat_sum
    pat_sum_a2 = grp["age"].transform(lambda x: (x**2).sum())
    # cross-product sum: sum(age_i * pred_i) for each patient, broadcast to all rows
    work["_ap"] = work["age"] * work["pred"]
    pat_sum_ap = grp["_ap"].transform("sum")

    # LOO sums (remove current row)
    loo_n = pat_n - 1
    loo_sum_a = pat_sum_a - work["age"]
    loo_sum_p = pat_sum - work["pred"]
    loo_sum_a2 = pat_sum_a2 - work["age"] ** 2
    loo_sum_ap = pat_sum_ap - work["_ap"]

    denom = loo_n * loo_sum_a2 - loo_sum_a ** 2
    has_enough = (loo_n >= 2) & (denom.abs() > 1e-10)
    loo_slope = np.where(has_enough, (loo_n * loo_sum_ap - loo_sum_a * loo_sum_p) / denom, np.nan)
    loo_intercept = np.where(
        has_enough,
        (loo_sum_p - loo_slope * loo_sum_a) / loo_n,
        np.nan,
    )
    work["ctx_loo_at_age"] = np.where(
        has_enough, loo_intercept + loo_slope * work["age"].values, np.nan
    )

    # ----------------------------------------------------------------
    # Restore original index order and build output frame
    # ----------------------------------------------------------------
    work = work.drop(columns=["_ap"]).sort_index()

    # Pass-through features from X_raw
    off_missing = X_raw["off"].isna().astype(float)
    on_missing = X_raw["on"].isna().astype(float)
    ledd_missing = X_raw["ledd"].isna().astype(float) if "ledd" in X_raw.columns else pd.Series(0.0, index=idx)

    feat = pd.DataFrame(
        {
            "stage1_pred": stage1_preds,
            "ctx_mean": work["ctx_mean"].values,
            "ctx_std": work["ctx_std"].values,
            "ctx_prev_pred": work["ctx_prev_pred"].values,
            "ctx_next_pred": work["ctx_next_pred"].values,
            "ctx_loo_at_age": work["ctx_loo_at_age"].values,
            "ctx_n_visits": work["ctx_n_visits"].values,
            "off": X_raw["off"].values,
            "on": X_raw["on"].values,
            "age": X_raw["age"].values,
            "off_missing": off_missing.values,
            "on_missing": on_missing.values,
            "ledd_missing": ledd_missing.values,
        },
        index=idx,
    )
    return feat


class PatientStackLearner(BaseEstimator, RegressorMixin):
    """Two-stage patient-context stacker for 07_patient_stack.

    Stage 1 is the 05_pk_rater pipeline (PK-decay + rater-bias +
    trajectory features + TableVectorizer + HGBR with ``HGBR_PARAMS_04``).
    Stage 2 corrects each visit's prediction using the stage-1 predictions
    of the *other* visits of the same patient, capturing the slowly-evolving
    true OFF score signal that is otherwise lost in visit-level noise.

    Fit protocol
    ------------
    1. Compute out-of-fold stage-1 predictions on the training rows via an
       inner ``GroupKFold(3)`` on ``patient_id`` (no target leakage).
    2. Refit stage 1 on all training rows.
    3. Build stage-2 features per visit using the OOF predictions of the
       patient's *other* visits (mean, std, prev/next, LOO-OLS, n_visits),
       plus the visit's own stage-1 OOF prediction, ``off``, ``on``, ``age``,
       and missingness flags.
    4. Fit stage 2 (a small HGBR) on these features against the target.

    Predict protocol
    ----------------
    Apply stage-1, build the same context features within each test patient,
    then apply stage 2.

    Constraints enforced
    --------------------
    * ``patient_id`` is never a model input; it is used only for grouping.
    * Stage 2 is trained exclusively on out-of-fold stage-1 predictions.
    * The target is never used to build any feature.

    Parameters
    ----------
    hgbr_params_s1 :
        HGBR hyperparameters for stage 1.  Defaults to ``HGBR_PARAMS_04``.
    hgbr_params_s2 :
        HGBR hyperparameters for stage 2.  Defaults to a small fast config.
    inner_cv_splits :
        Number of splits for the inner GroupKFold used to generate OOF
        stage-1 predictions (default 3).
    n_jobs :
        Passed to stage-2 HGBR (default −1).
    """

    def __init__(
        self,
        hgbr_params_s1: "dict | None" = None,
        hgbr_params_s2: "dict | None" = None,
        inner_cv_splits: int = 3,
        n_jobs: int = -1,
    ) -> None:
        self.hgbr_params_s1 = hgbr_params_s1
        self.hgbr_params_s2 = hgbr_params_s2
        self.inner_cv_splits = inner_cv_splits
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_stage1(self) -> "Pipeline":
        """Return a fresh, unfit 05_pk_rater pipeline."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import FunctionTransformer

        params = self.hgbr_params_s1 if self.hgbr_params_s1 is not None else HGBR_PARAMS_04
        feature_step = FunctionTransformer(_add_all_features_05)
        drop_step = FunctionTransformer(_drop_patient_id_05)
        tabular = skrub.tabular_pipeline("regressor")
        hgbr = HistGradientBoostingRegressor(**params)
        tabular_steps_no_last = tabular.steps[:-1]
        return Pipeline(
            steps=[
                ("add_features", feature_step),
                ("drop_patient_id", drop_step),
                *tabular_steps_no_last,
                ("histgradientboostingregressor", hgbr),
            ]
        )

    def _make_stage2(self) -> "HistGradientBoostingRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        # HGBR is internally parallelised via OpenMP; n_jobs is not a valid param.
        default = {
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "learning_rate": 0.05,
            "min_samples_leaf": 20,
            "random_state": 0,
        }
        params = self.hgbr_params_s2 if self.hgbr_params_s2 is not None else default
        return HistGradientBoostingRegressor(**params)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PatientStackLearner":
        import numpy as np
        from sklearn.model_selection import GroupKFold

        groups = X["patient_id"]
        gkf = GroupKFold(n_splits=self.inner_cv_splits)

        # --- Step 1: OOF stage-1 predictions ---
        oof_preds = np.full(len(y), np.nan)
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            s1_fold = self._make_stage1()
            s1_fold.fit(X_tr, y_tr)
            oof_preds[val_idx] = s1_fold.predict(X_val)

        # --- Step 2: Refit stage 1 on all training rows ---
        self._stage1 = self._make_stage1()
        self._stage1.fit(X, y)

        # --- Step 3: Build stage-2 features from OOF predictions ---
        feat2 = _build_stage2_features(
            patient_ids=groups,
            ages=X["age"],
            stage1_preds=oof_preds,
            X_raw=X,
        )

        # --- Step 4: Fit stage 2 ---
        self._stage2 = self._make_stage2()
        self._stage2.fit(feat2, y)

        return self

    def predict(self, X: pd.DataFrame) -> "np.ndarray":
        groups = X["patient_id"]
        s1_preds = self._stage1.predict(X)
        feat2 = _build_stage2_features(
            patient_ids=groups,
            ages=X["age"],
            stage1_preds=s1_preds,
            X_raw=X,
        )
        return self._stage2.predict(feat2)

    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {
            "hgbr_params_s1": self.hgbr_params_s1,
            "hgbr_params_s2": self.hgbr_params_s2,
            "inner_cv_splits": self.inner_cv_splits,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params: object) -> "PatientStackLearner":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_patient_stack_learner(
    hgbr_params_s1: "dict | None" = None,
    hgbr_params_s2: "dict | None" = None,
    inner_cv_splits: int = 3,
    n_jobs: int = -1,
    data_dir_preview: "str | Path | None" = None,  # noqa: ARG001
) -> "PatientStackLearner":
    """Return an unfit two-stage patient-context stacker for 07_patient_stack.

    Parameters
    ----------
    hgbr_params_s1 :
        HGBR hyperparameters for stage 1.  Defaults to ``HGBR_PARAMS_04``.
    hgbr_params_s2 :
        HGBR hyperparameters for stage 2.  Defaults to a small fast HGBR.
    inner_cv_splits :
        Inner GroupKFold splits for OOF stage-1 generation (default 3).
    n_jobs :
        Parallelism for stage-2 HGBR (default −1 = all cores).
    data_dir_preview :
        Unused.  Accepted for API consistency.

    Returns
    -------
    PatientStackLearner
        Sklearn-compatible estimator implementing fit / predict.
    """
    return PatientStackLearner(
        hgbr_params_s1=hgbr_params_s1,
        hgbr_params_s2=hgbr_params_s2,
        inner_cv_splits=inner_cv_splits,
        n_jobs=n_jobs,
    )
