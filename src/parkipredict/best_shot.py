"""Feature engineering and learner builders for 08_best_shot.

All features are computed on a stacked (train + test) X DataFrame so that
rater bias and per-patient statistics use the full patient history.
patient_id is used only for grouping and is dropped before the estimator.
rater_id is kept as a categorical feature for TableVectorizer.

``add_best_shot_features`` — full feature pipeline (blocks 1 + 2 below).
``build_best_shot_learner``  — Pipeline: features → drop_patient_id → TV → HGBR.
``build_best_shot_ensemble`` — 5-seed average of build_best_shot_learner.

Block 1 — Rater bias (per-rater mean deviation from patient mean):
  For off, on, ledd:
    dev = value − patient_mean(col)
    rater_bias_<col> = per-rater mean(dev)
    adjusted_<col>   = value − rater_bias_<col>
  For adjusted_off and adjusted_on: patient mean/min/max,
  previous and next adjusted values (1 and 2 steps by age), and
  linear interpolation between previous and next adjusted values.

Block 2 — Age-weighted interpolation and neighbours for off and on:
  interp_off / interp_on: for each visit, interpolate the column at the
    visit's age between the previous and next *other* observed values,
    weighted inversely by age distance (the visit's own value excluded).
  age_dist_nearest_off: age distance to the nearest observed off.
  n_observed_off: count of non-NaN off values per patient.
  local_slope_off / local_slope_on: (next_obs − prev_obs) / (age_next − age_prev).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Reference hyperparameters (fixed, for the reference arm of 08_best_shot)
# ---------------------------------------------------------------------------

HGBR_PARAMS_REF: dict = {
    "learning_rate": 0.05,
    "max_iter": 600,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 0,
}


# ---------------------------------------------------------------------------
# Feature engineering helpers (stateless, operates on X only)
# ---------------------------------------------------------------------------


def _patient_sorted(X: pd.DataFrame) -> pd.DataFrame:
    """Return X sorted by (patient_id, age); original index unchanged.

    Sorting by age within each patient is required before any shift-based
    temporal feature so that prev/next always means chronologically earlier/later.
    """
    return X.sort_values(["patient_id", "age"])


def _adjusted_neighbours(
    X: pd.DataFrame,
    col: str,
    adj_col: str,
) -> pd.DataFrame:
    """Add patient-mean, min, max, prev/next (1 and 2 steps) and linear interpolation
    for ``adj_col`` within each patient sorted by age.

    Columns added:
      {adj_col}_patient_mean
      {adj_col}_patient_min
      {adj_col}_patient_max
      {adj_col}_prev1, {adj_col}_prev2
      {adj_col}_next1, {adj_col}_next2
      {adj_col}_interp  (linear interp between prev1 and next1 adjusted)
    """
    grp = X.groupby("patient_id", sort=False)

    X[f"{adj_col}_patient_mean"] = grp[adj_col].transform("mean")
    X[f"{adj_col}_patient_min"] = grp[adj_col].transform("min")
    X[f"{adj_col}_patient_max"] = grp[adj_col].transform("max")

    X[f"{adj_col}_prev1"] = grp[adj_col].shift(1)
    X[f"{adj_col}_prev2"] = grp[adj_col].shift(2)
    X[f"{adj_col}_next1"] = grp[adj_col].shift(-1)
    X[f"{adj_col}_next2"] = grp[adj_col].shift(-2)

    # Linear interpolation between prev1 and next1 (age-distance weighted midpoint).
    # w_prev = (age_next − age) / (age_next − age_prev): weight proportional to
    # proximity of the NEXT visit (far next → visit is closer to prev → more weight
    # on prev value).  Zero-range guard prevents division by zero for same-age visits.
    age_prev = grp["age"].shift(1)
    age_next = grp["age"].shift(-1)
    prev_val = X[f"{adj_col}_prev1"]
    next_val = X[f"{adj_col}_next1"]
    age_range = (age_next - age_prev).replace(0, np.nan)
    w_prev = (age_next - X["age"]) / age_range
    w_next = (X["age"] - age_prev) / age_range
    both_known = prev_val.notna() & next_val.notna()
    # NaN when either neighbour is missing; HGBR handles the NaN natively.
    interp = np.where(both_known, w_prev * prev_val + w_next * next_val, np.nan)
    X[f"{adj_col}_interp"] = interp

    return X


def _age_weighted_interp(
    ages: np.ndarray,
    vals: np.ndarray,
    i: int,
) -> float:
    """Interpolate val[i] from the nearest observed values *excluding* i.

    For each visit i, find the previous and next visits (by age) where the
    column is observed and i is excluded. Interpolate at ages[i] weighted
    by inverse age distance.

    Returns NaN if fewer than 2 other observed values exist.
    """
    obs_mask = ~np.isnan(vals)
    obs_mask_excl = obs_mask.copy()
    obs_mask_excl[i] = False
    obs_ages = ages[obs_mask_excl]
    obs_vals = vals[obs_mask_excl]

    if len(obs_ages) < 2:
        # Only one or zero other observed values — return that value or NaN
        if len(obs_ages) == 1:
            return float(obs_vals[0])
        return np.nan

    ai = ages[i]
    # Find prev (largest obs_age <= ai) and next (smallest obs_age >= ai)
    prev_mask = obs_ages <= ai
    next_mask = obs_ages >= ai

    if not prev_mask.any() or not next_mask.any():
        # Extrapolation: the visit falls outside the observed range for this
        # patient; return the nearest observed value rather than extrapolating
        # linearly, which would be unreliable with sparse observations.
        dists = np.abs(obs_ages - ai)
        return float(obs_vals[dists.argmin()])

    prev_idx = np.where(prev_mask)[0][-1]
    next_idx = np.where(next_mask)[0][0]

    a_prev, v_prev = obs_ages[prev_idx], obs_vals[prev_idx]
    a_next, v_next = obs_ages[next_idx], obs_vals[next_idx]

    if a_prev == a_next:
        return float(v_prev)

    w_next = (ai - a_prev) / (a_next - a_prev)
    w_prev = 1.0 - w_next
    return float(w_prev * v_prev + w_next * v_next)


def add_best_shot_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add all 08_best_shot features to X (X-only, no target).

    Must be called on the full stacked (train + test) X so that rater bias
    and patient statistics are computed on the complete patient history.
    The original index order is restored before returning.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with columns: patient_id, age, on, off, ledd,
        rater_id, and all other feature columns.  Must NOT contain target.

    Returns
    -------
    pd.DataFrame
        Copy of X with additional feature columns; original index preserved.
    """
    X = X.copy()
    orig_index = X.index.copy()

    # Sort by patient_id, age for all within-patient computations
    X = _patient_sorted(X)

    # ------------------------------------------------------------------
    # Block 1 — Rater bias correction
    # ------------------------------------------------------------------
    # For each column c ∈ {off, on, ledd}:
    #   dev_i = c_i − patient_mean(c)         # deviation from own patient average
    #   rater_bias_c = mean(dev_i) per rater  # systematic over/under-scoring
    #   adjusted_c   = c − rater_bias_c       # score with rater effect removed
    # This de-biases the raw scores before building temporal features so that
    # neighbour comparisons reflect disease state rather than rater tendency.
    pat_grp = X.groupby("patient_id", sort=False)

    for col in ("off", "on", "ledd"):
        patient_mean = pat_grp[col].transform("mean")
        dev = X[col] - patient_mean

        tmp = pd.DataFrame({"rater_id": X["rater_id"], "dev": dev})
        # Per-rater mean deviation, broadcast back to each row of that rater.
        rater_bias = tmp.groupby("rater_id")["dev"].transform("mean")

        X[f"rater_bias_{col}"] = rater_bias
        X[f"adjusted_{col}"] = X[col] - rater_bias

    # Extended neighbours and interpolation for adjusted_off and adjusted_on
    for col in ("adjusted_off", "adjusted_on"):
        X = _adjusted_neighbours(X, col=col, adj_col=col)

    # ------------------------------------------------------------------
    # Block 2 — Age-weighted interpolation and per-patient stats for off/on
    # ------------------------------------------------------------------
    interp_off_vals: dict[int, float] = {}
    interp_on_vals: dict[int, float] = {}
    age_dist_vals: dict[int, float] = {}
    n_obs_vals: dict[int, int] = {}
    slope_off_vals: dict[int, float] = {}
    slope_on_vals: dict[int, float] = {}

    for pid, grp_df in pat_grp:
        ages = grp_df["age"].values.astype(float)
        offs = grp_df["off"].values.astype(float)
        ons = grp_df["on"].values.astype(float)
        idx = grp_df.index.tolist()
        n = len(ages)

        obs_off_mask = ~np.isnan(offs)
        n_obs_off = int(obs_off_mask.sum())
        obs_off_ages = ages[obs_off_mask]

        for i in range(n):
            row_idx = idx[i]
            # interp_off and interp_on
            interp_off_vals[row_idx] = _age_weighted_interp(ages, offs, i)
            interp_on_vals[row_idx] = _age_weighted_interp(ages, ons, i)

            # age distance to nearest observed off (excluding self)
            obs_mask_excl = obs_off_mask.copy()
            obs_mask_excl[i] = False
            if obs_mask_excl.any():
                dists = np.abs(ages[obs_mask_excl] - ages[i])
                age_dist_vals[row_idx] = float(dists.min())
            elif obs_off_mask[i]:
                age_dist_vals[row_idx] = 0.0
            else:
                age_dist_vals[row_idx] = np.nan

            n_obs_vals[row_idx] = n_obs_off

            # local slope of off and on between prev and next observed values
            # (excluding the current visit's own value)
            def _local_slope(col_vals: np.ndarray, i: int) -> float:
                # Rate of change between the nearest observed values bracketing
                # visit i (excluding i itself), expressed in score units per year.
                # Used as a local disease-progression signal for the HGBR.
                obs_m = ~np.isnan(col_vals)
                obs_m_excl = obs_m.copy()
                obs_m_excl[i] = False  # exclude self to avoid circularity
                obs_a = ages[obs_m_excl]
                obs_v = col_vals[obs_m_excl]
                if len(obs_a) < 2:
                    return np.nan
                ai = ages[i]
                prev_m = obs_a <= ai
                next_m = obs_a >= ai
                if not prev_m.any() or not next_m.any():
                    # Visit is outside the observed span — no bracketing pair.
                    return np.nan
                p = np.where(prev_m)[0][-1]
                nx = np.where(next_m)[0][0]
                if obs_a[p] == obs_a[nx]:
                    # Same age for both brackets — slope undefined.
                    return np.nan
                return float((obs_v[nx] - obs_v[p]) / (obs_a[nx] - obs_a[p]))

            slope_off_vals[row_idx] = _local_slope(offs, i)
            slope_on_vals[row_idx] = _local_slope(ons, i)

    X["interp_off"] = pd.Series(interp_off_vals)
    X["interp_on"] = pd.Series(interp_on_vals)
    X["age_dist_nearest_off"] = pd.Series(age_dist_vals)
    X["n_observed_off"] = pd.Series(n_obs_vals)
    X["local_slope_off"] = pd.Series(slope_off_vals)
    X["local_slope_on"] = pd.Series(slope_on_vals)

    # Restore original index order
    X = X.loc[orig_index]
    return X


# ---------------------------------------------------------------------------
# Module-level helpers for pickle compatibility (FunctionTransformer)
# ---------------------------------------------------------------------------


def _drop_patient_id_08(X: pd.DataFrame) -> pd.DataFrame:
    """Drop patient_id before TableVectorizer (module-level for pickle)."""
    return X.drop(columns=["patient_id"], errors="ignore")


# ---------------------------------------------------------------------------
# Learner builders
# These builders operate on X that has already been featurized by
# add_best_shot_features (called outside CV on the full stacked dataset).
# The pipeline only needs to drop patient_id, then run TV + HGBR.
# ---------------------------------------------------------------------------


def build_best_shot_learner(hgbr_params: dict | None = None) -> Pipeline:
    """Return an unfit 08_best_shot pipeline for pre-featurized X.

    Assumes X has already been transformed by ``add_best_shot_features``
    on the full stacked (train + test) dataset before being passed to
    this learner.  The pipeline then:

        drop_patient_id   (FunctionTransformer)
          -> TableVectorizer   (from skrub.tabular_pipeline)
          -> HistGradientBoostingRegressor

    patient_id is dropped here so it is never a model input.
    rater_id is kept as-is for TV categorical encoding.

    Parameters
    ----------
    hgbr_params : dict | None
        HGBR hyperparameter dict.  Defaults to ``HGBR_PARAMS_REF``.

    Returns
    -------
    Pipeline
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import FunctionTransformer

    params = hgbr_params if hgbr_params is not None else HGBR_PARAMS_REF
    tabular = skrub.tabular_pipeline("regressor")
    tabular_steps_no_last = tabular.steps[:-1]

    return Pipeline(
        steps=[
            ("drop_patient_id", FunctionTransformer(_drop_patient_id_08)),
            *tabular_steps_no_last,
            ("histgradientboostingregressor", HistGradientBoostingRegressor(**params)),
        ]
    )


class _BestShotEnsemble(BaseEstimator, RegressorMixin):
    """Average predictions of ``n_members`` HGBR pipelines (seeds 0..n_members-1).

    Operates on pre-featurized X (after add_best_shot_features).

    Parameters
    ----------
    hgbr_params : dict
        Base HGBR params; random_state is overridden per member.
    n_members : int
        Number of ensemble members (default 5).
    """

    def __init__(self, hgbr_params: dict, n_members: int = 5) -> None:
        self.hgbr_params = hgbr_params
        self.n_members = n_members

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BestShotEnsemble":
        self._members = []
        for seed in range(self.n_members):
            params = {**self.hgbr_params, "random_state": seed}
            m = build_best_shot_learner(hgbr_params=params)
            m.fit(X, y)
            self._members.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self._members], axis=0)
        return preds.mean(axis=0)

    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {"hgbr_params": self.hgbr_params, "n_members": self.n_members}

    def set_params(self, **params: object) -> "_BestShotEnsemble":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_best_shot_ensemble(
    hgbr_params: dict | None = None,
    n_members: int = 5,
) -> "_BestShotEnsemble":
    """Return an unfit 5-seed ensemble of build_best_shot_learner.

    Parameters
    ----------
    hgbr_params : dict | None
        Base HGBR params.  Defaults to ``HGBR_PARAMS_REF``.
    n_members : int
        Number of ensemble members (default 5).

    Returns
    -------
    _BestShotEnsemble
    """
    return _BestShotEnsemble(
        hgbr_params=hgbr_params if hgbr_params is not None else HGBR_PARAMS_REF,
        n_members=n_members,
    )
