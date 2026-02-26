"""
Stage 1 — LightGBM Regressors
================================
Trains one LightGBM regressor per continuous target with:
  - 5-fold cross-validation (stratified by yield_pct quartile)
  - Early stopping on a hold-out validation split
  - Logs RMSE and R² per fold and on the final test set
  - Serialises each trained model to ``ml/artifacts/<target>_lgbm.pkl``

Targets:
    yield_pct, steel_output_tons, energy_cost_usd,
    production_cost_usd, scrap_rate_pct

Usage:
    from ml.models.lgbm_regressors import train_all_regressors, load_regressor

    results = train_all_regressors(df_features, feature_cols, artifact_dir)
    model   = load_regressor("yield_pct", artifact_dir)
    preds   = model.predict(X_new)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from ml.feature_engineering import REGRESSION_TARGETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightGBM hyper-parameters (shared base; tuned per target via early stopping)
# ---------------------------------------------------------------------------
_BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": ["rmse"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

_N_ESTIMATORS = 2000          # upper bound; early stopping halts training
_EARLY_STOPPING_ROUNDS = 50
_N_FOLDS = 5
_TEST_SIZE = 0.15             # held-out final test fraction
_VALID_SIZE = 0.15            # inner validation for early stopping (of train)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_all_regressors(
    df: pd.DataFrame,
    feature_cols: list[str],
    artifact_dir: str | Path = "ml/artifacts",
) -> dict[str, dict[str, Any]]:
    """Train one LightGBM regressor per regression target.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame (output of ``build_features``).
    feature_cols:
        Ordered list of input feature column names.
    artifact_dir:
        Directory to save serialised models (``<target>_lgbm.pkl``).

    Returns
    -------
    dict
        Keys are target names.  Each value is a sub-dict::

            {
                "model"       : fitted LGBMRegressor,
                "cv_rmse"     : [fold RMSE × N_FOLDS],
                "cv_r2"       : [fold R²   × N_FOLDS],
                "mean_cv_rmse": float,
                "std_cv_rmse" : float,
                "mean_cv_r2"  : float,
                "test_rmse"   : float,
                "test_r2"     : float,
                "best_iteration": int,
                "feature_importances": pd.Series,
            }
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    for target in REGRESSION_TARGETS:
        if target not in df.columns:
            logger.warning("Target '%s' not found in DataFrame — skipping.", target)
            continue

        logger.info("=" * 60)
        logger.info("Training LightGBM regressor for: %s", target)
        logger.info("=" * 60)

        result = _train_single_regressor(df, feature_cols, target, artifact_dir)
        results[target] = result

        logger.info(
            "%s | Test RMSE=%.4f | Test R²=%.4f | CV RMSE=%.4f±%.4f",
            target,
            result["test_rmse"],
            result["test_r2"],
            result["mean_cv_rmse"],
            result["std_cv_rmse"],
        )

    return results


def load_regressor(target: str, artifact_dir: str | Path = "ml/artifacts") -> lgb.LGBMRegressor:
    """Load a serialised LGBMRegressor from disk.

    Parameters
    ----------
    target:
        One of the five regression target names.
    artifact_dir:
        Directory where ``<target>_lgbm.pkl`` is stored.

    Returns
    -------
    lgb.LGBMRegressor
    """
    path = Path(artifact_dir) / f"{target}_lgbm.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for '{target}' at '{path}'. "
            "Run train_all_regressors() first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _train_single_regressor(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """End-to-end training for a single target."""
    # --- Drop rows missing the target ---
    valid = df[feature_cols + [target]].dropna()
    X = valid[feature_cols].values.astype(np.float32)
    y = valid[target].values.astype(np.float64)

    # --- Final train/test split (stratified by quartile) ---
    quartiles = pd.qcut(y, q=4, labels=False, duplicates="drop")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=42, stratify=quartiles
    )

    # --- 5-fold CV on train+val portion ---
    cv_rmse: list[float] = []
    cv_r2: list[float] = []
    kf = KFold(n_splits=_N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_trainval), start=1):
        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

        fold_model = _build_lgbm()
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        preds = fold_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2 = float(r2_score(y_val, preds))
        cv_rmse.append(rmse)
        cv_r2.append(r2)
        logger.debug("  Fold %d/%d — RMSE=%.4f  R²=%.4f", fold, _N_FOLDS, rmse, r2)

    # --- Final model trained on full trainval set ---
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_trainval, y_trainval, test_size=_VALID_SIZE, random_state=0
    )
    final_model = _build_lgbm()
    final_model.fit(
        X_tr_final, y_tr_final,
        eval_set=[(X_val_final, y_val_final)],
        callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(period=-1)],
        feature_name=feature_cols,
    )

    # --- Test-set evaluation ---
    test_preds = final_model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    test_r2 = float(r2_score(y_test, test_preds))

    # --- Feature importance (gain-based) ---
    importance = pd.Series(
        final_model.feature_importances_,
        index=feature_cols,
        name=target,
    ).sort_values(ascending=False)

    # --- Persist ---
    artifact_path = artifact_dir / f"{target}_lgbm.pkl"
    joblib.dump(final_model, artifact_path)
    logger.info("  Saved model → %s", artifact_path)

    return {
        "model": final_model,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "mean_cv_rmse": float(np.mean(cv_rmse)),
        "std_cv_rmse": float(np.std(cv_rmse)),
        "mean_cv_r2": float(np.mean(cv_r2)),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_iteration": int(final_model.best_iteration_),
        "feature_importances": importance,
        "feature_cols": feature_cols,
        "target": target,
    }


def _build_lgbm() -> lgb.LGBMRegressor:
    """Instantiate a fresh LGBMRegressor with base params."""
    return lgb.LGBMRegressor(n_estimators=_N_ESTIMATORS, **_BASE_PARAMS)
