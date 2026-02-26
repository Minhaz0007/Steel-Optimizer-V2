"""
Stage 4 — Temporal Forecasting
================================
Forecasts next-shift ``yield_pct`` and ``energy_cost_usd`` given known
future inputs (planned grade, runtime, scheduled maintenance, etc.).

Approach:
    LightGBM regressor with explicit lag and rolling features constructed
    from the historical DataFrame.  This avoids the heavy PyTorch /
    pytorch-forecasting dependency while still capturing temporal
    patterns effectively.

Forecast horizon: 1 shift ahead (next shift)

Additional lag features used here (beyond feature_engineering.py):
    - yield_pct_lag{1,2,3}
    - energy_cost_usd_lag{1,2,3}
    - scrap_rate_pct_lag{1,2}
    - unplanned_downtime_minutes_lag{1}
    - yield_pct_roll7  (7-shift rolling mean)
    - energy_cost_usd_roll7

Usage::

    from ml.models.forecasting import TemporalForecaster

    forecaster = TemporalForecaster.from_data(df_features, artifact_dir)
    preds = forecaster.forecast_next_shift(
        history_df=recent_10_shifts_df,
        planned_inputs={
            "planned_runtime_hours": 8.0,
            "num_furnaces_running": 3.0,
            ...
        },
    )
    # preds["yield_pct"]       → float (next-shift predicted yield)
    # preds["energy_cost_usd"] → float (next-shift predicted energy cost)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

_FORECAST_TARGETS: list[str] = ["yield_pct", "energy_cost_usd"]

# Lag depths per target
_LAG_DEPTHS: dict[str, list[int]] = {
    "yield_pct":                  [1, 2, 3],
    "energy_cost_usd":            [1, 2, 3],
    "scrap_rate_pct":             [1, 2],
    "unplanned_downtime_minutes": [1],
}

_ROLL_WINDOWS: dict[str, int] = {
    "yield_pct":       7,
    "energy_cost_usd": 7,
}

_N_SPLITS = 5          # TimeSeriesSplit folds
_N_ESTIMATORS = 2000
_EARLY_STOPPING = 50

_BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": ["rmse"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class TemporalForecaster:
    """LightGBM-based next-shift forecaster for yield and energy cost.

    Parameters
    ----------
    models:
        Dict mapping target name → fitted LGBMRegressor.
    feature_cols:
        Feature column names used for each target (same for both targets).
    """

    def __init__(
        self,
        models: dict[str, lgb.LGBMRegressor],
        feature_cols: list[str],
    ) -> None:
        self.models = models
        self.feature_cols = feature_cols

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        df: pd.DataFrame,
        artifact_dir: str | Path = "ml/artifacts",
    ) -> "TemporalForecaster":
        """Build temporal lag features, train forecasting models, and save.

        Parameters
        ----------
        df:
            Feature-engineered DataFrame (output of ``build_features``),
            already sorted chronologically.
        artifact_dir:
            Directory to persist serialised models.

        Returns
        -------
        TemporalForecaster
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        df_lagged = _build_temporal_features(df)
        feature_cols = _get_temporal_feature_cols(df_lagged)

        models: dict[str, lgb.LGBMRegressor] = {}
        cv_summaries: dict[str, dict] = {}

        for target in _FORECAST_TARGETS:
            if target not in df_lagged.columns:
                logger.warning("Forecast target '%s' not found — skipping.", target)
                continue

            model, cv_summary = _train_forecast_model(df_lagged, feature_cols, target)
            models[target] = model
            cv_summaries[target] = cv_summary

            # Persist
            path = artifact_dir / f"{target}_forecast_lgbm.pkl"
            joblib.dump(model, path)
            logger.info(
                "Forecaster '%s' → %s | CV RMSE=%.4f±%.4f | CV R²=%.4f",
                target, path,
                cv_summary["mean_rmse"], cv_summary["std_rmse"],
                cv_summary["mean_r2"],
            )

        # Save feature cols list for inference
        joblib.dump(feature_cols, artifact_dir / "forecast_feature_cols.pkl")

        return cls(models=models, feature_cols=feature_cols)

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path = "ml/artifacts",
    ) -> "TemporalForecaster":
        """Load a previously trained TemporalForecaster from disk."""
        artifact_dir = Path(artifact_dir)
        fc_path = artifact_dir / "forecast_feature_cols.pkl"
        if not fc_path.exists():
            raise FileNotFoundError(
                f"Forecast feature cols not found at '{fc_path}'. "
                "Run TemporalForecaster.from_data() first."
            )
        feature_cols = joblib.load(fc_path)

        models: dict[str, lgb.LGBMRegressor] = {}
        for target in _FORECAST_TARGETS:
            path = artifact_dir / f"{target}_forecast_lgbm.pkl"
            if path.exists():
                models[target] = joblib.load(path)

        return cls(models=models, feature_cols=feature_cols)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forecast_next_shift(
        self,
        history_df: pd.DataFrame,
        planned_inputs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Predict next-shift yield_pct and energy_cost_usd.

        Parameters
        ----------
        history_df:
            DataFrame of recent shifts (chronological order).  Must contain
            the raw target columns so that lag features can be computed.
            Typically the last ~10 shifts are sufficient.
        planned_inputs:
            Dict of known-future controllable or context values for the
            next shift (e.g., ``planned_runtime_hours``, ``num_furnaces_running``).

        Returns
        -------
        dict mapping target → float prediction
        """
        df_lagged = _build_temporal_features(history_df)
        if len(df_lagged) == 0:
            raise ValueError("history_df is too short to compute lag features.")

        # Use the most recent row as the feature vector for next-shift prediction
        last_row = df_lagged.iloc[[-1]].copy()

        # Overlay planned inputs if provided
        if planned_inputs:
            for col, val in planned_inputs.items():
                if col in last_row.columns:
                    last_row[col] = float(val)

        preds: dict[str, float] = {}
        for target, model in self.models.items():
            available_cols = [c for c in self.feature_cols if c in last_row.columns]
            row_vec = last_row[available_cols].fillna(0.0).values.astype(np.float32)
            # Pad/reorder to match training feature cols
            full_vec = np.zeros((1, len(self.feature_cols)), dtype=np.float32)
            col_idx = {c: i for i, c in enumerate(self.feature_cols)}
            for i, c in enumerate(available_cols):
                if c in col_idx:
                    full_vec[0, col_idx[c]] = row_vec[0, i]
            preds[target] = float(model.predict(full_vec)[0])

        return preds


# ---------------------------------------------------------------------------
# Feature engineering helpers (temporal-specific)
# ---------------------------------------------------------------------------


def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features needed for temporal forecasting."""
    df = df.copy()

    # Lag features
    for col, lags in _LAG_DEPTHS.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling-mean features
    for col, window in _ROLL_WINDOWS.items():
        if col not in df.columns:
            continue
        df[f"{col}_roll{window}"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )

    # Drop rows with NaN in lag columns (need full history)
    lag_cols = [
        f"{col}_lag{lag}"
        for col, lags in _LAG_DEPTHS.items()
        for lag in lags
        if col in df.columns
    ]
    existing_lag_cols = [c for c in lag_cols if c in df.columns]
    if existing_lag_cols:
        df = df.dropna(subset=existing_lag_cols).reset_index(drop=True)

    return df


def _get_temporal_feature_cols(df: pd.DataFrame) -> list[str]:
    """Collect all feature columns relevant for temporal forecasting."""
    # Controllable inputs present in df
    from ml.feature_engineering import CONTROLLABLE_VARS, CONTEXT_VARS
    controllable = [c for c in CONTROLLABLE_VARS if c in df.columns]
    context      = [c for c in CONTEXT_VARS if c in df.columns]

    # Engineered temporal features
    temporal = [
        c for c in df.columns
        if (c.endswith(tuple(f"_lag{k}" for k in range(1, 8)))
            or c.endswith("_roll3") or c.endswith("_roll7")
            or c in ("shift_sin", "shift_cos",
                     "day_of_week", "month", "week_of_year",
                     "grade_change_x_scrap", "shifts_since_maintenance"))
        and c not in _FORECAST_TARGETS
    ]

    all_cols = controllable + context + temporal
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in all_cols:
        if c not in seen and c not in _FORECAST_TARGETS:
            seen.add(c)
            result.append(c)
    return result


def _train_forecast_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[lgb.LGBMRegressor, dict]:
    """Train one LightGBM forecaster with TimeSeriesSplit CV."""
    available = [c for c in feature_cols if c in df.columns]
    valid = df[available + [target]].dropna()

    X = valid[available].values.astype(np.float32)
    y = valid[target].values.astype(np.float64)

    tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
    cv_rmse: list[float] = []
    cv_r2:   list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        m = lgb.LGBMRegressor(n_estimators=_N_ESTIMATORS, **_BASE_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        preds = m.predict(X_val)
        cv_rmse.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        cv_r2.append(float(r2_score(y_val, preds)))
        logger.debug(
            "  TS Fold %d — RMSE=%.4f  R²=%.4f", fold, cv_rmse[-1], cv_r2[-1]
        )

    # Retrain on full data
    # Use last 15% as hold-out for early stopping
    split = max(1, int(len(X) * 0.85))
    final_model = lgb.LGBMRegressor(n_estimators=_N_ESTIMATORS, **_BASE_PARAMS)
    final_model.fit(
        X[:split], y[:split],
        eval_set=[(X[split:], y[split:])],
        callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(period=-1)],
        feature_name=available,
    )

    return final_model, {
        "mean_rmse": float(np.mean(cv_rmse)),
        "std_rmse":  float(np.std(cv_rmse)),
        "mean_r2":   float(np.mean(cv_r2)),
    }
