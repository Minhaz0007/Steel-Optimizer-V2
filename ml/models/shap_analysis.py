"""
Stage 1 — SHAP Analysis
=========================
Generates SHAP explanations for all Stage 1 LightGBM regressors and
CatBoost classifiers.

Outputs per model:
    - SHAP summary plot (``ml/artifacts/shap_<target>_summary.png``)
    - Ranked feature importance table separating controllable vs uncontrollable
      features (``ml/artifacts/shap_importance_table.csv``)

Usage:
    from ml.models.shap_analysis import run_shap_analysis

    tables = run_shap_analysis(
        regressor_results,
        classifier_results,
        df_features,
        feature_cols,
        controllable_cols,
        artifact_dir="ml/artifacts",
    )
    # tables["yield_pct"] → pd.DataFrame with SHAP importance per feature
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml.feature_engineering import CONTROLLABLE_VARS

logger = logging.getLogger(__name__)

_SHAP_SAMPLE_SIZE = 500   # cap background / explanation set for speed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_shap_analysis(
    regressor_results: dict[str, dict[str, Any]],
    classifier_results: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    controllable_cols: list[str] | None = None,
    artifact_dir: str | Path = "ml/artifacts",
) -> dict[str, pd.DataFrame]:
    """Compute SHAP values and produce plots + importance tables.

    Parameters
    ----------
    regressor_results:
        Output of ``train_all_regressors()`` — dict keyed by target name.
    classifier_results:
        Output of ``train_all_classifiers()`` — dict keyed by target name.
    df:
        Feature-engineered DataFrame (used to draw an explanation sample).
    feature_cols:
        Full ordered list of feature column names passed to every model.
    controllable_cols:
        Subset of ``feature_cols`` that are controllable.  Defaults to
        ``ml.feature_engineering.CONTROLLABLE_VARS``.
    artifact_dir:
        Directory to save plots and CSVs.

    Returns
    -------
    dict
        Keys are target names.  Each value is a DataFrame with columns
        ``["feature", "mean_abs_shap", "category"]`` sorted by importance.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if controllable_cols is None:
        controllable_cols = [c for c in CONTROLLABLE_VARS if c in feature_cols]

    # Sample a subset of the data for SHAP computation
    sample_df = df[feature_cols].dropna().sample(
        n=min(_SHAP_SAMPLE_SIZE, len(df)), random_state=42
    )
    X_sample = sample_df.values.astype(np.float32)

    importance_tables: dict[str, pd.DataFrame] = {}
    all_model_rows: list[dict] = []

    # ---- Regressors ----
    for target, res in regressor_results.items():
        model = res["model"]
        table = _shap_for_lgbm(
            model, X_sample, feature_cols, controllable_cols, target, artifact_dir
        )
        importance_tables[target] = table
        for _, row in table.iterrows():
            all_model_rows.append({"model": target, **row.to_dict()})

    # ---- Classifiers ----
    for target, res in classifier_results.items():
        model = res["model"]
        table = _shap_for_catboost(
            model, X_sample, feature_cols, controllable_cols, target, artifact_dir
        )
        importance_tables[target] = table
        for _, row in table.iterrows():
            all_model_rows.append({"model": target, **row.to_dict()})

    # ---- Combined importance CSV ----
    combined_df = pd.DataFrame(all_model_rows)
    combined_path = artifact_dir / "shap_importance_table.csv"
    combined_df.to_csv(combined_path, index=False)
    logger.info("SHAP combined importance table → %s", combined_path)

    return importance_tables


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _shap_for_lgbm(
    model: Any,
    X_sample: np.ndarray,
    feature_cols: list[str],
    controllable_cols: list[str],
    target: str,
    artifact_dir: Path,
) -> pd.DataFrame:
    """Generate SHAP values and summary plot for a LightGBM regressor."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    _save_summary_plot(shap_values, X_sample, feature_cols, target, artifact_dir)

    # Importance table
    return _build_importance_table(shap_values, feature_cols, controllable_cols)


def _shap_for_catboost(
    model: Any,
    X_sample: np.ndarray,
    feature_cols: list[str],
    controllable_cols: list[str],
    target: str,
    artifact_dir: Path,
) -> pd.DataFrame:
    """Generate SHAP values and summary plot for a CatBoost classifier."""
    # CatBoost TreeExplainer returns shape (n, features, 2) for binary
    # classification — take the positive-class slice.
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)
    if isinstance(raw, list):
        # SHAP returns list of arrays for multi-class; index 1 = positive class
        shap_values = raw[1] if len(raw) > 1 else raw[0]
    elif raw.ndim == 3:
        shap_values = raw[:, :, 1]
    else:
        shap_values = raw

    _save_summary_plot(shap_values, X_sample, feature_cols, target, artifact_dir)
    return _build_importance_table(shap_values, feature_cols, controllable_cols)


def _save_summary_plot(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_cols: list[str],
    target: str,
    artifact_dir: Path,
) -> None:
    """Render and save a SHAP beeswarm / dot summary plot."""
    plt.figure(figsize=(10, max(6, len(feature_cols) * 0.3)))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_cols,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title(f"SHAP Feature Importance — {target}", fontsize=13, pad=12)
    plt.tight_layout()
    plot_path = artifact_dir / f"shap_{target}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  SHAP summary plot → %s", plot_path)


def _build_importance_table(
    shap_values: np.ndarray,
    feature_cols: list[str],
    controllable_cols: list[str],
) -> pd.DataFrame:
    """Build a DataFrame ranking features by mean |SHAP| value.

    Returns a DataFrame with columns:
        feature, mean_abs_shap, category  (Controllable | Context | Engineered)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    controllable_set = set(controllable_cols)

    rows = []
    for feat, imp in zip(feature_cols, mean_abs):
        if feat in controllable_set:
            category = "Controllable"
        elif feat.endswith(("_roll3", "_lag1")) or feat in (
            "shift_sin", "shift_cos", "day_of_week", "month", "week_of_year",
            "grade_change_x_scrap", "shifts_since_maintenance",
        ):
            category = "Engineered"
        else:
            category = "Context"
        rows.append({"feature": feat, "mean_abs_shap": float(imp), "category": category})

    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    table.reset_index(drop=True, inplace=True)
    return table
