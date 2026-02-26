"""
Model Evaluation & Reporting
==============================
Generates a side-by-side comparison table of all Stage 1 model performance
metrics and prints/saves a structured evaluation report.

Outputs:
    - ``ml/artifacts/evaluation_report.csv``     — machine-readable metrics
    - ``ml/artifacts/evaluation_report.txt``     — human-readable summary
    - ``ml/artifacts/shap_importance_table.csv`` — written by shap_analysis.py

Usage::

    from ml.evaluation import generate_evaluation_report

    report_df = generate_evaluation_report(
        regressor_results,
        classifier_results,
        output_dir="ml/artifacts",
    )
    print(report_df.to_string(index=False))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_evaluation_report(
    regressor_results: dict[str, dict[str, Any]],
    classifier_results: dict[str, dict[str, Any]],
    shap_tables: dict[str, pd.DataFrame] | None = None,
    output_dir: str | Path = "ml/artifacts",
) -> pd.DataFrame:
    """Compile a unified evaluation DataFrame and save to disk.

    Parameters
    ----------
    regressor_results:
        Output of ``train_all_regressors()``.
    classifier_results:
        Output of ``train_all_classifiers()``.
    shap_tables:
        Output of ``run_shap_analysis()`` — optional; used for the
        top-5 feature section of the text report.
    output_dir:
        Directory where CSV and TXT reports are written.

    Returns
    -------
    pd.DataFrame
        One row per model with all evaluation metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # ---- Regression models ----
    for target, res in regressor_results.items():
        rows.append({
            "model_type": "LightGBM Regressor",
            "target": target,
            "task": "regression",
            # Test metrics
            "test_rmse": round(res.get("test_rmse", float("nan")), 4),
            "test_r2":   round(res.get("test_r2",   float("nan")), 4),
            # CV metrics
            "cv_rmse_mean": round(res.get("mean_cv_rmse", float("nan")), 4),
            "cv_rmse_std":  round(res.get("std_cv_rmse",  float("nan")), 4),
            "cv_r2_mean":   round(res.get("mean_cv_r2",   float("nan")), 4),
            # Training info
            "best_iteration": res.get("best_iteration", None),
            # Classification-specific (N/A for regressors)
            "test_f1":        None,
            "test_auc":       None,
            "test_precision": None,
            "test_recall":    None,
            "cv_f1_mean":     None,
            "cv_auc_mean":    None,
        })

    # ---- Classification models ----
    for target, res in classifier_results.items():
        rows.append({
            "model_type": "CatBoost Classifier",
            "target": target,
            "task": "classification",
            # Test metrics
            "test_rmse": None,
            "test_r2":   None,
            "cv_rmse_mean": None,
            "cv_rmse_std":  None,
            "cv_r2_mean":   None,
            "best_iteration": None,
            # Classification metrics
            "test_f1":        round(res.get("test_f1",        float("nan")), 4),
            "test_auc":       round(res.get("test_auc",       float("nan")), 4),
            "test_precision": round(res.get("test_precision", float("nan")), 4),
            "test_recall":    round(res.get("test_recall",    float("nan")), 4),
            "cv_f1_mean":     round(res.get("mean_cv_f1",     float("nan")), 4),
            "cv_auc_mean":    round(res.get("mean_cv_auc",    float("nan")), 4),
        })

    report_df = pd.DataFrame(rows)

    # ---- Save CSV ----
    csv_path = output_dir / "evaluation_report.csv"
    report_df.to_csv(csv_path, index=False)
    logger.info("Evaluation CSV → %s", csv_path)

    # ---- Save text report ----
    txt_path = output_dir / "evaluation_report.txt"
    _write_text_report(report_df, shap_tables, txt_path)

    return report_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _write_text_report(
    report_df: pd.DataFrame,
    shap_tables: dict[str, pd.DataFrame] | None,
    path: Path,
) -> None:
    """Write a formatted plain-text evaluation report."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  STEEL PLANT ML OPTIMIZATION — MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # ---- Stage 1a: Regression ----
    lines.append("STAGE 1A  —  LightGBM Regressors")
    lines.append("-" * 70)
    reg_df = report_df[report_df["task"] == "regression"].copy()
    header = (
        f"{'Target':<30} {'Test RMSE':>10} {'Test R²':>9} "
        f"{'CV RMSE':>10} {'±':>6} {'Best Iter':>10}"
    )
    lines.append(header)
    lines.append("-" * 70)
    for _, row in reg_df.iterrows():
        lines.append(
            f"{row['target']:<30} {_fmt(row['test_rmse']):>10} "
            f"{_fmt(row['test_r2']):>9} {_fmt(row['cv_rmse_mean']):>10} "
            f"{_fmt(row['cv_rmse_std']):>6} {_fmt(row['best_iteration'], '.0f'):>10}"
        )
    lines.append("")

    # ---- Stage 1b: Classification ----
    lines.append("STAGE 1B  —  CatBoost Classifiers")
    lines.append("-" * 70)
    clf_df = report_df[report_df["task"] == "classification"].copy()
    header2 = (
        f"{'Target':<30} {'F1':>8} {'AUC':>8} "
        f"{'Precision':>10} {'Recall':>8} {'CV F1':>8} {'CV AUC':>8}"
    )
    lines.append(header2)
    lines.append("-" * 70)
    for _, row in clf_df.iterrows():
        lines.append(
            f"{row['target']:<30} {_fmt(row['test_f1']):>8} "
            f"{_fmt(row['test_auc']):>8} {_fmt(row['test_precision']):>10} "
            f"{_fmt(row['test_recall']):>8} {_fmt(row['cv_f1_mean']):>8} "
            f"{_fmt(row['cv_auc_mean']):>8}"
        )
    lines.append("")

    # ---- SHAP feature importance per model ----
    if shap_tables:
        lines.append("SHAP FEATURE IMPORTANCE  —  Top 10 per Model")
        lines.append("=" * 70)
        for target, table in shap_tables.items():
            lines.append(f"\n  Model: {target}")
            lines.append(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12} {'Category':>15}")
            lines.append("  " + "-" * 70)
            for rank, (_, row) in enumerate(table.head(10).iterrows(), start=1):
                lines.append(
                    f"  {rank:<6} {row['feature']:<35} "
                    f"{row['mean_abs_shap']:>12.4f} {row['category']:>15}"
                )

        # ---- Split: controllable vs context ----
        lines.append("\n\nSHAP IMPORTANCE  —  Controllable vs Context Features")
        lines.append("=" * 70)
        for target, table in shap_tables.items():
            ctrl = table[table["category"] == "Controllable"]
            ctx  = table[table["category"] == "Context"]
            if len(ctrl) == 0 and len(ctx) == 0:
                continue
            lines.append(f"\n  Model: {target}")
            lines.append(f"  {'Type':<15} {'Feature':<35} {'Mean |SHAP|':>12}")
            lines.append("  " + "-" * 65)
            for _, row in ctrl.head(5).iterrows():
                lines.append(
                    f"  {'Controllable':<15} {row['feature']:<35} "
                    f"{row['mean_abs_shap']:>12.4f}"
                )
            for _, row in ctx.head(5).iterrows():
                lines.append(
                    f"  {'Context':<15} {row['feature']:<35} "
                    f"{row['mean_abs_shap']:>12.4f}"
                )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Evaluation report → %s", path)
    print("\n".join(lines))


def _fmt(val: Any, fmt: str = ".4f") -> str:
    """Format a possibly-None value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)
