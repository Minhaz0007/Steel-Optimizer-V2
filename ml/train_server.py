"""
Subprocess Training Script — for server.ts integration
========================================================
Reads a CSV from ``argv[1]``, trains the full pipeline, and streams
JSON-line progress + a final result object to **stdout**.

All Python logging goes to **stderr** so stdout stays clean for JSON.

Usage (called by server.ts):
    python3 -m ml.train_server <csv_path> [artifact_dir]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback

import numpy as np

# All logging → stderr; stdout is reserved for JSON-line output
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit(obj: dict) -> None:
    """Write a single JSON line to stdout and flush immediately."""
    print(json.dumps(obj, default=_json_safe), flush=True)


def _progress(label: str, pct: int) -> None:
    _emit({"type": "progress", "label": label, "pct": pct})


def _json_safe(obj):
    """Convert numpy / pandas scalars that aren't JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _to_float_dict(series_or_dict, n: int = 15) -> dict[str, float]:
    """Convert a pandas Series or dict to a plain {str: float} dict (top-n)."""
    if hasattr(series_or_dict, "head"):
        items = series_or_dict.head(n).items()
    else:
        items = list(series_or_dict.items())[:n]
    return {str(k): float(v) for k, v in items}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        _emit({"type": "error", "message": "Usage: python3 -m ml.train_server <csv_path> [artifact_dir]"})
        sys.exit(1)

    data_path = sys.argv[1]
    artifact_dir = sys.argv[2] if len(sys.argv) > 2 else "ml/artifacts"

    if not os.path.exists(data_path):
        _emit({"type": "error", "message": f"Data file not found: '{data_path}'"})
        sys.exit(1)

    try:
        # ---------------------------------------------------------------- #
        # 0. Load + feature engineer                                        #
        # ---------------------------------------------------------------- #
        _progress("Loading dataset and engineering features…", 5)
        import pandas as pd
        import joblib
        from pathlib import Path
        from ml.feature_engineering import build_features, get_feature_columns

        df_raw = pd.read_csv(data_path)
        df = build_features(df_raw)

        feature_meta = get_feature_columns(df)
        feature_cols = [c for c in feature_meta["all_features"] if c in df.columns]

        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_cols, f"{artifact_dir}/feature_cols.pkl")

        _progress(
            f"Features ready — {len(feature_cols)} columns | {len(df):,} rows",
            10,
        )

        # ---------------------------------------------------------------- #
        # 1a. LightGBM Regressors                                           #
        # ---------------------------------------------------------------- #
        _progress("Training LightGBM regressors (5 targets) …", 15)
        from ml.models.lgbm_regressors import train_all_regressors

        regressor_results = train_all_regressors(df, feature_cols, artifact_dir)
        _progress(f"LightGBM done — {len(regressor_results)} regressors trained", 45)

        # ---------------------------------------------------------------- #
        # 1b. CatBoost Classifiers                                          #
        # ---------------------------------------------------------------- #
        _progress("Training CatBoost classifiers (2 targets) …", 50)
        from ml.models.catboost_classifiers import train_all_classifiers

        classifier_results = train_all_classifiers(df, feature_cols, artifact_dir)
        _progress(f"CatBoost done — {len(classifier_results)} classifiers trained", 65)

        # ---------------------------------------------------------------- #
        # 3. Anomaly Detector                                               #
        # ---------------------------------------------------------------- #
        _progress("Training Isolation Forest anomaly detector …", 70)
        from ml.models.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector.from_data(df, feature_cols, artifact_dir)
        _progress(f"Anomaly detector ready (threshold={detector.threshold:.4f})", 75)

        # ---------------------------------------------------------------- #
        # 4. Temporal Forecaster                                            #
        # ---------------------------------------------------------------- #
        _progress("Training temporal forecaster (LightGBM + lag features) …", 78)
        from ml.models.forecasting import TemporalForecaster

        forecaster = TemporalForecaster.from_data(df, artifact_dir)
        _progress(f"Forecaster ready — targets: {list(forecaster.models.keys())}", 85)

        # ---------------------------------------------------------------- #
        # 1c. SHAP Analysis                                                 #
        # ---------------------------------------------------------------- #
        _progress("Running SHAP analysis (summary plots + importance tables) …", 88)
        from ml.models.shap_analysis import run_shap_analysis

        shap_tables = run_shap_analysis(
            regressor_results,
            classifier_results,
            df,
            feature_cols,
            feature_meta["controllable"],
            artifact_dir,
        )
        _progress("SHAP analysis complete", 93)

        # ---------------------------------------------------------------- #
        # Evaluation report                                                 #
        # ---------------------------------------------------------------- #
        _progress("Generating evaluation report …", 95)
        from ml.evaluation import generate_evaluation_report

        generate_evaluation_report(
            regressor_results, classifier_results, shap_tables, artifact_dir
        )

        _progress("All models trained and saved!", 99)

        # ---------------------------------------------------------------- #
        # Serialize result for frontend                                     #
        # ---------------------------------------------------------------- #
        regressors = [
            {
                "target": target,
                "model_type": "LightGBM",
                "test_rmse": round(float(res["test_rmse"]), 4),
                "test_r2": round(float(res["test_r2"]), 4),
                "cv_rmse_mean": round(float(res["mean_cv_rmse"]), 4),
                "cv_rmse_std": round(float(res["std_cv_rmse"]), 4),
                "cv_r2_mean": round(float(res["mean_cv_r2"]), 4),
                "best_iteration": int(res["best_iteration"]),
                "feature_importances": _to_float_dict(res["feature_importances"]),
            }
            for target, res in regressor_results.items()
        ]

        classifiers = [
            {
                "target": target,
                "model_type": "CatBoost",
                "test_f1": round(float(res["test_f1"]), 4),
                "test_precision": round(float(res["test_precision"]), 4),
                "test_recall": round(float(res["test_recall"]), 4),
                "test_auc": round(float(res["test_auc"]), 4),
                "cv_f1_mean": round(float(res["mean_cv_f1"]), 4),
                "cv_f1_std": round(float(res["std_cv_f1"]), 4),
                "cv_auc_mean": round(float(res["mean_cv_auc"]), 4),
                "cv_auc_std": round(float(res["std_cv_auc"]), 4),
                "feature_importances": _to_float_dict(res["feature_importances"]),
            }
            for target, res in classifier_results.items()
        ]

        shap_data: dict = {}
        for target, table in shap_tables.items():
            shap_data[target] = [
                {
                    "feature": str(row["feature"]),
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "category": str(row["category"]),
                }
                for _, row in table.head(15).iterrows()
            ]

        _emit({
            "type": "result",
            "regressors": regressors,
            "classifiers": classifiers,
            "anomaly": {"threshold": float(detector.threshold)},
            "forecaster": {"targets": list(forecaster.models.keys())},
            "shap": shap_data,
            "rows": int(len(df)),
            "features": int(len(feature_cols)),
        })

    except Exception as exc:
        _emit({
            "type": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
