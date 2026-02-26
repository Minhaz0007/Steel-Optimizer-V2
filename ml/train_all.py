"""
Master Training Script — train_all.py
=======================================
Runs the full model training pipeline end-to-end:

    Stage 1  — LightGBM regressors (5 targets)
    Stage 1  — CatBoost classifiers (2 targets)
    Stage 1  — SHAP analysis + plots
    Stage 2  — (no training required; Bayesian optimiser uses Stage 1 models)
    Stage 3  — Isolation Forest anomaly detector
    Stage 4  — LightGBM temporal forecaster

All artifacts are saved to ``ml/artifacts/`` (configurable).

Usage::

    # From the project root:
    python -m ml.train_all

    # Or with a custom data path:
    python -m ml.train_all --data path/to/data.csv --artifacts ml/artifacts

    # Skip SHAP (faster, no plots):
    python -m ml.train_all --no-shap
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_all")

# ------------------------------------------------------------------
# Default paths
# ------------------------------------------------------------------
_DEFAULT_DATA = "synthetic_steel_plant_5_years_shift_furnace_variable_output.csv"
_DEFAULT_ARTIFACTS = "ml/artifacts"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    data_path: str = _DEFAULT_DATA,
    artifact_dir: str = _DEFAULT_ARTIFACTS,
    run_shap: bool = True,
) -> None:
    """Orchestrate the full training pipeline.

    Parameters
    ----------
    data_path:
        Path to the CSV dataset.
    artifact_dir:
        Directory where all serialised models and plots are saved.
    run_shap:
        Whether to generate SHAP plots and importance tables.
    """
    t0 = time.time()
    artifact_dir_path = Path(artifact_dir)
    artifact_dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0. Load & feature-engineer the dataset                              #
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset: %s", data_path)
    try:
        df_raw = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Dataset not found: '%s'", data_path)
        sys.exit(1)

    logger.info("Raw dataset: %d rows × %d columns", *df_raw.shape)

    from ml.feature_engineering import build_features, get_feature_columns

    df = build_features(df_raw)
    logger.info("After feature engineering: %d rows × %d columns", *df.shape)

    feature_meta = get_feature_columns(df)
    feature_cols = feature_meta["all_features"]
    controllable_cols = feature_meta["controllable"]

    # Filter to only columns actually present
    feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info(
        "Feature columns: %d total (%d controllable, %d context, %d engineered)",
        len(feature_cols),
        len(controllable_cols),
        len(feature_meta["context"]),
        len(feature_meta["engineered"]),
    )

    # Save feature column list for inference-time use
    joblib.dump(feature_cols, artifact_dir_path / "feature_cols.pkl")
    logger.info("feature_cols.pkl saved (%d features)", len(feature_cols))

    # ------------------------------------------------------------------ #
    # 1a. LightGBM Regressors                                             #
    # ------------------------------------------------------------------ #
    logger.info("\n%s\nSTAGE 1A — LightGBM Regressors\n%s", "=" * 60, "=" * 60)
    from ml.models.lgbm_regressors import train_all_regressors

    regressor_results = train_all_regressors(df, feature_cols, artifact_dir)
    logger.info("Regressor training complete: %d models", len(regressor_results))

    # ------------------------------------------------------------------ #
    # 1b. CatBoost Classifiers                                            #
    # ------------------------------------------------------------------ #
    logger.info("\n%s\nSTAGE 1B — CatBoost Classifiers\n%s", "=" * 60, "=" * 60)
    from ml.models.catboost_classifiers import train_all_classifiers

    classifier_results = train_all_classifiers(df, feature_cols, artifact_dir)
    logger.info("Classifier training complete: %d models", len(classifier_results))

    # ------------------------------------------------------------------ #
    # 1c. SHAP Analysis                                                   #
    # ------------------------------------------------------------------ #
    shap_tables: dict = {}
    if run_shap:
        logger.info("\n%s\nSTAGE 1C — SHAP Analysis\n%s", "=" * 60, "=" * 60)
        from ml.models.shap_analysis import run_shap_analysis

        shap_tables = run_shap_analysis(
            regressor_results=regressor_results,
            classifier_results=classifier_results,
            df=df,
            feature_cols=feature_cols,
            controllable_cols=controllable_cols,
            artifact_dir=artifact_dir,
        )
        logger.info("SHAP analysis complete: %d models", len(shap_tables))
    else:
        logger.info("SHAP analysis skipped (--no-shap flag).")

    # ------------------------------------------------------------------ #
    # 2. (No training for Bayesian Optimizer — uses Stage 1 surrogates)   #
    # ------------------------------------------------------------------ #
    logger.info(
        "\nSTAGE 2 — Bayesian Optimizer: no training required "
        "(uses Stage 1 models as surrogates at inference time)."
    )

    # ------------------------------------------------------------------ #
    # 3. Isolation Forest Anomaly Detector                                #
    # ------------------------------------------------------------------ #
    logger.info("\n%s\nSTAGE 3 — Isolation Forest Anomaly Detector\n%s",
                "=" * 60, "=" * 60)
    from ml.models.anomaly_detection import AnomalyDetector

    detector = AnomalyDetector.from_data(df, feature_cols, artifact_dir)
    logger.info("Anomaly detector trained (threshold=%.4f).", detector.threshold)

    # ------------------------------------------------------------------ #
    # 4. Temporal Forecaster                                              #
    # ------------------------------------------------------------------ #
    logger.info("\n%s\nSTAGE 4 — Temporal Forecaster (LightGBM + Lags)\n%s",
                "=" * 60, "=" * 60)
    from ml.models.forecasting import TemporalForecaster

    forecaster = TemporalForecaster.from_data(df, artifact_dir)
    logger.info("Temporal forecaster trained: targets=%s", list(forecaster.models.keys()))

    # ------------------------------------------------------------------ #
    # 5. Evaluation Report                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n%s\nEVALUATION REPORT\n%s", "=" * 60, "=" * 60)
    from ml.evaluation import generate_evaluation_report

    report_df = generate_evaluation_report(
        regressor_results=regressor_results,
        classifier_results=classifier_results,
        shap_tables=shap_tables if run_shap else None,
        output_dir=artifact_dir,
    )

    elapsed = time.time() - t0
    logger.info("\nAll done!  Total time: %.1f s", elapsed)
    logger.info("Artifacts saved to: %s", artifact_dir_path.resolve())
    _print_artifact_summary(artifact_dir_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _print_artifact_summary(artifact_dir: Path) -> None:
    """Print a tree of saved artifacts."""
    files = sorted(artifact_dir.glob("*"))
    logger.info("\nSaved artifacts:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        logger.info("  %-50s  %8.1f KB", f.name, size_kb)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all steel plant ML optimization models."
    )
    parser.add_argument(
        "--data",
        default=_DEFAULT_DATA,
        help=f"Path to the CSV dataset (default: {_DEFAULT_DATA})",
    )
    parser.add_argument(
        "--artifacts",
        default=_DEFAULT_ARTIFACTS,
        help=f"Output directory for model artifacts (default: {_DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis (faster, no plots generated)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        data_path=args.data,
        artifact_dir=args.artifacts,
        run_shap=not args.no_shap,
    )
