"""
Stage 1 — CatBoost Classifiers
================================
Trains one CatBoost binary classifier per classification target with:
  - Automatic class-weight balancing for imbalanced targets
  - 5-fold stratified cross-validation
  - Early stopping on an internal validation split
  - Logs F1, Precision, Recall, and AUC-ROC per fold and on the final test set
  - Serialises each trained model to ``ml/artifacts/<target>_catboost.pkl``

Targets:
    quality_grade_pass   (1 = pass, 0 = fail  → maximise pass rate)
    rework_required      (1 = rework needed   → minimise)

Usage:
    from ml.models.catboost_classifiers import train_all_classifiers, load_classifier

    results = train_all_classifiers(df_features, feature_cols, artifact_dir)
    clf     = load_classifier("quality_grade_pass", artifact_dir)
    proba   = clf.predict_proba(X_new)[:, 1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ml.feature_engineering import CLASSIFICATION_TARGETS

logger = logging.getLogger(__name__)

_N_FOLDS = 5
_TEST_SIZE = 0.15
_VALID_SIZE = 0.15
_EARLY_STOPPING_ROUNDS = 50
_IMBALANCE_RATIO_THRESHOLD = 3.0  # apply class weighting if majority/minority > this


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_all_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    artifact_dir: str | Path = "ml/artifacts",
) -> dict[str, dict[str, Any]]:
    """Train one CatBoost classifier per binary classification target.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame.
    feature_cols:
        Ordered list of input feature column names.
    artifact_dir:
        Directory to save serialised models (``<target>_catboost.pkl``).

    Returns
    -------
    dict
        Keys are target names.  Each value contains::

            {
                "model"       : fitted CatBoostClassifier,
                "cv_f1"       : [fold F1 × N_FOLDS],
                "cv_auc"      : [fold AUC × N_FOLDS],
                "mean_cv_f1"  : float,
                "mean_cv_auc" : float,
                "test_f1"     : float,
                "test_precision": float,
                "test_recall" : float,
                "test_auc"    : float,
                "class_weights": dict or None,
                "feature_importances": pd.Series,
            }
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    for target in CLASSIFICATION_TARGETS:
        if target not in df.columns:
            logger.warning("Target '%s' not found in DataFrame — skipping.", target)
            continue

        logger.info("=" * 60)
        logger.info("Training CatBoost classifier for: %s", target)
        logger.info("=" * 60)

        result = _train_single_classifier(df, feature_cols, target, artifact_dir)
        results[target] = result

        logger.info(
            "%s | Test F1=%.4f | AUC=%.4f | Precision=%.4f | Recall=%.4f",
            target,
            result["test_f1"],
            result["test_auc"],
            result["test_precision"],
            result["test_recall"],
        )

    return results


def load_classifier(
    target: str,
    artifact_dir: str | Path = "ml/artifacts",
) -> CatBoostClassifier:
    """Load a serialised CatBoostClassifier from disk.

    Parameters
    ----------
    target:
        One of the two classification target names.
    artifact_dir:
        Directory where ``<target>_catboost.pkl`` is stored.
    """
    path = Path(artifact_dir) / f"{target}_catboost.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for '{target}' at '{path}'. "
            "Run train_all_classifiers() first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _train_single_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """End-to-end training for a single classification target."""
    # ---- Prepare data ----
    valid = df[feature_cols + [target]].dropna()
    X = valid[feature_cols].values.astype(np.float32)
    y = valid[target].values.astype(int)

    # ---- Class weight check ----
    class_weights, scale_pos_weight = _compute_weights(y)
    logger.info(
        "  Class distribution: %s  |  weighted=%s",
        dict(zip(*np.unique(y, return_counts=True))),
        class_weights is not None,
    )

    # ---- Train / test split ----
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=42, stratify=y
    )

    # ---- 5-fold stratified CV ----
    cv_f1: list[float] = []
    cv_auc: list[float] = []
    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

        fold_model = _build_catboost(class_weights, scale_pos_weight)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        preds = fold_model.predict(X_val)
        proba = fold_model.predict_proba(X_val)[:, 1]

        f1 = float(f1_score(y_val, preds, zero_division=0))
        auc = float(roc_auc_score(y_val, proba))
        cv_f1.append(f1)
        cv_auc.append(auc)
        logger.debug("  Fold %d/%d — F1=%.4f  AUC=%.4f", fold, _N_FOLDS, f1, auc)

    # ---- Final model on full trainval ----
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_trainval, y_trainval, test_size=_VALID_SIZE, random_state=0, stratify=y_trainval
    )
    final_model = _build_catboost(class_weights, scale_pos_weight)
    final_model.fit(
        X_tr_final, y_tr_final,
        eval_set=(X_val_final, y_val_final),
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )

    # ---- Test-set metrics ----
    test_preds = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]

    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_auc = float(roc_auc_score(y_test, test_proba))

    # ---- Feature importance ----
    importance = pd.Series(
        final_model.get_feature_importance(),
        index=feature_cols,
        name=target,
    ).sort_values(ascending=False)

    # ---- Persist ----
    artifact_path = artifact_dir / f"{target}_catboost.pkl"
    joblib.dump(final_model, artifact_path)
    logger.info("  Saved model → %s", artifact_path)

    return {
        "model": final_model,
        "cv_f1": cv_f1,
        "cv_auc": cv_auc,
        "mean_cv_f1": float(np.mean(cv_f1)),
        "std_cv_f1": float(np.std(cv_f1)),
        "mean_cv_auc": float(np.mean(cv_auc)),
        "std_cv_auc": float(np.std(cv_auc)),
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_auc": test_auc,
        "class_weights": class_weights,
        "feature_importances": importance,
        "feature_cols": feature_cols,
        "target": target,
    }


def _compute_weights(
    y: np.ndarray,
) -> tuple[dict[int, float] | None, float | None]:
    """Return class-weight dict and scale_pos_weight if class imbalance is detected."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None, None

    ratio = float(counts.max()) / float(counts.min())
    if ratio < _IMBALANCE_RATIO_THRESHOLD:
        return None, None  # balanced enough

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

    # scale_pos_weight = negative count / positive count (used by some CB params)
    neg_count = int(counts[classes == 0][0]) if 0 in classes else 1
    pos_count = int(counts[classes == 1][0]) if 1 in classes else 1
    spw = neg_count / pos_count

    logger.info(
        "  Imbalance ratio=%.2f — applying class weights: %s",
        ratio,
        weight_dict,
    )
    return weight_dict, spw


def _build_catboost(
    class_weights: dict[int, float] | None,
    scale_pos_weight: float | None,
) -> CatBoostClassifier:
    """Instantiate a CatBoostClassifier with optional class weighting."""
    params: dict[str, Any] = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "border_count": 128,
        "eval_metric": "AUC",
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }
    if class_weights is not None:
        params["class_weights"] = class_weights

    return CatBoostClassifier(**params)
