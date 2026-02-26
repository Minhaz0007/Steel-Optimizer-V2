"""
Stage 3 — Anomaly Detection (Isolation Forest)
================================================
Trains an Isolation Forest on *normal* operating shifts, defined as shifts
where ``quality_grade_pass == 1`` AND ``rework_required == 0``.

Purpose:
    Before applying Bayesian optimisation recommendations, flag shifts whose
    current operating conditions fall outside the learned normal envelope.
    Anomalous shifts may indicate sensor faults, unusual material batches, or
    process upsets that the surrogate models were not trained on.

Output:
    - Fitted ``IsolationForest`` serialised to ``ml/artifacts/anomaly_iforest.pkl``
    - Contamination threshold determined automatically from training residuals

Usage::

    from ml.models.anomaly_detection import AnomalyDetector

    detector = AnomalyDetector.from_data(df_features, feature_cols)
    result   = detector.predict(X_new_row)
    # result.is_anomaly   → bool
    # result.anomaly_score → float (lower = more anomalous; range ≈ -0.5 to 0.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

_CONTAMINATION = "auto"     # let sklearn estimate contamination from data
_N_ESTIMATORS  = 200
_RANDOM_STATE  = 42
_ANOMALY_LABEL = -1          # sklearn convention: -1 = anomaly, 1 = normal


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    """Result of a single anomaly detection query."""

    is_anomaly: bool
    anomaly_score: float     # decision_function score; lower → more anomalous
    label: int               # sklearn raw label: -1 (anomaly) or 1 (normal)

    def __str__(self) -> str:
        status = "ANOMALY" if self.is_anomaly else "Normal"
        return f"AnomalyResult({status}, score={self.anomaly_score:.4f})"


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Wraps a trained Isolation Forest for anomaly detection.

    Parameters
    ----------
    model:
        Fitted ``sklearn.ensemble.IsolationForest``.
    feature_cols:
        Ordered list of feature columns matching model training.
    threshold:
        Decision-function score below which a shift is flagged as anomalous.
        Computed from training data as the 5th percentile of normal-sample
        scores (i.e., we expect ~5% false-positive rate on clean data).
    """

    def __init__(
        self,
        model: IsolationForest,
        feature_cols: list[str],
        threshold: float,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        df: pd.DataFrame,
        feature_cols: list[str],
        artifact_dir: str | Path = "ml/artifacts",
        false_positive_rate: float = 0.05,
    ) -> "AnomalyDetector":
        """Train an Isolation Forest on normal-operating-condition shifts.

        "Normal" is defined as rows where both ``quality_grade_pass == 1``
        and ``rework_required == 0``.  If neither column exists, all rows
        are used (graceful degradation).

        Parameters
        ----------
        df:
            Feature-engineered DataFrame.
        feature_cols:
            Ordered list of feature column names.
        artifact_dir:
            Where to save the fitted model.
        false_positive_rate:
            Fraction of normal samples allowed to be flagged as anomalous.
            Controls the decision threshold (not the IF contamination param).
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # ---- Filter to normal operating conditions ----
        normal_mask = _get_normal_mask(df)
        logger.info(
            "Anomaly detector: %d normal shifts out of %d total (%.1f%%)",
            normal_mask.sum(),
            len(df),
            100.0 * normal_mask.sum() / len(df),
        )

        normal_df = df.loc[normal_mask, feature_cols].dropna()
        if len(normal_df) < 50:
            logger.warning(
                "Very few normal shifts (%d). Falling back to full dataset.", len(normal_df)
            )
            normal_df = df[feature_cols].dropna()

        X_normal = normal_df.values.astype(np.float32)

        # ---- Fit Isolation Forest ----
        iforest = IsolationForest(
            n_estimators=_N_ESTIMATORS,
            contamination=_CONTAMINATION,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        iforest.fit(X_normal)

        # ---- Determine threshold from training scores ----
        scores = iforest.decision_function(X_normal)
        threshold = float(np.percentile(scores, false_positive_rate * 100))
        logger.info(
            "Anomaly threshold (%.0f%% FPR): %.4f", false_positive_rate * 100, threshold
        )

        # ---- Persist ----
        artifact_path = artifact_dir / "anomaly_iforest.pkl"
        joblib.dump({"model": iforest, "threshold": threshold, "feature_cols": feature_cols},
                    artifact_path)
        logger.info("Anomaly detector saved → %s", artifact_path)

        return cls(model=iforest, feature_cols=feature_cols, threshold=threshold)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path = "ml/artifacts",
    ) -> "AnomalyDetector":
        """Load a previously trained AnomalyDetector from disk."""
        path = Path(artifact_dir) / "anomaly_iforest.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Anomaly model not found at '{path}'. "
                "Run AnomalyDetector.from_data() first."
            )
        bundle = joblib.load(path)
        return cls(
            model=bundle["model"],
            feature_cols=bundle["feature_cols"],
            threshold=bundle["threshold"],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray | pd.DataFrame) -> AnomalyResult:
        """Predict whether a single shift is anomalous.

        Parameters
        ----------
        X:
            Single shift as a 1-D array, 2-D array (1, n_features), or a
            dict / DataFrame row.  Features must be ordered to match
            ``self.feature_cols``.

        Returns
        -------
        AnomalyResult
        """
        X_arr = self._coerce(X)
        score = float(self.model.decision_function(X_arr)[0])
        label = int(self.model.predict(X_arr)[0])
        is_anomaly = score < self.threshold
        return AnomalyResult(is_anomaly=is_anomaly, anomaly_score=score, label=label)

    def predict_batch(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict anomaly status for every row in a DataFrame.

        Parameters
        ----------
        df:
            DataFrame with (at least) the columns in ``self.feature_cols``.

        Returns
        -------
        pd.DataFrame with three added columns:
            ``anomaly_score``, ``anomaly_label``, ``is_anomaly``
        """
        X = df[self.feature_cols].fillna(0.0).values.astype(np.float32)
        scores = self.model.decision_function(X)
        labels = self.model.predict(X)
        out = df.copy()
        out["anomaly_score"] = scores
        out["anomaly_label"] = labels
        out["is_anomaly"]    = scores < self.threshold
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _coerce(self, X: Any) -> np.ndarray:
        """Normalise input to a (1, n_features) float32 array."""
        if isinstance(X, dict):
            X = np.array([float(X.get(c, 0.0)) for c in self.feature_cols], dtype=np.float32)
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].fillna(0.0).values.astype(np.float32)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_normal_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask selecting normal operating condition rows."""
    mask = pd.Series(True, index=df.index)
    if "quality_grade_pass" in df.columns:
        mask &= df["quality_grade_pass"].fillna(0).astype(int) == 1
    if "rework_required" in df.columns:
        mask &= df["rework_required"].fillna(1).astype(int) == 0
    return mask
