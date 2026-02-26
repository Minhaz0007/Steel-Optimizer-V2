"""
Unified Inference & Recommendation API
========================================
``predict_and_recommend`` is the single entry-point for production use.

Pipeline (in order):
    1. Anomaly check — is the current operating context anomalous?
    2. Bayesian optimisation — find the best controllable setpoints
    3. Surrogate prediction — evaluate all targets at the optimal setpoints
    4. Return a structured ``RecommendationResult``

Quick-start::

    from ml.predict_and_recommend import predict_and_recommend

    result = predict_and_recommend(
        context_dict={
            "ambient_temperature_c": 31.0,
            "humidity_pct": 58.0,
            "raw_material_quality_index": 0.82,
            "moisture_content_pct": 4.5,
            "power_supply_stability_index": 0.95,
            "product_grade": 2.0,
            "operator_experience_level": 3.0,
            "maintenance_status": 0.0,
            "grade_change_flag": 0.0,
        },
        artifact_dir="ml/artifacts",
        n_opt_trials=300,
    )

    print(result.summary())

For forecasting next-shift outcomes given planned inputs::

    from ml.predict_and_recommend import forecast_next_shift

    forecast = forecast_next_shift(
        history_df=last_10_shifts_df,
        planned_inputs={"planned_runtime_hours": 8.0, "num_furnaces_running": 3},
        artifact_dir="ml/artifacts",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.models.anomaly_detection import AnomalyDetector, AnomalyResult
from ml.models.forecasting import TemporalForecaster
from ml.optimization.bayesian_optimizer import (
    BayesianSetpointOptimizer,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RecommendationResult:
    """Full output of the ``predict_and_recommend`` pipeline.

    Attributes
    ----------
    anomaly:
        Result of the anomaly check on the provided context.
    optimization:
        Bayesian optimisation result (setpoints + predicted outcomes).
    recommended_setpoints:
        Dict of controllable variable → recommended value.
    predicted_yield_pct:
        Predicted yield at the recommended setpoints.
    predicted_energy_cost_usd:
        Predicted energy cost at the recommended setpoints.
    predicted_steel_output_tons:
        Predicted steel output at the recommended setpoints.
    predicted_scrap_rate_pct:
        Predicted scrap rate at the recommended setpoints.
    predicted_production_cost_usd:
        Predicted production cost at the recommended setpoints.
    quality_pass_probability:
        P(quality_grade_pass=1) at the recommended setpoints.
    rework_probability:
        P(rework_required=1) at the recommended setpoints.
    warnings:
        List of human-readable warnings (anomaly alerts, constraint violations).
    """

    anomaly: AnomalyResult | None                      = None
    optimization: OptimizationResult | None             = None
    recommended_setpoints: dict[str, float]            = field(default_factory=dict)
    predicted_yield_pct: float                         = float("nan")
    predicted_energy_cost_usd: float                   = float("nan")
    predicted_steel_output_tons: float                 = float("nan")
    predicted_scrap_rate_pct: float                    = float("nan")
    predicted_production_cost_usd: float               = float("nan")
    quality_pass_probability: float                    = float("nan")
    rework_probability: float                          = float("nan")
    warnings: list[str]                                = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the recommendation."""
        lines = []
        lines.append("=" * 65)
        lines.append("  STEEL PLANT SHIFT OPTIMISATION RECOMMENDATION")
        lines.append("=" * 65)

        # Anomaly status
        if self.anomaly is not None:
            status = "⚠ ANOMALOUS CONDITIONS" if self.anomaly.is_anomaly else "✓ Normal operating conditions"
            lines.append(f"\nAnomaly Check  : {status}")
            lines.append(f"Anomaly Score  : {self.anomaly.anomaly_score:.4f}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        lines.append("\nRecommended Setpoints:")
        for k, v in self.recommended_setpoints.items():
            lines.append(f"  {k:<35s}  {v:>10.3f}")

        lines.append("\nPredicted Outcomes:")
        lines.append(f"  {'yield_pct':<35s}  {self.predicted_yield_pct:>10.3f}  %")
        lines.append(f"  {'steel_output_tons':<35s}  {self.predicted_steel_output_tons:>10.3f}  t")
        lines.append(f"  {'energy_cost_usd':<35s}  {self.predicted_energy_cost_usd:>10.2f}  USD")
        lines.append(f"  {'production_cost_usd':<35s}  {self.predicted_production_cost_usd:>10.2f}  USD")
        lines.append(f"  {'scrap_rate_pct':<35s}  {self.predicted_scrap_rate_pct:>10.3f}  %")
        lines.append(f"  {'P(quality_grade_pass)':<35s}  {self.quality_pass_probability:>10.4f}")
        lines.append(f"  {'P(rework_required)':<35s}  {self.rework_probability:>10.4f}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (e.g., for JSON API responses)."""
        return {
            "is_anomaly": self.anomaly.is_anomaly if self.anomaly else None,
            "anomaly_score": self.anomaly.anomaly_score if self.anomaly else None,
            "recommended_setpoints": self.recommended_setpoints,
            "predicted_outcomes": {
                "yield_pct": self.predicted_yield_pct,
                "steel_output_tons": self.predicted_steel_output_tons,
                "energy_cost_usd": self.predicted_energy_cost_usd,
                "production_cost_usd": self.predicted_production_cost_usd,
                "scrap_rate_pct": self.predicted_scrap_rate_pct,
            },
            "quality_pass_probability": self.quality_pass_probability,
            "rework_probability": self.rework_probability,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def predict_and_recommend(
    context_dict: dict[str, float],
    artifact_dir: str | Path = "ml/artifacts",
    n_opt_trials: int = 300,
    skip_anomaly_check: bool = False,
) -> RecommendationResult:
    """End-to-end prediction and setpoint recommendation pipeline.

    Parameters
    ----------
    context_dict:
        Dict of uncontrollable context variables for the current shift.
        Expected keys (all optional; missing values default to 0):

        ``ambient_temperature_c``, ``humidity_pct``,
        ``raw_material_quality_index``, ``moisture_content_pct``,
        ``power_supply_stability_index``, ``product_grade``,
        ``operator_experience_level``, ``maintenance_status``,
        ``grade_change_flag``, plus any engineered features computed
        beforehand (lag/rolling values).

    artifact_dir:
        Directory containing all serialised model files produced by
        ``train_all.py``.

    n_opt_trials:
        Optuna trial budget.  Increase for better setpoints at the cost
        of longer runtime (300 trials ≈ 1–5 seconds with surrogate models).

    skip_anomaly_check:
        If ``True``, bypass the Isolation Forest step.

    Returns
    -------
    RecommendationResult
    """
    artifact_dir = Path(artifact_dir)
    result = RecommendationResult()
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # Step 1: Anomaly check                                               #
    # ------------------------------------------------------------------ #
    if not skip_anomaly_check:
        try:
            detector = AnomalyDetector.from_artifact(artifact_dir)
            anomaly = detector.predict(context_dict)
            result.anomaly = anomaly

            if anomaly.is_anomaly:
                msg = (
                    f"Current operating conditions are anomalous "
                    f"(score={anomaly.anomaly_score:.4f}). "
                    "Optimisation recommendations may be unreliable."
                )
                warnings.append(msg)
                logger.warning(msg)
            else:
                logger.info("Anomaly check passed (score=%.4f).", anomaly.anomaly_score)

        except FileNotFoundError:
            logger.warning("Anomaly model not found — skipping anomaly check.")

    # ------------------------------------------------------------------ #
    # Step 2: Bayesian optimisation                                        #
    # ------------------------------------------------------------------ #
    try:
        optimizer = BayesianSetpointOptimizer.from_artifacts(artifact_dir)
        opt_result = optimizer.optimize(
            context=context_dict,
            n_trials=n_opt_trials,
        )
        result.optimization = opt_result
        result.recommended_setpoints = opt_result.recommended_setpoints

        # Populate predicted outcomes from optimisation result
        outcomes = opt_result.predicted_outcomes
        result.predicted_yield_pct          = outcomes.get("yield_pct", float("nan"))
        result.predicted_energy_cost_usd    = outcomes.get("energy_cost_usd", float("nan"))
        result.predicted_steel_output_tons  = outcomes.get("steel_output_tons", float("nan"))
        result.predicted_scrap_rate_pct     = outcomes.get("scrap_rate_pct", float("nan"))
        result.predicted_production_cost_usd = outcomes.get("production_cost_usd", float("nan"))
        result.quality_pass_probability     = opt_result.quality_pass_prob
        result.rework_probability           = opt_result.rework_prob

        # Constraint warning
        if opt_result.quality_pass_prob < 0.85:
            warnings.append(
                f"Quality pass probability ({opt_result.quality_pass_prob:.2%}) "
                "is below the 85% threshold. Review material quality and process settings."
            )

        logger.info(
            "Optimisation complete — yield=%.2f%%, energy=$%.0f, quality_prob=%.2f%%",
            result.predicted_yield_pct,
            result.predicted_energy_cost_usd,
            result.quality_pass_probability * 100,
        )

    except FileNotFoundError as exc:
        msg = f"Optimisation models not found: {exc}. Run train_all.py first."
        warnings.append(msg)
        logger.error(msg)

    result.warnings = warnings
    return result


# ---------------------------------------------------------------------------
# Forecasting convenience wrapper
# ---------------------------------------------------------------------------


def forecast_next_shift(
    history_df: pd.DataFrame,
    planned_inputs: dict[str, float] | None = None,
    artifact_dir: str | Path = "ml/artifacts",
) -> dict[str, float]:
    """Forecast next-shift yield and energy cost from recent history.

    Parameters
    ----------
    history_df:
        DataFrame of recent shifts (must include raw target columns for lag
        computation).  Chronological order, recent shifts at the end.
    planned_inputs:
        Known future values for controllable/context variables
        (e.g., planned_runtime_hours, product_grade).
    artifact_dir:
        Where forecasting models live.

    Returns
    -------
    dict: ``{"yield_pct": float, "energy_cost_usd": float}``
    """
    forecaster = TemporalForecaster.from_artifacts(artifact_dir)
    return forecaster.forecast_next_shift(history_df, planned_inputs)
