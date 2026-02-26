"""
Stage 2 — Bayesian Optimization via Optuna
============================================
Uses Stage 1 LightGBM regressors and CatBoost classifiers as surrogate
models to find the optimal controllable setpoints for a given shift context.

Objective (multi-objective, scalarised):
    Maximise:  yield_pct,  steel_output_tons
    Minimise:  energy_cost_usd,  scrap_rate_pct
    Constraint: P(quality_grade_pass=1) > 0.85

The composite objective is::

    score = w_yield * yield_norm
          + w_output * output_norm
          - w_energy * energy_norm
          - w_scrap  * scrap_norm

where each term is normalised by its approximate range so that the four
objectives are on a comparable scale.  The constraint is enforced by a
heavy penalty when the quality-pass probability falls below 0.85.

Usage::

    from ml.optimization.bayesian_optimizer import BayesianSetpointOptimizer

    optimizer = BayesianSetpointOptimizer.from_artifacts("ml/artifacts")
    result = optimizer.optimize(
        context={"ambient_temperature_c": 28.5, "humidity_pct": 62.0, ...},
        n_trials=200,
    )
    # result.recommended_setpoints → dict of controllable var → value
    # result.predicted_outcomes    → dict of target → predicted value
    # result.quality_pass_prob     → float probability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd

from ml.feature_engineering import (
    CONTROLLABLE_VARS,
    CONTEXT_VARS,
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGETS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controllable variable search bounds
# (derived from domain knowledge; adjust to match your data distribution)
# ---------------------------------------------------------------------------
SEARCH_BOUNDS: dict[str, tuple[float, float]] = {
    "avg_furnace_temperature_c": (1400.0, 1700.0),
    "oxygen_flow_rate":          (500.0, 3000.0),
    "charge_weight_tons":        (50.0, 300.0),
    "scrap_ratio_pct":           (10.0, 60.0),
    "iron_ore_ratio_pct":        (10.0, 60.0),
    "alloy_addition_kg":         (0.0, 500.0),
    "flux_addition_kg":          (0.0, 800.0),
    "num_furnaces_running":      (1.0, 6.0),      # treated as float, rounded
    "labor_count":               (20.0, 100.0),
    "planned_runtime_hours":     (4.0, 12.0),
}

# Objective weights (all positives; direction handled in score formula)
_WEIGHTS: dict[str, float] = {
    "yield_pct": 0.35,
    "steel_output_tons": 0.25,
    "energy_cost_usd": 0.25,
    "scrap_rate_pct": 0.15,
}

# Approximate ranges for normalisation (adjust after seeing real data)
_NORM_RANGES: dict[str, float] = {
    "yield_pct": 20.0,            # e.g., range 75–95 %
    "steel_output_tons": 200.0,   # e.g., range 100–300 t/shift
    "energy_cost_usd": 50_000.0,  # e.g., range 10k–60k USD
    "scrap_rate_pct": 10.0,       # e.g., range 0–10 %
}

_QUALITY_PASS_THRESHOLD = 0.85  # minimum acceptable P(pass)
_QUALITY_PENALTY = 1e6          # score penalty when constraint violated


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Container for a single optimisation run result."""

    recommended_setpoints: dict[str, float]  = field(default_factory=dict)
    predicted_outcomes: dict[str, float]     = field(default_factory=dict)
    quality_pass_prob: float                 = 0.0
    rework_prob: float                       = 0.0
    best_score: float                        = float("-inf")
    n_trials: int                            = 0

    def summary(self) -> str:
        lines = ["=== Optimization Result ==="]
        lines.append("\nRecommended Setpoints:")
        for k, v in self.recommended_setpoints.items():
            lines.append(f"  {k:<30s} {v:>10.3f}")
        lines.append("\nPredicted Outcomes:")
        for k, v in self.predicted_outcomes.items():
            lines.append(f"  {k:<30s} {v:>10.3f}")
        lines.append(f"\n  quality_pass_prob          {self.quality_pass_prob:>10.4f}")
        lines.append(f"  rework_prob                {self.rework_prob:>10.4f}")
        lines.append(f"\n  Composite score            {self.best_score:>10.4f}")
        lines.append(f"  Trials run                 {self.n_trials:>10d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------


class BayesianSetpointOptimizer:
    """Wraps Optuna TPE sampler over LightGBM/CatBoost surrogate models.

    Parameters
    ----------
    regressors:
        Dict mapping regression target name → fitted LGBMRegressor.
    quality_classifier:
        Fitted CatBoostClassifier for ``quality_grade_pass``.
    rework_classifier:
        Fitted CatBoostClassifier for ``rework_required``.
    feature_cols:
        Ordered list of ALL feature columns (controllable + context + engineered).
        Must exactly match the order used during model training.
    context_defaults:
        Default values for context/engineered features not provided by the
        caller at inference time.  Any missing column is filled from here.
    """

    def __init__(
        self,
        regressors: dict[str, Any],
        quality_classifier: Any,
        rework_classifier: Any,
        feature_cols: list[str],
        context_defaults: dict[str, float] | None = None,
    ) -> None:
        self.regressors = regressors
        self.quality_classifier = quality_classifier
        self.rework_classifier = rework_classifier
        self.feature_cols = feature_cols
        self.context_defaults: dict[str, float] = context_defaults or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path = "ml/artifacts",
        feature_cols: list[str] | None = None,
        context_defaults: dict[str, float] | None = None,
    ) -> "BayesianSetpointOptimizer":
        """Load all surrogate models from serialised artifacts.

        Parameters
        ----------
        artifact_dir:
            Directory containing ``<target>_lgbm.pkl`` and
            ``<target>_catboost.pkl`` files.
        feature_cols:
            Feature column list.  If ``None``, tries to load from
            ``ml/artifacts/feature_cols.pkl``.
        context_defaults:
            Optional default context values for missing columns.
        """
        artifact_dir = Path(artifact_dir)

        # Load regressors
        regressors: dict[str, Any] = {}
        for target in REGRESSION_TARGETS:
            path = artifact_dir / f"{target}_lgbm.pkl"
            if path.exists():
                regressors[target] = joblib.load(path)
                logger.info("Loaded regressor: %s", path)
            else:
                logger.warning("Regressor not found: %s", path)

        # Load classifiers
        q_path = artifact_dir / "quality_grade_pass_catboost.pkl"
        r_path = artifact_dir / "rework_required_catboost.pkl"
        quality_clf = joblib.load(q_path) if q_path.exists() else None
        rework_clf  = joblib.load(r_path) if r_path.exists() else None

        # Load feature column list
        fc_path = artifact_dir / "feature_cols.pkl"
        if feature_cols is None:
            if fc_path.exists():
                feature_cols = joblib.load(fc_path)
            else:
                raise FileNotFoundError(
                    f"feature_cols not found at '{fc_path}'. "
                    "Pass feature_cols explicitly or run train_all.py first."
                )

        return cls(
            regressors=regressors,
            quality_classifier=quality_clf,
            rework_classifier=rework_clf,
            feature_cols=feature_cols,
            context_defaults=context_defaults,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        context: dict[str, float],
        n_trials: int = 300,
        n_startup_trials: int = 30,
        seed: int = 42,
    ) -> OptimizationResult:
        """Run Bayesian optimisation for a given shift context.

        Parameters
        ----------
        context:
            Dict of uncontrollable variable values for this shift, e.g.::

                {
                    "ambient_temperature_c": 30.2,
                    "humidity_pct": 55.0,
                    "raw_material_quality_index": 0.87,
                    ...
                }

        n_trials:
            Total Optuna trials.  More trials → better optimum but slower.
        n_startup_trials:
            Random-search trials before TPE takes over.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        OptimizationResult
        """
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials, seed=seed
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = self._make_objective(context)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score  = study.best_value

        # Re-evaluate best solution to get all predicted outcomes
        feature_vec = self._build_feature_vector(best_params, context)
        predicted_outcomes = self._predict_all(feature_vec)
        quality_prob = self._quality_pass_prob(feature_vec)
        rework_prob  = self._rework_prob(feature_vec)

        return OptimizationResult(
            recommended_setpoints={
                k: (round(v) if k == "num_furnaces_running" else round(v, 3))
                for k, v in best_params.items()
            },
            predicted_outcomes=predicted_outcomes,
            quality_pass_prob=quality_prob,
            rework_prob=rework_prob,
            best_score=float(best_score),
            n_trials=n_trials,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_objective(self, context: dict[str, float]):
        """Return the Optuna objective closure over the given context."""

        def objective(trial: optuna.Trial) -> float:
            # Suggest values for controllable variables
            params: dict[str, float] = {}
            for var, (lo, hi) in SEARCH_BOUNDS.items():
                params[var] = trial.suggest_float(var, lo, hi)

            # Build complete feature vector
            feature_vec = self._build_feature_vector(params, context)

            # Predict regression targets
            preds = self._predict_all(feature_vec)

            # Compute constraint: quality pass probability
            q_prob = self._quality_pass_prob(feature_vec)
            if q_prob < _QUALITY_PASS_THRESHOLD:
                return -_QUALITY_PENALTY * (_QUALITY_PASS_THRESHOLD - q_prob)

            # Compute composite score (maximising)
            score = (
                _WEIGHTS["yield_pct"]        *  preds.get("yield_pct", 0)        / _NORM_RANGES["yield_pct"]
                + _WEIGHTS["steel_output_tons"] *  preds.get("steel_output_tons", 0) / _NORM_RANGES["steel_output_tons"]
                - _WEIGHTS["energy_cost_usd"]   *  preds.get("energy_cost_usd", 0)  / _NORM_RANGES["energy_cost_usd"]
                - _WEIGHTS["scrap_rate_pct"]    *  preds.get("scrap_rate_pct", 0)   / _NORM_RANGES["scrap_rate_pct"]
            )
            return float(score)

        return objective

    def _build_feature_vector(
        self,
        controllable: dict[str, float],
        context: dict[str, float],
    ) -> np.ndarray:
        """Assemble a single-row feature array matching ``self.feature_cols``."""
        merged: dict[str, float] = {}
        # Fill from defaults first (lowest priority)
        merged.update(self.context_defaults)
        # Fill context variables
        merged.update(context)
        # Fill controllable variables (highest priority)
        merged.update(controllable)

        row = np.array(
            [float(merged.get(col, 0.0)) for col in self.feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)
        return row

    def _predict_all(self, feature_vec: np.ndarray) -> dict[str, float]:
        """Run all regression surrogates on ``feature_vec``."""
        preds: dict[str, float] = {}
        for target, model in self.regressors.items():
            preds[target] = float(model.predict(feature_vec)[0])
        return preds

    def _quality_pass_prob(self, feature_vec: np.ndarray) -> float:
        """Return P(quality_grade_pass=1)."""
        if self.quality_classifier is None:
            return 1.0  # optimistic fallback when no classifier loaded
        proba = self.quality_classifier.predict_proba(feature_vec)
        return float(proba[0, 1])

    def _rework_prob(self, feature_vec: np.ndarray) -> float:
        """Return P(rework_required=1)."""
        if self.rework_classifier is None:
            return 0.0
        proba = self.rework_classifier.predict_proba(feature_vec)
        return float(proba[0, 1])


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_optimization(
    context: dict[str, float],
    artifact_dir: str | Path = "ml/artifacts",
    n_trials: int = 300,
) -> OptimizationResult:
    """One-shot helper: load models, run optimisation, return result.

    Parameters
    ----------
    context:
        Uncontrollable context variables for the target shift.
    artifact_dir:
        Where serialised models live.
    n_trials:
        Optuna trial budget.

    Returns
    -------
    OptimizationResult
    """
    optimizer = BayesianSetpointOptimizer.from_artifacts(artifact_dir)
    return optimizer.optimize(context, n_trials=n_trials)
