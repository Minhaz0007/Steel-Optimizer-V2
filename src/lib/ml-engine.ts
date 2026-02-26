/**
 * ML Engine — Type Definitions
 * ===============================
 * All JavaScript ML model implementations (Linear Regression, Extra Trees,
 * Gradient Boosting) have been replaced by the Python backend pipeline:
 *
 *   Stage 1  — LightGBM regressors       (ml/models/lgbm_regressors.py)
 *   Stage 1  — CatBoost classifiers      (ml/models/catboost_classifiers.py)
 *   Stage 1  — SHAP analysis             (ml/models/shap_analysis.py)
 *   Stage 2  — Bayesian optimisation     (ml/optimization/bayesian_optimizer.py)
 *   Stage 3  — Isolation Forest          (ml/models/anomaly_detection.py)
 *   Stage 4  — Temporal forecasting      (ml/models/forecasting.py)
 *
 * Training:  POST /api/train     (streams SSE progress events)
 * Inference: POST /api/recommend (returns JSON recommendation)
 */

// ---------------------------------------------------------------------------
// Stage 1 — Predictive model results (returned by /api/train)
// ---------------------------------------------------------------------------

export interface RegressionModelResult {
  target: string;
  model_type: 'LightGBM';
  /** Test-set RMSE */
  test_rmse: number;
  /** Test-set R² */
  test_r2: number;
  /** Mean 5-fold CV RMSE */
  cv_rmse_mean: number;
  /** Std 5-fold CV RMSE */
  cv_rmse_std: number;
  /** Mean 5-fold CV R² */
  cv_r2_mean: number;
  /** Boosting rounds used (early stopping) */
  best_iteration: number;
  /** Top-15 gain-based feature importances {feature: importance} */
  feature_importances: Record<string, number>;
}

export interface ClassifierResult {
  target: string;
  model_type: 'CatBoost';
  test_f1: number;
  test_precision: number;
  test_recall: number;
  test_auc: number;
  cv_f1_mean: number;
  cv_f1_std: number;
  cv_auc_mean: number;
  cv_auc_std: number;
  feature_importances: Record<string, number>;
}

export interface ShapEntry {
  feature: string;
  mean_abs_shap: number;
  category: 'Controllable' | 'Context' | 'Engineered';
}

// ---------------------------------------------------------------------------
// TrainingSession — one full pipeline run (all 7 models + extras)
// ---------------------------------------------------------------------------

export interface TrainingSession {
  id: string;
  datasetId: string;
  trainedAt: string;
  rows: number;
  features: number;
  regressors: RegressionModelResult[];
  classifiers: ClassifierResult[];
  anomaly: { threshold: number };
  forecaster: { targets: string[] };
  /** SHAP importance tables keyed by target name */
  shap: Record<string, ShapEntry[]>;
}

// ---------------------------------------------------------------------------
// Recommendation — returned by /api/recommend
// ---------------------------------------------------------------------------

export interface PredictedOutcomes {
  yield_pct: number;
  steel_output_tons: number;
  energy_cost_usd: number;
  production_cost_usd: number;
  scrap_rate_pct: number;
}

export interface RecommendationResult {
  is_anomaly: boolean | null;
  anomaly_score: number | null;
  recommended_setpoints: Record<string, number>;
  predicted_outcomes: PredictedOutcomes;
  quality_pass_probability: number;
  rework_probability: number;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// OptimizationRecord — stored per recommendation run
// ---------------------------------------------------------------------------

export interface OptimizationRecord {
  id: string;
  timestamp: string;
  context: Record<string, number>;
  result: RecommendationResult;
}

// ---------------------------------------------------------------------------
// Legacy stubs — kept so any stray import doesn't break the build.
// Nothing in the active app uses these at runtime.
// ---------------------------------------------------------------------------

/** @deprecated Replaced by TrainingSession + Python backend. */
export interface TrainedModel {
  id: string;
  type: string;
  metrics: { rmse: number; mae: number; r2: number; mape: number; accuracy: number };
  cvMetrics?: any;
  scalerJSON?: any;
  modelInstance: null;
  modelJSON?: any;
  featureImportance?: { feature: string; importance: number }[];
  config: { targetVariable: string; features: string[]; testSplit: number; models: string[] };
}

/** @deprecated */
export interface TrainingConfig {
  targetVariable: string;
  features: string[];
  testSplit: number;
  models: string[];
}
