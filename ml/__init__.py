"""
Steel Plant ML Optimization Package
====================================
Modular Python ML pipeline for steel furnace optimization.

Stages:
    1. Feature Engineering  — ml.feature_engineering
    2. Predictive Models    — ml.models.lgbm_regressors, ml.models.catboost_classifiers
    3. SHAP Analysis        — ml.models.shap_analysis
    4. Bayesian Optimization— ml.optimization.bayesian_optimizer
    5. Anomaly Detection    — ml.models.anomaly_detection
    6. Temporal Forecasting — ml.models.forecasting
    7. Unified API          — ml.predict_and_recommend
    8. Evaluation           — ml.evaluation
"""

__version__ = "1.0.0"
__author__ = "Steel-Optimizer-V2"
