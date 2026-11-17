"""Model Instantiation and Training Definitions."""

from typing import Any


def get_model(model_type: str = "xgboost", random_state: int = 42) -> Any:
    """Instantiate a regression model with standardized hyperparameters across all pipelines.

    Supported types:
    - 'xgboost': XGBoost Regressor (Primary benchmark).
    - 'random_forest': Random Forest Regressor (Noboa et al. baseline comparison).
    - 'gradient_boosting': Scikit-Learn HistGradientBoostingRegressor (Robust fallback).
    """
    if model_type == "xgboost":
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=120,
                max_depth=6,
                learning_rate=0.08,
                random_state=random_state,
            )

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_type == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'xgboost', 'random_forest', or 'gradient_boosting'.")
