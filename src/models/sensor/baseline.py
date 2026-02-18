"""
Baseline ML Models for Sensor-based Defect Prediction
Models: Random Forest, XGBoost, LightGBM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, Dict, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class BaselineModel:
    """Wrapper for baseline ML models"""

    def __init__(self, model_type: str = "xgboost", seed: int = 42):
        """
        Args:
            model_type: 'random_forest', 'xgboost', or 'lightgbm'
            seed: Random seed
        """
        self.model_type = model_type
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()

        self._build_model()

    def _build_model(self):
        """Build the specified model"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.seed,
                n_jobs=-1
            )

        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=-1,
                eval_metric='logloss'
            )

        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        if self.model_type == "xgboost" and X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        logger.info(f"Model '{self.model_type}' trained successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        """Save model and scaler"""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model and scaler"""
        self.model = joblib.load(f"{path}_model.pkl")
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
