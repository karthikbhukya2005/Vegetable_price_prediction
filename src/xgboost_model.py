import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.config import RANDOM_SEED

class XGBoostForecaster:
    def __init__(self, horizon=1, commodity="Tomato"):
        self.horizon = horizon
        self.commodity = commodity
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, scaler=None):
        """
        Fits XGBoost regressor using training data with validation early stopping.
        """
        self.feature_names = list(X_train.columns)
        self.scaler = scaler
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            early_stopping_rounds=20,
            n_jobs=4
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is not None and not isinstance(X.iloc[0, 0] if len(X) > 0 else 0, (np.floating, float)):
            X_in = pd.DataFrame(self.scaler.transform(X[self.feature_names]), columns=self.feature_names)
        else:
            X_in = X[self.feature_names]
        return self.model.predict(X_in)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "horizon": self.horizon, "scaler": self.scaler}, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls(horizon=data["horizon"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.scaler = data.get("scaler")
        return instance
