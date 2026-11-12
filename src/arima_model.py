import os
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

class ARIMAForecaster:
    def __init__(self, order=(1, 1, 1), commodity="Tomato", horizon=1):
        self.order = order
        self.commodity = commodity
        self.horizon = horizon
        self.model_res = None
        
    def fit(self, train_series: np.ndarray):
        """
        Fits ARIMA(p, d, q) strictly on training price series.
        """
        try:
            model = ARIMA(train_series, order=self.order)
            self.model_res = model.fit()
        except Exception as e:
            # Fallback to ARIMA(1, 1, 0) if convergence issue
            print(f"ARIMA{self.order} error: {e}. Falling back to (1, 1, 0)")
            self.order = (1, 1, 0)
            model = ARIMA(train_series, order=self.order)
            self.model_res = model.fit()
        return self

    def predict_test_rolling(self, full_series: np.ndarray, test_start_idx: int, horizon: int = 1):
        """
        Produces multi-step h-ahead forecasts for the test partition.
        At time t in test set, forecasts y_{t+h} given history up to t.
        """
        preds = []
        n_total = len(full_series)
        
        # In multi-step forecasting, forecast horizon h steps ahead
        for t in range(test_start_idx, n_total - horizon + 1):
            history = full_series[:t]
            # Efficient forecast using existing parameters or local AR update
            try:
                # Use fitted parameters with appended history
                res = self.model_res.apply(history)
                forecast = res.forecast(steps=horizon)[-1]
            except Exception:
                forecast = history[-1]
            preds.append(float(forecast))
            
        return np.array(preds)

    @classmethod
    def select_best_order(cls, train_series: np.ndarray, max_p=3, max_q=3):
        """
        Grid-search (p, 1, q) orders based on AIC on training set only.
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(train_series, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, 1, q)
                except Exception:
                    continue
                    
        return best_order
