import numpy as np
import pandas as pd

class NaiveModel:
    """
    Random Walk / Persistence Model:
    y_hat(t + h) = y(t)
    For weekly series, the last observed price at time t serves as benchmark forecast.
    """
    def __init__(self):
        pass
        
    def fit(self, y_train):
        pass
        
    def predict(self, current_prices: np.ndarray, h: int = 1) -> np.ndarray:
        # Returns current price as the forecast for all horizons
        return np.asarray(current_prices, dtype=np.float64)
