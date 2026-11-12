import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import joblib

from src.config import BASE_DIR, MODELS_DIR, SEQUENCE_LENGTH, HORIZONS
from src.data_loader import load_raw_data
from src.features import engineer_features, get_feature_subsets
from src.hmm_model import MarketRegimeHMM
from src.hmm_gru_model import HMMGRUForecaster
from src.gru_model import GRUForecaster
from src.lstm_model import LSTMForecaster
from src.xgboost_model import XGBoostForecaster
from src.arima_model import ARIMAForecaster

def generate_future_forecast(commodity: str, horizon: int, model_name: str = "HMM_GRU") -> dict:
    """
    Produces genuine out-of-sample future predictions strictly beyond the latest available date.
    
    Forecasting Strategy:
    - Direct Multi-Horizon: For each horizon h in {1, 4, 8, 12}, the model f_h is trained to map
      the lookback window of features ending at time T directly to the target price at time T + h.
    - Zero Future Leakage: All input features (prices, weather lags, technical indicators, HMM probabilities)
      are strictly computed on past observations up to forecast origin T. No future prices or actual future
      weather observations are used.
    - Forecast target date: T + h * 7 days (strictly > T).
    """
    comm_name = commodity.capitalize()
    veg_lower = commodity.lower()
    
    if veg_lower not in {"tomato", "onion", "potato"}:
        raise ValueError(f"Unknown commodity: {commodity}. Must be tomato, onion, or potato.")
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}. Must be one of {HORIZONS}.")
        
    df = load_raw_data(comm_name)
    last_idx = len(df) - 1
    last_date = df['date'].iloc[last_idx]
    last_price = float(df['price'].iloc[last_idx])
    
    target_date = last_date + pd.Timedelta(weeks=horizon)
    assert target_date > last_date, "Target date must be strictly later than last observed date."
    
    comm_model_dir = MODELS_DIR / veg_lower
    
    # 1. Compute causal engineered features up to origin T
    hmm_path = comm_model_dir / "hmm_model.pkl"
    if not hmm_path.exists():
        raise FileNotFoundError(f"Trained HMM model not found: {hmm_path}")
        
    hmm = MarketRegimeHMM.load(hmm_path)
    # Forward filtering ensures causality up to time T
    probs = hmm.predict_proba_filtered(df['price'].values)
    
    feat_df = engineer_features(df)
    for k in range(hmm.n_components):
        feat_df[f'hmm_prob_state_{k}'] = probs[:, k]
    feat_df['hmm_state'] = np.argmax(probs, axis=1)
    
    subsets = get_feature_subsets()
    cols_d = subsets["D"]
    cols_hmm_gru = subsets["D"] + [f'hmm_prob_state_{k}' for k in range(hmm.n_components)]
    
    model_name_clean = model_name.upper().replace("-", "_")
    predicted_price = None
    
    if model_name_clean in {"HMM_GRU", "HMMGRU"}:
        model_file = comm_model_dir / f"hmm_gru_h{horizon}.pt"
        scaler_file = comm_model_dir / f"scaler_hmm_gru_h{horizon}.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Artifacts missing for HMM-GRU horizon {horizon}: {model_file}")
            
        scaler = joblib.load(scaler_file)
        # Take the last SEQUENCE_LENGTH observations ending at T
        last_seq = feat_df[cols_hmm_gru].iloc[-SEQUENCE_LENGTH:].copy()
        X_scaled = scaler.transform(last_seq)
        X_tensor = X_scaled.reshape(1, SEQUENCE_LENGTH, len(cols_hmm_gru))
        
        forecaster = HMMGRUForecaster(
            input_dim=len(cols_hmm_gru), hidden_dim=64, num_layers=2, dropout=0.2,
            horizon=horizon, commodity=comm_name
        )
        ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
        forecaster.model.load_state_dict(ckpt['state_dict'])
        forecaster.model.eval()
        predicted_price = float(forecaster.predict(X_tensor)[0])
        
    elif model_name_clean == "XGBOOST":
        model_file = comm_model_dir / f"xgboost_h{horizon}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Artifact missing for XGBoost horizon {horizon}: {model_file}")
        bundle = joblib.load(model_file)
        estimator = bundle["model"]
        scaler = bundle.get("scaler")
        feature_names = bundle["feature_names"]
        x_last = feat_df[feature_names].tail(1)
        if scaler is not None:
            x_in = pd.DataFrame(scaler.transform(x_last), columns=feature_names)
        else:
            x_in = x_last
        predicted_price = float(estimator.predict(x_in)[0])
        
    elif model_name_clean == "GRU":
        model_file = comm_model_dir / f"gru_h{horizon}.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Artifact missing for GRU horizon {horizon}: {model_file}")
        # Scale with feature set D on train split
        n_train = int(len(df) * 0.70)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(feat_df[cols_d].iloc[:n_train])
        last_seq = feat_df[cols_d].iloc[-SEQUENCE_LENGTH:].copy()
        X_scaled = sc.transform(last_seq)
        X_tensor = X_scaled.reshape(1, SEQUENCE_LENGTH, len(cols_d))
        
        forecaster = GRUForecaster(input_dim=len(cols_d), hidden_dim=64, num_layers=2, dropout=0.2, horizon=horizon)
        ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
        forecaster.model.load_state_dict(ckpt['state_dict'])
        forecaster.model.eval()
        predicted_price = float(forecaster.predict(X_tensor)[0])
        
    elif model_name_clean == "LSTM":
        model_file = comm_model_dir / f"lstm_h{horizon}.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Artifact missing for LSTM horizon {horizon}: {model_file}")
        n_train = int(len(df) * 0.70)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(feat_df[cols_d].iloc[:n_train])
        last_seq = feat_df[cols_d].iloc[-SEQUENCE_LENGTH:].copy()
        X_scaled = sc.transform(last_seq)
        X_tensor = X_scaled.reshape(1, SEQUENCE_LENGTH, len(cols_d))
        
        forecaster = LSTMForecaster(input_dim=len(cols_d), hidden_dim=64, num_layers=2, dropout=0.2, horizon=horizon)
        ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
        forecaster.model.load_state_dict(ckpt['state_dict'])
        forecaster.model.eval()
        predicted_price = float(forecaster.predict(X_tensor)[0])
        
    elif model_name_clean == "NAIVE":
        # Random Walk persistence: y_{T+h} = y_T
        predicted_price = last_price
        
    else:
        raise ValueError(f"Unsupported model for future forecast: {model_name}")
        
    # Prevent negative price outputs in physical domain
    predicted_price = max(0.0, predicted_price)
    
    return {
        "vegetable": veg_lower,
        "model": "HMM-GRU (Proposed)" if model_name_clean in {"HMM_GRU", "HMMGRU"} else model_name,
        "horizon_weeks": horizon,
        "forecast_origin_date": last_date.strftime("%Y-%m-%d"),
        "last_observed_date": last_date.strftime("%Y-%m-%d"),
        "forecast_target_date": target_date.strftime("%Y-%m-%d"),
        "last_observed_price": round(last_price, 2),
        "predicted_price": round(predicted_price, 2),
        "price_change": round(predicted_price - last_price, 2),
        "percentage_change": round(((predicted_price - last_price) / (last_price + 1e-6)) * 100.0, 2),
        "is_future": True,
        "data_type": "synthetic_simulated",
        "forecasting_strategy": "Direct multi-horizon HMM-GRU forecaster strictly evaluated from forecast origin T.",
        "weather_assumption": "Zero future leakage: model uses strictly past observed weather and price features up to forecast origin T.",
        "dataset_freshness": {
            "status": "up_to_date",
            "last_dataset_date": last_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "disclaimer": "Simulated benchmark dataset (2003 - 2026-11-08). Experimental model forecast for research evaluation."
        }
    }

def generate_multi_horizon_future(commodity: str, model_name: str = "HMM_GRU") -> list:
    """
    Returns genuine future forecasts across all four horizons [1, 4, 8, 12].
    """
    results = []
    for h in HORIZONS:
        try:
            fc = generate_future_forecast(commodity, h, model_name=model_name)
            results.append(fc)
        except Exception as e:
            results.append({
                "vegetable": commodity.lower(),
                "model": model_name,
                "horizon_weeks": h,
                "error": str(e)
            })
    return results

def get_forecast_chart_data(commodity: str, horizon: int = 1, model_name: str = "HMM_GRU", lookback_weeks: int = 52) -> dict:
    """
    Assembles the exact payload required for the interactive Price Forecast Visualization:
    - Historical price series up to forecast origin date (e.g. 2026-11-08)
    - Distinct future forecast trajectory strictly starting after the forecast origin
    - Clear separator demarcation at the forecast origin
    """
    comm_name = commodity.capitalize()
    veg_lower = commodity.lower()
    df = load_raw_data(comm_name)
    
    last_date = df['date'].iloc[-1].strftime("%Y-%m-%d")
    last_price = float(df['price'].iloc[-1])
    
    # Historical slice for visual context (e.g. past 52 weeks)
    hist_slice = df.tail(lookback_weeks).copy()
    historical_points = []
    for _, row in hist_slice.iterrows():
        historical_points.append({
            "date": row['date'].strftime("%Y-%m-%d"),
            "actual": round(float(row['price']), 2),
            "predicted": None,
            "period": "historical"
        })
        
    # Generate multi-horizon forecasts
    future_forecasts = generate_multi_horizon_future(commodity, model_name=model_name)
    valid_futures = [f for f in future_forecasts if "predicted_price" in f]
    
    # Anchor point at forecast origin: connects the chart line seamlessly at the boundary
    forecast_points = []
    
    # Selected horizon single forecast point
    selected_forecast = next((f for f in valid_futures if f["horizon_weeks"] == horizon), None)
    
    # All future points for multi-step trajectory
    future_curve = []
    # Include origin point for visual continuity
    future_curve.append({
        "date": last_date,
        "actual": round(last_price, 2),
        "predicted": round(last_price, 2),
        "period": "origin"
    })
    
    for f in valid_futures:
        future_curve.append({
            "date": f["forecast_target_date"],
            "actual": None,
            "predicted": f["predicted_price"],
            "horizon": f["horizon_weeks"],
            "period": "future"
        })
        
    return {
        "vegetable": veg_lower,
        "model": model_name,
        "selected_horizon": horizon,
        "forecast_origin_date": last_date,
        "last_observed_price": round(last_price, 2),
        "selected_forecast": selected_forecast,
        "historical_series": historical_points,
        "future_curve": future_curve,
        "multi_horizon_forecasts": valid_futures,
        "data_type": "synthetic_simulated",
        "disclaimer": "Simulated benchmark dataset through 2026-11-08. Forecasts are model outputs strictly beyond historical data."
    }

