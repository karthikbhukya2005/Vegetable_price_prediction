import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.future_forecast import generate_future_forecast
from src.config import HORIZONS, PREDICTIONS_DIR, RESULTS_DIR

def run_future_forecasts():
    print("=" * 80)
    print("GENERATING GENUINE FUTURE FORECASTS (STRICTLY > 2026-11-08)")
    print("VEGETABLE PRICE PREDICTION USING WEATHER-AWARE HYBRID HMM-GRU")
    print("=" * 80)
    
    commodities = ["Tomato", "Onion", "Potato"]
    all_forecasts = []
    
    for comm in commodities:
        print(f"\nProcessing genuine future forecasts for {comm.upper()}...")
        for h in HORIZONS:
            try:
                fc = generate_future_forecast(comm, horizon=h, model_name="HMM_GRU")
                all_forecasts.append(fc)
                print(f"  [OK] Horizon {h}W: Target Date = {fc['forecast_target_date']} | Origin Date = {fc['forecast_origin_date']} | Origin Price = {fc['last_observed_price']} | Predicted Price = {fc['predicted_price']} | Change = {fc['price_change']} ({fc['percentage_change']}%)")
            except Exception as e:
                print(f"  [ERROR] Horizon {h}W: {e}")
                
    if all_forecasts:
        df_out = pd.DataFrame([{
            "Vegetable": f["vegetable"],
            "Model": f["model"],
            "Horizon_Weeks": f["horizon_weeks"],
            "Forecast_Origin_Date": f["forecast_origin_date"],
            "Forecast_Target_Date": f["forecast_target_date"],
            "Last_Observed_Price": f["last_observed_price"],
            "Predicted_Price": f["predicted_price"],
            "Price_Change": f["price_change"],
            "Percentage_Change": f["percentage_change"],
            "Is_Strictly_Future": f["is_future"],
            "Forecasting_Strategy": f["forecasting_strategy"],
            "Weather_Assumption": f["weather_assumption"]
        } for f in all_forecasts])
        
        out_csv1 = PREDICTIONS_DIR / "all_future_forecasts.csv"
        out_csv2 = RESULTS_DIR / "tables" / "all_future_forecasts.csv"
        df_out.to_csv(out_csv1, index=False)
        df_out.to_csv(out_csv2, index=False)
        print(f"\n[SAVED] Master Future Forecasts CSV: {out_csv1}")
        print(f"[SAVED] Master Future Forecasts CSV: {out_csv2}")
        print(df_out[["Vegetable", "Horizon_Weeks", "Forecast_Origin_Date", "Forecast_Target_Date", "Last_Observed_Price", "Predicted_Price", "Price_Change"]])
    else:
        print("\nNo future forecasts generated.")
        
    return all_forecasts

if __name__ == "__main__":
    run_future_forecasts()

