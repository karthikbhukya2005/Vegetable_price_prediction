import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import BASE_DIR, PREDICTIONS_DIR, RESULTS_DIR

MODEL_COLUMNS = {
    "Naive": "Naive",
    "ARIMA": "ARIMA",
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "HMM-GRU": "HMM_GRU"
}

MODELS_ORDER = ["Naive", "ARIMA", "XGBoost", "LSTM", "GRU", "HMM-GRU"]
VEGETABLES = ["Tomato", "Onion", "Potato"]
HORIZONS = [1, 4, 8, 12]

# Published Research Benchmarks exactly as specified by reference literature
# Reference Paper 1: Price Forecasting of TOP Commodities using Hidden Markov-based Deep Learning Approach (2024)
RESEARCH_PAPER_1 = [
    {"Vegetable": "Tomato", "Model": "GRU", "RMSE": 220.87, "MAE": 163.11, "MAPE": 9.40, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Tomato", "Model": "HM-GRU", "RMSE": 193.94, "MAE": 135.98, "MAPE": 8.41, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Tomato", "Model": "Best Hybrid (HM-LSTM)", "RMSE": 186.45, "MAE": 133.40, "MAPE": 8.02, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Onion", "Model": "GRU", "RMSE": 116.69, "MAE": 84.78, "MAPE": 6.16, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Onion", "Model": "HM-GRU", "RMSE": 99.16, "MAE": 78.68, "MAPE": 5.91, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Onion", "Model": "Best Hybrid (HM-RNN)", "RMSE": 88.69, "MAE": 63.21, "MAPE": 4.62, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Potato", "Model": "GRU", "RMSE": 122.18, "MAE": 90.95, "MAPE": 9.03, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Potato", "Model": "HM-GRU", "RMSE": 82.35, "MAE": 64.22, "MAPE": 6.56, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
    {"Vegetable": "Potato", "Model": "Best Hybrid (HM-RNN)", "RMSE": 79.58, "MAE": 58.86, "MAPE": 5.80, "Source": "Ref Paper 1 (2024)", "Category": "Published Research"},
]

# Reference Paper 2: Forecasting of Retail Price Rise in Tomato, Onion and Potato Commodities in India (2024)
# Uses monthly CPI/WPI indices - contextual benchmark
RESEARCH_PAPER_2 = [
    {"Vegetable": "Tomato", "Metric": "MAPE", "Value": 4.58, "Note": "Monthly CPI/WPI index benchmark", "Source": "Ref Paper 2 (2024)"},
    {"Vegetable": "Onion", "Metric": "MAPE", "Value": 2.02, "Note": "Monthly CPI/WPI index benchmark", "Source": "Ref Paper 2 (2024)"},
    {"Vegetable": "Potato", "Metric": "MAPE", "Value": 1.30, "Note": "Monthly CPI/WPI index benchmark", "Source": "Ref Paper 2 (2024)"},
]

# Reference Paper 3: Price Dynamics of Tomato, Onion and Potato (TOP) in India (2024)
RESEARCH_PAPER_3 = {
    "Title": "Price Dynamics of Tomato, Onion and Potato (TOP) in India",
    "Year": 2024,
    "Focus": "GARCH-based price volatility analysis",
    "Note": "Related research context examining ARCH/GARCH volatility clustering in TOP commodities. Focuses on volatility dynamics rather than direct MAE/RMSE/MAPE point forecasting benchmarks."
}

# Reference Paper 4: HMM-based multi-horizon forecasting preprint
RESEARCH_PAPER_4 = [
    {"Horizon": "1 week", "Horizon_Weeks": 1, "Model": "HM-GRU", "RMSE": 79.69, "MAE": 61.72, "MAPE": 4.30, "Source": "Preprint (2024)"},
    {"Horizon": "1 week", "Horizon_Weeks": 1, "Model": "GRU", "RMSE": 99.84, "MAE": 76.17, "MAPE": 5.13, "Source": "Preprint (2024)"},
    {"Horizon": "4 weeks", "Horizon_Weeks": 4, "Model": "HM-GRU", "RMSE": 113.86, "MAE": 91.29, "MAPE": 6.49, "Source": "Preprint (2024)"},
    {"Horizon": "4 weeks", "Horizon_Weeks": 4, "Model": "GRU", "RMSE": 131.91, "MAE": 106.84, "MAPE": 8.12, "Source": "Preprint (2024)"},
    {"Horizon": "8 weeks", "Horizon_Weeks": 8, "Model": "HM-GRU", "RMSE": 157.07, "MAE": 123.60, "MAPE": 9.10, "Source": "Preprint (2024)"},
    {"Horizon": "8 weeks", "Horizon_Weeks": 8, "Model": "GRU", "RMSE": 162.96, "MAE": 126.85, "MAPE": 9.23, "Source": "Preprint (2024)"},
    {"Horizon": "12 weeks", "Horizon_Weeks": 12, "Model": "HM-GRU", "RMSE": 232.34, "MAE": 172.77, "MAPE": 12.63, "Source": "Preprint (2024)"},
    {"Horizon": "12 weeks", "Horizon_Weeks": 12, "Model": "GRU", "RMSE": 261.82, "MAE": 196.28, "MAPE": 14.20, "Source": "Preprint (2024)"},
]

def calculate_metrics_clean(actual, predicted):
    """
    Computes MAE, RMSE, and MAPE strictly using original-scale prices.
    Handles zero values safely. Validates finite arrays.
    """
    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(predicted, dtype=np.float64)
    
    # Filter out NaNs
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"MAE": None, "RMSE": None, "MAPE": None}
        
    y_t = y_true[valid]
    y_p = y_pred[valid]
    
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    
    # Avoid zero division in MAPE
    non_zero = y_t != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0)
    else:
        mape = None
        
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if mape is not None else None
    }

def compute_and_save_accuracy_comparison():
    """
    Computes all accuracy metrics across all commodities, models, and horizons
    from the verified test-set prediction CSV files.
    Ranks models for each vegetable and horizon (Rank 1 = best/lowest error).
    Saves results to results/model_accuracy_comparison.csv.
    """
    records = []
    
    for veg in VEGETABLES:
        veg_lower = veg.lower()
        for h in HORIZONS:
            pred_file = PREDICTIONS_DIR / f"{veg_lower}_h{h}_predictions.csv"
            if not pred_file.exists():
                continue
                
            try:
                df_pred = pd.read_csv(pred_file)
            except Exception:
                continue
                
            if "Actual_Price" not in df_pred.columns:
                continue
                
            actual = df_pred["Actual_Price"].values
            
            group_metrics = []
            for display_name in MODELS_ORDER:
                col_name = MODEL_COLUMNS.get(display_name, display_name)
                if col_name in df_pred.columns:
                    m = calculate_metrics_clean(actual, df_pred[col_name].values)
                    group_metrics.append({
                        "Vegetable": veg,
                        "Horizon": h,
                        "Model": display_name,
                        "MAE": m["MAE"],
                        "RMSE": m["RMSE"],
                        "MAPE": m["MAPE"],
                        "Test_Records": len(df_pred),
                        "Date_Start": str(df_pred["Date"].iloc[0]),
                        "Date_End": str(df_pred["Date"].iloc[-1])
                    })
                else:
                    group_metrics.append({
                        "Vegetable": veg,
                        "Horizon": h,
                        "Model": display_name,
                        "MAE": None,
                        "RMSE": None,
                        "MAPE": None,
                        "Test_Records": len(df_pred),
                        "Date_Start": str(df_pred["Date"].iloc[0]),
                        "Date_End": str(df_pred["Date"].iloc[-1])
                    })
                    
            # Compute Ranks within this vegetable & horizon group
            df_grp = pd.DataFrame(group_metrics)
            
            # Rank for MAE (lower is better)
            if df_grp["MAE"].notnull().any():
                df_grp["Rank_MAE"] = df_grp["MAE"].rank(ascending=True, method="min").fillna(999).astype(int)
            else:
                df_grp["Rank_MAE"] = None
                
            # Rank for RMSE (lower is better)
            if df_grp["RMSE"].notnull().any():
                df_grp["Rank_RMSE"] = df_grp["RMSE"].rank(ascending=True, method="min").fillna(999).astype(int)
            else:
                df_grp["Rank_RMSE"] = None
                
            # Rank for MAPE (lower is better)
            if df_grp["MAPE"].notnull().any():
                df_grp["Rank_MAPE"] = df_grp["MAPE"].rank(ascending=True, method="min").fillna(999).astype(int)
            else:
                df_grp["Rank_MAPE"] = None
                
            # Overall Rank (based on RMSE primary metric)
            df_grp["Rank"] = df_grp["Rank_RMSE"]
            
            records.extend(df_grp.to_dict("records"))
            
    df_all = pd.DataFrame(records)
    out_path = RESULTS_DIR / "model_accuracy_comparison.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[ACCURACY] Saved comprehensive model accuracy comparison to {out_path}")
    return df_all

def get_accuracy_comparison_df():
    """
    Loads model_accuracy_comparison.csv or computes it on first call.
    """
    path = RESULTS_DIR / "model_accuracy_comparison.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return compute_and_save_accuracy_comparison()

def get_accuracy_summary_data(vegetable: str = "tomato", horizon: int = 1):
    """
    Returns the four summary cards: Best MAE, Best RMSE, Best MAPE, Best Performing Model.
    """
    df = get_accuracy_comparison_df()
    veg_capital = vegetable.capitalize()
    
    sub = df[(df["Vegetable"].str.lower() == vegetable.lower()) & (df["Horizon"] == horizon)].copy()
    if sub.empty:
        # Fallback to any vegetable if specific horizon not found
        sub = df[df["Vegetable"].str.lower() == vegetable.lower()].copy()
        
    if sub.empty:
        return {
            "vegetable": vegetable,
            "horizon": horizon,
            "best_mae": {"model": "—", "value": None},
            "best_rmse": {"model": "—", "value": None},
            "best_mape": {"model": "—", "value": None},
            "best_model": {"model": "—", "description": "No data available"}
        }
        
    valid_mae = sub.dropna(subset=["MAE"]).sort_values("MAE")
    valid_rmse = sub.dropna(subset=["RMSE"]).sort_values("RMSE")
    valid_mape = sub.dropna(subset=["MAPE"]).sort_values("MAPE")
    
    best_mae_row = valid_mae.iloc[0] if not valid_mae.empty else None
    best_rmse_row = valid_rmse.iloc[0] if not valid_rmse.empty else None
    best_mape_row = valid_mape.iloc[0] if not valid_mape.empty else None
    
    # Best model is determined by best Rank (lowest RMSE rank)
    best_overall = valid_rmse.iloc[0] if not valid_rmse.empty else None
    
    return {
        "vegetable": veg_capital,
        "horizon": horizon,
        "best_mae": {
            "model": best_mae_row["Model"] if best_mae_row is not None else "—",
            "value": float(best_mae_row["MAE"]) if best_mae_row is not None else None
        },
        "best_rmse": {
            "model": best_rmse_row["Model"] if best_rmse_row is not None else "—",
            "value": float(best_rmse_row["RMSE"]) if best_rmse_row is not None else None
        },
        "best_mape": {
            "model": best_mape_row["Model"] if best_mape_row is not None else "—",
            "value": float(best_mape_row["MAPE"]) if best_mape_row is not None else None
        },
        "best_model": {
            "model": best_overall["Model"] if best_overall is not None else "—",
            "rmse": float(best_overall["RMSE"]) if best_overall is not None else None,
            "mae": float(best_overall["MAE"]) if best_overall is not None else None,
            "mape": float(best_overall["MAPE"]) if best_overall is not None else None,
        },
        "total_models_evaluated": len(sub),
        "data_type": "synthetic_simulated",
        "disclaimer": "Project evaluation uses synthetic/simulated vegetable price and weather data. These metrics describe model performance on the generated dataset and should not be interpreted as verified real-market forecasting performance."
    }

def get_comparison_table_and_charts(vegetable: str = "tomato", horizon: int = 1, model_filter: str = "all"):
    """
    Returns:
    - comparison_table: list of rows for the requested vegetable and horizon
    - horizon_vs_rmse: chart data showing RMSE across horizons 1, 4, 8, 12 for all models
    - commodity_comparison: chart data comparing models across Tomato, Onion, Potato at horizon
    - metric_comparison: MAE, RMSE, MAPE across models for the selected vegetable & horizon
    """
    df = get_accuracy_comparison_df()
    veg_cap = vegetable.capitalize()
    
    # Filter for table
    sub = df[(df["Vegetable"].str.lower() == vegetable.lower()) & (df["Horizon"] == horizon)].copy()
    if model_filter and model_filter.lower() != "all":
        sub = sub[sub["Model"].str.lower() == model_filter.lower()]
        
    table_rows = []
    for _, r in sub.sort_values(by="Rank", na_position="last").iterrows():
        table_rows.append({
            "Model": r["Model"],
            "MAE": round(float(r["MAE"]), 2) if pd.notnull(r["MAE"]) else "N/A",
            "RMSE": round(float(r["RMSE"]), 2) if pd.notnull(r["RMSE"]) else "N/A",
            "MAPE": round(float(r["MAPE"]), 2) if pd.notnull(r["MAPE"]) else "N/A",
            "Rank": int(r["Rank"]) if pd.notnull(r["Rank"]) and r["Rank"] < 900 else "N/A",
            "Rank_MAE": int(r["Rank_MAE"]) if pd.notnull(r["Rank_MAE"]) and r["Rank_MAE"] < 900 else "N/A",
            "Rank_RMSE": int(r["Rank_RMSE"]) if pd.notnull(r["Rank_RMSE"]) and r["Rank_RMSE"] < 900 else "N/A",
            "Rank_MAPE": int(r["Rank_MAPE"]) if pd.notnull(r["Rank_MAPE"]) and r["Rank_MAPE"] < 900 else "N/A",
            "is_best": (int(r["Rank"]) == 1) if pd.notnull(r["Rank"]) and r["Rank"] < 900 else False
        })
        
    # Chart 1, 2, 3: Metric Comparison across models for (vegetable, horizon)
    metric_chart = []
    for row in table_rows:
        if row["RMSE"] != "N/A":
            metric_chart.append({
                "model": row["Model"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "MAPE": row["MAPE"],
            })
            
    # Chart 4: Horizon vs RMSE for selected vegetable
    horizon_df = df[df["Vegetable"].str.lower() == vegetable.lower()]
    horizon_chart = []
    for h in HORIZONS:
        h_sub = horizon_df[horizon_df["Horizon"] == h]
        entry = {"horizon": f"{h}W", "horizon_weeks": h}
        for _, r in h_sub.iterrows():
            if pd.notnull(r["RMSE"]):
                entry[r["Model"]] = round(float(r["RMSE"]), 2)
        horizon_chart.append(entry)
        
    # Chart 5: Tomato vs Onion vs Potato comparison at selected horizon
    veg_comp_chart = []
    h_all_veg = df[df["Horizon"] == horizon]
    for model_name in MODELS_ORDER:
        entry = {"model": model_name}
        for v in VEGETABLES:
            val_row = h_all_veg[(h_all_veg["Vegetable"] == v) & (h_all_veg["Model"] == model_name)]
            if not val_row.empty and pd.notnull(val_row.iloc[0]["RMSE"]):
                entry[v] = round(float(val_row.iloc[0]["RMSE"]), 2)
        veg_comp_chart.append(entry)
        
    return {
        "vegetable": veg_cap,
        "horizon": horizon,
        "model_filter": model_filter,
        "table": table_rows,
        "metric_comparison": metric_chart,
        "horizon_vs_rmse": horizon_chart,
        "commodity_comparison": veg_comp_chart
    }

def get_research_comparison_data(vegetable: str = "tomato", horizon: int = 1):
    """
    Prepares structured published research comparisons:
    - Side-by-side comparison of our GRU and HMM-GRU vs Paper 1 (2024)
    - Paper 2 (retail price rise MAPE)
    - Paper 3 (GARCH volatility overview)
    - Paper 4 (Preprint multi-horizon comparison)
    """
    df = get_accuracy_comparison_df()
    veg_cap = vegetable.capitalize()
    
    # Filter our project results for selected vegetable and horizon
    our_sub = df[(df["Vegetable"].str.lower() == vegetable.lower()) & (df["Horizon"] == horizon)]
    
    our_gru = our_sub[our_sub["Model"] == "GRU"]
    our_hmm_gru = our_sub[our_sub["Model"] == "HMM-GRU"]
    
    our_gru_metrics = {
        "RMSE": round(float(our_gru.iloc[0]["RMSE"]), 2) if not our_gru.empty and pd.notnull(our_gru.iloc[0]["RMSE"]) else None,
        "MAE": round(float(our_gru.iloc[0]["MAE"]), 2) if not our_gru.empty and pd.notnull(our_gru.iloc[0]["MAE"]) else None,
        "MAPE": round(float(our_gru.iloc[0]["MAPE"]), 2) if not our_gru.empty and pd.notnull(our_gru.iloc[0]["MAPE"]) else None,
    }
    
    our_hmm_gru_metrics = {
        "RMSE": round(float(our_hmm_gru.iloc[0]["RMSE"]), 2) if not our_hmm_gru.empty and pd.notnull(our_hmm_gru.iloc[0]["RMSE"]) else None,
        "MAE": round(float(our_hmm_gru.iloc[0]["MAE"]), 2) if not our_hmm_gru.empty and pd.notnull(our_hmm_gru.iloc[0]["MAE"]) else None,
        "MAPE": round(float(our_hmm_gru.iloc[0]["MAPE"]), 2) if not our_hmm_gru.empty and pd.notnull(our_hmm_gru.iloc[0]["MAPE"]) else None,
    }
    
    # Find matching paper 1 entries for this vegetable
    p1_rows = [r for r in RESEARCH_PAPER_1 if r["Vegetable"].lower() == vegetable.lower()]
    
    # Comparative side-by-side table
    comparison_table = []
    
    # Our HMM-GRU
    comparison_table.append({
        "Model": "HMM-GRU (Our Project)",
        "Vegetable": veg_cap,
        "Horizon": f"{horizon} Week{'s' if horizon>1 else ''}",
        "RMSE": our_hmm_gru_metrics["RMSE"] if our_hmm_gru_metrics["RMSE"] is not None else "N/A",
        "MAE": our_hmm_gru_metrics["MAE"] if our_hmm_gru_metrics["MAE"] is not None else "N/A",
        "MAPE": our_hmm_gru_metrics["MAPE"] if our_hmm_gru_metrics["MAPE"] is not None else "N/A",
        "Source": "Our Project (Simulated Data)",
        "Category": "Our Project Results"
    })
    
    # Our GRU
    comparison_table.append({
        "Model": "GRU Baseline (Our Project)",
        "Vegetable": veg_cap,
        "Horizon": f"{horizon} Week{'s' if horizon>1 else ''}",
        "RMSE": our_gru_metrics["RMSE"] if our_gru_metrics["RMSE"] is not None else "N/A",
        "MAE": our_gru_metrics["MAE"] if our_gru_metrics["MAE"] is not None else "N/A",
        "MAPE": our_gru_metrics["MAPE"] if our_gru_metrics["MAPE"] is not None else "N/A",
        "Source": "Our Project (Simulated Data)",
        "Category": "Our Project Results"
    })
    
    # Published Paper 1 rows
    for r in p1_rows:
        comparison_table.append({
            "Model": r["Model"],
            "Vegetable": r["Vegetable"],
            "Horizon": "Test Split",
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "MAPE": r["MAPE"],
            "Source": r["Source"],
            "Category": "Published Research (Ref Paper 1)"
        })
        
    # Bar Chart comparison: Our vs Paper 1
    # We compare GRU and HMM-GRU / HM-GRU
    p1_gru = next((r for r in p1_rows if r["Model"] == "GRU"), None)
    p1_hm_gru = next((r for r in p1_rows if r["Model"] == "HM-GRU"), None)
    
    grouped_bar_data = [
        {
            "model_type": "GRU Baseline",
            "Our_Project_RMSE": our_gru_metrics["RMSE"],
            "Published_Paper1_RMSE": p1_gru["RMSE"] if p1_gru else None,
            "Our_Project_MAE": our_gru_metrics["MAE"],
            "Published_Paper1_MAE": p1_gru["MAE"] if p1_gru else None,
            "Our_Project_MAPE": our_gru_metrics["MAPE"],
            "Published_Paper1_MAPE": p1_gru["MAPE"] if p1_gru else None,
        },
        {
            "model_type": "HMM-GRU Hybrid",
            "Our_Project_RMSE": our_hmm_gru_metrics["RMSE"],
            "Published_Paper1_RMSE": p1_hm_gru["RMSE"] if p1_hm_gru else None,
            "Our_Project_MAE": our_hmm_gru_metrics["MAE"],
            "Published_Paper1_MAE": p1_hm_gru["MAE"] if p1_hm_gru else None,
            "Our_Project_MAPE": our_hmm_gru_metrics["MAPE"],
            "Published_Paper1_MAPE": p1_hm_gru["MAPE"] if p1_hm_gru else None,
        }
    ]
    
    return {
        "vegetable": veg_cap,
        "horizon": horizon,
        "comparison_table": comparison_table,
        "grouped_bar_data": grouped_bar_data,
        "reference_paper_1": RESEARCH_PAPER_1,
        "reference_paper_2": RESEARCH_PAPER_2,
        "reference_paper_3": RESEARCH_PAPER_3,
        "reference_paper_4_preprint": RESEARCH_PAPER_4,
        "notice": "Notice: Differences in dataset, time frequency, test periods, data preprocessing, and synthetic/simulated generation mean cross-paper metrics are contextual reference benchmarks rather than identical experimental conditions."
    }

