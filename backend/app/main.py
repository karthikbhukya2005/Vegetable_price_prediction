from pathlib import Path
from datetime import date
import sys
import warnings

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "Data"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"
VEGETABLES = {"tomato", "onion", "potato"}
HORIZONS = {1, 4, 8, 12}

app = FastAPI(title="Vegetable Price Prediction API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_records(df: pd.DataFrame):
    clean = df.replace([np.inf, -np.inf], np.nan)
    clean = clean.astype(object).where(pd.notnull(clean), None)
    return clean.to_dict("records")

def read_table(path: Path):
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read {path.name}: {exc}") from exc

def find_history_file(vegetable: str) -> Path:
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if not DATA_ROOT.exists():
        raise HTTPException(status_code=503, detail=f"Data directory not found: {DATA_ROOT}")
    candidates = sorted(
        p for p in DATA_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and vegetable in p.name.lower()
        and not any(token in p.name.lower() for token in ("prediction", "metric", "result", "ablation"))
    )
    # Prefer the canonical dataset over duplicate copies in original/processed.
    candidates.sort(key=lambda p: (p.parent.name.lower() != "data", "original" in str(p.parent).lower()))
    for path in candidates:
        try:
            cols = {str(c).strip().lower() for c in read_table(path).columns}
            if ({"date", "price"} <= cols or {"index", "imputed price"} <= cols):
                return path
        except HTTPException:
            continue
    raise HTTPException(status_code=404, detail=f"No valid historical dataset found for {vegetable} under Data/.")

def normalise_history(path: Path):
    df = read_table(path)
    colmap = {str(c).strip().lower(): c for c in df.columns}
    date_col = next((colmap[k] for k in ("date", "index") if k in colmap), None)
    price_col = next((colmap[k] for k in ("price", "imputed price", "modal price", "modal_price") if k in colmap), None)
    temp_col = next((colmap[k] for k in ("temperature", "temp") if k in colmap), None)
    rain_col = next((colmap[k] for k in ("precipitation", "rainfall", "rain") if k in colmap), None)
    if date_col is None or price_col is None:
        raise HTTPException(status_code=422, detail=f"{path.name} must contain Date/index and Price/Imputed Price columns.")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "price": pd.to_numeric(df[price_col], errors="coerce"),
        "temperature": pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan,
        "precipitation": pd.to_numeric(df[rain_col], errors="coerce") if rain_col else np.nan,
    }).dropna(subset=["date", "price"]).sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise HTTPException(status_code=422, detail=f"{path.name} contains no valid date/price rows.")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return safe_records(out)

def find_result_file(filename: str):
    if not RESULTS_ROOT.exists():
        return None
    matches = list(RESULTS_ROOT.rglob(filename))
    return matches[0] if matches else None

@app.get("/")
def root():
    return {
        "message": "Vegetable Price Prediction API",
        "status": "running",
        "docs": "/docs",
        "data_mode": "synthetic_simulated",
        "disclaimer": "Benchmark dataset consists of simulated price and weather series updated through 2026-11-08 for ML evaluation."
    }

@app.get("/api/health")
def health():
    return {
        "status": "ok" if DATA_ROOT.exists() and RESULTS_ROOT.exists() else "degraded",
        "project_root": str(PROJECT_ROOT),
        "data_mode": "synthetic_simulated",
        "latest_dataset_date": "2026-11-08",
        "data_folder_exists": DATA_ROOT.exists(),
        "results_folder_exists": RESULTS_ROOT.exists(),
        "models_folder_exists": MODELS_ROOT.exists(),
    }

@app.get("/api/overview")
def overview():
    rows = []
    for veg in sorted(VEGETABLES):
        try:
            path = find_history_file(veg)
            data = normalise_history(path)
            prices = [r["price"] for r in data if r["price"] is not None]
            rows.append({
                "vegetable": veg, "available": True, "source_file": path.name,
                "records": len(data), "first_date": data[0]["date"], "latest_date": data[-1]["date"],
                "latest_price": prices[-1], "average_price": float(np.mean(prices)),
            })
        except HTTPException as exc:
            rows.append({"vegetable": veg, "available": False, "error": exc.detail})
    return {
        "data_mode": "synthetic_simulated",
        "disclaimer": "All datasets are synthetic/simulated benchmark series ending 2026-11-08 for algorithmic evaluation.",
        "vegetables": rows
    }

@app.get("/api/history/{vegetable}")
def history(vegetable: str):
    path = find_history_file(vegetable)
    return {"vegetable": vegetable.lower(), "source_file": path.name, "data": normalise_history(path)}

RESULT_FILES = {
    "metrics": "all_results.csv",
    "ablation": "ablation_results.csv",
    "hmm": "hmm_state_selection.csv",
    "dm": "diebold_mariano_results.csv",
    "future": "all_future_forecasts.csv",
}

@app.get("/api/results/{kind}")
def get_results(kind: str):
    if kind not in RESULT_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown result type '{kind}'.")
    filename = RESULT_FILES[kind]
    path = find_result_file(filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{filename} not found under results/.")
    df = read_table(path)
    return {"filename": path.name, "columns": list(df.columns), "data": safe_records(df)}

@app.get("/api/predictions/{vegetable}/{horizon}")
def predictions(vegetable: str, horizon: int):
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    filename = f"{vegetable}_h{horizon}_predictions.csv"
    path = RESULTS_ROOT / "predictions" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {filename}")
    df = read_table(path)
    required = {"Date", "Actual_Price"}
    if not required.issubset(df.columns):
        raise HTTPException(status_code=422, detail=f"{filename} is missing required columns: {sorted(required - set(df.columns))}")
    return {"vegetable": vegetable, "horizon": horizon, "filename": filename, "columns": list(df.columns), "data": safe_records(df)}

@app.get("/api/regimes/{vegetable}")
def get_regimes(vegetable: str):
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    filename = f"{vegetable}_regime_stats.csv"
    path = RESULTS_ROOT / "tables" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Regime statistics not found: {filename}")
    df = read_table(path)
    required = {"regime", "count", "percentage", "mean_price", "std_price", "min_price", "max_price", "median_price"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"{filename} is missing columns: {sorted(missing)}")
    return {"vegetable": vegetable, "filename": filename, "data": safe_records(df)}

@app.get("/api/future/{vegetable}/{horizon}")
def future_forecast(vegetable: str, horizon: int, model: str = Query("HMM_GRU")):
    """
    Direct h-week-ahead genuine future forecast strictly beyond the latest observed date.
    Uses trained HMM-GRU (or other selected model) and causal features up to origin T.
    """
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    try:
        from src.future_forecast import generate_future_forecast
        return generate_future_forecast(commodity=vegetable, horizon=horizon, model_name=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Future forecast failed: {type(exc).__name__}: {exc}")

@app.get("/api/future/{vegetable}")
def future_multi_horizon(vegetable: str, model: str = Query("HMM_GRU")):
    """
    Returns multi-horizon future forecast trajectory across all horizons [1, 4, 8, 12].
    """
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    try:
        from src.future_forecast import generate_multi_horizon_future
        return {
            "vegetable": vegetable,
            "model": model,
            "forecasts": generate_multi_horizon_future(commodity=vegetable, model_name=model)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multi-horizon forecast failed: {type(exc).__name__}: {exc}")

@app.get("/api/forecast_chart_data/{vegetable}/{horizon}")
def forecast_chart_data(vegetable: str, horizon: int, model: str = Query("HMM_GRU"), lookback: int = Query(52)):
    """
    Assembles historical context + demarcation boundary + future forecast points
    specifically formatted for the Price Forecast Visualization chart.
    """
    vegetable = vegetable.lower()
    if vegetable not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    try:
        from src.future_forecast import get_forecast_chart_data
        return get_forecast_chart_data(commodity=vegetable, horizon=horizon, model_name=model, lookback_weeks=lookback)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chart payload generation failed: {type(exc).__name__}: {exc}")

@app.get("/api/accuracy/summary")
def accuracy_summary(vegetable: str = Query("tomato"), horizon: int = Query(1)):
    """
    Returns summary accuracy metrics (Best MAE, Best RMSE, Best MAPE, Best Model)
    for the selected vegetable and horizon.
    """
    veg = vegetable.lower()
    if veg not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    try:
        from src.accuracy_metrics import get_accuracy_summary_data
        return get_accuracy_summary_data(vegetable=veg, horizon=horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Accuracy summary failed: {type(exc).__name__}: {exc}")

@app.get("/api/accuracy/comparison")
def accuracy_comparison(vegetable: str = Query("tomato"), horizon: int = Query(1), model: str = Query("all")):
    """
    Returns full accuracy comparison table and chart breakdowns:
    - Ranked model performance table
    - Metric comparison across models (MAE, RMSE, MAPE)
    - Horizon vs RMSE decay curve
    - Cross-commodity accuracy comparison
    """
    veg = vegetable.lower()
    if veg not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    try:
        from src.accuracy_metrics import get_comparison_table_and_charts
        return get_comparison_table_and_charts(vegetable=veg, horizon=horizon, model_filter=model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Accuracy comparison failed: {type(exc).__name__}: {exc}")

@app.get("/api/accuracy/predictions")
def accuracy_predictions(vegetable: str = Query("tomato"), horizon: int = Query(1), model: str = Query("HMM-GRU")):
    """
    Returns aligned actual vs predicted time series for interactive charting.
    """
    veg = vegetable.lower()
    if veg not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 4, 8 or 12.")
    
    filename = f"{veg}_h{horizon}_predictions.csv"
    path = RESULTS_ROOT / "predictions" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {filename}")
        
    df = read_table(path)
    if "Date" not in df.columns or "Actual_Price" not in df.columns:
        raise HTTPException(status_code=422, detail=f"Required columns missing in {filename}")
        
    # Find matching model column
    model_col = None
    for c in df.columns:
        if c.lower().replace("-", "_") == model.lower().replace("-", "_"):
            model_col = c
            break
            
    if model_col is None:
        model_col = "HMM_GRU" if "HMM_GRU" in df.columns else df.columns[-1]
        
    data = []
    actuals = []
    preds = []
    for _, row in df.iterrows():
        act = float(row["Actual_Price"]) if pd.notnull(row["Actual_Price"]) else None
        prd = float(row[model_col]) if pd.notnull(row[model_col]) else None
        res = round(act - prd, 2) if (act is not None and prd is not None) else None
        data.append({
            "date": str(row["Date"]),
            "actual": act,
            "predicted": prd,
            "residual": res
        })
        if act is not None and prd is not None:
            actuals.append(act)
            preds.append(prd)
            
    from src.accuracy_metrics import calculate_metrics_clean
    metrics = calculate_metrics_clean(actuals, preds)
    
    return {
        "vegetable": veg,
        "horizon": horizon,
        "model": model,
        "metrics": metrics,
        "data": data
    }

@app.get("/api/accuracy/research-benchmarks")
def accuracy_research_benchmarks(vegetable: str = Query("tomato"), horizon: int = Query(1)):
    """
    Returns research paper benchmark comparisons (Ref Paper 1, Ref Paper 2, Ref Paper 3, Ref Paper 4).
    """
    veg = vegetable.lower()
    if veg not in VEGETABLES:
        raise HTTPException(status_code=400, detail="Vegetable must be tomato, onion or potato.")
    try:
        from src.accuracy_metrics import get_research_comparison_data
        return get_research_comparison_data(vegetable=veg, horizon=horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research benchmarks failed: {type(exc).__name__}: {exc}")
