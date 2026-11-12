import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

def preprocess_and_split(df: pd.DataFrame, commodity: str):
    """
    Validates chronological order, checks continuity, splits without shuffling,
    and returns (train_df, val_df, test_df).
    Saves processed full CSV to data/processed/{commodity}_clean.csv.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort chronologically
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Verify exact weekly frequency
    deltas = df['date'].diff().dt.days.dropna()
    is_weekly = bool((deltas == 7).all())
    
    # Missing values check
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        df = df.ffill().fillna(0)
        
    # Chronological partition
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val
    
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    # Save full clean data
    out_path = DATA_PROCESSED_DIR / f"{commodity.lower()}_clean.csv"
    df.to_csv(out_path, index=False)
    
    return {
        "full_df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "splits_info": {
            "n_total": n_total,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "train_dates": (train_df['date'].min().strftime('%Y-%m-%d'), train_df['date'].max().strftime('%Y-%m-%d')),
            "val_dates": (val_df['date'].min().strftime('%Y-%m-%d'), val_df['date'].max().strftime('%Y-%m-%d')),
            "test_dates": (test_df['date'].min().strftime('%Y-%m-%d'), test_df['date'].max().strftime('%Y-%m-%d')),
            "is_regular_weekly": is_weekly
        }
    }
