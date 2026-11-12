import os
import pandas as pd
from pathlib import Path
from src.config import DATA_RAW_DIR, COMMODITY_FILES

def load_raw_data(commodity: str) -> pd.DataFrame:
    """
    Loads raw Excel file for specified commodity and standardizes column names.
    Columns returned: ['date', 'price', 'temperature', 'precipitation']
    """
    if commodity not in COMMODITY_FILES:
        raise ValueError(f"Unknown commodity '{commodity}'. Available: {list(COMMODITY_FILES.keys())}")
        
    file_path = DATA_RAW_DIR / COMMODITY_FILES[commodity]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_excel(file_path)
    
    # Rename columns to standard lowercase
    col_mapping = {
        'index': 'date',
        'Imputed Price': 'price',
        'Temperature': 'temperature',
        'Precipitation': 'precipitation'
    }
    df = df.rename(columns=col_mapping)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Ensure float types
    for col in ['price', 'temperature', 'precipitation']:
        df[col] = df[col].astype(float)
        
    return df
