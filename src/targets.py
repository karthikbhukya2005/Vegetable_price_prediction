import pandas as pd
from src.config import HORIZONS

def create_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """
    Constructs forward prediction targets:
    target_h1:  price(t+1)
    target_h4:  price(t+4)
    target_h8:  price(t+8)
    target_h12: price(t+12)
    
    At time t, target_h is future price.
    """
    target_df = pd.DataFrame(index=df.index)
    for h in horizons:
        target_df[f'target_h{h}'] = df['price'].shift(-h)
    return target_df
