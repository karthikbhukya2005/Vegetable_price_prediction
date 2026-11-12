import os
import random
import numpy as np
import pandas as pd
from scipy import stats
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, y_naive=None):
    """
    Computes MAE, RMSE, MAPE, R2, Theil's U1, and Theil's U2.
    All inputs should be 1D numpy arrays or lists.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Avoid zero division in MAPE
    mask = (y_true != 0)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    # Theil's U1 (Normalized Root Mean Squared Error)
    denom_u1 = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    theil_u1 = rmse / (denom_u1 + 1e-12)
    
    # Theil's U2 (relative to naive baseline forecast)
    if y_naive is not None and len(y_naive) == len(y_true):
        y_naive = np.asarray(y_naive, dtype=np.float64)
        rmse_naive = np.sqrt(np.mean((y_true - y_naive) ** 2))
        theil_u2 = rmse / (rmse_naive + 1e-12)
    else:
        theil_u2 = np.nan
        
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "Theil_U1": float(theil_u1),
        "Theil_U2": float(theil_u2)
    }

def diebold_mariano_test(e1, e2, h=1, loss_type='sq'):
    """
    Diebold-Mariano test with Harvey, Leybourne, Newbold (1997) adjustment for horizon h.
    e1: errors of model 1 (e.g. baseline)
    e2: errors of model 2 (e.g. proposed)
    H0: both models have equal predictive accuracy
    H1: model 2 is statistically more accurate (if DM > 0 and e1^2 > e2^2)
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    n = len(e1)
    
    if loss_type == 'sq':
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)
        
    d_mean = np.mean(d)
    
    # Autocovariance up to lag h-1
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for i in range(1, h):
        gamma_i = np.mean((d[i:] - d_mean) * (d[:-i] - d_mean))
        var_d += 2 * (1 - i / h) * gamma_i
        
    var_d = max(var_d, 1e-12)
    dm_stat = d_mean / np.sqrt(var_d / n)
    
    # Harvey-Leybourne-Newbold modification factor
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_mod = dm_stat * hln_factor
    
    # Two-sided p-value from Student-t with n-1 df
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_mod), df=n - 1))
    
    return {
        "DM_stat": float(dm_stat_mod),
        "p_value": float(p_value),
        "significant_05": bool(p_value < 0.05)
    }
