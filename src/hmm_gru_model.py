import os
import numpy as np
import torch
import torch.nn as nn
from src.gru_model import PyTorchGRU, TimeSeriesDataset, GRUForecaster

class HMMGRUForecaster(GRUForecaster):
    """
    Primary Proposed Hybrid Model:
    Integrates weather, technical indicators, seasonality, and HMM latent regime probabilities
    into a sequential GRU network for multi-step forecasting.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, horizon=1, commodity="Tomato"):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, horizon=horizon)
        self.commodity = commodity
