import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from pathlib import Path
from src.config import HMM_CANDIDATE_STATES, RANDOM_SEED, HMM_PLOTS_DIR, MODELS_DIR

def select_best_hmm_states(train_series: np.ndarray, candidate_states=HMM_CANDIDATE_STATES):
    """
    Evaluates Gaussian HMM with 2, 3, 4, 5, 6 hidden states on training series.
    Computes Log-Likelihood, AIC, and BIC.
    AIC = 2*k - 2*ln(L)
    BIC = ln(N)*k - 2*ln(L)
    where k = K*(K-1) + 2*K - 1 + K (transitions, initial probs, means, variances)
    """
    X = train_series.reshape(-1, 1)
    N = len(X)
    
    results = []
    best_bic = float('inf')
    best_k = 3
    best_model = None
    
    for k in candidate_states:
        try:
            model = hmm.GaussianHMM(
                n_components=k,
                covariance_type="diag",
                n_iter=200,
                random_state=RANDOM_SEED
            )
            model.fit(X)
            log_lik = model.score(X)
            
            # Number of free parameters:
            # Transition matrix: k*(k-1)
            # Initial state probs: k-1
            # Means: k
            # Variances: k
            n_params = k * (k - 1) + (k - 1) + 2 * k
            aic = 2 * n_params - 2 * log_lik
            bic = np.log(N) * n_params - 2 * log_lik
            
            results.append({
                "states": k,
                "log_likelihood": float(log_lik),
                "aic": float(aic),
                "bic": float(bic),
                "model": model
            })
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = model
        except Exception as e:
            print(f"Error fitting HMM with {k} states: {e}")
            
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
    return best_k, best_model, res_df

class MarketRegimeHMM:
    def __init__(self, n_components=3, commodity="Tomato"):
        self.n_components = n_components
        self.commodity = commodity
        self.model = None
        self.state_order = None # Maps raw HMM states to sorted order by price mean (0: lowest, K-1: highest)
        
    def fit(self, train_prices: np.ndarray):
        """
        Fits Gaussian HMM strictly on training prices (or log returns).
        Standardizes state ordering such that state 0 has the lowest mean price,
        and state K-1 has the highest mean price for clear economic interpretation.
        """
        X = train_prices.reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=300,
            random_state=RANDOM_SEED
        )
        self.model.fit(X)
        
        # Sort states by their learned mean price
        raw_means = self.model.means_.flatten()
        self.state_order = np.argsort(raw_means) # raw state index sorted by mean
        self.raw_to_ordered = {raw_s: ordered_s for ordered_s, raw_s in enumerate(self.state_order)}
        return self

    def predict_states(self, prices: np.ndarray):
        """
        Predicts the most likely hidden regime sequence using Viterbi algorithm.
        Returns ordered regimes (0: Low, 1: Normal, 2: High/Crisis).
        """
        X = prices.reshape(-1, 1)
        raw_states = self.model.predict(X)
        ordered_states = np.array([self.raw_to_ordered[s] for s in raw_states])
        return ordered_states

    def predict_proba(self, prices: np.ndarray):
        """
        Returns posterior state probabilities for each time step.
        Ordered consistently with state_order.
        """
        X = np.asarray(prices, dtype=float).reshape(-1, 1)
        raw_probs = self.model.predict_proba(X)
        ordered_probs = raw_probs[:, self.state_order]
        return ordered_probs
    def predict_proba_filtered(self, prices: np.ndarray):
        """
        Calculates causal HMM probabilities using forward filtering.
        At time t, probabilities use observations only up to time t.
        """
        from scipy.special import logsumexp

        X = np.asarray(prices, dtype=float).reshape(-1, 1)

        if self.model is None:
            raise ValueError("HMM model has not been fitted.")

        if len(X) == 0:
            return np.empty((0, self.n_components))

        log_emissions = self.model._compute_log_likelihood(X)

        log_start = np.log(np.clip(self.model.startprob_, 1e-300, None))
        log_trans = np.log(np.clip(self.model.transmat_, 1e-300, None))

        filtered = np.zeros((len(X), self.n_components))

        log_alpha = log_start + log_emissions[0]
        log_alpha -= logsumexp(log_alpha)
        filtered[0] = np.exp(log_alpha)

        for t in range(1, len(X)):
            log_alpha = (
                log_emissions[t]
                + logsumexp(
                    log_alpha[:, None] + log_trans,
                    axis=0
                )
            )
            log_alpha -= logsumexp(log_alpha)
            filtered[t] = np.exp(log_alpha)

        return filtered[:, self.state_order]

    def predict_states_filtered(self, prices: np.ndarray):
        probabilities = self.predict_proba_filtered(prices)
        return np.argmax(probabilities, axis=1)


    def get_regime_stats(self, prices: np.ndarray, dates: pd.Series):
        """
        Calculates regime descriptive statistics.
        """
        states = self.predict_states(prices)
        df = pd.DataFrame({'date': dates, 'price': prices, 'state': states})
        
        stats = []
        for s in range(self.n_components):
            sub = df[df['state'] == s]['price']
            if len(sub) > 0:
                stats.append({
                    "regime": s,
                    "count": len(sub),
                    "percentage": len(sub) / len(df) * 100.0,
                    "mean_price": float(sub.mean()),
                    "std_price": float(sub.std()),
                    "min_price": float(sub.min()),
                    "max_price": float(sub.max()),
                    "median_price": float(sub.median())
                })
        return pd.DataFrame(stats)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "n_components": self.n_components,
            "commodity": self.commodity,
            "state_order": self.state_order,
            "raw_to_ordered": self.raw_to_ordered
        }, filepath)
        print(f"[OK] Saved HMM model to {filepath}")

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls(n_components=data["n_components"], commodity=data["commodity"])
        instance.model = data["model"]
        instance.state_order = data["state_order"]
        instance.raw_to_ordered = data["raw_to_ordered"]
        return instance
    

def plot_hmm_regimes(dates, prices, states, commodity: str, save_dir=HMM_PLOTS_DIR):
    """
    Renders regime overlay plots on historical price time series.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    palette = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']
    regime_names = {0: "Low/Bearish", 1: "Normal/Equilibrium", 2: "High/Volatile", 3: "Extreme Spike", 4: "State 4", 5: "State 5"}
    
    fig, ax = plt.subplots(figsize=(14, 5))
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'price': prices, 'state': states})
    
    for s in np.unique(states):
        sub = df[df['state'] == s]
        label = f"Regime {s}: {regime_names.get(s, f'Regime {s}')}"
        ax.scatter(sub['date'], sub['price'], color=palette[s % len(palette)], label=label, s=12, alpha=0.85)
        
    ax.plot(df['date'], df['price'], color='gray', alpha=0.35, linewidth=0.8)
    ax.set_title(f'{commodity}: HMM Discovered Latent Market Regimes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (Rs. / Quintal)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', frameon=True)
    fig.savefig(save_dir / f"{commodity.lower()}_hmm_regimes.png", dpi=200)
    plt.close(fig)
