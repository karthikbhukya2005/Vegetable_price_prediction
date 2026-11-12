import os
import sys
from pathlib import Path

# Ensure base directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.config import (
    COMMODITY_FILES, HORIZONS, RANDOM_SEED, SEQUENCE_LENGTH,
    MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR,
    HMM_PLOTS_DIR, PRED_PLOTS_DIR, COMP_PLOTS_DIR, ABLATION_PLOTS_DIR,
    REPORTS_DIR, ABLATION_LEVELS
)
from src.utils import set_seed, calculate_metrics, diebold_mariano_test
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_and_split
from src.features import engineer_features, get_feature_subsets
from src.targets import create_targets
from src.hmm_model import MarketRegimeHMM, select_best_hmm_states, plot_hmm_regimes
from src.naive_model import NaiveModel
from src.arima_model import ARIMAForecaster
from src.xgboost_model import XGBoostForecaster
from src.lstm_model import LSTMForecaster
from src.gru_model import GRUForecaster
from src.hmm_gru_model import HMMGRUForecaster

def create_sequences(X_arr: np.ndarray, y_arr: np.ndarray, seq_len: int = SEQUENCE_LENGTH):
    """
    Creates rolling 3D sequence tensors (N - seq_len, seq_len, features) for PyTorch.
    At index i, features correspond to [i : i + seq_len], and target corresponds to i + seq_len - 1.
    """
    X_seq, y_seq = [], []
    for i in range(len(X_arr) - seq_len + 1):
        X_seq.append(X_arr[i : i + seq_len])
        y_seq.append(y_arr[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

def run_all():
    print("=" * 80)
    print("STARTING COMPLETE EXPERIMENT WORKFLOW")
    print("VEGETABLE PRICE PREDICTION USING WEATHER-AWARE HYBRID HMM-GRU MODELS")
    print("=" * 80)
    
    set_seed(RANDOM_SEED)
    
    # Ensure all results directories exist
    for d in [METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR, HMM_PLOTS_DIR, PRED_PLOTS_DIR, COMP_PLOTS_DIR, ABLATION_PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    all_metrics_records = []
    ablation_records = []
    dm_test_records = []
    hmm_selection_records = []
    
    for commodity in ["Tomato", "Onion", "Potato"]:
        print(f"\n{'#' * 60}\nPROCESSING COMMODITY: {commodity.upper()}\n{'#' * 60}")
        comm_model_dir = MODELS_DIR / commodity.lower()
        comm_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Data
        df = load_raw_data(commodity)
        first_date_str = df['date'].min().strftime('%Y-%m-%d')
        last_date_str = df['date'].max().strftime('%Y-%m-%d')
        print(f"[DATASET] Commodity: {commodity}", flush=True)
        print(f"  - Total records: {len(df)}", flush=True)
        print(f"  - First date: {first_date_str}", flush=True)
        print(f"  - Latest date: {last_date_str}", flush=True)
        
        # 2. Preprocess and chronological split
        splits = preprocess_and_split(df, commodity)
        train_df = splits["train_df"]
        val_df = splits["val_df"]
        test_df = splits["test_df"]
        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
        print(f"[SPLITS] Train = {n_train} ({splits['splits_info']['train_dates'][0]} to {splits['splits_info']['train_dates'][1]})", flush=True)
        print(f"         Val   = {n_val} ({splits['splits_info']['val_dates'][0]} to {splits['splits_info']['val_dates'][1]})", flush=True)
        print(f"         Test  = {n_test} ({splits['splits_info']['test_dates'][0]} to {splits['splits_info']['test_dates'][1]})", flush=True)
        
        # 3. Fit HMM on Training Split Only (No Leakage)
        print(f"Selecting best HMM state count on training data...")
        best_k, best_hmm, hmm_summary = select_best_hmm_states(train_df['price'].values)
        hmm_summary['commodity'] = commodity
        hmm_selection_records.append(hmm_summary)
        print(f"Optimal HMM states for {commodity}: K = {best_k}")
        
        hmm_model = MarketRegimeHMM(n_components=best_k, commodity=commodity)
        hmm_model.fit(train_df['price'].values)
        hmm_model.save(comm_model_dir / "hmm_model.pkl")
        
        # Predict states across full dataset
        # Filtered probabilities are causal: they use only observations
        # available up to each individual time step.
        full_probs = hmm_model.predict_proba_filtered(df['price'].values)

        # Viterbi states are retained only for descriptive visualization.
        full_states = hmm_model.predict_states(df['price'].values)

        plot_hmm_regimes(
            df['date'].values,
            df['price'].values,
            full_states,
            commodity
        )

        regime_stats = hmm_model.get_regime_stats(
            df['price'].values,
            df['date']
        )
        regime_stats.to_csv(TABLES_DIR / f"{commodity.lower()}_regime_stats.csv", index=False)
        print(f"HMM Regimes Summary:\n{regime_stats}")
        
        # 4. Feature Engineering
        print("Constructing engineered features...")
        feat_df = engineer_features(df)
        
        # Add HMM regime probabilities as features
        for k in range(best_k):
            feat_df[f'hmm_prob_state_{k}'] = full_probs[:, k]
        feat_df['hmm_state'] = full_states
        
        # 5. Create Horizon Targets
        target_df = create_targets(df, horizons=HORIZONS)
        
        # 6. Feature Subsets for Ablation & Models
        subsets = get_feature_subsets()
        feature_sets = {
            "Ablation_A_PriceOnly": subsets["A"],
            "Ablation_B_Weather": subsets["B"],
            "Ablation_C_Tech": subsets["C"],
            "Ablation_D_WeatherTech": subsets["D"],
            "HMM_GRU_Full": subsets["D"] + [f'hmm_prob_state_{k}' for k in range(best_k)]
        }
        
        # Select best ARIMA order on training set
        print(f"Selecting best ARIMA order on training set...")
        best_arima_order = ARIMAForecaster.select_best_order(train_df['price'].values, max_p=2, max_q=2)
        print(f"Best ARIMA order for {commodity}: {best_arima_order}")
        
        # Loop through horizons
        for h in HORIZONS:
            print(f"\n--- [{commodity.upper()}] Training & Evaluating for Horizon H = {h} week(s) ---", flush=True)
            y_target = target_df[f'target_h{h}']
            
            # Align target with features (drop the last h rows where future target is NaN)
            valid_mask = ~y_target.isna()
            valid_feat = feat_df[valid_mask].copy().reset_index(drop=True)
            valid_y = y_target[valid_mask].copy().reset_index(drop=True)
            # The target date for y_target at observation t is df['date'][t + h]
            target_dates = df['date'].iloc[h:].copy().reset_index(drop=True)
            
            # Recalculate split cutoffs on aligned data
            n_tot_h = len(valid_feat)
            n_train_h = n_train
            n_val_h = n_val
            n_test_h = n_tot_h - n_train_h - n_val_h
            
            train_idx = slice(0, n_train_h)
            val_idx = slice(n_train_h, n_train_h + n_val_h)
            test_idx = slice(n_train_h + n_val_h, n_tot_h)
            
            y_train = valid_y.iloc[train_idx].values
            y_val = valid_y.iloc[val_idx].values
            y_test = valid_y.iloc[test_idx].values
            test_target_dates = target_dates.iloc[test_idx].values
            
            # Naive baseline: y_hat(t+h) = price(t)
            naive_model = NaiveModel()
            y_pred_naive = naive_model.predict(valid_feat['price'].iloc[test_idx].values, h=h)
            m_naive = calculate_metrics(y_test, y_pred_naive, y_naive=y_pred_naive)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "Naive", "Horizon": h, **m_naive
            })
            
            # ARIMA model
            arima_forecaster = ARIMAForecaster(order=best_arima_order, commodity=commodity, horizon=h)
            arima_forecaster.fit(train_df['price'].values)
            # Predict rolling on test set
            y_pred_arima = arima_forecaster.predict_test_rolling(df['price'].values, test_start_idx=n_train + n_val, horizon=h)
            # Ensure shape match
            y_pred_arima = y_pred_arima[:len(y_test)]
            m_arima = calculate_metrics(y_test, y_pred_arima, y_naive=y_pred_naive)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "ARIMA", "Horizon": h, **m_arima
            })
            
            # XGBoost Model (using full feature set D)
            xgb_cols = subsets["D"]
            scaler_xgb = StandardScaler()
            # FIT ONLY ON TRAIN SET
            X_train_xgb = pd.DataFrame(scaler_xgb.fit_transform(valid_feat[xgb_cols].iloc[train_idx]), columns=xgb_cols)
            X_val_xgb = pd.DataFrame(scaler_xgb.transform(valid_feat[xgb_cols].iloc[val_idx]), columns=xgb_cols)
            X_test_xgb = pd.DataFrame(scaler_xgb.transform(valid_feat[xgb_cols].iloc[test_idx]), columns=xgb_cols)
            
            xgb_forecaster = XGBoostForecaster(horizon=h, commodity=commodity)
            xgb_forecaster.fit(X_train_xgb, y_train, X_val_xgb, y_val, scaler=scaler_xgb)
            y_pred_xgb = xgb_forecaster.predict(X_test_xgb)
            m_xgb = calculate_metrics(y_test, y_pred_xgb, y_naive=y_pred_naive)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "XGBoost", "Horizon": h, **m_xgb
            })
            xgb_forecaster.save(comm_model_dir / f"xgboost_h{h}.pkl")
            
            # Neural Sequence Data Preparation (Scaled strictly on train split)
            # We scale each feature set cleanly
            predictions_dict = {
                "Naive": y_pred_naive,
                "ARIMA": y_pred_arima,
                "XGBoost": y_pred_xgb
            }
            
            # 1. Standard LSTM (using Full Feature Set D)
            cols_lstm = subsets["D"]
            scaler_lstm = StandardScaler()
            X_all_lstm = scaler_lstm.fit_transform(valid_feat[cols_lstm].iloc[train_idx])
            X_val_scaled = scaler_lstm.transform(valid_feat[cols_lstm].iloc[val_idx])
            X_test_scaled = scaler_lstm.transform(valid_feat[cols_lstm].iloc[test_idx])
            
            # Create sequence tensors
            X_tr_seq, y_tr_seq = create_sequences(X_all_lstm, y_train)
            X_va_seq, y_va_seq = create_sequences(X_val_scaled, y_val)
            X_te_seq, y_te_seq = create_sequences(X_test_scaled, y_test)
            
            # Adjust y_test for sequence offset
            y_test_seq = y_test[SEQUENCE_LENGTH - 1:]
            y_naive_seq = y_pred_naive[SEQUENCE_LENGTH - 1:]
            test_dates_seq = test_target_dates[SEQUENCE_LENGTH - 1:]
            formatted_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in test_dates_seq]
            
            lstm_model = LSTMForecaster(input_dim=len(cols_lstm), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h)
            lstm_model.fit(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq)
            y_pred_lstm = lstm_model.predict(X_te_seq)
            m_lstm = calculate_metrics(y_test_seq, y_pred_lstm, y_naive=y_naive_seq)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "LSTM", "Horizon": h, **m_lstm
            })
            predictions_dict["LSTM"] = y_pred_lstm
            lstm_model.save(comm_model_dir / f"lstm_h{h}.pt")
            
            # 2. Standard GRU (using Feature Set D)
            gru_model = GRUForecaster(input_dim=len(cols_lstm), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h)
            gru_model.fit(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq)
            y_pred_gru = gru_model.predict(X_te_seq)
            m_gru = calculate_metrics(y_test_seq, y_pred_gru, y_naive=y_naive_seq)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "GRU", "Horizon": h, **m_gru
            })
            predictions_dict["GRU"] = y_pred_gru
            gru_model.save(comm_model_dir / f"gru_h{h}.pt")
            
            # 3. HMM-GRU Hybrid (Primary Proposed Model: Feature Set D + HMM Probs)
            cols_hmm_gru = feature_sets["HMM_GRU_Full"]
            scaler_hmm_gru = StandardScaler()
            X_tr_hg = scaler_hmm_gru.fit_transform(valid_feat[cols_hmm_gru].iloc[train_idx])
            X_va_hg = scaler_hmm_gru.transform(valid_feat[cols_hmm_gru].iloc[val_idx])
            X_te_hg = scaler_hmm_gru.transform(valid_feat[cols_hmm_gru].iloc[test_idx])
            
            X_tr_hg_seq, y_tr_hg_seq = create_sequences(X_tr_hg, y_train)
            X_va_hg_seq, y_va_hg_seq = create_sequences(X_va_hg, y_val)
            X_te_hg_seq, y_te_hg_seq = create_sequences(X_te_hg, y_test)
            
            hmm_gru_model = HMMGRUForecaster(input_dim=len(cols_hmm_gru), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h, commodity=commodity)
            hmm_gru_model.fit(X_tr_hg_seq, y_tr_hg_seq, X_va_hg_seq, y_va_hg_seq)
            y_pred_hmm_gru = hmm_gru_model.predict(X_te_hg_seq)
            m_hmm_gru = calculate_metrics(y_test_seq, y_pred_hmm_gru, y_naive=y_naive_seq)
            all_metrics_records.append({
                "Vegetable": commodity, "Model": "HMM-GRU", "Horizon": h, **m_hmm_gru
            })
            predictions_dict["HMM-GRU"] = y_pred_hmm_gru
            hmm_gru_model.save(comm_model_dir / f"hmm_gru_h{h}.pt")
            # Save scaler for online serving
            joblib.dump(scaler_hmm_gru, comm_model_dir / f"scaler_hmm_gru_h{h}.pkl")
            
            # Compute Diebold-Mariano Tests
            dm_gru = diebold_mariano_test(y_test_seq - y_pred_gru, y_test_seq - y_pred_hmm_gru, h=h)
            dm_arima = diebold_mariano_test(y_test_seq - y_pred_arima[SEQUENCE_LENGTH - 1:], y_test_seq - y_pred_hmm_gru, h=h)
            dm_test_records.append({
                "Vegetable": commodity, "Horizon": h,
                "Comparison": "HMM-GRU vs GRU",
                "DM_Stat": dm_gru["DM_stat"], "p_value": dm_gru["p_value"], "Significant_05": dm_gru["significant_05"]
            })
            dm_test_records.append({
                "Vegetable": commodity, "Horizon": h,
                "Comparison": "HMM-GRU vs ARIMA",
                "DM_Stat": dm_arima["DM_stat"], "p_value": dm_arima["p_value"], "Significant_05": dm_arima["significant_05"]
            })
            
            # Save Predictions CSV
            pred_df = pd.DataFrame({
                "Date": formatted_dates,
                "Actual_Price": y_test_seq,
                "Naive": y_naive_seq,
                "ARIMA": y_pred_arima[SEQUENCE_LENGTH - 1:],
                "XGBoost": y_pred_xgb[SEQUENCE_LENGTH - 1:],
                "LSTM": y_pred_lstm,
                "GRU": y_pred_gru,
                "HMM_GRU": y_pred_hmm_gru
            })
            pred_file_path = PREDICTIONS_DIR / f"{commodity.lower()}_h{h}_predictions.csv"
            pred_df.to_csv(pred_file_path, index=False)
            print(f"  [SAVED] Prediction CSV: {pred_file_path} ({len(pred_df)} records, {pred_df['Date'].iloc[0]} to {pred_df['Date'].iloc[-1]})", flush=True)
            
            # Plot Actual vs Predicted for H=h
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(pd.to_datetime(test_dates_seq), y_test_seq, label="Actual Price", color="black", linewidth=1.5)
            # Align the full-test Naive predictions with the sequence-model
            # dates, which start after the lookback window.
            y_pred_naive_seq = y_pred_naive[SEQUENCE_LENGTH - 1:]
            ax.plot(pd.to_datetime(test_dates_seq), y_pred_naive_seq, label="Naive Baseline", color="gray", linestyle="--", alpha=0.7)
            ax.plot(pd.to_datetime(test_dates_seq), y_pred_arima[SEQUENCE_LENGTH - 1:], label="ARIMA", color="orange", linestyle=":", alpha=0.7)
            ax.plot(pd.to_datetime(test_dates_seq), y_pred_xgb[SEQUENCE_LENGTH - 1:], label="XGBoost", color="green", alpha=0.7)
            ax.plot(pd.to_datetime(test_dates_seq), y_pred_gru, label="GRU", color="blue", alpha=0.8)
            ax.plot(pd.to_datetime(test_dates_seq), y_pred_hmm_gru, label="HMM-GRU (Proposed)", color="red", linewidth=1.8)
            ax.set_title(f"{commodity}: Actual vs Predicted Prices (Horizon H = {h} Week{'s' if h>1 else ''})", fontsize=11, fontweight="bold")
            ax.set_ylabel("Price (Rs./Quintal)")
            ax.set_xlabel("Date")
            ax.legend(loc="upper left")
            fig.savefig(PRED_PLOTS_DIR / f"{commodity.lower()}_h{h}_actual_vs_predicted.png", dpi=200)
            plt.close(fig)
            
            # ==========================================================
            # 4. ABLATION STUDY FOR HORIZON h
            # ==========================================================
            print(f"Running Ablation Study for {commodity} H={h}...")
            # Level A: Price Only + GRU
            cols_A = feature_sets["Ablation_A_PriceOnly"]
            sc_A = StandardScaler()
            X_tr_A = sc_A.fit_transform(valid_feat[cols_A].iloc[train_idx])
            X_va_A = sc_A.transform(valid_feat[cols_A].iloc[val_idx])
            X_te_A = sc_A.transform(valid_feat[cols_A].iloc[test_idx])
            X_tr_A_s, y_tr_A_s = create_sequences(X_tr_A, y_train)
            X_va_A_s, y_va_A_s = create_sequences(X_va_A, y_val)
            X_te_A_s, y_te_A_s = create_sequences(X_te_A, y_test)
            m_A_gru = GRUForecaster(input_dim=len(cols_A), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h)
            m_A_gru.fit(X_tr_A_s, y_tr_A_s, X_va_A_s, y_va_A_s)
            y_pred_A = m_A_gru.predict(X_te_A_s)
            ablation_records.append({
                "Vegetable": commodity, "Level": "A", "Description": ABLATION_LEVELS["A"], "Horizon": h,
                **calculate_metrics(y_test_seq, y_pred_A, y_naive=y_naive_seq)
            })
            
            # Level B: Price + Weather + GRU
            cols_B = feature_sets["Ablation_B_Weather"]
            sc_B = StandardScaler()
            X_tr_B = sc_B.fit_transform(valid_feat[cols_B].iloc[train_idx])
            X_va_B = sc_B.transform(valid_feat[cols_B].iloc[val_idx])
            X_te_B = sc_B.transform(valid_feat[cols_B].iloc[test_idx])
            X_tr_B_s, y_tr_B_s = create_sequences(X_tr_B, y_train)
            X_va_B_s, y_va_B_s = create_sequences(X_va_B, y_val)
            X_te_B_s, y_te_B_s = create_sequences(X_te_B, y_test)
            m_B_gru = GRUForecaster(input_dim=len(cols_B), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h)
            m_B_gru.fit(X_tr_B_s, y_tr_B_s, X_va_B_s, y_va_B_s)
            y_pred_B = m_B_gru.predict(X_te_B_s)
            ablation_records.append({
                "Vegetable": commodity, "Level": "B", "Description": ABLATION_LEVELS["B"], "Horizon": h,
                **calculate_metrics(y_test_seq, y_pred_B, y_naive=y_naive_seq)
            })
            
            # Level C: Price + Technical Indicators + GRU
            cols_C = feature_sets["Ablation_C_Tech"]
            sc_C = StandardScaler()
            X_tr_C = sc_C.fit_transform(valid_feat[cols_C].iloc[train_idx])
            X_va_C = sc_C.transform(valid_feat[cols_C].iloc[val_idx])
            X_te_C = sc_C.transform(valid_feat[cols_C].iloc[test_idx])
            X_tr_C_s, y_tr_C_s = create_sequences(X_tr_C, y_train)
            X_va_C_s, y_va_C_s = create_sequences(X_va_C, y_val)
            X_te_C_s, y_te_C_s = create_sequences(X_te_C, y_test)
            m_C_gru = GRUForecaster(input_dim=len(cols_C), hidden_dim=64, num_layers=2, dropout=0.2, horizon=h)
            m_C_gru.fit(X_tr_C_s, y_tr_C_s, X_va_C_s, y_va_C_s)
            y_pred_C = m_C_gru.predict(X_te_C_s)
            ablation_records.append({
                "Vegetable": commodity, "Level": "C", "Description": ABLATION_LEVELS["C"], "Horizon": h,
                **calculate_metrics(y_test_seq, y_pred_C, y_naive=y_naive_seq)
            })
            
            # Level D: Price + Weather + Technical Indicators + GRU (matches m_gru)
            ablation_records.append({
                "Vegetable": commodity, "Level": "D", "Description": ABLATION_LEVELS["D"], "Horizon": h,
                **m_gru
            })
            
            # Level E: Full Hybrid (Price + Weather + Technical + HMM + GRU) (matches m_hmm_gru)
            ablation_records.append({
                "Vegetable": commodity, "Level": "E", "Description": ABLATION_LEVELS["E"], "Horizon": h,
                **m_hmm_gru
            })

    # ==============================================================
    # COMPILE AND SAVE MASTER RESULTS TABLES
    # ==============================================================
    print("\n" + "=" * 80)
    print("COMPILING EXPERIMENTAL RESULTS AND TABLES")
    print("=" * 80)
    
    # 1. All Results Table
    df_all_metrics = pd.DataFrame(all_metrics_records)
    df_all_metrics.to_csv(METRICS_DIR / "all_results.csv", index=False)
    print(f"Saved {METRICS_DIR / 'all_results.csv'}")
    
    # 2. Ablation Results Table
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(BASE_DIR / "results" / "ablation_results.csv", index=False)
    df_ablation.to_csv(TABLES_DIR / "ablation_summary.csv", index=False)
    print(f"Saved {BASE_DIR / 'results' / 'ablation_results.csv'}")
    
    # 3. Diebold-Mariano Results Table
    df_dm = pd.DataFrame(dm_test_records)
    df_dm.to_csv(TABLES_DIR / "diebold_mariano_results.csv", index=False)
    print(f"Saved {TABLES_DIR / 'diebold_mariano_results.csv'}")
    
    # 4. HMM Selection Results Table
    df_hmm_all = pd.concat(hmm_selection_records, ignore_index=True)
    df_hmm_all.to_csv(TABLES_DIR / "hmm_state_selection.csv", index=False)
    print(f"Saved {TABLES_DIR / 'hmm_state_selection.csv'}")
    
    # ==============================================================
    # GENERATE COMPARISON & ABLATION PLOTS
    # ==============================================================
    print("Generating Comparison & Ablation Plots...")
    
    # 1. Model Comparison Bar Chart (RMSE across models for H=1 and H=4)
    for h in [1, 4]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub_df = df_all_metrics[df_all_metrics['Horizon'] == h]
        sns.barplot(data=sub_df, x='Vegetable', y='RMSE', hue='Model', palette='tab10', ax=ax)
        ax.set_title(f"Model RMSE Comparison Across Commodities (Horizon H = {h} Week{'s' if h>1 else ''})", fontsize=11, fontweight='bold')
        ax.set_ylabel("RMSE (Rs./Quintal)")
        fig.savefig(COMP_PLOTS_DIR / f"model_rmse_comparison_h{h}.png", dpi=200)
        plt.close(fig)
        
    # 2. Horizon Decay Plot (RMSE vs Horizon for each model)
    for comm in ["Tomato", "Onion", "Potato"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sub_df = df_all_metrics[df_all_metrics['Vegetable'] == comm]
        for model_name in sub_df['Model'].unique():
            m_sub = sub_df[sub_df['Model'] == model_name]
            ax.plot(m_sub['Horizon'], m_sub['RMSE'], marker='o', label=model_name, linewidth=1.5)
        ax.set_title(f"{comm}: Forecast Error vs Horizon (H = 1, 4, 8, 12 Weeks)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Horizon (Weeks)")
        ax.set_ylabel("RMSE (Rs./Quintal)")
        ax.set_xticks(HORIZONS)
        ax.legend()
        fig.savefig(COMP_PLOTS_DIR / f"{comm.lower()}_rmse_vs_horizon.png", dpi=200)
        plt.close(fig)

    # 3. Ablation Study Bar Charts (Level A through E)
    for comm in ["Tomato", "Onion", "Potato"]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sub_abl = df_ablation[df_ablation['Vegetable'] == comm]
        sns.barplot(data=sub_abl, x='Level', y='RMSE', hue='Horizon', palette='viridis', ax=ax)
        ax.set_title(f"{comm}: Ablation Study Across Feature Tiers (A -> E)", fontsize=11, fontweight='bold')
        ax.set_ylabel("RMSE (Rs./Quintal)")
        ax.set_xlabel("Ablation Tier (A: Price, B: +Weather, C: +Tech, D: +Weather+Tech, E: +HMM+GRU)")
        fig.savefig(ABLATION_PLOTS_DIR / f"{comm.lower()}_ablation_rmse.png", dpi=200)
        plt.close(fig)

    # Refresh the dashboard accuracy cache from newly generated test predictions.
    from src.accuracy_metrics import compute_and_save_accuracy_comparison
    compute_and_save_accuracy_comparison()

    print("\nAll plots and tables successfully generated and saved!")
    return df_all_metrics, df_ablation

if __name__ == "__main__":
    run_all()
