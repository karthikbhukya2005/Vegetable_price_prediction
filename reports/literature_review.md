# Literature Review: Vegetable Price Forecasting and Market Dynamics

**Project**: Vegetable Price Prediction Using Weather-Aware Hybrid HMM-GRU Models  
**Date**: September 2026  
**Lead ML Research & Engineering Team**  

---

## 1. Executive Summary

This literature review provides a critical evaluation and synthesis of four foundational research papers provided for this project:
1. **Ajmal et al. (2024)**: *Price Dynamics of Tomato, Onion and Potato (TOP) in India* (Asian Journal of Agricultural Extension, Economics & Sociology).
2. **Avinash et al. (2024a)**: *Hidden Markov Guided Deep Learning Models for Forecasting Highly Volatile Agricultural Commodity Prices* (Preprint / Working Paper).
3. **Vani et al. (2024)**: *Forecasting of Retail Price Rise in Tomato, Onion and Potato Commodities in India* (Indian Journal of Agricultural Economics).
4. **Avinash et al. (2024b)**: *Price Forecasting of TOP (Tomato, Onion and Potato) Commodities using Hidden Markov-based Deep Learning Approach* (Statistics and Applications).

We examine the specific datasets, empirical horizons, modeling strategies, hidden regime formulations, and limitations in each work. We then explicitly delineate what our project adapts, what differences and novel extensions we introduce, and how this guides our experimental design.

---

## 2. In-Depth Review of Provided Research Papers

### 2.1 Ajmal et al. (2024)
* **Full Title**: *Price Dynamics of Tomato, Onion and Potato (TOP) in India*
* **Authors**: Ajmal S., S. Rohith, Unniravisankar P., and Osman Nabay.
* **Publication**: *Asian Journal of Agricultural Extension, Economics & Sociology*, Vol. 42, Issue 3, pp. 134–143 (March 2024). DOI: `10.9734/ajaees/2024/v42i32387`.
* **Dataset & Period**: Wholesale monthly price data collected from AGMARKNET spanning **January 2005 to December 2021** (17 years) across major producing states (Andhra Pradesh, Karnataka, Madhya Pradesh, Maharashtra, Uttar Pradesh, West Bengal, Bihar, Gujarat, Rajasthan, Punjab) and all-India aggregated series.
* **Frequency**: Monthly.
* **Features & Variables**: Nominal and real wholesale prices of Tomato, Onion, Potato.
* **Preprocessing**: Deflation using Wholesale Price Index (WPI), logarithmic transformations for growth rate calculation, and moving average adjustments for seasonal indices.
* **Methodologies**:
  - Compound Annual Growth Rate (CAGR) for long-term trends.
  - Cuddy Della Valle Index (CDVI) for measuring instability adjusted for trend.
  - Ratio-to-moving-average method for seasonal indices.
  - ARCH/GARCH(1,1) econometric models to examine price volatility persistence and clustering.
* **Key Findings**:
  - High volatility clustering was established across all three commodities ($\alpha_1 + \beta_1$ close to 1 in GARCH), confirming that price shocks persist over prolonged durations.
  - Tomatoes exhibited the highest instability index and sharp seasonality (peaking July–November).
  - Onions exhibited pronounced seasonal peaks between September and December due to the post-monsoon lean arrival period.
  - Potatoes showed moderate growth and seasonal trough during peak rabi harvest (February–April).
* **Limitations**:
  - Focuses on monthly aggregated series; does not provide short-to-medium term multi-step weekly forecasts.
  - Purely econometric and descriptive; does not employ machine learning or recurrent deep neural architectures.
  - Did not integrate exogenous meteorological variables (temperature or rainfall).

---

### 2.2 Avinash et al. (2024a - Hidden Markov Guided Deep Learning)
* **Full Title**: *Hidden Markov Guided Deep Learning Models for Forecasting Highly Volatile Agricultural Commodity Prices*
* **Authors**: G. Avinash, Ramasubramanian V., Mrinmoy Ray, Ranjit Kumar Paul, Samarth Godara, Harish Nayak, Rajeev Ranjan Kumar, B. Manjunatha.
* **Dataset & Period**: Weekly potato modal wholesale price series (Rs./Quintal) from Champadanga mandi, Hooghly district, West Bengal, spanning **January 1, 2006 to May 21, 2023** (907 observations).
* **Frequency**: Weekly.
* **Features Used (18 features)**:
  - Price measures: Modal price, Minimum price, Maximum price.
  - Trend / Momentum indicators: Simple Moving Average (SMA 7 weeks, SMA 21 weeks).
  - Volatility indicators: Bollinger Bands (Upper, Middle, Lower).
  - Frequency domain features: Fast Fourier Transform (FFT) extracted periodic components.
* **Preprocessing**: Missing values filled via linear interpolation; data partitioned chronologically into 90% training (first 816 weeks) and 10% testing (last 91 weeks). MinMax normalization scaled between [0, 1].
* **HMM Methodology**:
  - Gaussian Hidden Markov Model (HMM) fitted on price series and technical indicators.
  - State selection investigated between 2 and 6 states using Log-Likelihood, AIC, and BIC.
  - Inferred hidden state sequence ($S_t$) was transformed into one-hot or discrete regime feature and concatenated into the input tensor for sequential neural networks.
* **Models Evaluated**:
  - Classical / Standalone DL: Simple RNN, CNN-1D, LSTM, GRU, BiLSTM, BiGRU.
  - Hybrid HMM-DL: HM-RNN, HM-CNN, HM-LSTM, HM-GRU, HM-BiLSTM, HM-BiGRU.
* **Forecast Horizons**: 1-week ahead and 4-weeks ahead.
* **Evaluation Metrics**: RMSE, MAE, MAPE, and Diebold-Mariano (DM) test for statistical difference.
* **Key Findings**:
  - Hybrid HM-DL models consistently surpassed standalone DL models across both 1-week and 4-week horizons.
  - **HM-GRU achieved the lowest RMSE (79.69), MAE (61.72), and MAPE (4.30%)** for 1-week ahead potato forecasting, outperforming HM-LSTM and standalone GRU.
  - For 4-week ahead forecasting, HM-GRU maintained superiority with RMSE 113.86 and MAPE 6.49%.
* **Limitations**:
  - Restricted to a single commodity (Potato) and a single mandi (Champadanga).
  - Exogenous weather conditions (temperature, precipitation) were explicitly excluded. The authors explicitly stated in Section 5: *"Future research avenues can explore the inclusion of weather conditions, market demand, and spatial dependencies."*
  - Forecast horizon did not extend beyond 4 weeks (8-week and 12-week horizons were not tested).

---

### 2.3 Vani et al. (2024)
* **Full Title**: *Forecasting of Retail Price Rise in Tomato, Onion and Potato Commodities in India*
* **Authors**: G.K. Vani, A. Verma, D. Singh, and A. Shrivastav.
* **Publication**: *Indian Journal of Agricultural Economics*, Vol. 79, No. 3, pp. 394–406 (2024). DOI: `10.63040/25827510.2024.03.005`.
* **Dataset & Period**: Monthly CPI and WPI sub-indices from **January 2014 to April 2024** (124 months).
* **Frequency**: Monthly.
* **Features & Variables**: Consumer Price Index (retail) and Wholesale Price Index (wholesale) sub-indices for Tomato, Onion, and Potato.
* **Preprocessing**: Stationarity checks via Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) tests; differencing applied to achieve stationarity.
* **Methodologies**:
  - Autoregressive (AR) and ARIMA models.
  - Ordinary Least Squares (OLS) regression modeling wholesale price pass-through to retail prices.
  - Evaluation of predictive performance against Naive baseline models using Theil’s $U_2$ coefficient and MAPE.
* **Key Findings**:
  - Strong transmission was found from wholesale to retail prices, with wholesale price fluctuations leading retail prices.
  - Autoregressive models attained MAPE of 4.58% (Tomato), 2.02% (Onion), and 1.30% (Potato) for short-term retail index forecasting.
  - Demonstrated the mathematical necessity of using Theil's $U_2$ metric to verify whether a complex forecast beats a naive random walk ($U_2 < 1.0$).
* **Limitations**:
  - Conducted solely on monthly index level numbers rather than raw physical market prices (Rs./Quintal).
  - Macroeconomic index series smooths out sharp intra-month and weekly price spikes.
  - Deep learning and non-linear regime-switching were outside the scope of the paper.

---

### 2.4 Avinash et al. (2024b - HMM-based Deep Learning Approach)
* **Full Title**: *Price Forecasting of TOP (Tomato, Onion and Potato) Commodities using Hidden Markov-based Deep Learning Approach*
* **Authors**: G. Avinash, Ramasubramanian V., Mrinmoy Ray, Ranjit Kumar Paul, Samarth Godara, Harish Nayak, Rajeev Ranjan Kumar, B. Manjunatha.
* **Publication**: *Statistics and Applications*, Vol. 22, No. 2, pp. 63–90 (2024). ISSN: 2454-7395.
* **Dataset & Period**: Weekly wholesale modal prices (Rs./Quintal) for Tomato, Onion, and Potato from **Azadpur Mandi, Delhi** (the largest agricultural terminal wholesale market in India).
* **Frequency**: Weekly.
* **Methodology**:
  - Addressed non-linear and non-stationary dynamics of agricultural price time series.
  - Applied Gaussian HMM on the price series to uncover underlying market states (bearish, normal, bullish/volatile regimes).
  - Extracted state probabilities and state assignments were fed into recurrent neural architectures (MLP, RNN, GRU, LSTM).
  - Benchmarked baseline DL models against HM-DL hybrids.
* **Forecast Horizons**: 1-week and 4-week horizons.
* **Evaluation Metrics**: RMSE, MAE, MAPE, and Diebold-Mariano test.
* **Key Findings**:
  - Confirmed that across all three commodities (Tomato, Onion, Potato), the integration of HMM hidden state representations significantly reduced forecasting errors compared to unguided deep neural networks.
  - GRU architectures delivered lower training latency and superior parameter efficiency compared to LSTM networks with equal or better predictive accuracy.
* **Limitations**:
  - Lacked exogenous meteorological covariates.
  - Horizon evaluations stopped at 4 weeks. Medium-to-long term horizons (8 and 12 weeks) critical for harvest planning were not analyzed.
  - No systematic ablation study isolating the individual contributions of price lags, weather covariates, technical indicators, and HMM state features was provided.

---

## 3. Synthesis: Comparative Matrix of Prior Studies vs. Current Project

| Dimension | Ajmal et al. (2024) | Avinash et al. (2024a) | Vani et al. (2024) | Avinash et al. (2024b) | Current Project (Our Work) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Commodities** | Tomato, Onion, Potato | Potato only | Tomato, Onion, Potato | Tomato, Onion, Potato | **Tomato, Onion, Potato** |
| **Time Frame** | 2005 – 2021 (Monthly) | 2006 – 2023 (Weekly, 907 obs) | 2014 – 2024 (Monthly, 124 obs) | Weekly (Azadpur Mandi) | **2003 – 2025 (Weekly, ~1,170 obs per crop)** |
| **Data Length** | 17 years | 17.5 years | 10 years | ~15 years | **22.4 years continuous weekly records** |
| **Exogenous Weather** | None | None | None | None | **Temperature (°C) & Precipitation (mm)** |
| **Technical Indicators**| None | SMA, BB, FFT | None | Lagged price indicators | **SMA7, SMA21, EMA, MACD, Momentum, Bollinger Bands** |
| **Cyclical / Seasonality**| Static monthly index | None explicit | None explicit | Fourier terms | **sin/cos weekly harmonics, month, quarter, week-of-year** |
| **HMM Market Regimes** | No (GARCH used) | Gaussian HMM (2–6 states) | No (ARIMA used) | Gaussian HMM | **Gaussian HMM fitted strictly on train split** |
| **Forecasting Models** | Econometric (GARCH) | DL & HMM-DL (RNN, GRU, LSTM) | Time series (ARIMA, OLS) | MLP, RNN, GRU, LSTM + HMM | **Naive, ARIMA, XGBoost, LSTM, GRU, HMM-GRU** |
| **Forecast Horizons** | 1 month | 1 and 4 weeks | 1 month | 1 and 4 weeks | **1, 4, 8, and 12 weeks** |
| **Theil's U Metric** | Not reported | Not reported | $U_2$ calculated on index | Not reported | **Theil's $U_1$ and $U_2$ rigorously calculated** |
| **Ablation Study** | None | None | None | None | **5-tier ablation (A, B, C, D, E) across all 4 horizons** |
| **Interactive Dashboard**| None | None | None | None | **FastAPI backend + React/Vite web application** |

---

## 4. Methodological Adaptations & Distinct Innovations

### 4.1 What We Are Adapting from the Papers
1. **Gaussian HMM for Latent Regime Extraction**: Following Avinash et al. (2024a, 2024b), we adopt Gaussian HMM to decompose the volatile price series into discrete market regimes (e.g., bearish, moderate, high volatility/inflationary spike).
2. **HMM-GRU Hybrid Integration**: As demonstrated by Avinash et al., GRU matches or outperforms LSTM while avoiding parameter bloat. We adapt the sequential feeding of HMM-derived state representations as engineered inputs into GRU cells.
3. **Technical Indicators**: We incorporate moving averages (SMA 7, SMA 21) and Bollinger Bands as validated by Avinash et al., and expand them with momentum and trend-following indicators (EMA, MACD).
4. **Rigorous Baseline Benchmarking**: We adopt the Naive random walk, Box-Jenkins ARIMA, and standard deep learning baselines (LSTM, GRU) as established across the papers.

### 4.2 What is Different in Our Project
1. **Weather-Aware Modeling**: None of the four papers incorporated meteorological features into their predictive models. Our project directly answers the open research recommendation in Avinash et al. by integrating weekly mean **Temperature** and **Precipitation** together with lagged weather dynamics.
2. **Extended Multi-Step Horizons (1, 4, 8, 12 Weeks)**: Prior studies evaluated only 1-week and 4-week forecasts. We rigorously test medium-to-long term horizons (8 weeks / 2 months, and 12 weeks / 1 quarter), which are essential for farmer planting decisions and government buffer stock interventions.
3. **5-Level Systematic Ablation Study**: We conduct an explicit ablation study decomposing the predictive lift of:
   - Level A: Price only + GRU
   - Level B: Price + Weather + GRU
   - Level C: Price + Technical Indicators + GRU
   - Level D: Price + Weather + Technical Indicators + GRU
   - Level E: Full Hybrid (Price + Weather + Technical Indicators + HMM + GRU)
4. **Leakage-Free Regime Inference**: We implement a strict temporal splitting protocol: the HMM transition matrix, emission distributions, and feature scalers are fitted exclusively on the training partition ($t \le T_{\text{train}}$) and applied sequentially without look-ahead bias.
5. **Interactive Full-Stack Deployment**: We translate the academic pipeline into a live, interactive decision-support application with a FastAPI backend and React frontend.
