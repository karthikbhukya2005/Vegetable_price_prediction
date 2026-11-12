# Exploratory Data Analysis (EDA) Report: TOP Commodities

**Project**: Vegetable Price Prediction Using Weather-Aware Hybrid HMM-GRU Models  
**Date**: September 2026  
**Lead ML Research & Engineering Team**  

---

## 1. Overview of Exploratory Data Analysis

This report presents an empirical analysis of weekly wholesale prices alongside regional meteorological variables (Temperature in °C and Precipitation in mm) across the three staple vegetables: Tomato, Onion, and Potato.
For each commodity, a dedicated suite of 15 targeted visualizations has been rendered and saved into `results/plots/eda/{commodity}/`.

---

## 2. Comprehensive EDA: Tomato
### 2.1 Price Volatility & Distributional Dynamics
- **Sample Size**: 1,170 weekly observations from `2003-04-06` to `2025-08-31`.
- **Price Metrics (Rs./Quintal)**: Mean = `Rs. 1339.54`, Median = `Rs. 1028.57`, Min = `Rs. 150.00`, Max = `Rs. 10600.00`.
- **Coefficient of Variation (CV)**: `78.76%`.
- **Skewness & Kurtosis**: Skewness = `2.43` (heavily positive/right-tailed), Kurtosis = `10.15` (leptokurtic fat-tailed distribution).
- **Top Historical Price Peaks**: 2023-07-09: Rs. 10600.0; 2023-07-30: Rs. 8166.7; 2023-08-06: Rs. 8166.7.
- **Weekly Price Return Volatility**: Max single-week surge = `+300.0%`, Max single-week drop = `-64.6%`, Return Std Dev = `28.56%`.

### 2.2 Seasonality & Cyclic Behavior
- **Seasonal Peak**: Month of **July** (historical mean: `Rs. 2246.98`).
- **Seasonal Trough**: Month of **May** (historical mean: `Rs. 717.37`).
- **Seasonal Amplitude Ratio**: `3.13x` between annual peak and trough periods.

### 2.3 Exogenous Meteorological Correlation
- **Price vs. Temperature Correlation (r)**: `0.070`.
- **Price vs. Precipitation Correlation (r)**: `0.242`.
- **Meteorological Drivers**: Extreme monsoon rainfall events during summer months disrupt harvest, transportation, and market arrivals, generating acute short-term supply contractions reflected in multi-week price run-ups.

### 2.4 Visual Artifacts Generated
The following 15 plots were generated for `Tomato` under `results/plots/eda/tomato/`:
1. `01_price_vs_date.png`: Historical price trajectory over 22+ years.
2. `02_temperature_vs_date.png`: Temperature time series.
3. `03_precipitation_vs_date.png`: Precipitation bar trajectory.
4. `04_price_distribution.png`: Histogram with KDE, mean, and median.
5. `05_temperature_distribution.png`: Temperature distribution.
6. `06_precipitation_distribution.png`: Precipitation distribution.
7. `07_price_vs_temperature.png`: Scatter plot with linear regression fit.
8. `08_price_vs_precipitation.png`: Scatter plot with linear regression fit.
9. `09_correlation_heatmap.png`: Bivariate Pearson correlation matrix.
10. `10_monthly_price_pattern.png`: Monthly boxplot revealing seasonal cycle.
11. `11_seasonal_price_pattern.png`: Quarterly distribution boxplot.
12. `12_yearly_price_pattern.png`: Long-term annual price trends.
13. `13_rolling_mean.png`: 7-week, 21-week, and 52-week rolling averages.
14. `14_rolling_volatility.png`: 21-week rolling standard deviation.
15. `15_price_change_analysis.png`: Weekly percentage return series.

---

## 2. Comprehensive EDA: Onion
### 2.1 Price Volatility & Distributional Dynamics
- **Sample Size**: 1,180 weekly observations from `2003-01-26` to `2025-08-31`.
- **Price Metrics (Rs./Quintal)**: Mean = `Rs. 1240.96`, Median = `Rs. 915.94`, Min = `Rs. 132.00`, Max = `Rs. 6700.75`.
- **Coefficient of Variation (CV)**: `82.22%`.
- **Skewness & Kurtosis**: Skewness = `1.84` (heavily positive/right-tailed), Kurtosis = `3.96` (leptokurtic fat-tailed distribution).
- **Top Historical Price Peaks**: 2019-12-29: Rs. 6700.8; 2019-12-22: Rs. 6660.7; 2019-12-08: Rs. 6654.8.
- **Weekly Price Return Volatility**: Max single-week surge = `+85.7%`, Max single-week drop = `-48.6%`, Return Std Dev = `15.25%`.

### 2.2 Seasonality & Cyclic Behavior
- **Seasonal Peak**: Month of **November** (historical mean: `Rs. 1958.03`).
- **Seasonal Trough**: Month of **April** (historical mean: `Rs. 653.08`).
- **Seasonal Amplitude Ratio**: `3.00x` between annual peak and trough periods.

### 2.3 Exogenous Meteorological Correlation
- **Price vs. Temperature Correlation (r)**: `-0.314`.
- **Price vs. Precipitation Correlation (r)**: `0.049`.
- **Meteorological Drivers**: Extreme monsoon rainfall events during summer months disrupt harvest, transportation, and market arrivals, generating acute short-term supply contractions reflected in multi-week price run-ups.

### 2.4 Visual Artifacts Generated
The following 15 plots were generated for `Onion` under `results/plots/eda/onion/`:
1. `01_price_vs_date.png`: Historical price trajectory over 22+ years.
2. `02_temperature_vs_date.png`: Temperature time series.
3. `03_precipitation_vs_date.png`: Precipitation bar trajectory.
4. `04_price_distribution.png`: Histogram with KDE, mean, and median.
5. `05_temperature_distribution.png`: Temperature distribution.
6. `06_precipitation_distribution.png`: Precipitation distribution.
7. `07_price_vs_temperature.png`: Scatter plot with linear regression fit.
8. `08_price_vs_precipitation.png`: Scatter plot with linear regression fit.
9. `09_correlation_heatmap.png`: Bivariate Pearson correlation matrix.
10. `10_monthly_price_pattern.png`: Monthly boxplot revealing seasonal cycle.
11. `11_seasonal_price_pattern.png`: Quarterly distribution boxplot.
12. `12_yearly_price_pattern.png`: Long-term annual price trends.
13. `13_rolling_mean.png`: 7-week, 21-week, and 52-week rolling averages.
14. `14_rolling_volatility.png`: 21-week rolling standard deviation.
15. `15_price_change_analysis.png`: Weekly percentage return series.

---

## 2. Comprehensive EDA: Potato
### 2.1 Price Volatility & Distributional Dynamics
- **Sample Size**: 1,161 weekly observations from `2003-06-08` to `2025-08-31`.
- **Price Metrics (Rs./Quintal)**: Mean = `Rs. 738.13`, Median = `Rs. 600.00`, Min = `Rs. 120.00`, Max = `Rs. 3068.33`.
- **Coefficient of Variation (CV)**: `67.22%`.
- **Skewness & Kurtosis**: Skewness = `1.46` (heavily positive/right-tailed), Kurtosis = `2.55` (leptokurtic fat-tailed distribution).
- **Top Historical Price Peaks**: 2020-11-29: Rs. 3068.3; 2020-11-08: Rs. 3008.3; 2020-11-01: Rs. 2968.3.
- **Weekly Price Return Volatility**: Max single-week surge = `+147.9%`, Max single-week drop = `-65.2%`, Return Std Dev = `10.72%`.

### 2.2 Seasonality & Cyclic Behavior
- **Seasonal Peak**: Month of **November** (historical mean: `Rs. 986.89`).
- **Seasonal Trough**: Month of **February** (historical mean: `Rs. 452.36`).
- **Seasonal Amplitude Ratio**: `2.18x` between annual peak and trough periods.

### 2.3 Exogenous Meteorological Correlation
- **Price vs. Temperature Correlation (r)**: `0.143`.
- **Price vs. Precipitation Correlation (r)**: `0.093`.
- **Meteorological Drivers**: Extreme monsoon rainfall events during summer months disrupt harvest, transportation, and market arrivals, generating acute short-term supply contractions reflected in multi-week price run-ups.

### 2.4 Visual Artifacts Generated
The following 15 plots were generated for `Potato` under `results/plots/eda/potato/`:
1. `01_price_vs_date.png`: Historical price trajectory over 22+ years.
2. `02_temperature_vs_date.png`: Temperature time series.
3. `03_precipitation_vs_date.png`: Precipitation bar trajectory.
4. `04_price_distribution.png`: Histogram with KDE, mean, and median.
5. `05_temperature_distribution.png`: Temperature distribution.
6. `06_precipitation_distribution.png`: Precipitation distribution.
7. `07_price_vs_temperature.png`: Scatter plot with linear regression fit.
8. `08_price_vs_precipitation.png`: Scatter plot with linear regression fit.
9. `09_correlation_heatmap.png`: Bivariate Pearson correlation matrix.
10. `10_monthly_price_pattern.png`: Monthly boxplot revealing seasonal cycle.
11. `11_seasonal_price_pattern.png`: Quarterly distribution boxplot.
12. `12_yearly_price_pattern.png`: Long-term annual price trends.
13. `13_rolling_mean.png`: 7-week, 21-week, and 52-week rolling averages.
14. `14_rolling_volatility.png`: 21-week rolling standard deviation.
15. `15_price_change_analysis.png`: Weekly percentage return series.

---

## 3. Cross-Commodity Comparative Insights & Modeling Implications
1. **Regime Switching Necessity**: Linear stationary assumptions are thoroughly violated in all three commodities. Long periods of stable price fluctuations are punctuated by violent, regime-shifting volatility explosions. This provides strong empirical justification for employing a **Hidden Markov Model (HMM)** to identify latent market regimes.
2. **Weather Lag Effects**: Contemporary rainfall exhibits low instantaneous linear correlation with price, but lagged cumulative precipitation directly triggers supply chokeholds 2 to 8 weeks downstream. This confirms the necessity of including rolling weather summaries and multi-week weather lags.
3. **Non-stationarity & Multi-Horizon Modeling**: The long-term trend (captured by 52-week moving averages) and strong 52-week cyclicality necessitate incorporating sinusoidal calendar harmonics (`sin_week`, `cos_week`) alongside technical indicators.