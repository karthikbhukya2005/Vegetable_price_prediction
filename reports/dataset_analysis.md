# Dataset Analysis Report: TOP Commodities (Tomato, Onion, Potato)

**Project**: Vegetable Price Prediction Using Weather-Aware Hybrid HMM-GRU Models  
**Date**: September 2026  
**Dataset Source**: Weekly physical wholesale modal prices and localized weather records  

---

## 1. Overview and Data Integrity

This document provides a comprehensive audit of the historical weekly price and meteorological datasets for three essential Indian vegetable commodities: Tomato, Onion, and Potato (the 'TOP' crops).
All datasets were extracted from the primary archive `TOP_data_prices.zip` and placed under `data/original/` without altering or deleting the original distribution.

---

## 2. Detailed Dataset Profile: Tomato
- **Filename**: `Tomato_weekly_price_weather.xlsx`
- **Row count**: `1,170`
- **Column count**: `4`
- **Column names & types**: `index` (datetime64[ns]), `Imputed Price` (float64), `Temperature` (float64), `Precipitation` (float64)
- **Observation date range**: `2003-04-06` to `2025-08-31` (22.4 years)
- **Temporal frequency**: Weekly regular (`7-day` delta on Sundays, `100%` adherence)
- **Missing values**: `0` across all 4 columns (100% complete)
- **Duplicate rows**: `0` (no redundant observations)
- **Duplicate timestamps**: `0` (strictly monotonic time series)
- **Date gaps**: `0` (exact match between expected calendar Sundays and recorded rows: 1170 weeks expected vs 1170 rows)

### Numerical Distributions
| Variable | Min | Max | Mean | Median | Std Dev | CV (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Price (Rs./Quintal)** | 150.00 | 10600.00 | 1339.54 | 1028.57 | 1054.96 | 78.76% |
| **Temperature (°C)** | 8.56 | 39.56 | 25.55 | 27.30 | 7.60 | 29.73% |
| **Precipitation (mm)** | 0.00 | 32.88 | 2.20 | 0.29 | 4.04 | 183.75% |

---

## 2. Detailed Dataset Profile: Onion
- **Filename**: `Onion_weekly_price_weather.xlsx`
- **Row count**: `1,180`
- **Column count**: `4`
- **Column names & types**: `index` (datetime64[ns]), `Imputed Price` (float64), `Temperature` (float64), `Precipitation` (float64)
- **Observation date range**: `2003-01-26` to `2025-08-31` (22.6 years)
- **Temporal frequency**: Weekly regular (`7-day` delta on Sundays, `100%` adherence)
- **Missing values**: `0` across all 4 columns (100% complete)
- **Duplicate rows**: `0` (no redundant observations)
- **Duplicate timestamps**: `0` (strictly monotonic time series)
- **Date gaps**: `0` (exact match between expected calendar Sundays and recorded rows: 1180 weeks expected vs 1180 rows)

### Numerical Distributions
| Variable | Min | Max | Mean | Median | Std Dev | CV (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Price (Rs./Quintal)** | 132.00 | 6700.75 | 1240.96 | 915.94 | 1020.30 | 82.22% |
| **Temperature (°C)** | 16.34 | 33.95 | 25.17 | 24.69 | 3.69 | 14.67% |
| **Precipitation (mm)** | 0.00 | 28.43 | 2.48 | 0.14 | 4.35 | 175.30% |

---

## 2. Detailed Dataset Profile: Potato
- **Filename**: `Potato_weekly_price_weather.xlsx`
- **Row count**: `1,161`
- **Column count**: `4`
- **Column names & types**: `index` (datetime64[ns]), `Imputed Price` (float64), `Temperature` (float64), `Precipitation` (float64)
- **Observation date range**: `2003-06-08` to `2025-08-31` (22.23 years)
- **Temporal frequency**: Weekly regular (`7-day` delta on Sundays, `100%` adherence)
- **Missing values**: `0` across all 4 columns (100% complete)
- **Duplicate rows**: `0` (no redundant observations)
- **Duplicate timestamps**: `0` (strictly monotonic time series)
- **Date gaps**: `0` (exact match between expected calendar Sundays and recorded rows: 1161 weeks expected vs 1161 rows)

### Numerical Distributions
| Variable | Min | Max | Mean | Median | Std Dev | CV (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Price (Rs./Quintal)** | 120.00 | 3068.33 | 738.13 | 600.00 | 496.17 | 67.22% |
| **Temperature (°C)** | 9.79 | 40.00 | 25.90 | 27.75 | 7.33 | 28.32% |
| **Precipitation (mm)** | 0.00 | 26.73 | 2.24 | 0.22 | 4.08 | 182.53% |

---

## 3. Comparative Synthesis

| Parameter | Tomato | Onion | Potato |
| :--- | :--- | :--- | :--- |
| **Filename** | `Tomato_weekly_price_weather.xlsx` | `Onion_weekly_price_weather.xlsx` | `Potato_weekly_price_weather.xlsx` |
| **Number of Rows** | 1170 | 1180 | 1161 |
| **Number of Columns** | 4 | 4 | 4 |
| **Start Date** | 2003-04-06 | 2003-01-26 | 2003-06-08 |
| **End Date** | 2025-08-31 | 2025-08-31 | 2025-08-31 |
| **Frequency** | Weekly (Sunday) | Weekly (Sunday) | Weekly (Sunday) |
| **Missing Values** | 0 | 0 | 0 |
| **Duplicate Rows** | 0 | 0 | 0 |
| **Duplicate Dates** | 0 | 0 | 0 |
| **Date Gaps** | 0 | 0 | 0 |
| **Minimum Price** | Rs. 150.00 | Rs. 132.00 | Rs. 120.00 |
| **Maximum Price** | Rs. 10600.00 | Rs. 6700.75 | Rs. 3068.33 |
| **Mean Price** | Rs. 1339.54 | Rs. 1240.96 | Rs. 738.13 |
| **Median Price** | Rs. 1028.57 | Rs. 915.94 | Rs. 600.00 |
| **Price Std Dev** | Rs. 1054.96 | Rs. 1020.30 | Rs. 496.17 |
| **Price CV (%)** | **78.76%** | **82.22%** | **67.22%** |
| **Min Temperature** | 8.56 °C | 16.34 °C | 9.79 °C |
| **Max Temperature** | 39.56 °C | 33.95 °C | 40.00 °C |
| **Mean Temperature** | 25.55 °C | 25.17 °C | 25.90 °C |
| **Min Precipitation** | 0.00 mm | 0.00 mm | 0.00 mm |
| **Max Precipitation** | 32.88 mm | 28.43 mm | 26.73 mm |
| **Mean Precipitation**| 2.20 mm | 2.48 mm | 2.24 mm |

---

## 4. Key Analytical Takeaways
1. **Volatility Hierarchy**: Onion ($CV = 82.22%$) and Tomato ($CV = 78.76%$) exhibit massive price fluctuations driven by high perishability and severe weather sensitivity. Potato exhibits comparatively moderate volatility ($CV = 67.22%$) attributable to extensive commercial cold storage infrastructure across northern and eastern India.
2. **Price Spikes and Skewness**: All three series display significant positive right-skewness (mean substantially exceeding median). The maximum price for Tomato (Rs. 10,600) is over 10 times its median, reflecting sharp episodic monsoon disruptions (e.g., July 2023).
3. **Continuous Uniform Quality**: The zero missing values and exact 7-day cadence allow us to preserve the temporal index without distortion, making them ideal for multi-step lag generation, HMM state extraction, and recurrent neural modeling.