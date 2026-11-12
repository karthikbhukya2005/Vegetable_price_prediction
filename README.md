# AgriSense --- Vegetable Price Prediction

**Project title:** Vegetable Price Prediction Using Weather-Aware Hybrid
HMM-GRU Models\
**Author:** Karthik Bhukya\
**Roll Number:** 23BCS031\
**Department:** Computer Science and Engineering, IIIT Dharwad

AgriSense is a dashboard-oriented project for exploring vegetable price
history, generating price forecasts, and evaluating forecasting models.
It focuses on Tomato, Onion, and Potato (TOP), combining time-series
modelling with Hidden Markov Model (HMM) regime information and
GRU-based forecasting.

> **Important data disclosure:** The currently prepared weekly price and
> weather datasets are synthetic/simulated. Results produced from these
> files demonstrate the project pipeline and model behaviour on
> generated data. They must not be presented as verified real-market
> forecasts or real observed market prices.

## 1. Project objectives

-   Explore historical weekly prices for Tomato, Onion, and Potato.
-   Examine price patterns and HMM-derived market regimes.
-   Build and compare forecasting approaches.
-   Evaluate forecasts using MAE, RMSE, and MAPE.
-   Present price analytics, forecasts, model performance, HMM regimes,
    and ablation results in the AgriSense dashboard.
-   Support research-oriented comparison while distinguishing this
    project's results from published-paper benchmarks.

## 2. Technology stack

The project has used or is intended to use the following technologies.
Check the repository dependency files for exact installed versions.

  Area                             Technologies
  -------------------------------- ------------------------------------------------
  Data processing                  Python, Pandas, NumPy
  Visualisation                    Matplotlib, Seaborn, frontend charting library
  Machine learning                 scikit-learn, XGBoost
  Time-series / regime modelling   statsmodels, hmmlearn
  Deep learning                    PyTorch, GRU/LSTM
  Model persistence                joblib
  Backend                          FastAPI
  Frontend                         React, Vite
  Data files                       Excel (`.xlsx`)

## 3. Existing dashboard features

1.  **Overview** --- high-level dataset and price summary.
2.  **Price Analytics** --- historical price analysis.
3.  **Forecasting** --- forecast visualisation.
4.  **Model Performance** --- model evaluation views.
5.  **HMM Regimes** --- regime-related analysis.
6.  **Ablation Study** --- comparison of model configurations.

A planned enhancement is **Forecast Accuracy & Research Comparison**, to
be added immediately below Ablation Study in the existing sidebar. It is
intended to show MAE, RMSE, and MAPE calculations, model comparison
tables, charts, and clearly labelled reference-paper benchmarks inside
the current dashboard. It is not intended to be a separate application.

## 4. Models considered

The experiment workflow has considered these forecasting baselines and
models:

-   Naive
-   ARIMA
-   XGBoost
-   LSTM
-   GRU
-   HMM-GRU

The exact models and horizons available in a particular run depend on
the experiment outputs generated in the repository. A dashboard should
display unavailable results as `N/A`, not as invented values.

## 5. Forecast horizons and evaluation metrics

The intended forecast horizons are 1, 4, 8, and 12 weeks.

### Mean Absolute Error (MAE)

`MAE = (1/n) * sum(|actual - predicted|)`

MAE is the average absolute difference between actual and predicted
values. Lower is better.

### Root Mean Squared Error (RMSE)

`RMSE = sqrt((1/n) * sum((actual - predicted)^2))`

RMSE gives greater weight to larger errors. Lower is better.

### Mean Absolute Percentage Error (MAPE)

`MAPE = (100/n) * sum(|(actual - predicted) / actual|)`

MAPE expresses error as a percentage. Zero actual values require special
handling because division by zero is undefined. The implementation
should document how zero-valued observations are treated.

### Evaluation requirements

-   Evaluate on the held-out test set, not the training set.
-   Convert predictions back to the original price scale before
    calculating metrics.
-   Align actual and predicted values by timestamp.
-   Compare models on the same test observations for a given vegetable
    and horizon.
-   Keep chronological train/validation/test splits.
-   Prevent future observations from leaking into features or
    predictions.
-   Report the evaluation period and number of test observations with
    each metric.
-   Do not compare metrics across different datasets or protocols as if
    they were directly equivalent.

## 6. Dataset files

The project has used these dataset filenames:

``` text
Data/
├── Tomato_weekly_price_weather.xlsx
├── Onion_weekly_price_weather.xlsx
└── Potato_weekly_price_weather.xlsx
```

These are the generated synthetic/simulated datasets currently described
for the project. Inspect each workbook and the data-loading code to
confirm its columns, date range, units, and missing-value handling
before running experiments.

Do not describe generated data as Agmarknet observations or any other
real data source unless the repository contains verified source data and
provenance.

## 7. Indicative repository structure

The actual repository may differ; use checked-in files as the source of
truth.

``` text
vegetable-price-prediction/
├── Data/
│   ├── Tomato_weekly_price_weather.xlsx
│   ├── Onion_weekly_price_weather.xlsx
│   └── Potato_weekly_price_weather.xlsx
├── src/
│   ├── config.py
│   ├── features.py
│   ├── hmm_model.py
│   └── ...                  # Other modelling and preprocessing modules
├── backend/
│   ├── app/
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── styles.css
├── models/
├── results/
├── run_experiments.py
└── README.md
```

## 8. Setup

Use the dependency files already present in the repository. The commands
below are typical Windows PowerShell commands; adjust paths if the
repository uses a different virtual-environment location.

### Python environment

From the repository root:

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

If the project already has a working environment under `backend\.venv`,
activate that environment instead of creating a second one.

### Frontend dependencies

``` powershell
cd frontend
npm install
```

## 9. Running the project

Start the backend and frontend in separate terminals. Confirm the exact
FastAPI import path and frontend scripts in the repository before using
these commands.

### Backend

From the repository root, if the FastAPI application is exposed as
`app.main:app`:

``` powershell
uvicorn app.main:app --reload --app-dir backend
```

If this import path does not match the repository, use the command
documented by the backend package or its existing run configuration.

### Frontend

``` powershell
cd frontend
npm run dev
```

Vite typically prints the local development URL in the terminal. Open
that URL in a browser.

## 10. Running experiments

The experiment entry point discussed during development is:

``` powershell
python run_experiments.py
```

Before running, check configured input paths, required packages,
chronological split logic, feature generation, timestamp alignment,
model output paths, and whether the run overwrites existing files in
`models/` or `results/`.

Back up important model and result artifacts before a full experiment
run. It may take time because it evaluates multiple vegetables, models,
horizons, and ablation configurations.

## 11. Research benchmark comparison

Keep published-paper values separate from this project's own experiment
results. For every benchmark, record:

-   Paper title and publication year.
-   Model name.
-   Vegetable or series, when explicitly identified by the paper.
-   Metric and value.
-   Forecast horizon, if reported.
-   Dataset frequency and units, when available.
-   Whether the paper is peer-reviewed or a preprint.
-   Comparability limitations.

Do not merge research-paper values into the project's own test metrics.
Different datasets, time frequencies, price units, test periods, and
evaluation protocols can make numerical comparisons contextual rather
than like-for-like.

## 12. Reproducibility and responsible reporting

-   Preserve source datasets and document whether they are synthetic or
    real.
-   Keep experiment configuration and random seeds where applicable.
-   Store predictions with timestamps and model/horizon identifiers.
-   Store metric outputs with the evaluation period and test sample
    count.
-   Do not report placeholder values as experimental results.
-   Do not claim that a model is best unless the comparison uses valid,
    matching test observations.
-   Clearly distinguish generated-data demonstrations from real-world
    forecasting validation.

## 13. Current development note

The next planned enhancement is **Forecast Accuracy & Research
Comparison**, immediately below **Ablation Study** in the existing
AgriSense sidebar. It should render in the existing dashboard's main
content area and reuse the current application shell, styling, backend,
and frontend architecture.

The implementation should first inspect the repository's actual
experiment outputs and prediction files. If a model/horizon result is
missing, show it as unavailable and explain what experiment needs to be
run.

------------------------------------------------------------------------

## Author

**Karthik Bhukya**\
Roll Number: **23BCS031**\
Department of Computer Science and Engineering\
Indian Institute of Information Technology Dharwad (IIIT Dharwad)
