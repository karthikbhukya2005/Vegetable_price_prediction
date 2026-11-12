import { useEffect, useState } from "react";
import { api } from "./api";


import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";


import {
  Leaf,
  LayoutDashboard,
  BarChart3,
  TrendingUp,
  Activity,
  BrainCircuit,
  FlaskConical,
  RefreshCw,
  Download,
} from "lucide-react";

// Dashboard navigation
const sections = [
  "Overview",
  "Price Analytics",
  "Forecasting",
  "Model Performance",
  "HMM Regimes",
  "Ablation Study",
  "Forecast Accuracy & Research Comparison",
];

// Available vegetables
const vegetables = ["tomato", "onion", "potato"];

// Format price in Indian Rupees
const money = (value) => {
  if (value == null || Number.isNaN(Number(value))) {
    return "—";
  }

  return `₹${Number(value).toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  })}`;
};

// Reusable table component
function Table({ rows }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="empty">
        No CSV data found for this section yet.
      </div>
    );
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>
                {column.replaceAll("_", " ")}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.slice(0, 50).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>
                  {row[column] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Reusable dashboard statistic card
function StatCard({ title, value, subtitle, icon: Icon }) {
  return (
    <article className="stat-card">
      <div className="stat-icon">
        <Icon size={20} />
      </div>

      <small>{title}</small>

      <h2>{value}</h2>

      <span>{subtitle}</span>
    </article>
  );
}

function ModelComparison({ rows, vegetable, horizon }) {
  const [metric, setMetric] = useState("RMSE");

  if (!rows || rows.length === 0) {
    return (
      <div className="empty">
        No model results found. Check results/all_results.csv.
      </div>
    );
  }

  // Normalize column names to handle different CSV naming styles.
  const normalize = (value) =>
    String(value).toLowerCase().replace(/[^a-z0-9]/g, "");

  const columns = Object.keys(rows[0]);

  const findColumn = (names) => {
    for (const name of names) {
      const found = columns.find(
        (column) => normalize(column) === normalize(name)
      );
      if (found) return found;
    }

    return null;
  };

  const vegetableColumn = findColumn(["Vegetable", "Commodity"]);
  const modelColumn = findColumn(["Model", "Model_Name", "Algorithm"]);
  const horizonColumn = findColumn([
    "Horizon",
    "Horizon_Weeks",
    "Forecast_Horizon",
    "Weeks_Ahead",
  ]);

  const metricColumns = {
    RMSE: findColumn(["RMSE", "Root_Mean_Squared_Error"]),
    MAE: findColumn(["MAE", "Mean_Absolute_Error"]),
    MAPE: findColumn(["MAPE", "Mean_Absolute_Percentage_Error"]),
  };

  const metricColumn = metricColumns[metric];

  if (!modelColumn || !metricColumn) {
    return (
      <div className="error">
        Could not identify the Model or {metric} column in all_results.csv.
        <br />
        Available columns: {columns.join(", ")}
      </div>
    );
  }

  let filteredRows = rows;

  if (vegetableColumn) {
    filteredRows = filteredRows.filter(
      (row) =>
        String(row[vegetableColumn]).toLowerCase().includes(vegetable)
    );
  }

  if (horizonColumn) {
    filteredRows = filteredRows.filter((row) => {
      const value = String(row[horizonColumn]);
      const number = value.match(/\d+/);
      return number && Number(number[0]) === Number(horizon);
    });
  }

  const chartData = filteredRows
    .map((row) => ({
      model: String(row[modelColumn]),
      value: Number(row[metricColumn]),
    }))
    .filter((row) => Number.isFinite(row.value));

  if (chartData.length === 0) {
    return (
      <div className="empty">
        No matching model results for {vegetable} at {horizon} week(s).
        Check the column names and values in all_results.csv.
      </div>
    );
  }

  return (
    <div>
      <div className="metric-filter">
        <label>
          EVALUATION METRIC
          <select value={metric} onChange={(event) => setMetric(event.target.value)}
          >
            {Object.keys(metricColumns).map((name) => (
              <option key={name} value={name} disabled={!metricColumns[name]}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="chart large">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 15, right: 25, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />

            <XAxis dataKey="model" interval={0} angle={-15} textAnchor="end" height={65} />

            <YAxis />

            <Tooltip formatter={(value) => [
              Number(value).toFixed(3),
              metric,
            ]}
            />

            <Legend />

            <Bar dataKey="value" name={metric} fill="#3978cf" radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <Table rows={chartData.map((item) => ({
        Model: item.model,
        [metric]: item.value,
      }))}
      />
    </div>
  );
}


function FutureForecastVisualization({ chartData, vegetable, horizon }) {
  if (!chartData) {
    return <div className="loading">Loading genuine future forecast data...</div>;
  }

  const {
    forecast_origin_date,
    last_observed_price,
    selected_forecast,
    historical_series,
    future_curve,
    multi_horizon_forecasts,
  } = chartData;

  // Build combined dataset for line chart
  const combinedChartData = [];

  if (historical_series) {
    historical_series.forEach((pt) => {
      combinedChartData.push({
        date: pt.date,
        actual: pt.actual,
        future_predicted: null,
        period: "historical",
      });
    });
  }

  // Future curve includes origin point (2026-11-08) and future target points (2026-11-15, etc.)
  if (future_curve) {
    future_curve.forEach((pt) => {
      if (pt.period === "origin") {
        const existing = combinedChartData.find((d) => d.date === pt.date);
        if (existing) {
          existing.future_predicted = pt.predicted;
        } else {
          combinedChartData.push({
            date: pt.date,
            actual: pt.actual,
            future_predicted: pt.predicted,
            period: "origin",
          });
        }
      } else {
        combinedChartData.push({
          date: pt.date,
          actual: null,
          future_predicted: pt.predicted,
          horizon: pt.horizon,
          period: "future",
        });
      }
    });
  }

  return (
    <div className="future-forecast-container">
      {/* KPI STAT CARDS */}
      <div className="cards">
        <StatCard
          title="Forecast Origin Date"
          value={forecast_origin_date || "2026-11-08"}
          subtitle={`Observed Price: ${money(last_observed_price)}`}
          icon={Activity}
        />
        <StatCard
          title={`Target Date (${horizon} Week${horizon > 1 ? "s" : ""})`}
          value={selected_forecast?.forecast_target_date || "—"}
          subtitle="Strictly future date"
          icon={TrendingUp}
        />
        <StatCard
          title="Predicted Future Price"
          value={selected_forecast ? money(selected_forecast.predicted_price) : "—"}
          subtitle={
            selected_forecast
              ? `Expected: ${selected_forecast.price_change >= 0 ? "+" : ""}${money(selected_forecast.price_change)} (${selected_forecast.percentage_change >= 0 ? "+" : ""}${selected_forecast.percentage_change}%)`
              : "HMM-GRU Output"
          }
          icon={TrendingUp}
        />
        <StatCard
          title="Forecasting Model"
          value="HMM-GRU"
          subtitle="Direct Multi-Horizon (Proposed)"
          icon={BrainCircuit}
        />
      </div>

      <div className="warning data-simulation-notice" style={{ margin: "14px 0" }}>
        <strong>Genuine Future Prediction:</strong> The forecast origin is strictly fixed at the latest historical observation (<strong>{forecast_origin_date}</strong>). Future projections are computed strictly for future dates without look-ahead actual prices or future weather leakage.
      </div>

      {/* CHART */}
      <div className="chart large" style={{ height: "390px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={combinedChartData} margin={{ top: 25, right: 30, left: 10, bottom: 15 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" minTickGap={35} />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => money(v)} />
            <Tooltip
              formatter={(value, name) => [money(value), name]}
              labelFormatter={(label) => {
                if (label === forecast_origin_date) return `${label} — FORECAST ORIGIN`;
                if (label > forecast_origin_date) return `${label} — GENUINE FUTURE FORECAST`;
                return `${label} — Historical Observation`;
              }}
            />
            <Legend />
            <ReferenceLine
              x={forecast_origin_date}
              stroke="#d97706"
              strokeDasharray="4 4"
              strokeWidth={2}
              label={{
                value: `Origin (${forecast_origin_date})`,
                position: "top",
                fill: "#b45309",
                fontSize: 11,
                fontWeight: "bold",
              }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              name="Historical Price (Observed)"
              stroke="#16a34a"
              strokeWidth={2.5}
              dot={false}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="future_predicted"
              name="Future Forecast (HMM-GRU)"
              stroke="#2563eb"
              strokeWidth={2.5}
              strokeDasharray="5 5"
              dot={{ r: 5, fill: "#2563eb" }}
              connectNulls={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* MULTI-HORIZON TABLE */}
      <div className="forecast-summary">
        <h3>Multi-Horizon Future Forecasts (Strictly &gt; {forecast_origin_date})</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Target Date</th>
                <th>Origin Date</th>
                <th>Origin Price</th>
                <th>Predicted Price</th>
                <th>Expected Change</th>
                <th>Percentage Change</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {multi_horizon_forecasts && multi_horizon_forecasts.map((row) => (
                <tr
                  key={row.horizon_weeks}
                  style={row.horizon_weeks === horizon ? { background: "#eff6ff", fontWeight: "bold" } : {}}
                >
                  <td>
                    {row.horizon_weeks} week{row.horizon_weeks > 1 ? "s" : ""}{" "}
                    {row.horizon_weeks === horizon ? "★ (Selected)" : ""}
                  </td>
                  <td><strong>{row.forecast_target_date}</strong></td>
                  <td>{row.forecast_origin_date}</td>
                  <td>{money(row.last_observed_price)}</td>
                  <td style={{ color: "#1d4ed8", fontWeight: "bold" }}>{money(row.predicted_price)}</td>
                  <td style={{ color: row.price_change >= 0 ? "#15803d" : "#b91c1c" }}>
                    {row.price_change >= 0 ? "+" : ""}{money(row.price_change)}
                  </td>
                  <td style={{ color: row.percentage_change >= 0 ? "#15803d" : "#b91c1c" }}>
                    {row.percentage_change >= 0 ? "+" : ""}{row.percentage_change}%
                  </td>
                  <td>{row.model}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function HistoricalEvaluationChart({ rows }) {
  const [selectedModel, setSelectedModel] = useState("HMM_GRU");

  const models = [
    "Naive",
    "ARIMA",
    "XGBoost",
    "LSTM",
    "GRU",
    "HMM_GRU",
  ];

  if (!rows || rows.length === 0) {
    return (
      <div className="empty">
        No test evaluation records found.
        Check your prediction CSV files.
      </div>
    );
  }

  const chartData = rows.map((row) => ({
    date: row.Date,
    actual: Number(row.Actual_Price),
    predicted: Number(row[selectedModel]),
  })).filter(
    (row) =>
      Number.isFinite(row.actual) &&
      Number.isFinite(row.predicted)
  );

  if (chartData.length === 0) {
    return (
      <div className="empty">
        No valid test forecast values found for this model.
      </div>
    );
  }

  return (
    <>
      <div className="metric-filter">
        <label>
          SELECT EVALUATION MODEL
          <select value={selectedModel} onChange={(event) =>
            setSelectedModel(event.target.value)
          }
          >
            {models.map((model) => (
              <option key={model} value={model}>
                {model === "HMM_GRU"
                  ? "HMM-GRU (Proposed Model)"
                  : model}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="chart large">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 15, right: 25, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" minTickGap={35} />
            <YAxis scale="sqrt" domain={[0, "auto"]} />
            <Tooltip formatter={(value) => money(value)} />
            <Legend />
            <Line type="monotone" dataKey="actual" name="Actual Test Price" stroke="#26966f" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="predicted" name={selectedModel} stroke="#3978cf" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="forecast-summary">
        <h3>Out-of-sample test split predictions</h3>
        <Table rows={chartData.map((row) => ({
          Date: row.date,
          Actual_Price: row.actual,
          [selectedModel]: row.predicted,
          Absolute_Error: Math.abs(
            row.actual - row.predicted
          ).toFixed(2),
        }))}
        />
      </div>
    </>
  );
}



function AblationComparison({ rows, vegetable, horizon }) {
  const [metric, setMetric] = useState("RMSE");
  const field = (row, name) => {
    const key = Object.keys(row || {}).find((item) =>
      item.toLowerCase().replace(/[^a-z0-9]/g, "") === name.toLowerCase().replace(/[^a-z0-9]/g, "")
    );
    return key ? row[key] : undefined;
  };
  const selected = rows.filter((r) =>
    String(field(r, "Vegetable") ?? field(r, "Commodity") ?? "").toLowerCase() === vegetable &&
    Number(field(r, "Horizon") ?? field(r, "Horizon_Weeks")) === Number(horizon)
  );
  const metricNames = ["MAE", "RMSE", "MAPE", "R2", "Theil_U1", "Theil_U2"];
  const data = selected.map((r) => ({
    level: `Level ${field(r, "Level") ?? "?"}`,
    description: field(r, "Description") ?? "Feature configuration",
    value: Number(field(r, metric)),
  })).filter((r) => Number.isFinite(r.value));

  if (!data.length) return <div className="empty">No ablation rows for {vegetable} at {horizon} weeks.</div>;

  const best = [...data].sort((x, y) =>
    metric === "R2" ? y.value - x.value : x.value - y.value
  )[0];

  return (
    <>
      <div className="section-head">
        <div>
          <h3>Feature ablation comparison</h3>
          <p className="muted">Compare configurations A–E for the selected commodity and horizon. Lower MAE, RMSE, MAPE and Theil values are better; higher R² is better.</p>
        </div>
        <label className="control-label">SELECT EVALUATION METRIC
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {metricNames.map((m) => <option key={m} value={m}>{m.replaceAll("_", " ")}</option>)}
          </select>
        </label>
      </div>
      <div className="cards">
        <StatCard title="Best configuration" value={best.level} subtitle={best.description} icon={FlaskConical} />
        <StatCard title={`Best ${metric}`} value={best.value.toFixed(3)} subtitle="For selected vegetable and horizon" icon={Activity} />
        <StatCard title="Experiments compared" value={data.length} subtitle="Expected five feature configurations" icon={BarChart3} />
      </div>
      <div className="ablation-chart-panel">
        <h3>{metric.replaceAll("_", " ")} by feature configuration</h3>
        <div className="chart large">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 15, right: 20, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="level" />
              <YAxis domain={metric === "R2" ? ["auto", "auto"] : [0, "auto"]} />
              <Tooltip formatter={(v) => Number(v).toFixed(3)} />
              <Legend />
              <Bar dataKey="value" name={metric.replaceAll("_", " ")} fill="#3978cf" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <h3>Detailed ablation metrics</h3>
      <Table rows={selected} />
    </>
  );
}

function HMMRegimeChart({ rows, vegetable }) {
  const [view, setView] = useState("mean_price");

  if (!rows || rows.length === 0) {
    return (
      <div className="empty">
        No HMM regime statistics available for {vegetable}.
        Check results/tables/{vegetable}_regime_stats.csv.
      </div>
    );
  }

  const chartData = rows.map((row) => ({
    regime: `State ${row.regime}`,
    mean_price: Number(row.mean_price),
    count: Number(row.count),
    percentage: Number(row.percentage),
    min_price: Number(row.min_price),
    max_price: Number(row.max_price),
    median_price: Number(row.median_price),
    std_price: Number(row.std_price),
  }));

  const metricOptions = [
    { value: "mean_price", label: "Average Price" },
    { value: "count", label: "Observation Count" },
    { value: "percentage", label: "Regime Percentage" },
  ];

  const selectedMetric = metricOptions.find(
    (item) => item.value === view
  );

  const totalObservations = chartData.reduce(
    (sum, row) => sum + row.count,
    0
  );

  const highestRegime = [...chartData].sort(
    (a, b) => b.mean_price - a.mean_price
  )[0];

  return (
    <>
      <div className="cards">
        <StatCard title="Number of HMM regimes" value={chartData.length} subtitle="Identified hidden states"
          icon={BrainCircuit} />

        <StatCard title="Total observations" value={totalObservations.toLocaleString("en-IN")}
          subtitle="Observations assigned to regimes" icon={Activity} />

        <StatCard title="Highest mean-price regime" value={highestRegime.regime} subtitle={money(highestRegime.mean_price)}
          icon={TrendingUp} />

        <StatCard title="Highest regime percentage" value={`${Math.max(...chartData.map((row) => row.percentage)
        ).toFixed(2)}%`}
          subtitle="Share of observations"
          icon={BarChart3}
        />
      </div>

      <section>
        <div className="section-head">
          <h3>HMM Regime Distribution</h3>

          <select value={view} onChange={(event) => setView(event.target.value)}
          >
            {metricOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="chart large">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />

              <XAxis dataKey="regime" />

              <YAxis />

              <Tooltip formatter={(value) =>
                view === "mean_price"
                  ? money(value)
                  : view === "percentage"
                    ? `${Number(value).toFixed(2)}%`
                    : Number(value).toLocaleString("en-IN")
              }
              />

              <Legend />

              <Bar dataKey={view} name={selectedMetric.label} fill="#3978cf" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <h3>Price Range Across Regimes</h3>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Regime</th>
                <th>Mean Price</th>
                <th>Median Price</th>
                <th>Minimum</th>
                <th>Maximum</th>
                <th>Std. Deviation</th>
              </tr>
            </thead>

            <tbody>
              {chartData.map((row) => (
                <tr key={row.regime}>
                  <td>{row.regime}</td>
                  <td>{money(row.mean_price)}</td>
                  <td>{money(row.median_price)}</td>
                  <td>{money(row.min_price)}</td>
                  <td>{money(row.max_price)}</td>
                  <td>{money(row.std_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3>Regime Observation Distribution</h3>
        <Table rows={chartData.map((row) => ({
          Regime: row.regime,
          Observations: row.count,
          Percentage: `${row.percentage.toFixed(2)}%`,
        }))}
        />
      </section>
    </>
  );
}



function AccuracyResearchComparison({ vegetable, horizon }) {
  const [summary, setSummary] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [research, setResearch] = useState(null);
  const [predictionRows, setPredictionRows] = useState([]);
  const [selectedModel, setSelectedModel] = useState("HMM_GRU");
  const [requestError, setRequestError] = useState("");
  const [requestLoading, setRequestLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setRequestLoading(true);
    setRequestError("");

    Promise.all([
      api.get("/accuracy/summary", { params: { vegetable, horizon } }),
      api.get("/accuracy/comparison", { params: { vegetable, horizon } }),
      api.get("/accuracy/research-benchmarks", { params: { vegetable, horizon } }),
      api.get("/accuracy/predictions", { params: { vegetable, horizon, model: selectedModel } }),
    ]).then(([summaryResponse, comparisonResponse, researchResponse, predictionResponse]) => {
      if (!active) return;
      setSummary(summaryResponse.data);
      setComparison(comparisonResponse.data);
      setResearch(researchResponse.data);
      setPredictionRows(predictionResponse.data.data || []);
    }).catch((err) => {
      if (!active) return;
      setRequestError(err.response?.data?.detail || "Could not load accuracy comparison data. Check the backend and prediction CSV files.");
      setSummary(null);
      setComparison(null);
      setResearch(null);
      setPredictionRows([]);
    }).finally(() => {
      if (active) setRequestLoading(false);
    });

    return () => { active = false; };
  }, [vegetable, horizon, selectedModel]);

  const metricCards = [
    { title: "Best MAE", item: summary?.best_mae, suffix: "price units", icon: Activity },
    { title: "Best RMSE", item: summary?.best_rmse, suffix: "price units", icon: BarChart3 },
    { title: "Best MAPE", item: summary?.best_mape, suffix: "%", icon: TrendingUp },
  ];

  const metricNames = ["MAE", "RMSE", "MAPE"];
  const metricRows = comparison?.metric_comparison || [];
  const horizonRows = comparison?.horizon_vs_rmse || [];
  const commodityRows = comparison?.commodity_comparison || [];
  const researchRows = research?.comparison_table || [];
  const predictionChartRows = predictionRows.map((row) => ({
    date: row.date,
    actual: row.actual,
    predicted: row.predicted,
  })).filter((row) => Number.isFinite(Number(row.actual)) && Number.isFinite(Number(row.predicted)));

  if (requestLoading && !summary && !comparison) {
    return <div className="loading">Calculating and loading forecast accuracy metrics...</div>;
  }

  return (
    <>
      <div className="warning data-simulation-notice">
        <strong>Synthetic-data disclosure:</strong> Project evaluation uses synthetic/simulated vegetable price and weather data. These metrics describe performance on the generated dataset and are not verified real-market forecasting results.
      </div>

      {requestError && <div className="error">{requestError}</div>}

      <section>
        <div className="section-head">
          <div>
            <h3>Forecast Accuracy Overview</h3>
            <p className="muted">{vegetable.toUpperCase()} · {horizon}-week-ahead held-out test evaluation. Lower MAE, RMSE and MAPE indicate smaller errors.</p>
          </div>
          <span>{summary?.total_models_evaluated ?? 0} model rows</span>
        </div>
        <div className="cards">
          {metricCards.map(({ title, item, suffix, icon }) => (
            <StatCard key={title} title={title} value={item?.value == null ? "N/A" : Number(item.value).toFixed(3)} subtitle={item?.model ? `Best model: ${item.model} · ${suffix}` : "No verified result available"} icon={icon} />
          ))}
          <StatCard title="Best by RMSE" value={summary?.best_model?.model || "N/A"} subtitle={summary?.best_model?.rmse == null ? "No verified result available" : `RMSE: ${Number(summary.best_model.rmse).toFixed(3)}`} icon={BrainCircuit} />
        </div>
      </section>

      <section>
        <div className="section-head">
          <div>
            <h3>Model Accuracy Comparison</h3>
            <p className="muted">Metrics are calculated from aligned test-set prediction files in the original price scale.</p>
          </div>
        </div>
        {comparison?.table?.length ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE (%)</th><th>Overall Rank</th><th>MAE Rank</th><th>RMSE Rank</th><th>MAPE Rank</th></tr></thead>
              <tbody>
                {comparison.table.map((row) => (
                  <tr key={row.Model} style={row.is_best ? { background: "#effaf2", fontWeight: 700 } : undefined}>
                    <td>{row.Model}{row.is_best ? <span className="badge-best">Best RMSE</span> : null}</td>
                    <td>{row.MAE}</td><td>{row.RMSE}</td><td>{row.MAPE}</td>
                    <td>{row.Rank}</td><td>{row.Rank_MAE}</td><td>{row.Rank_RMSE}</td><td>{row.Rank_MAPE}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty">No accuracy metrics are available for this vegetable and horizon yet. Run the experiment pipeline to generate test predictions.</div>}
      </section>

      <section>
        <h3>Metric-wise Model Comparison</h3>
        <div className="charts-grid-3">
          {metricNames.map((metric) => (
            <div className="ablation-chart-panel" key={metric}>
              <h3>{metric} (lower is better)</h3>
              <div className="chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricRows} margin={{ top: 10, right: 8, left: 0, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="model" angle={-25} textAnchor="end" interval={0} height={55} />
                    <YAxis />
                    <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                    <Bar dataKey={metric} name={metric} fill="#3978cf" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3>Forecast Horizon vs RMSE</h3>
        <p className="muted">How error changes as the forecast horizon increases for {vegetable.toUpperCase()}.</p>
        <div className="chart large">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={horizonRows}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="horizon" />
              <YAxis />
              <Tooltip />
              <Legend />
              {["Naive", "ARIMA", "XGBoost", "LSTM", "GRU", "HMM-GRU"].map((model, index) => (
                <Line key={model} type="monotone" dataKey={model} name={model} stroke={["#64748b", "#d97706", "#16a34a", "#8b5cf6", "#2563eb", "#dc2626"][index]} connectNulls dot={{ r: 3 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <h3>Commodity-wise RMSE Comparison</h3>
        <p className="muted">Compare Tomato, Onion and Potato at the selected {horizon}-week horizon. Values are in the project's price units.</p>
        <div className="chart large">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={commodityRows}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Tomato" fill="#dc6b57" />
              <Bar dataKey="Onion" fill="#8b78c6" />
              <Bar dataKey="Potato" fill="#d5a441" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section>
        <div className="section-head">
          <div>
            <h3>Actual vs Predicted Test Prices</h3>
            <p className="muted">Timestamp-aligned out-of-sample predictions for {vegetable.toUpperCase()} · {horizon} weeks.</p>
          </div>
          <label className="control-label">MODEL
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              <option value="Naive">Naive</option><option value="ARIMA">ARIMA</option><option value="XGBoost">XGBoost</option><option value="LSTM">LSTM</option><option value="GRU">GRU</option><option value="HMM_GRU">HMM-GRU</option>
            </select>
          </label>
        </div>
        {predictionChartRows.length ? (
          <>
            <div className="chart large">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictionChartRows}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" minTickGap={35} />
                  <YAxis />
                  <Tooltip formatter={(value) => money(value)} />
                  <Legend />
                  <Line type="monotone" dataKey="actual" name="Actual test price" stroke="#26966f" strokeWidth={2.3} dot={false} />
                  <Line type="monotone" dataKey="predicted" name={selectedModel === "HMM_GRU" ? "HMM-GRU prediction" : `${selectedModel} prediction`} stroke="#3978cf" strokeWidth={2.3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="muted">MAE: {predictionRows.metrics?.MAE ?? "N/A"} · RMSE: {predictionRows.metrics?.RMSE ?? "N/A"} · MAPE: {predictionRows.metrics?.MAPE ?? "N/A"}</p>
          </>
        ) : <div className="empty">No aligned test predictions available for {selectedModel}.</div>}
      </section>

      <section className="research-section">
        <h3>Comparison with Published Research</h3>
        <p className="muted">Project metrics and published-paper metrics are displayed in separate categories. Cross-study values are contextual, not direct like-for-like rankings.</p>
        <div className="research-meta">
          <h4>Reference Paper 1 — TOP HMM-based Deep Learning (2024)</h4>
          Test-set benchmarks for GRU, HM-GRU and the paper's best hybrid model. The paper uses a different dataset and evaluation protocol.
        </div>
        <Table rows={researchRows} />
        <div className="research-meta">
          <h4>Reference Paper 2 — Retail Price Rise Forecasting (2024)</h4>
          Monthly CPI/WPI index MAPE benchmarks; these are contextual and not directly equivalent to this project's weekly price-series metrics.
        </div>
        <Table rows={(research?.reference_paper_2 || []).filter((row) => String(row.Vegetable).toLowerCase() === vegetable)} />
        <div className="research-meta">
          <h4>Reference Paper 3 — TOP Price Dynamics (2024)</h4>
          {research?.reference_paper_3?.Note || "GARCH-based volatility research; not a direct MAE/RMSE/MAPE forecasting benchmark."}
        </div>
        <div className="research-meta">
          <h4>Reference Paper 4 — HMM Multi-horizon Preprint (2024)</h4>
          Preprint benchmark. The commodity/series identity is not assigned here unless verified in the source.
        </div>
        <Table rows={research?.reference_paper_4_preprint || []} />
        <div className="warning">{research?.notice || "Differences in dataset, frequency, test period and evaluation protocol limit direct cross-paper comparisons."}</div>
      </section>
    </>
  );
}

// Main application
export default function App() {
  const [section, setSection] = useState("Overview");

  const [vegetable, setVegetable] = useState("tomato");

  const [horizon, setHorizon] = useState(1);

  // API data states
  const [overview, setOverview] = useState([]);
  const [history, setHistory] = useState([]);

  const [metrics, setMetrics] = useState([]);
  const [ablation, setAblation] = useState([]);
  const [hmm, setHmm] = useState([]);
  const [regimes, setRegimes] = useState([]);
  const [dm, setDm] = useState([]);

  const [predictions, setPredictions] = useState([]);
  const [futureForecast, setFutureForecast] = useState(null);
  const [futureError, setFutureError] = useState("");
  const [forecastChartData, setForecastChartData] = useState(null);
  const [forecastChartError, setForecastChartError] = useState("");

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Load all dashboard data

  async function loadData() {
    setLoading(true);
    setError("");

    try {
      // Load overview and selected vegetable history
      const [overviewResponse, historyResponse] =
        await Promise.all([
          api.get("/overview"),
          api.get(`/history/${vegetable}`),
        ]);

      setOverview(
        overviewResponse.data.vegetables || []
      );

      setHistory(
        historyResponse.data.data || []
      );

    } catch (err) {
      console.error("Dashboard API error:", err);

      setError(
        err.response?.data?.detail ||
        "Cannot connect to FastAPI. Start the backend on port 8000."
      );

      setOverview([]);
      setHistory([]);
    }

    // Load experiment result CSV files
    const loadResult = async (endpoint, setter) => {
      try {
        const response = await api.get(
          `/results/${endpoint}`
        );

        setter(response.data.data || []);

      } catch (err) {
        console.error(`${endpoint} loading error:`, err);
        setter([]);
      }
    };

    await Promise.all([
      loadResult("metrics", setMetrics),
      loadResult("ablation", setAblation),
      loadResult("hmm", setHmm),
      loadResult("dm", setDm),
    ]);

    // Load HMM regime statistics for selected vegetable
    try {
      const regimeResponse = await api.get(
        `/regimes/${vegetable}`
      );

      setRegimes(
        regimeResponse.data.data || []
      );

    } catch (err) {
      console.error("HMM regime loading error:", err);
      setRegimes([]);
    }

    // Load selected vegetable forecast
    try {
      const predictionResponse = await api.get(
        `/predictions/${vegetable}/${horizon}`
      );

      setPredictions(
        predictionResponse.data.data || []
      );

    } catch (err) {
      console.error("Prediction loading error:", err);
      setPredictions([]);
    }

    try {
      const futureResponse = await api.get(`/future/${vegetable}/${horizon}`);
      setFutureForecast(futureResponse.data);
      setFutureError("");
    } catch (err) {
      setFutureForecast(null);
      setFutureError(err.response?.data?.detail || "Future forecast is unavailable.");
    }

    try {
      const chartResp = await api.get(`/forecast_chart_data/${vegetable}/${horizon}?model=HMM_GRU`);
      setForecastChartData(chartResp.data);
      setForecastChartError("");
    } catch (err) {
      console.error("Forecast chart error:", err);
      setForecastChartData(null);
      setForecastChartError(err.response?.data?.detail || "Future forecast chart unavailable.");
    }

    setLoading(false);
  }


  // Reload data when filters change
  useEffect(() => {
    loadData();
  }, [vegetable, horizon]);

  // Latest recorded price
  const latest = history.length
    ? history[history.length - 1]
    : null;

  // Average historical price
  const average = history.length
    ? history.reduce(
      (sum, row) => sum + Number(row.price || 0),
      0
    ) / history.length
    : null;

  // Export data to CSV
  function download(rows, filename) {
    if (!rows || rows.length === 0) {
      window.alert("No data available to export.");
      return;
    }

    const columns = Object.keys(rows[0]);

    const csv = [
      columns.join(","),

      ...rows.map((row) =>
        columns
          .map((column) =>
            JSON.stringify(row[column] ?? "")
          )
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);

    link.click();

    link.remove();

    URL.revokeObjectURL(url);
  }

  // Historical price chart
  const priceChart = (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={history}>
        <defs>
          <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3978cf" stopOpacity={0.25} />

            <stop offset="100%" stopColor="#3978cf" stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" vertical={false} />

        <XAxis dataKey="date" minTickGap={45} />

        <YAxis scale="sqrt" domain={[0, "auto"]} tickFormatter={(value) => money(value)} />

        <Tooltip formatter={(value) => money(value)}
        />

        <Area type="monotone" dataKey="price" name="Price" stroke="#3978cf" fill="url(#priceFill)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );

  // Sidebar icons
  const navigationIcons = [
    LayoutDashboard,
    BarChart3,
    TrendingUp,
    Activity,
    BrainCircuit,
    FlaskConical,
    BarChart3,
  ];

  return (
    <div className="layout">

      {/* SIDEBAR */}
      <aside>
        <div className="brand">
          <Leaf />

          <span>
            AgriSense
            <small>PRICE INTELLIGENCE</small>
          </span>
        </div>

        <div className="nav-label">
          WORKSPACE
        </div>

        {sections.map((item, index) => {
          const Icon = navigationIcons[index];

          return (
            <button key={item} className={section === item ? "nav active" : "nav"} onClick={() => setSection(item)}
            >
              <Icon size={17} />
              {item}
            </button>
          );
        })}

        <div className="side-footer">
          Vegetable Price Prediction
          <br />
          HMM-GRU Dashboard
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main>

        {/* HEADER */}
        <header>
          <div>
            <small>
              Workspace / {section}
            </small>

            <h1>{section}</h1>
          </div>

          <button className="refresh" onClick={loadData} disabled={loading}>
            <RefreshCw size={15} />
            Refresh
          </button>
        </header>

        {/* FILTERS */}
        <div className="filters">

          <label>
            COMMODITY

            <select value={vegetable} onChange={(event) =>
              setVegetable(event.target.value)
            }
            >
              {vegetables.map((item) => (
                <option key={item} value={item}>
                  {item.charAt(0).toUpperCase() +
                    item.slice(1)}
                </option>
              ))}
            </select>
          </label>

          <label>
            FORECAST HORIZON

            <select value={horizon} onChange={(event) =>
              setHorizon(Number(event.target.value))
            }
            >
              {[1, 4, 8, 12].map((item) => (
                <option key={item} value={item}>
                  {item} week{item > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </label>

        </div>

        {/* ERROR MESSAGE */}
        {error && (
          <div className="error">
            {error}
            <br />
            Check your backend and CSV file locations.
          </div>
        )}

        {/* LOADING */}
        {loading ? (
          <div className="loading">
            Loading dashboard data...
          </div>
        ) : (
          <>

            {/* ================= OVERVIEW ================= */}
            {section === "Overview" && (
              <>
                <div className="hero">
                  <span>
                    AGRICULTURAL PRICE INTELLIGENCE
                  </span>

                  <h2>
                    Understand the market.
                    <br />

                    <em>
                      Anticipate the next move.
                    </em>
                  </h2>

                  <p>
                    Weekly vegetable price insights
                    powered by historical data, weather
                    features and hybrid HMM-GRU experiments.
                  </p>
                </div>

                <div className="warning data-simulation-notice">
                  <strong>Research Benchmark Notice:</strong> Datasets are synthetic/simulated weekly price and weather benchmark series updated through <strong>{latest?.date || "2026-11-08"}</strong>. Forecasts are experimental machine learning outputs for algorithmic evaluation.
                </div>

                <div className="cards">

                  <StatCard title="Latest recorded price" value={money(latest?.price)} subtitle={latest?.date
                    || "No data loaded"} icon={TrendingUp} />

                  <StatCard title="Historical observations" value={history.length.toLocaleString("en-IN")}
                    subtitle="Weekly records" icon={Activity} />

                  <StatCard title="Average price" value={money(average)} subtitle="Across loaded history" icon={BarChart3} />

                  <StatCard title="Forecast horizon" value={`${horizon} ${horizon === 1 ? "week" : "weeks"}`} subtitle="Selected horizon"
                    icon={BrainCircuit} />

                </div>

                <section>
                  <div className="section-head">
                    <h3>
                      Historical price movement
                    </h3>

                    <span>
                      {vegetable.toUpperCase()}
                    </span>
                  </div>

                  <div className="chart">
                    {history.length > 0
                      ? priceChart
                      : (
                        <div className="empty">
                          No historical CSV loaded.
                          Check your Data folder.
                        </div>
                      )}
                  </div>
                </section>

                <section>
                  <h3>Commodity snapshot</h3>

                  <div className="commodity-list">
                    {overview.map((item) => (
                      <button key={item.vegetable} onClick={() =>
                        setVegetable(item.vegetable)
                      }
                      >
                        <b>
                          {item.vegetable.toUpperCase()}
                        </b>

                        <span>
                          {item.available
                            ? `${item.records} observations`
                            : "Dataset not found"}
                        </span>

                        <strong>
                          {money(item.latest_price)}
                        </strong>
                      </button>
                    ))}
                  </div>
                </section>
              </>
            )}

            {/* ================= PRICE ANALYTICS ================= */}
            {section === "Price Analytics" && (
              <>
                <section>
                  <div className="section-head">
                    <h3>
                      Historical price series
                    </h3>

                    <button onClick={() =>
                      download(
                        history,
                        `${vegetable}_history.csv`
                      )
                    }
                    >
                      <Download size={14} />
                      Export CSV
                    </button>
                  </div>

                  <div className="chart large">
                    {history.length > 0
                      ? priceChart
                      : (
                        <div className="empty">
                          No historical data.
                        </div>
                      )}
                  </div>
                </section>

                <section>
                  <h3>Weather observations</h3>

                  <div className="chart">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" />

                        <XAxis dataKey="date" minTickGap={45} />

                        <YAxis />

                        <Tooltip />

                        <Line dataKey="temperature" name="Temperature" stroke="#d58b43" dot={false} />

                        <Line dataKey="precipitation" name="Precipitation" stroke="#3999b4" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </>
            )}


            {section === "Forecasting" && (
              <>
                <section>
                  <div className="section-head">
                    <div>
                      <h3>Genuine Future Price Forecasting (HMM-GRU)</h3>
                      <p className="muted">
                        {vegetable.toUpperCase()} · Multi-horizon forecasts strictly beyond the latest historical observation (2026-11-08)
                      </p>
                    </div>
                  </div>

                  <FutureForecastVisualization
                    chartData={forecastChartData}
                    vegetable={vegetable}
                    horizon={horizon}
                  />
                </section>

                <section>
                  <div className="section-head">
                    <div>
                      <h3>Historical Out-of-Sample Test Evaluation</h3>
                      <p className="muted">
                        Out-of-sample backtesting evaluation on the 15% test split for {vegetable.toUpperCase()} at {horizon} week(s).
                      </p>
                    </div>

                    <button onClick={() =>
                      download(
                        predictions,
                        `${vegetable}_h${horizon}_predictions.csv`
                      )
                    }
                    >
                      <Download size={14} />
                      Export Test Predictions
                    </button>
                  </div>

                  <HistoricalEvaluationChart rows={predictions} />
                </section>
              </>
            )}



            {section === "Model Performance" && (
              <>
                <section>
                  <div className="section-head">
                    <h3>Model Performance Comparison</h3>
                    <span>
                      {vegetable.toUpperCase()} · {horizon} WEEK(S)
                    </span>
                  </div>

                  <p className="muted">
                    Compare forecasting models using your experimental evaluation results.
                    Lower RMSE, MAE and MAPE values indicate smaller prediction errors.
                  </p>

                  <ModelComparison rows={metrics} vegetable={vegetable} horizon={horizon} />
                </section>

                <section>
                  <h3>Complete Experimental Metrics</h3>
                  <Table rows={metrics} />
                </section>

                <section>
                  <h3>Diebold-Mariano Test Results</h3>
                  <Table rows={dm} />
                </section>
              </>
            )}


            {/* ================= HMM REGIMES ================= */}

            {section === "HMM Regimes" && (
              <section>
                <div className="section-head">
                  <div>
                    <h3>Hidden Markov Model — Market Regimes</h3>
                    <p className="muted">
                      {vegetable.toUpperCase()} · Statistical price regime analysis
                    </p>
                  </div>
                </div>

                <p className="muted">
                  The Hidden Markov Model identifies latent price regimes
                  from historical price patterns. Each state summarizes
                  observations with similar statistical price characteristics.
                </p>

                <HMMRegimeChart rows={regimes} vegetable={vegetable} />
              </section>
            )}


            {/* ================= ABLATION STUDY ================= */}
            {section === "Ablation Study" && (
              <section>
                <h3>
                  Feature ablation results
                </h3>

                <p className="muted">
                  A: Price only
                  <br />
                  B: Price + Weather
                  <br />
                  C: Price + Technical indicators
                  <br />
                  D: Price + Weather + Technical indicators
                  <br />
                  E: Price + Weather + Technical indicators + HMM
                </p>

                <AblationComparison rows={ablation} vegetable={vegetable} horizon={horizon} />
              </section>
            )}

            {/* ================= FORECAST ACCURACY & RESEARCH COMPARISON ================= */}
            {section === "Forecast Accuracy & Research Comparison" && (
              <AccuracyResearchComparison vegetable={vegetable} horizon={horizon} />
            )}

          </>
        )}

        {/* FOOTER */}
        <footer>
          AgriSense · Vegetable Price Prediction
          · HMM-GRU Research Benchmark (Simulated Data through 2026-11-08)
        </footer>

      </main>
    </div>
  );
}