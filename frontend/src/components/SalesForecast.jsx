import React, { useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import "../styles/SalesForecast.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function SalesForecast({ categories }) {
  const [category, setCategory] = useState(categories[0] || "");
  const [steps, setSteps] = useState(30);
  const [forecastData, setForecastData] = useState(null);
  const [trend, setTrend] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setForecastData(null);
    setMetrics(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await axios.post(`${API_URL}/forecast`, { category, steps });

      // Chart data
      setForecastData(res.data.forecast_data);

      // Trend insight
      setTrend(res.data.trend);

      // Metrics
      setMetrics({
        mae: res.data.mae,
        rmse: res.data.rmse,
        accuracy: res.data.accuracy_pct,
      });
    } catch (err) {
      console.error("API call failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = forecastData
    ? {
        labels: forecastData.dates,
        datasets: [
          {
            label: "Actual Sales",
            data: forecastData.actual,
            borderColor: "#3b82f6",
            backgroundColor: "#3b82f6",
            tension: 0.2,
          },
          {
            label: "Forecasted Sales",
            data: forecastData.forecast,
            borderColor: "#f59e0b",
            backgroundColor: "#f59e0b",
            borderDash: [5, 5],
            tension: 0.2,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: `Sales Forecast for ${category}` },
    },
    scales: {
      y: { beginAtZero: true },
    },
  };

  return (
    <div className="app-container">
      <div className="form-container">
        <h2 className="title">📈 Sales Forecast Dashboard</h2>
        <form onSubmit={handleSubmit}>
          <label htmlFor="category-select" className="styled-label">Product Category</label>
          <select
            id="category-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="select"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>

          <label htmlFor="steps-input" className="styled-label">Forecast Days</label>
          <input
            id="steps-input"
            type="number"
            value={steps}
            onChange={(e) => setSteps(parseInt(e.target.value, 10))}
            className="input"
          />

          <button type="submit" className="button" disabled={isLoading}>
            {isLoading ? "Forecasting..." : "Predict"}
          </button>
        </form>

        {forecastData && (
          <div className="output">
            <Line data={chartData} options={chartOptions} />
            <p style={{ marginTop: "10px" }}><strong>Trend Insight:</strong> {trend}</p>

            {metrics && (
              <p style={{ marginTop: "10px" }}>
                <strong>Metrics:</strong> <br />
                MAE: {metrics.mae} <br />
                RMSE: {metrics.rmse} <br />
                Accuracy: {metrics.accuracy}%
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default SalesForecast;
