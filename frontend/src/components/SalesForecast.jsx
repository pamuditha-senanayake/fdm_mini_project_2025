import React, { useState } from "react";
import axios from "axios";
import "../styles/SalesForecast.css";

function SalesForecast({ categories }) {
  const [category, setCategory] = useState(categories[0] || "");
  const [steps, setSteps] = useState(30);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult("Processing forecast...");
    try {
      const API_URL = "http://localhost:8000";
      const res = await axios.post(`${API_URL}/forecast`, { category, steps });
      const { forecast, trend, mae, rmse, accuracy_pct } = res.data;


      setResult(
        `Forecast for ${category} (${steps} days):\n${forecast.join(
          "\n"
        )}
          \n\nTrend Insight: ${trend}\n `
          // MAE: ${mae}, RMSE: ${rmse}, Accuracy: ${accuracy_pct}%`
      );


    } catch (err) {
      console.error("API call failed:", err);
      setResult(
        "Error fetching forecast. Please ensure the backend is running and reachable."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div
        className="blob"
        style={{ top: "-20%", left: "-10%", width: "400px", height: "400px" }}
      />
      <div
        className="blob"
        style={{
          bottom: "-20%",
          right: "-10%",
          width: "500px",
          height: "500px",
          animationDelay: "5s",
        }}
      />
      <div className="form-container">
        <h2 className="title">📈 Sales Forecast Dashboard</h2>
        <form onSubmit={handleSubmit}>
          <label htmlFor="category-select" className="styled-label">
            Product Category
          </label>
          <div className="select-container">
            <select
              id="category-select"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="select"
            >
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          <label htmlFor="steps-input" className="styled-label">
            Forecast Days
          </label>
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

        {result &&
            <div className="output">
                {result}
            </div>
        }

      </div>
    </div>
  );
}

export default SalesForecast;
