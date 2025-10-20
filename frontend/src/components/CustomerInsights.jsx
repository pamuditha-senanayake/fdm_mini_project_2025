import React, { useState } from "react";
import axios from "axios";
import { Markup } from "interweave";
import "../styles/CustomerInsights.css";

function CustomerInsights() {
  const [insights, setInsights] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchInsights = async () => {
    setIsLoading(true);
    setError("");
    setInsights(null);

    try {
      // const API_URL = "http://localhost:8000";
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.get(`${API_URL}/insights`);
      setInsights(response.data);
    } catch (err) {
      const errorMessage =
        err.response?.data?.detail ||
        "Failed to fetch insights. Please ensure the backend is running.";
      setError(errorMessage);
      console.error("API call failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="insights-container">
      <button
        className="insights-button"
        onClick={fetchInsights}
        disabled={isLoading}
      >
        {isLoading ? "Analyzing..." : "🔍 Generate Comprehensive Insights"}
      </button>

      {isLoading && (
        <p className="loading-text">Loading model insights, please wait...</p>
      )}
      {error && <p className="error-text">Error: {error}</p>}

      {insights && (
        <div className="insights-wrapper">
          <div className="insight-block">
            <h3 className="insight-title">📈 Model Metrics</h3>
            <div className="insight-content">
              <Markup content={insights.metrics.replace(/\n/g, "<br />")} />
            </div>
          </div>

          <div className="insight-block">
            <h3 className="insight-title">📋 Descriptive Insights</h3>
            <div className="insight-content">
              <Markup content={insights.descriptive.replace(/\n/g, "<br />")} />
            </div>
          </div>

          <div className="insight-block">
            <h3 className="insight-title">🧩 Customer Segmentation Insights</h3>
            <div className="insight-content">
              <Markup content={insights.segmentation.replace(/\n/g, "<br />")} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CustomerInsights;
