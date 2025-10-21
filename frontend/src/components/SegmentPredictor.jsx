import React, { useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import '../styles/SegmentPredictor.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function SegmentPredictor({ incomeLevels }) {
  const [form, setForm] = useState({ age: 35, income: incomeLevels[0], total_purchases: 5, amount: 250 });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: name === 'income' ? value : Number(value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/api/predict-segment`, form);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Please check inputs and backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = result
    ? {
        labels: Object.keys(result.probabilities),
        datasets: [
          {
            label: 'Probability',
            data: Object.values(result.probabilities).map(p => p * 100),
            backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'],
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Segment Prediction Probabilities (%)' },
    },
    scales: {
      y: { beginAtZero: true, max: 100 }
    }
  };

  return (
    <div className="form-container">
      <h2 className="title">🛍️ Customer Segment Prediction Tool</h2>
      <form className="form-grid" onSubmit={handleSubmit}>
        <div className="form-field">
          <label className="label" htmlFor="age">Age (18-100)</label>
          <input className="input" type="number" id="age" name="age" value={form.age} onChange={handleChange} min="18" max="100" />
        </div>
        <div className="form-field">
          <label className="label" htmlFor="income">Income Level</label>
          <div className="select-container">
            <select className="select" id="income" name="income" value={form.income} onChange={handleChange}>
              {incomeLevels.map(i => <option key={i} value={i}>{i}</option>)}
            </select>
          </div>
        </div>
        <div className="form-field">
          <label className="label" htmlFor="total_purchases">Total Purchases</label>
          <input className="input" type="number" id="total_purchases" name="total_purchases" value={form.total_purchases} onChange={handleChange} min="0" />
        </div>
        <div className="form-field">
          <label className="label" htmlFor="amount">Total Amount ($)</label>
          <input className="input" type="number" id="amount" name="amount" value={form.amount} onChange={handleChange} min="0" step="0.01" />
        </div>
        <button className="button" type="submit" disabled={isLoading}>
          {isLoading ? 'Predicting...' : 'Predict Segment'}
        </button>
      </form>

      {error && (
        <div className="output">
          <h3>Error</h3>
          <p className="error-text">{error}</p>
        </div>
      )}

      {result && (
        <div className="output">
          <h3>Prediction Result</h3>
          <p><strong>Predicted Segment:</strong> {result.predicted_segment}</p>

          <h3>Prediction Confidence</h3>
          <Bar data={chartData} options={chartOptions} />

          <h3>Marketing Recommendation</h3>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  );
}

export default SegmentPredictor;
