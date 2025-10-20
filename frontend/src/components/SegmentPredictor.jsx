import React, { useState } from 'react';
import axios from 'axios';
import { Markup } from 'interweave';
import '../styles/SegmentPredictor.css';

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
          <ul>
            {Object.entries(result.probabilities).map(([segment, prob]) => (
              <li key={segment}>{segment}: {(prob * 100).toFixed(2)}%</li>
            ))}
          </ul>
          <h3>Marketing Recommendation</h3>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  );
}

export default SegmentPredictor;
