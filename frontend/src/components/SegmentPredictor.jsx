// src/components/SegmentPredictor.jsx
import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import { Markup } from 'interweave';

// Reusing styles from other components for consistency
const FormContainer = styled.div`
  background: rgba(255, 255, 255, 0.05); padding: 40px; border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.18); max-width: 800px;
  width: 100%; margin: 40px auto; z-index: 2; position: relative;
`;
const Title = styled.h2`
  text-align: center; margin-bottom: 2rem; font-size: 1.8rem; font-weight: 600; color: #FFD966;
`;
const FormGrid = styled.form`
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem 1.5rem;
  @media (max-width: 768px) { grid-template-columns: 1fr; }
`;
const FormField = styled.div`
  display: flex; flex-direction: column;
`;
const StyledLabel = styled.label`
  display: block; margin-bottom: 8px; font-weight: 400; color: rgba(255, 255, 255, 0.8); text-align: left;
`;
const Input = styled.input`
  width: 100%; padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1); color: #fff; font-size: 1rem; transition: all 0.3s ease;
  &:focus { outline: none; border: 1px solid #FFD966; box-shadow: 0 0 15px rgba(255, 217, 102, 0.2); }
`;
const SelectContainer = styled.div`
  position: relative; width: 100%;
  &::after {
    content: '▼'; font-size: 1rem; color: #FFD966; position: absolute; right: 20px;
    top: 50%; transform: translateY(-50%); pointer-events: none;
  }
`;
const Select = styled.select`
  width: 100%; padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1); color: #fff; font-size: 1rem;
  -webkit-appearance: none; -moz-appearance: none; appearance: none; cursor: pointer;
  &:focus { outline: none; border: 1px solid #FFD966; box-shadow: 0 0 15px rgba(255, 217, 102, 0.2); }
  option { background: #4a2f16; color: #f0f0f0; }
`;
const Button = styled.button`
  width: 100%; padding: 15px 30px; margin-top: 1rem; border: none;
  background: linear-gradient(45deg, #FFC371, #FF5F6D); color: #fff; font-weight: 600;
  font-size: 1.1rem; border-radius: 10px; cursor: pointer; transition: all 0.3s ease;
  grid-column: 1 / -1;
  &:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); }
  &:disabled { background: #555; cursor: not-allowed; }
`;
const Output = styled.div`
  margin-top: 25px; background: rgba(0, 0, 0, 0.3); padding: 25px; border-radius: 15px;
  border: 1px solid rgba(255, 217, 102, 0.2); text-align: left;
  h3 { color: #FFD966; margin-bottom: 1rem; }
  p { margin: 0.5rem 0; line-height: 1.6; }
  ul { list-style-position: inside; padding-left: 0; }
`;

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
    <FormContainer>
      <Title>🛍️ Customer Segment Prediction Tool</Title>
      <FormGrid onSubmit={handleSubmit}>
        <FormField>
          <StyledLabel htmlFor="age">Age (18-100)</StyledLabel>
          <Input type="number" id="age" name="age" value={form.age} onChange={handleChange} min="18" max="100" />
        </FormField>
        <FormField>
          <StyledLabel htmlFor="income">Income Level</StyledLabel>
          <SelectContainer>
            <Select id="income" name="income" value={form.income} onChange={handleChange}>
              {incomeLevels.map(i => <option key={i} value={i}>{i}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="total_purchases">Total Purchases</StyledLabel>
          <Input type="number" id="total_purchases" name="total_purchases" value={form.total_purchases} onChange={handleChange} min="0" />
        </FormField>
        <FormField>
          <StyledLabel htmlFor="amount">Total Amount ($)</StyledLabel>
          <Input type="number" id="amount" name="amount" value={form.amount} onChange={handleChange} min="0" step="0.01" />
        </FormField>
        <Button type="submit" disabled={isLoading}>{isLoading ? 'Predicting...' : 'Predict Segment'}</Button>
      </FormGrid>

      {error && <Output><h3>Error</h3><p style={{color: '#FF5F6D'}}>{error}</p></Output>}
      {result && (
        <Output>
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
        </Output>
      )}
    </FormContainer>
  );
}

export default SegmentPredictor;