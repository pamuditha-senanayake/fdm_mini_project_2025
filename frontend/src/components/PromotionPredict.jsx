import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";

// --- STYLES (for consistency with the theme) ---

const FormContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 40px;
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.18);
  max-width: 800px;
  width: 100%;
  margin: 40px auto;
  z-index: 2;
  position: relative;
`;

const Title = styled.h2`
  text-align: center;
  margin-bottom: 2rem;
  font-size: 1.8rem;
  font-weight: 600;
  color: #FFD966;
`;

const FormGrid = styled.form`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem 1.5rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const FormField = styled.div`
  display: flex;
  flex-direction: column;
`;

const StyledLabel = styled.label`
  display: block;
  margin-bottom: 8px;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.8);
  text-align: left;
`;

const SelectContainer = styled.div`
  position: relative;
  width: 100%;

  &::after {
    content: '▼';
    font-size: 1rem;
    color: #FFD966;
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 14px 18px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  cursor: pointer;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border: 1px solid #FFD966;
    box-shadow: 0 0 15px rgba(255, 217, 102, 0.2);
  }

  option {
    background: #4a2f16;
    color: #f0f0f0;
  }
`;

const Button = styled.button`
  width: 100%;
  padding: 15px 30px;
  margin-top: 1rem;
  border: none;
  background: linear-gradient(45deg, #FFC371, #FF5F6D);
  color: #fff;
  font-weight: 600;
  font-size: 1.1rem;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  grid-column: 1 / -1;

  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
`;

const Output = styled.div`
  margin-top: 25px;
  background: rgba(0, 0, 0, 0.3);
  padding: 25px;
  border-radius: 15px;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.95rem;
  color: #FFD966;
  white-space: pre-wrap;
  word-wrap: break-word;
  border: 1px solid rgba(255, 217, 102, 0.2);
  max-height: 300px;
  overflow-y: auto;
  text-align: left;
`;


// --- COMPONENT ---

function PromotionPredict({ categories, segments, shipping, payment, genders, incomes }) {
  const [form, setForm] = useState({
    product_category: categories[0] || "",
    customer_segment: segments[0] || "",
    shipping_method: shipping[0] || "",
    payment_method: payment[0] || "",
    gender: genders[0] || "",
    income: incomes[0] || ""
  });
  const [result, setResult] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult("Finding best promotion...");
    try {
      // Use the environment variable for the API URL, falling back to localhost for development
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

      // Make the real API call to the backend
      const res = await axios.post(`${API_URL}/promotion`, form);

      // Set the result with the REAL data received from the backend
      setResult(res.data.recommendation);

    } catch (err) {
      console.error("API call failed:", err); // Log the actual error for debugging
      setResult("Error fetching recommendation. Please ensure the backend is running and reachable.");
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <FormContainer>
      <Title>🛒 Promotion Recommendation Engine</Title>
      <FormGrid onSubmit={handleSubmit}>
        <FormField>
          <StyledLabel htmlFor="product-category">Product Category</StyledLabel>
          <SelectContainer>
            <Select id="product-category" name="product_category" value={form.product_category} onChange={handleChange}>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="customer-segment">Customer Segment</StyledLabel>
          <SelectContainer>
            <Select id="customer-segment" name="customer_segment" value={form.customer_segment} onChange={handleChange}>
              {segments.map(s => <option key={s} value={s}>{s}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="shipping-method">Shipping Method</StyledLabel>
          <SelectContainer>
            <Select id="shipping-method" name="shipping_method" value={form.shipping_method} onChange={handleChange}>
              {shipping.map(s => <option key={s} value={s}>{s}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="payment-method">Payment Method</StyledLabel>
          <SelectContainer>
            <Select id="payment-method" name="payment_method" value={form.payment_method} onChange={handleChange}>
              {payment.map(p => <option key={p} value={p}>{p}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="gender">Gender</StyledLabel>
          <SelectContainer>
            <Select id="gender" name="gender" value={form.gender} onChange={handleChange}>
              {genders.map(g => <option key={g} value={g}>{g}</option>)}
            </Select>
          </SelectContainer>
        </FormField>
        <FormField>
          <StyledLabel htmlFor="income">Income Level</StyledLabel>
          <SelectContainer>
            <Select id="income" name="income" value={form.income} onChange={handleChange}>
              {incomes.map(i => <option key={i} value={i}>{i}</option>)}
            </Select>
          </SelectContainer>
        </FormField>

        <Button type="submit" disabled={isLoading}>{isLoading ? "Analyzing..." : "Get Recommendation"}</Button>
      </FormGrid>
      {result && <Output>{result}</Output>}
    </FormContainer>
  );
}

export default PromotionPredict;