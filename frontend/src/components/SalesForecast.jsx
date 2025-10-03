import React, { useState } from "react";
import styled, { createGlobalStyle, keyframes } from "styled-components";
import axios from "axios";

// --- STYLES ---

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  /* Add a base background color to the body in case the component doesn't fill the whole screen */
  body {
    background-color: #1a0e05;
  }
`;

// <<< MODIFIED COMPONENT
// The background effect is now applied here
const AppContainer = styled.div`
  font-family: 'Poppins', sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  color: #f0f0f0;
  overflow: hidden; /* Crucial for clipping the blurred edges */
  position: relative;
  isolation: isolate; /* Creates a new stacking context */

  /* This pseudo-element creates the soft, blurred background */
  &::before {
    content: '';
    position: absolute;
    z-index: -1; /* Place it behind the blobs and form */
    
    /* Make it larger than the container to ensure edges are blurred */
    top: -50px;
    left: -50px;
    right: -50px;
    bottom: -50px;

    /* The gradient is now on this element */
    background: linear-gradient(135deg, #2d1a0b, #4a2f16);
    
    /* The magic property that creates the soft edge effect */
    filter: blur(80px);
  }
`;


const move = keyframes`
  0% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(100px, 50px) scale(1.2); }
  100% { transform: translate(0, 0) scale(1); }
`;

const Blob = styled.div`
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(45deg, rgba(255, 217, 102, 0.3), rgba(239, 131, 84, 0.3));
  filter: blur(100px);
  z-index: 1; /* Sits above the blurred background but below the form */
  animation: ${move} 20s infinite alternate;
`;

const FormContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 40px;
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.18);
  max-width: 550px;
  width: 90%;
  z-index: 2; /* Sits on top of everything */
  position: relative;
`;

const Title = styled.h2`
  text-align: center;
  margin-bottom: 2rem;
  font-size: 1.8rem;
  font-weight: 600;
  color: #FFD966;
`;

const StyledLabel = styled.label`
  display: block;
  margin-bottom: 8px;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.8);
`;

const Input = styled.input`
  width: 100%;
  padding: 14px 18px;
  margin-bottom: 1.5rem;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border: 1px solid #FFD966;
    box-shadow: 0 0 15px rgba(255, 217, 102, 0.2);
  }
`;

const SelectContainer = styled.div`
  position: relative;
  width: 100%;
  margin-bottom: 1.5rem;

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
`;

// --- COMPONENT ---

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
      // Mock API call for demonstration
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate network delay
      const mockData = {
          forecast: Array.from({length: 5}, () => (Math.random() * 1000 + 500).toFixed(2)),
          trend: "Positive growth trend detected.",
          mae: (Math.random() * 10).toFixed(2),
          rmse: (Math.random() * 20).toFixed(2),
          accuracy_pct: (Math.random() * 10 + 88).toFixed(2)
      };
      const { forecast, trend, mae, rmse, accuracy_pct } = mockData;

      setResult(
        `Forecast for ${category} (${steps} days):\n${forecast.join("\n")}\n\nTrend Insight: ${trend}\nMAE: ${mae}, RMSE: ${rmse}, Accuracy: ${accuracy_pct}%`
      );
    } catch (err) {
      setResult("Error fetching forecast. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <GlobalStyle />
      <AppContainer>
        <Blob style={{ top: "-20%", left: "-10%", width: "400px", height: "400px" }} />
        <Blob style={{ bottom: "-20%", right: "-10%", width: "500px", height: "500px", animationDelay: "5s" }} />
        <FormContainer>
          <Title>📈 Sales Forecast Dashboard</Title>
          <form onSubmit={handleSubmit}>
            <StyledLabel htmlFor="category-select">Product Category</StyledLabel>
            <SelectContainer>
              <Select id="category-select" value={category} onChange={(e) => setCategory(e.target.value)}>
                {categories.map((cat) => <option key={cat} value={cat}>{cat}</option>)}
              </Select>
            </SelectContainer>

            <StyledLabel htmlFor="steps-input">Forecast Days</StyledLabel>
            <Input id="steps-input" type="number" value={steps} onChange={(e) => setSteps(parseInt(e.target.value, 10))} />

            <Button type="submit" disabled={isLoading}>{isLoading ? "Forecasting..." : "Predict"}</Button>
          </form>

          {result && <Output>{result}</Output>}
        </FormContainer>
      </AppContainer>
    </>
  );
}

export default SalesForecast;