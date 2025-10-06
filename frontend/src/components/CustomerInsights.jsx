// src/components/CustomerInsights.jsx

import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import { Markup } from 'interweave'; // A library to safely render HTML/Markdown from string

// --- STYLES ---

const InsightsContainer = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 40px;
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.18);
  width: 100%;
  max-width: 900px;
  margin: 20px auto;
  text-align: center;
`;

const Button = styled.button`
  padding: 15px 40px;
  margin-bottom: 2rem;
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

  &:disabled {
    background: #555;
    cursor: not-allowed;
  }
`;

const InsightsWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-top: 1rem;
`;

const InsightBlock = styled.div`
  background: rgba(0, 0, 0, 0.2);
  padding: 25px;
  border-radius: 15px;
  border: 1px solid rgba(255, 217, 102, 0.2);
  text-align: left;
`;

const InsightTitle = styled.h3`
  color: #FFD966;
  margin-bottom: 1rem;
  font-size: 1.4rem;
`;

const InsightContent = styled.div`
  color: rgba(255, 255, 255, 0.9);
  line-height: 1.6;
  white-space: pre-wrap;

  strong {
    color: #FFD966;
  }
`;

const LoadingText = styled.p`
  font-size: 1.1rem;
  color: #FFD966;
`;

const ErrorText = styled(LoadingText)`
  color: #FF5F6D;
`;


// --- COMPONENT ---

function CustomerInsights() {
  const [insights, setInsights] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchInsights = async () => {
    setIsLoading(true);
    setError('');
    setInsights(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.get(`${API_URL}/insights`);
      setInsights(response.data);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || "Failed to fetch insights. Please ensure the backend is running.";
      setError(errorMessage);
      console.error("API call failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <InsightsContainer>
      <Button onClick={fetchInsights} disabled={isLoading}>
        {isLoading ? 'Analyzing...' : '🔍 Generate Comprehensive Insights'}
      </Button>

      {isLoading && <LoadingText>Loading model insights, please wait...</LoadingText>}
      {error && <ErrorText>Error: {error}</ErrorText>}

      {insights && (
        <InsightsWrapper>
          <InsightBlock>
            <InsightTitle>📈 Model Metrics</InsightTitle>
            <InsightContent>
              <Markup content={insights.metrics.replace(/\n/g, '<br />')} />
            </InsightContent>
          </InsightBlock>

          <InsightBlock>
            <InsightTitle>📋 Descriptive Insights</InsightTitle>
            <InsightContent>
                <Markup content={insights.descriptive.replace(/\n/g, '<br />')} />
            </InsightContent>
          </InsightBlock>

          <InsightBlock>
            <InsightTitle>🧩 Customer Segmentation Insights</InsightTitle>
            <InsightContent>
                <Markup content={insights.segmentation.replace(/\n/g, '<br />')} />
            </InsightContent>
          </InsightBlock>
        </InsightsWrapper>
      )}
    </InsightsContainer>
  );
}

export default CustomerInsights;