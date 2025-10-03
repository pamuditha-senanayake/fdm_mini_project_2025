import React from "react";
import { useState, useEffect } from "react";
import styled, { createGlobalStyle, keyframes } from "styled-components";
import SalesForecast from "../components/SalesForecast";
import PromotionPredict from "../components/PromotionPredict";

// === STYLES ===

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Poppins', sans-serif;
    background-color: #1a0e05; /* A solid dark color for the page edges */
    color: #f0f0f0;
    overflow-x: hidden;
  }

  html {
    scroll-behavior: smooth;
  }
`;

// Background Blob Animation
const move = keyframes`
  0% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(150px, -100px) scale(1.3); }
  100% { transform: translate(0, 0) scale(1); }
`;

const Blob = styled.div`
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(45deg, rgba(255, 217, 102, 0.2), rgba(239, 131, 84, 0.2));
  filter: blur(120px);
  z-index: 1;
  animation: ${move} 25s infinite alternate;
`;

const PageWrapper = styled.div`
  position: relative;
  width: 100%;
  overflow: hidden; 
  isolation: isolate;

  &::before {
    content: '';
    position: absolute;
    z-index: -1;
    top: -50px;
    left: -50px;
    right: -50px;
    bottom: -50px;
    background: linear-gradient(135deg, #4a2f16, #2d1a0b);
    filter: blur(80px);
  }
`;

// Navigation Bar
const Navbar = styled.nav`
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 25px;
  padding: 12px 30px;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  z-index: 1000;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
`;

const NavLink = styled.a`
  font-family: 'Poppins', sans-serif;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
  text-decoration: none;
  position: relative;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    color: #FFD966;
  }

  &:after {
    content: '';
    position: absolute;
    width: 0%;
    height: 2px;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    background: #FFD966;
    transition: width 0.3s ease;
    border-radius: 2px;
  }

  &:hover:after {
    width: 100%;
  }
`;

// Section Styling
const Section = styled.section`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 100px 20px;
  text-align: center;
  position: relative;
  overflow: hidden;
`;

const SectionContent = styled.div`
  position: relative;
  z-index: 10;
  width: 100%;
  max-width: 1100px;
  margin: 0 auto;
  text-align: center;
`;

// Typography
const Title = styled.h1`
  font-size: 3.5rem;
  margin-bottom: 1rem;
  font-weight: 700;
  color: #f0f0f0;

   @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const GradientTitle = styled(Title)`
  font-size: 6rem;
  font-weight: 800;
  background: linear-gradient(90deg, #FFD966, #FFC371, #FF5F6D);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;


  @media (max-width: 768px) {
    font-size: 4rem;
  }
`;

const Subtitle = styled.p`
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.7);
  max-width: 600px;
  margin: 0 auto 30px auto;
  line-height: 1.6;

  @media (max-width: 768px) {
    font-size: 1rem;
  }
`;

// ===== MainPage Component =====
function MainPage() {
    const [categories, setCategories] = useState([]);
      useEffect(() => {
    // Fetch categories from backend or hardcode for demo
    setCategories(["Electronics", "Apparel & Fashion", "Books & Media", "Home Goods"]);
  }, []);

  return (
    <>
      <GlobalStyle />
      <PageWrapper id="main-page">
        {/* Animated Blobs in the background */}
        <Blob style={{ top: "5%", left: "10%", width: "400px", height: "400px" }} />
        <Blob style={{ top: "50%", left: "50%", width: "600px", height: "600px", animationDelay: "5s" }} />
        <Blob style={{ top: "80%", left: "20%", width: "300px", height: "300px", animationDelay: "10s" }} />


        <Navbar>
            <NavLink href="#home">Home</NavLink>
            <NavLink href="#stock">Forecast</NavLink>
            {/* Changed "Audit" to "Promotions" for clarity */}
            <NavLink href="#insights">Promotions</NavLink>
        </Navbar>

        {/* Hero Section */}
        <Section id="home">
            <SectionContent>
            <GradientTitle>RetailIQ</GradientTitle>
            <Subtitle>
                Harnessing AI to deliver intelligent, actionable retail insights for sales forecasting and trend analysis.
            </Subtitle>
            </SectionContent>
        </Section>

        {/* Forecast Section */}
        <Section id="stock">
            <SectionContent>
            <Title>Trend & Sales Insights</Title>
            <Subtitle>
                Select a product category and forecast future sales demand with our predictive model.
            </Subtitle>
            {categories.length > 0 && <SalesForecast categories={categories} />}
            </SectionContent>
        </Section>

        {/* Promotion Insights Section */}
        <Section id="insights">
            <SectionContent>
                <Title>Promotion & Customer Insights</Title>
                <Subtitle>
                Identify high-purchasing customers and get actionable promotion recommendations.
                </Subtitle>
                {categories.length > 0 && (
                <PromotionPredict
                    categories={["Electronics","Apparel & Fashion","Books & Media","Home Goods"]}
                    segments={["Retail","Wholesale"]}
                    shipping={["Standard","Express"]}
                    payment={["Credit Card","PayPal","Cash"]}
                    genders={["Male","Female","Other"]}
                    incomes={["Low","Medium","High"]}
                />
                )}
            </SectionContent>
        </Section>

      </PageWrapper>
    </>
  );
}

export default MainPage;