import React, { useState, useEffect } from "react";
import SalesForecast from "../components/SalesForecast";
import PromotionPredict from "../components/PromotionPredict";
import CustomerInsights from "../components/CustomerInsights";
import SegmentPredictor from "../components/SegmentPredictor";
import "../styles/MainPage.css";

function MainPage() {
  const [categories, setCategories] = useState([]);

  const incomeLevels = ["Low", "Medium", "High"];
  const segments = ["Regular", "Premium", "Occasional"];
  const shipping = ["Standard", "Express", "Same-Day"];
  const payment = ["Credit Card", "PayPal", "Debit Card"];
  const genders = ["Male", "Female"];

  useEffect(() => {
    setCategories(["Electronics", "Grocery", "Clothing", "Home Decor", "Books"]);
  }, []);

  return (
    <>
      <div id="main-page" className="page-wrapper">
        <div className="blob" style={{ top: "5%", left: "10%", width: "400px", height: "400px" }} />
        <div className="blob" style={{ top: "50%", left: "50%", width: "600px", height: "600px", animationDelay: "5s" }} />
        <div className="blob" style={{ top: "80%", left: "20%", width: "300px", height: "300px", animationDelay: "10s" }} />

        <nav className="navbar">
          <a href="#home" className="nav-link">Home</a>
          <a href="#stock" className="nav-link">Forecast</a>
          <a href="#insights" className="nav-link">Promotions</a>
          <a href="#segmentation" className="nav-link">Segmentation</a>
        </nav>

        <section id="home" className="section">
          <div className="section-content">
            <h1 className="gradient-title">RetailIQ</h1>
            <p className="subtitle">
              Harnessing AI to deliver intelligent, actionable retail insights for sales forecasting and trend analysis.
            </p>
          </div>
        </section>

        <section id="stock" className="section">
          <div className="section-content">
            <h1 className="title">Trend & Sales Insights</h1>
            <p className="subtitle">
              Select a product category and forecast future sales demand with our predictive model.
            </p>
            {categories.length > 0 && <SalesForecast categories={categories} />}
          </div>
        </section>

        <section id="insights" className="section">
          <div className="section-content">
            <h1 className="title">Promotion & Customer Insights</h1>
            <p className="subtitle">
              Identify high-purchasing customers and get actionable promotion recommendations.
            </p>
            {categories.length > 0 && (
              <PromotionPredict
                categories={["Electronics", "Grocery", "Clothing", "Home Decor", "Books"]}
                segments={segments}
                shipping={shipping}
                payment={payment}
                genders={genders}
                incomes={incomeLevels}
              />
            )}
          </div>
        </section>

        <section id="audit" className="section">
          <div className="section-content">
            <h1 className="title">Comprehensive Business Audit</h1>
            <p className="subtitle">
              Generate a complete overview of model performance, descriptive analytics, and customer segmentation insights with a single click.
            </p>
            <CustomerInsights />
          </div>
        </section>

        <section id="segmentation" className="section">
          <div className="section-content">
            <h1 className="title">Live Customer Segmentation</h1>
            <p className="subtitle">
              Enter customer details to predict their segment and receive targeted marketing recommendations in real-time.
            </p>
            <SegmentPredictor incomeLevels={incomeLevels} />
          </div>
        </section>
      </div>
    </>
  );
}

export default MainPage;
