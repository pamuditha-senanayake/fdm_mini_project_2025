import React, { useState, useEffect } from "react";
import "./MainPage.css"; // external CSS file
import FormSection from "../components/FormSection";

function MainPage() {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    setCategories(["Electronics", "Grocery", "Clothing", "Home Decor", "Books"]);
  }, []);

  return (
    <div className="page-wrapper" id="main-page">
      {/* Animated Blobs */}
      <div className="blob blob1"></div>
      <div className="blob blob2"></div>
      <div className="blob blob3"></div>

      {/* Navbar */}
      <nav className="navbar">
        <a href="#home" className="nav-link">Home</a>
        <a href="#stock" className="nav-link">Forecast</a>
        <a href="#insights" className="nav-link">Promotions</a>
        <a href="#segmentation" className="nav-link">Segmentation</a>
      </nav>

      {/* Sections */}
      <section id="home" className="section">
        <div className="section-content">
          <h1 className="gradient-title">SLIITXtream</h1>
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
          <FormSection />
        </div>
      </section>

      <section id="insights" className="section">
        <div className="section-content">
          <h1 className="title">Promotion & Customer Insights</h1>
          <p className="subtitle">
            Identify high-purchasing customers and get actionable promotion recommendations.
          </p>
        </div>
      </section>

      <section id="audit" className="section">
        <div className="section-content">
          <h1 className="title">Comprehensive Business Audit</h1>
          <p className="subtitle">
            Generate a complete overview of model performance, descriptive analytics, and customer segmentation insights with a single click.
          </p>
        </div>
      </section>

      <section id="segmentation" className="section">
        <div className="section-content">
          <h1 className="title">Live Customer Segmentation</h1>
          <p className="subtitle">
            Enter customer details to predict their segment and receive targeted marketing recommendations in real-time.
          </p>
        </div>
      </section>
    </div>
  );
}

export default MainPage;
