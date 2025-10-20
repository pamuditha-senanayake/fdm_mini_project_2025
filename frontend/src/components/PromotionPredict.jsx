import React, { useState } from "react";
import axios from "axios";
import "../styles/PromotionPredict.css";

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
    const payload = {};
    Object.keys(form).forEach(key => {
      payload[key] = form[key].trim();
    });

    try {
      const API_URL = 'http://localhost:8000';
      const res = await axios.post(`${API_URL}/promotion`, payload);
      setResult(res.data.recommendation);
    } catch (err) {
      console.error("API call failed:", err);
      setResult("Error fetching recommendation. Please ensure the backend is running and reachable.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2 className="title">🛒 Promotion Recommendation System</h2>
      <form className="form-grid" onSubmit={handleSubmit}>
        <div className="form-field">
          <label className="styled-label" htmlFor="product_category">Product Category</label>
          <div className="select-container">
            <select id="product_category" name="product_category" value={form.product_category} onChange={handleChange}>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>

        <div className="form-field">
          <label className="styled-label" htmlFor="customer_segment">Customer Segment</label>
          <div className="select-container">
            <select id="customer_segment" name="customer_segment" value={form.customer_segment} onChange={handleChange}>
              {segments.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div className="form-field">
          <label className="styled-label" htmlFor="shipping_method">Shipping Method</label>
          <div className="select-container">
            <select id="shipping_method" name="shipping_method" value={form.shipping_method} onChange={handleChange}>
              {shipping.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div className="form-field">
          <label className="styled-label" htmlFor="payment_method">Payment Method</label>
          <div className="select-container">
            <select id="payment_method" name="payment_method" value={form.payment_method} onChange={handleChange}>
              {payment.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>

        <div className="form-field">
          <label className="styled-label" htmlFor="gender">Gender</label>
          <div className="select-container">
            <select id="gender" name="gender" value={form.gender} onChange={handleChange}>
              {genders.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>
        </div>

        <div className="form-field">
          <label className="styled-label" htmlFor="income">Income Level</label>
          <div className="select-container">
            <select id="income" name="income" value={form.income} onChange={handleChange}>
              {incomes.map(i => <option key={i} value={i}>{i}</option>)}
            </select>
          </div>
        </div>

        <button type="submit" className="button" disabled={isLoading}>
          {isLoading ? "Analyzing..." : "Predict"}
        </button>
      </form>
      {result && <div className="output">{result}</div>}
    </div>
  );
}

export default PromotionPredict;
