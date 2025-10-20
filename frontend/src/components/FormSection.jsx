import React, { useState } from "react";
import "./FormSection.css"; // external CSS file

export default function FormSection() {
  const [name, setName] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://localhost:8000/form/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      setResponse(data.message);
    } catch (error) {
      console.error("Error:", error);
      setResponse("Server error. Please try again later.");
    }
  };

  return (
    <div className="form-container">
      <h3 className="form-title">Submit Your Name</h3>
      <form onSubmit={handleSubmit} className="form-box">
        <input
          type="text"
          placeholder="Enter name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="form-input"
        />
        <button type="submit" className="form-button">Send</button>
      </form>
      {response && <p className="form-response">Server says: {response}</p>}
    </div>
  );
}
