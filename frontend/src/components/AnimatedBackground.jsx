import React from "react";
import "../styles/AnimatedBackground.css";

const AnimatedBackground = () => {
  return (
    <div className="background-container">
      <div
        className="blob"
        style={{
          "--size": "400px",
          "--duration": "18s",
          "--rotateDuration": "35s",
          "--color1": "#fffacd",
          "--color2": "#ffffff",
          top: "10%",
          left: "-10%",
        }}
      />
      <div
        className="blob"
        style={{
          "--size": "500px",
          "--duration": "22s",
          "--rotateDuration": "45s",
          "--color1": "#ffffe0",
          "--color2": "#fffaf0",
          top: "50%",
          left: "80%",
        }}
      />
      <div
        className="blob"
        style={{
          "--size": "300px",
          "--duration": "16s",
          "--rotateDuration": "30s",
          "--color1": "#f0e68c",
          "--color2": "#fffff0",
          top: "70%",
          left: "10%",
        }}
      />
      <div
        className="blob"
        style={{
          "--size": "450px",
          "--duration": "20s",
          "--rotateDuration": "40s",
          "--color1": "#fff8dc",
          "--color2": "#fffff0",
          top: "-5%",
          left: "40%",
        }}
      />
      <div
        className="blob"
        style={{
          "--size": "350px",
          "--duration": "24s",
          "--rotateDuration": "50s",
          "--color1": "#fafad2",
          "--color2": "#ffffff",
          top: "30%",
          left: "-30%",
        }}
      />
    </div>
  );
};

export default AnimatedBackground;
