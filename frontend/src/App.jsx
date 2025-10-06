// src/App.jsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage'; 

function App() {
  return (
    <Router>
      <Routes>
        {/* The single-page AI website is mapped to the root URL */}
        <Route path="/" element={<MainPage />} />
      </Routes>
    </Router>
  );
}

export default App;