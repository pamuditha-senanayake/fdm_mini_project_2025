// src/App.jsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import MainPage2 from './pages/MainPage2';

function App() {
  return (
    <Router>
      <Routes>
        {/* The single-page AI website is mapped to the root URL */}
        <Route path="/" element={<MainPage />} />
          <Route path="/main" element={<MainPage2 />} />
      </Routes>
    </Router>
  );
}

export default App;