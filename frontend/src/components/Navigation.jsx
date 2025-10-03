// src/components/Navigation.jsx

import React from 'react';

// Helper for Smooth Scrolling (kept here for encapsulation)
const scrollToSection = (id) => {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
};

const Navigation = () => {
  const links = [
    { name: 'HOME', id: 'home' },
    { name: 'PREDICT', id: 'predict' },
    { name: 'MODELS', id: 'models' },
  ];

  return (
    <nav className="fixed top-6 left-1/2 transform -translate-x-1/2 z-[100]">
      <div
        className="flex space-x-4 p-2.5 rounded-full border border-indigo-500/30 shadow-2xl shadow-indigo-900/50"
        // Super CSS: Glassmorphism/Neumorphism style with a strong blur and dark transparent background
        style={{ backdropFilter: 'blur(16px)', backgroundColor: 'rgba(5, 5, 20, 0.6)' }}
      >
        {links.map((link) => (
          <button
            key={link.id}
            onClick={() => scrollToSection(link.id)}
            className="text-sm font-semibold uppercase px-5 py-2 tracking-wider rounded-full text-indigo-200 transition duration-300
                       hover:bg-indigo-700/50 hover:text-white hover:shadow-lg hover:shadow-cyan-500/30
                       focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {link.name}
          </button>
        ))}
      </div>
    </nav>
  );
};

export default Navigation;