import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/home';
import Design from './pages/Design';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/design" element={<Design />} />
    </Routes>
  );
}

export default App;