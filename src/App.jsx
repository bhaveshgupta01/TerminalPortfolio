import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/home';
import Design from './pages/Design';
import Publications from './pages/Publications';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/design" element={<Design />} />
        <Route path="/publications" element={<Publications />} />
      </Routes>
    </Router>
  );
}

export default App;