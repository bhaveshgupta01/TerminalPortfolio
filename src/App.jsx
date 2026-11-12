import React, { lazy, Suspense } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Story from './pages/Story';

// Archived versions load on demand so the home page ships only its own code.
const Home = lazy(() => import('./pages/Home'));
const Design = lazy(() => import('./pages/Design'));
const WalletHome = lazy(() => import('./pages/WalletHome'));
const LabHome = lazy(() => import('./pages/LabHome'));
const V2Home = lazy(() => import('./pages/V2Home'));
const PandaCompanion = lazy(() => import('./components/PandaCompanion'));

function App() {
  // The home page has its own line system; the cursor panda stays with the
  // archived versions.
  const isHome = useLocation().pathname === '/';
  return (
    <Suspense fallback={null}>
      <Routes>
        <Route path="/" element={<Story />} />
        {/* v1 — the original scrolling portfolio */}
        <Route path="/v1" element={<Home />} />
        <Route path="/wallet" element={<WalletHome />} />
        <Route path="/lab" element={<LabHome />} />
        {/* v2 — Apple-Wallet-style stack with deep-linkable card state */}
        <Route path="/v2" element={<V2Home />} />
        <Route path="/v2/:cardId" element={<V2Home />} />
        <Route path="/v2/:cardId/:mode" element={<V2Home />} />
        <Route path="/design" element={<Design />} />
      </Routes>
      {/* Cursor companion — lives outside the Routes so it persists across
          navigation without remounting. Auto-hides on touch + reduced-motion. */}
      {!isHome && <PandaCompanion />}
    </Suspense>
  );
}

export default App;
