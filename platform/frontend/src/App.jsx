import { useState } from 'react'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import CCTVTracker from "./pages/CCTVTracker";
import CalibratedSpeed from './pages/CalibratedSpeed';
import TOFSpeed from './pages/TOFSpeed';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CCTVTracker />} />
        <Route path="/calibrated-speed" element={<CalibratedSpeed />} />
        <Route path="/tof-speed" element={<TOFSpeed />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App
