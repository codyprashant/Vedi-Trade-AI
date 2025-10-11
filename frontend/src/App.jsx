import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './pages/Layout'
import LiveMonitor from './pages/LiveMonitor'
import Strategies from './pages/Strategies'
import SignalHistory from './pages/SignalHistory'
import Backtest from './pages/Backtest'

function App() {
  return (
    <BrowserRouter data-filename="pages/Migration" data-linenumber="418" data-visual-selector-id="pages/Migration418">
      <Routes data-filename="pages/Migration" data-linenumber="419" data-visual-selector-id="pages/Migration419">
        <Route data-filename="pages/Migration" data-linenumber="420" data-visual-selector-id="pages/Migration420" path="/" element={<Navigate data-filename='pages/Migration' data-linenumber='420' data-visual-selector-id='pages/Migration420' to="/live-monitor" replace />} />
        <Route data-filename="pages/Migration" data-linenumber="421" data-visual-selector-id="pages/Migration421" path="/live-monitor" element={<Layout data-filename='pages/Migration' data-linenumber='421' data-visual-selector-id='pages/Migration421' currentPageName="LiveMonitor"><LiveMonitor data-filename='pages/Migration' data-linenumber='421' data-visual-selector-id='pages/Migration421' /></Layout>} />
        <Route data-filename="pages/Migration" data-linenumber="422" data-visual-selector-id="pages/Migration422" path="/strategies" element={<Layout data-filename='pages/Migration' data-linenumber='422' data-visual-selector-id='pages/Migration422' currentPageName="Strategies"><Strategies data-filename='pages/Migration' data-linenumber='422' data-visual-selector-id='pages/Migration422' /></Layout>} />
        <Route data-filename="pages/Migration" data-linenumber="423" data-visual-selector-id="pages/Migration423" path="/signal-history" element={<Layout data-filename='pages/Migration' data-linenumber='423' data-visual-selector-id='pages/Migration423' currentPageName="SignalHistory"><SignalHistory data-filename='pages/Migration' data-linenumber='423' data-visual-selector-id='pages/Migration423' /></Layout>} />
        <Route data-filename="pages/Migration" data-linenumber="424" data-visual-selector-id="pages/Migration424" path="/backtest" element={<Layout data-filename='pages/Migration' data-linenumber='424' data-visual-selector-id='pages/Migration424' currentPageName="Backtest"><Backtest data-filename='pages/Migration' data-linenumber='424' data-visual-selector-id='pages/Migration424' /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}

export default App