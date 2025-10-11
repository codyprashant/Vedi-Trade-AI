import Layout from "./Layout.jsx";

import Strategies from "./Strategies";

import SignalHistory from "./SignalHistory";

import Backtest from "./Backtest";

import LiveMonitor from "./LiveMonitor";

import Migration from "./Migration";

import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';

const PAGES = {
    
    Strategies: Strategies,
    
    SignalHistory: SignalHistory,
    
    Backtest: Backtest,
    
    LiveMonitor: LiveMonitor,
    
    Migration: Migration,
    
}

function _getCurrentPage(url) {
    if (url.endsWith('/')) {
        url = url.slice(0, -1);
    }
    let urlLastPart = url.split('/').pop();
    if (urlLastPart.includes('?')) {
        urlLastPart = urlLastPart.split('?')[0];
    }

    const pageName = Object.keys(PAGES).find(page => page.toLowerCase() === urlLastPart.toLowerCase());
    return pageName || Object.keys(PAGES)[0];
}

// Create a wrapper component that uses useLocation inside the Router context
function PagesContent() {
    const location = useLocation();
    const currentPage = _getCurrentPage(location.pathname);
    
    return (
        <Layout currentPageName={currentPage}>
            <Routes>            
                
                    <Route path="/" element={<Strategies />} />
                
                
                <Route path="/Strategies" element={<Strategies />} />
                
                <Route path="/SignalHistory" element={<SignalHistory />} />
                
                <Route path="/Backtest" element={<Backtest />} />
                
                <Route path="/LiveMonitor" element={<LiveMonitor />} />
                
                <Route path="/Migration" element={<Migration />} />
                
            </Routes>
        </Layout>
    );
}

export default function Pages() {
    return (
        <Router>
            <PagesContent />
        </Router>
    );
}