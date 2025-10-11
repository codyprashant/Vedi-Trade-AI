import React, { useState } from "react";
import { TestTube, Plus, List, Eye, Play } from "lucide-react";
import RunGenerator from "../components/backtest/RunGenerator";
import RunsDashboard from "../components/backtest/RunsDashboard";
import SignalsViewer from "../components/backtest/SignalsViewer";
import ExecuteSimulation from "../components/backtest/ExecuteSimulation";

export default function Backtest() {
  const [activeTab, setActiveTab] = useState('generator');
  const [selectedRun, setSelectedRun] = useState(null);

  const tabs = [
    { id: 'generator', label: 'Generate Run', icon: Plus },
    { id: 'runs', label: 'All Runs', icon: List },
    { id: 'signals', label: 'View Signals', icon: Eye },
    { id: 'simulate', label: 'Execute Simulation', icon: Play },
  ];

  const handleRunGenerated = (result) => {
    setSelectedRun({ manual_run_id: result.manual_run_id });
    setActiveTab('runs');
  };

  const handleSelectRun = (run) => {
    setSelectedRun(run);
    setActiveTab('signals');
  };

  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-1">
          <div 
            className="w-8 h-8 rounded-xl flex items-center justify-center shadow-lg"
            style={{ 
              background: 'linear-gradient(to bottom right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
              boxShadow: '0 10px 25px rgba(var(--theme-primary), 0.25)'
            }}
          >
            <TestTube className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Strategy Backtesting</h1>
            <p className="text-xs text-white/60">Test your strategies on historical data</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div 
        className="flex gap-2 p-1 rounded-xl backdrop-blur-xl mb-4"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg transition-all duration-300`}
              style={{
                background: activeTab === tab.id 
                  ? 'linear-gradient(to right, rgba(var(--theme-primary), 0.3), rgba(var(--theme-secondary), 0.3))' 
                  : 'transparent',
                border: activeTab === tab.id ? '1px solid rgba(var(--theme-primary), 0.3)' : '1px solid transparent'
              }}
            >
              <Icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-white' : 'text-white/50'}`} />
              <span className={`text-sm font-medium ${activeTab === tab.id ? 'text-white' : 'text-white/50'}`}>
                {tab.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div>
        {activeTab === 'generator' && (
          <RunGenerator onRunGenerated={handleRunGenerated} />
        )}
        {activeTab === 'runs' && (
          <RunsDashboard onSelectRun={handleSelectRun} />
        )}
        {activeTab === 'signals' && (
          <SignalsViewer 
            runId={selectedRun?.manual_run_id} 
            onBack={() => setActiveTab('runs')}
          />
        )}
        {activeTab === 'simulate' && (
          <ExecuteSimulation runId={selectedRun?.manual_run_id} />
        )}
      </div>
    </div>
  );
}