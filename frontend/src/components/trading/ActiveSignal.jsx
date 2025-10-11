import React from "react";
import { TrendingUp, TrendingDown, Clock, Zap, Activity } from "lucide-react";

const generateMockSignal = () => {
  const verdict = ["BUY", "SELL", "HOLD"][Math.floor(Math.random() * 3)];
  const strength = Math.floor(Math.random() * 10) + 91;
  
  return {
    timestamp: new Date().toLocaleString(),
    verdict,
    strength,
    strategies: {
      rsiDivergence: Math.floor(Math.random() * 20) + 80,
      macdCross: Math.floor(Math.random() * 20) + 80,
      emaBreakout: Math.floor(Math.random() * 20) + 80,
      smaTrend: Math.floor(Math.random() * 20) + 80,
    }
  };
};

export default function ActiveSignal() {
  const activeSignal = generateMockSignal();

  return (
    <div 
      className="rounded-2xl p-4 backdrop-blur-xl sticky top-4"
      style={{ 
        background: 'rgba(0, 0, 0, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Zap className="w-4 h-4" style={{ color: 'rgb(var(--theme-primary))' }} />
        <h3 className="text-sm font-semibold text-white">Active Signal</h3>
      </div>

      {/* Verdict Badge */}
      <div 
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl mb-3`}
        style={{
          background: activeSignal.verdict === "BUY" 
            ? 'rgba(34, 197, 94, 0.2)' 
            : activeSignal.verdict === "SELL"
            ? 'rgba(239, 68, 68, 0.2)'
            : 'rgba(59, 130, 246, 0.2)'
        }}
      >
        {activeSignal.verdict === "BUY" ? (
          <TrendingUp className="w-4 h-4 text-green-400" />
        ) : activeSignal.verdict === "SELL" ? (
          <TrendingDown className="w-4 h-4 text-red-400" />
        ) : (
          <Activity className="w-4 h-4 text-blue-400" />
        )}
        <span className={`text-base font-bold ${
          activeSignal.verdict === "BUY" 
            ? "text-green-400" 
            : activeSignal.verdict === "SELL"
            ? "text-red-400"
            : "text-blue-400"
        }`}>
          {activeSignal.verdict}
        </span>
      </div>

      {/* Timestamp */}
      <div className="flex items-center gap-1.5 mb-4 text-xs text-white/60">
        <Clock className="w-3.5 h-3.5" />
        <span>{activeSignal.timestamp}</span>
      </div>

      {/* Signal Strength */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-white/70">Signal Strength</span>
          <span className="text-xl font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
            {activeSignal.strength}%
          </span>
        </div>
        <div 
          className="w-full h-2 rounded-full overflow-hidden"
          style={{ background: 'rgba(255, 255, 255, 0.1)' }}
        >
          <div 
            className="h-full rounded-full transition-all duration-1000"
            style={{ 
              width: `${activeSignal.strength}%`,
              background: `linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))`
            }}
          ></div>
        </div>
      </div>

      {/* Strategy Breakdown */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-white mb-2">Strategy Breakdown</h4>
        
        <div className="space-y-2">
          <div 
            className="flex items-center justify-between p-2 rounded-xl backdrop-blur-sm"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <span className="text-xs text-white/70">RSI Divergence</span>
            <span className="text-xs font-bold text-white">{activeSignal.strategies.rsiDivergence}%</span>
          </div>
          
          <div 
            className="flex items-center justify-between p-2 rounded-xl backdrop-blur-sm"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <span className="text-xs text-white/70">MACD Cross</span>
            <span className="text-xs font-bold text-white">{activeSignal.strategies.macdCross}%</span>
          </div>
          
          <div 
            className="flex items-center justify-between p-2 rounded-xl backdrop-blur-sm"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <span className="text-xs text-white/70">EMA Breakout</span>
            <span className="text-xs font-bold text-white">{activeSignal.strategies.emaBreakout}%</span>
          </div>
          
          <div 
            className="flex items-center justify-between p-2 rounded-xl backdrop-blur-sm"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <span className="text-xs text-white/70">SMA Trend</span>
            <span className="text-xs font-bold text-white">{activeSignal.strategies.smaTrend}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}