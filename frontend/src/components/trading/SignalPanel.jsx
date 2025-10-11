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

const generateMockHistory = () => {
  return Array.from({ length: 5 }, (_, i) => ({
    id: i,
    timestamp: new Date(Date.now() - i * 3600000).toLocaleString(),
    verdict: ["BUY", "SELL", "HOLD"][Math.floor(Math.random() * 3)],
    strength: Math.floor(Math.random() * 10) + 91,
  }));
};

export default function SignalPanel() {
  const activeSignal = generateMockSignal();
  const recentSignals = generateMockHistory();

  return (
    <div className="space-y-3">
      {/* Active Signal */}
      <div 
        className="rounded-2xl p-3 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center gap-1.5 mb-2">
          <Zap className="w-3.5 h-3.5" style={{ color: 'rgb(var(--theme-primary))' }} />
          <h3 className="text-xs font-semibold text-white">Active Signal</h3>
        </div>

        {/* Verdict Badge */}
        <div 
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg mb-2`}
          style={{
            background: activeSignal.verdict === "BUY" 
              ? 'rgba(34, 197, 94, 0.2)' 
              : activeSignal.verdict === "SELL"
              ? 'rgba(239, 68, 68, 0.2)'
              : 'rgba(59, 130, 246, 0.2)'
          }}
        >
          {activeSignal.verdict === "BUY" ? (
            <TrendingUp className="w-3.5 h-3.5 text-green-400" />
          ) : activeSignal.verdict === "SELL" ? (
            <TrendingDown className="w-3.5 h-3.5 text-red-400" />
          ) : (
            <Activity className="w-3.5 h-3.5 text-blue-400" />
          )}
          <span className={`text-sm font-bold ${
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
        <div className="flex items-center gap-1 mb-2 text-[10px] text-white/60">
          <Clock className="w-3 h-3" />
          <span>{activeSignal.timestamp}</span>
        </div>

        {/* Signal Strength */}
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-medium text-white/70">Signal Strength</span>
            <span className="text-base font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
              {activeSignal.strength}%
            </span>
          </div>
          <div 
            className="w-full h-1.5 rounded-full overflow-hidden"
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
        <div className="space-y-1.5">
          <h4 className="text-[10px] font-semibold text-white mb-1.5">Strategy Breakdown</h4>
          
          <div className="space-y-1">
            <div 
              className="flex items-center justify-between p-1.5 rounded-lg backdrop-blur-sm"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <span className="text-[10px] text-white/70">RSI Divergence</span>
              <span className="text-[10px] font-bold text-white">{activeSignal.strategies.rsiDivergence}%</span>
            </div>
            
            <div 
              className="flex items-center justify-between p-1.5 rounded-lg backdrop-blur-sm"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <span className="text-[10px] text-white/70">MACD Cross</span>
              <span className="text-[10px] font-bold text-white">{activeSignal.strategies.macdCross}%</span>
            </div>
            
            <div 
              className="flex items-center justify-between p-1.5 rounded-lg backdrop-blur-sm"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <span className="text-[10px] text-white/70">EMA Breakout</span>
              <span className="text-[10px] font-bold text-white">{activeSignal.strategies.emaBreakout}%</span>
            </div>
            
            <div 
              className="flex items-center justify-between p-1.5 rounded-lg backdrop-blur-sm"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <span className="text-[10px] text-white/70">SMA Trend</span>
              <span className="text-[10px] font-bold text-white">{activeSignal.strategies.smaTrend}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Signals */}
      <div 
        className="rounded-2xl p-3 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <h3 className="text-xs font-semibold text-white mb-2">Recent Signals</h3>
        <div className="space-y-1.5">
          {recentSignals.map(signal => (
            <div 
              key={signal.id}
              className="flex items-center justify-between p-2 rounded-lg backdrop-blur-sm transition-all duration-300 cursor-pointer"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <div className="flex items-center gap-1.5">
                {signal.verdict === "BUY" ? (
                  <div 
                    className="w-6 h-6 rounded-lg flex items-center justify-center"
                    style={{ background: 'rgba(34, 197, 94, 0.2)' }}
                  >
                    <TrendingUp className="w-3 h-3 text-green-400" />
                  </div>
                ) : signal.verdict === "SELL" ? (
                  <div 
                    className="w-6 h-6 rounded-lg flex items-center justify-center"
                    style={{ background: 'rgba(239, 68, 68, 0.2)' }}
                  >
                    <TrendingDown className="w-3 h-3 text-red-400" />
                  </div>
                ) : (
                  <div 
                    className="w-6 h-6 rounded-lg flex items-center justify-center"
                    style={{ background: 'rgba(59, 130, 246, 0.2)' }}
                  >
                    <Activity className="w-3 h-3 text-blue-400" />
                  </div>
                )}
                <div>
                  <div className={`text-[10px] font-medium ${
                    signal.verdict === "BUY" 
                      ? "text-green-400" 
                      : signal.verdict === "SELL"
                      ? "text-red-400"
                      : "text-blue-400"
                  }`}>
                    {signal.verdict}
                  </div>
                  <div className="text-[9px] text-white/50">{signal.timestamp}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] font-bold text-white">{signal.strength}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}