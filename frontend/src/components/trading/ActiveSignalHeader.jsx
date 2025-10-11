import React, { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Clock, Zap, AlertCircle, ChevronDown, ChevronUp, BarChart3 } from "lucide-react";
import { signalsApi } from "../services/signalsApi";

export default function ActiveSignalHeader() {
  const [activeSignal, setActiveSignal] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const fetchLatestSignal = async () => {
      try {
        setLoading(true);
        const signal = await signalsApi.getLatestSignal();
        setActiveSignal(signal);
      } catch (error) {
        console.error('Failed to fetch latest signal:', error);
        setActiveSignal(null);
      } finally {
        setLoading(false);
      }
    };

    fetchLatestSignal();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchLatestSignal, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div 
        className="rounded-2xl p-4 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center justify-center gap-3 py-2">
          <div className="w-5 h-5 border-2 border-white/20 border-t-amber-500 rounded-full animate-spin"></div>
          <span className="text-sm text-white/60">Loading signal...</span>
        </div>
      </div>
    );
  }

  // No signal in last 30 minutes
  if (!activeSignal) {
    return (
      <div 
        className="rounded-2xl p-4 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center justify-center gap-3 py-2">
          <AlertCircle className="w-5 h-5 text-white/40" />
          <div className="text-center">
            <h3 className="text-sm font-semibold text-white/70">No Active Signal</h3>
            <p className="text-xs text-white/40">No signals detected in the last 30 minutes</p>
          </div>
        </div>
      </div>
    );
  }

  const verdict = activeSignal.signal_type;
  const strength = activeSignal.final_signal_strength;
  
  // Sort contributions by value for better display
  const sortedContributions = Object.entries(activeSignal.indicator_contributions || {})
    .sort(([, a], [, b]) => b - a);

  return (
    <div 
      className="rounded-2xl backdrop-blur-xl overflow-hidden transition-all duration-300"
      style={{ 
        background: 'rgba(0, 0, 0, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
        border: '1px solid rgba(251, 191, 36, 0.2)'
      }}
    >
      <div className="p-4 space-y-4">
        {/* Header: Signal Type & Expand Button */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div 
              className="w-12 h-12 rounded-xl flex items-center justify-center relative overflow-hidden shadow-lg"
              style={{
                background: verdict === "BUY" 
                  ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(34, 197, 94, 0.1))' 
                  : 'linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.1))',
                border: `2px solid ${verdict === "BUY" ? "rgba(34, 197, 94, 0.5)" : "rgba(239, 68, 68, 0.5)"}`,
              }}
            >
              {verdict === "BUY" ? (
                <TrendingUp className="w-6 h-6 text-green-400" />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-400" />
              )}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
            </div>

            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-xl font-black ${verdict === "BUY" ? "text-green-400" : "text-red-400"}`}>
                  {verdict}
                </span>
                <span 
                  className="px-2 py-0.5 rounded-full text-xs font-bold"
                  style={{
                    background: verdict === "BUY" ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                    color: verdict === "BUY" ? '#4ade80' : '#f87171'
                  }}
                >
                  {strength}%
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-white/60">
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>{activeSignal.minutesAgo} min ago</span>
                </div>
                <div className="flex items-center gap-1">
                  <Zap className="w-3 h-3" style={{ color: 'rgb(var(--theme-primary))' }} />
                  <span>#{activeSignal.id}</span>
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 hover:bg-white/10 rounded-lg transition-all duration-300"
          >
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-white/60" />
            ) : (
              <ChevronDown className="w-5 h-5 text-white/60" />
            )}
          </button>
        </div>

        {/* Signal Strength Bar */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-white/70">Signal Strength</span>
            <span className="text-lg font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
              {strength}%
            </span>
          </div>
          <div 
            className="w-full h-2.5 rounded-full overflow-hidden relative"
            style={{ background: 'rgba(255, 255, 255, 0.1)' }}
          >
            <div 
              className="h-full rounded-full transition-all duration-1000 relative overflow-hidden"
              style={{ 
                width: `${strength}%`,
                background: `linear-gradient(90deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))`,
                boxShadow: `0 0 15px rgba(var(--theme-primary), 0.5)`
              }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
            </div>
          </div>
        </div>

        {/* Price Levels - Vertical Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <div 
            className="p-3 rounded-xl text-center"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <div className="text-xs text-white/50 mb-1">Entry</div>
            <div className="text-sm font-bold text-white">${activeSignal.entry_price}</div>
          </div>
          <div 
            className="p-3 rounded-xl text-center"
            style={{ background: 'rgba(239, 68, 68, 0.1)' }}
          >
            <div className="text-xs text-red-400/70 mb-1">Stop Loss</div>
            <div className="text-sm font-semibold text-red-400">${activeSignal.stop_loss_price}</div>
          </div>
          <div 
            className="p-3 rounded-xl text-center"
            style={{ background: 'rgba(34, 197, 94, 0.1)' }}
          >
            <div className="text-xs text-green-400/70 mb-1">Take Profit</div>
            <div className="text-sm font-semibold text-green-400">${activeSignal.take_profit_price}</div>
          </div>
          <div 
            className="p-3 rounded-xl text-center"
            style={{ background: 'rgba(251, 191, 36, 0.1)' }}
          >
            <div className="text-xs text-amber-400/70 mb-1">Risk:Reward</div>
            <div className="text-sm font-bold text-amber-400">1:{activeSignal.risk_reward_ratio}</div>
          </div>
        </div>

        {/* Indicator Contributions - Always Visible Vertically */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4" style={{ color: 'rgb(var(--theme-primary))' }} />
            <h4 className="text-sm font-bold text-white">Indicator Contributions</h4>
          </div>
          <div className="space-y-2">
            {sortedContributions.map(([name, value], idx) => (
              <div 
                key={name}
                className="p-2.5 rounded-xl backdrop-blur-sm hover:bg-white/10 transition-all duration-300"
                style={{ 
                  background: 'rgba(255, 255, 255, 0.05)',
                  animation: `fadeInRight 0.4s ease-out ${idx * 0.05}s backwards`
                }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-semibold text-white">{name}</span>
                  <span className="text-sm font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
                    {value}%
                  </span>
                </div>
                <div 
                  className="w-full h-2 rounded-full overflow-hidden relative"
                  style={{ background: 'rgba(255, 255, 255, 0.1)' }}
                >
                  <div 
                    className="h-full rounded-full relative overflow-hidden"
                    style={{ 
                      width: `${value}%`,
                      background: 'linear-gradient(90deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
                      boxShadow: '0 0 10px rgba(var(--theme-primary), 0.4)',
                      transition: 'width 1s ease-out'
                    }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Expanded: Trends & Volatility */}
        {expanded && (
          <div style={{ animation: 'fadeIn 0.3s ease-out' }}>
            <h4 className="text-sm font-bold text-white mb-3">Market Context</h4>
            <div className="grid grid-cols-3 gap-3">
              <div 
                className="p-3 rounded-xl text-center"
                style={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <div className="text-xs text-white/50 mb-1">H1 Trend</div>
                <div className={`text-sm font-bold ${
                  activeSignal.h1_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {activeSignal.h1_trend_direction}
                </div>
              </div>
              <div 
                className="p-3 rounded-xl text-center"
                style={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <div className="text-xs text-white/50 mb-1">H4 Trend</div>
                <div className={`text-sm font-bold ${
                  activeSignal.h4_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {activeSignal.h4_trend_direction}
                </div>
              </div>
              <div 
                className="p-3 rounded-xl text-center"
                style={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <div className="text-xs text-white/50 mb-1">Volatility</div>
                <div className={`text-sm font-bold ${
                  activeSignal.volatility_state === 'Low' 
                    ? 'text-green-400' 
                    : activeSignal.volatility_state === 'High'
                    ? 'text-red-400'
                    : 'text-blue-400'
                }`}>
                  {activeSignal.volatility_state}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInRight {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }

        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </div>
  );
}