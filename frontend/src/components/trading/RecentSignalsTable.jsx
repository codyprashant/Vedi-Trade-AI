
import React, { useState, useEffect } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { signalsApi } from "../services/signalsApi";
import { APP_CONFIG } from "../config/appConfig";

export default function RecentSignalsTable({ limit = APP_CONFIG.SIGNAL_LIMITS.DASHBOARD }) {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        setLoading(true);
        const data = await signalsApi.getRecentSignals(limit);
        setSignals(data.signals || []);
      } catch (error) {
        console.error('Failed to fetch signals:', error);
        setSignals([]);
      } finally {
        setLoading(false);
      }
    };

    fetchSignals();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchSignals, 30000);
    return () => clearInterval(interval);
  }, [limit]);

  if (loading) {
    return (
      <div 
        className="rounded-2xl backdrop-blur-xl p-8 text-center" 
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="w-8 h-8 border-2 border-white/20 border-t-amber-500 rounded-full animate-spin mx-auto mb-2"></div>
        <p className="text-sm text-white/60">Loading signals...</p>
      </div>
    );
  }

  if (signals.length === 0) {
    return (
      <div 
        className="rounded-2xl backdrop-blur-xl p-8 text-center" 
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <p className="text-sm text-white/60">No signals available</p>
      </div>
    );
  }

  return (
    <div 
      className="rounded-2xl backdrop-blur-xl overflow-hidden" 
      style={{ 
        background: 'rgba(0, 0, 0, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
      }}
    >
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
            <tr>
              <th className="text-left p-3 text-xs font-semibold text-white/70">ID</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">Time</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">Signal</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">Strength</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">Entry</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">R:R</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">H1 Trend</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">H4 Trend</th>
              <th className="text-left p-3 text-xs font-semibold text-white/70">Volatility</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((signal, index) => {
              const signalTime = new Date(signal.timestamp);
              const timeStr = signalTime.toLocaleString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                hour: '2-digit', 
                minute: '2-digit' 
              });
              
              return (
                <tr 
                  key={signal.id}
                  className="transition-all duration-300 hover:bg-white/5"
                  style={{ 
                    background: index % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.02)',
                    animation: `fadeInUp 0.3s ease-out ${index * 0.1}s backwards` 
                  }}
                >
                  <td className="p-3 text-xs text-white/70">#{signal.id}</td>
                  <td className="p-3 text-xs text-white/70">{timeStr}</td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      {signal.signal_type === "BUY" ? (
                        <>
                          <div 
                            className="w-7 h-7 rounded-lg flex items-center justify-center"
                            style={{ background: 'rgba(34, 197, 94, 0.2)' }}
                          >
                            <TrendingUp className="w-3.5 h-3.5 text-green-400" />
                          </div>
                          <span className="text-xs font-semibold text-green-400">BUY</span>
                        </>
                      ) : (
                        <>
                          <div 
                            className="w-7 h-7 rounded-lg flex items-center justify-center"
                            style={{ background: 'rgba(239, 68, 68, 0.2)' }}
                          >
                            <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                          </div>
                          <span className="text-xs font-semibold text-red-400">SELL</span>
                        </>
                      )}
                    </div>
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-16 h-1.5 rounded-full overflow-hidden"
                        style={{ background: 'rgba(255, 255, 255, 0.1)' }}
                      >
                        <div 
                          className="h-full rounded-full"
                          style={{ 
                            width: `${signal.final_signal_strength}%`,
                            background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
                          }}
                        ></div>
                      </div>
                      <span className="text-xs font-bold text-white">{signal.final_signal_strength}%</span>
                    </div>
                  </td>
                  <td className="p-3 text-xs font-medium text-white/80">${signal.entry_price}</td>
                  <td className="p-3 text-xs font-bold text-green-400">1:{signal.risk_reward_ratio}</td>
                  <td className="p-3">
                    <span className={`text-xs font-medium ${
                      signal.h1_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {signal.h1_trend_direction}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className={`text-xs font-medium ${
                      signal.h4_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {signal.h4_trend_direction}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                      signal.volatility_state === 'Low' 
                        ? 'bg-green-500/20 text-green-400' 
                        : signal.volatility_state === 'High'
                        ? 'bg-red-500/20 text-red-400'
                        : 'bg-blue-500/20 text-blue-400'
                    }`}>
                      {signal.volatility_state}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
