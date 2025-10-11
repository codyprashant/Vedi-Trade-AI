import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Loader2, ArrowLeft, ChevronDown, ChevronUp, BarChart3 } from "lucide-react";
import { backtestApi } from "../services/backtestApi";

export default function SignalsViewer({ runId, onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSignal, setExpandedSignal] = useState(null);
  const [filter, setFilter] = useState({ type: 'all', minStrength: 0 });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const result = await backtestApi.fetchSignals(runId);
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (runId) {
      fetchData();
    }
  }, [runId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'rgb(var(--theme-primary))' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-400">{error}</p>
        <Button onClick={onBack} className="mt-4">Go Back</Button>
      </div>
    );
  }

  const filteredSignals = data?.signals?.filter(signal => {
    const typeMatch = filter.type === 'all' || signal.signal_type === filter.type;
    const strengthMatch = signal.final_signal_strength >= filter.minStrength;
    return typeMatch && strengthMatch;
  }) || [];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div 
        className="rounded-2xl p-4 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Button onClick={onBack} size="sm" variant="ghost">
              <ArrowLeft className="w-4 h-4" />
            </Button>
            <div>
              <h2 className="text-sm font-bold text-white">Signals for {runId}</h2>
              <p className="text-xs text-white/50">{filteredSignals.length} signals found</p>
            </div>
          </div>

          {/* Filters */}
          <div className="flex gap-2">
            <select
              value={filter.type}
              onChange={(e) => setFilter({...filter, type: e.target.value})}
              className="px-3 py-1 rounded-lg bg-white/5 border border-white/10 text-xs"
              style={{ color: 'white' }}
            >
              <option value="all" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>All Types</option>
              <option value="BUY" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>BUY Only</option>
              <option value="SELL" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>SELL Only</option>
            </select>
            <input
              type="number"
              value={filter.minStrength}
              onChange={(e) => setFilter({...filter, minStrength: parseInt(e.target.value) || 0})}
              placeholder="Min Strength"
              className="w-24 px-3 py-1 rounded-lg bg-white/5 border border-white/10 text-white placeholder-white/40 text-xs"
            />
          </div>
        </div>
      </div>

      {/* Signals Table */}
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
                <th className="text-left p-3 text-xs font-semibold text-white/70">Time</th>
                <th className="text-left p-3 text-xs font-semibold text-white/70">Signal</th>
                <th className="text-right p-3 text-xs font-semibold text-white/70">Entry</th>
                <th className="text-right p-3 text-xs font-semibold text-white/70">Stop Loss</th>
                <th className="text-right p-3 text-xs font-semibold text-white/70">Take Profit</th>
                <th className="text-center p-3 text-xs font-semibold text-white/70">Strength</th>
                <th className="text-center p-3 text-xs font-semibold text-white/70">R:R</th>
                <th className="text-center p-3 text-xs font-semibold text-white/70">Volatility</th>
                <th className="text-center p-3 text-xs font-semibold text-white/70"></th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.map((signal, index) => (
                <React.Fragment key={index}>
                  <tr
                    className="hover:bg-white/5 transition-colors"
                    style={{
                      background: index % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.02)'
                    }}
                  >
                    <td className="p-3 text-xs text-white/70">
                      {new Date(signal.timestamp).toLocaleString()}
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        {signal.signal_type === "BUY" ? (
                          <>
                            <TrendingUp className="w-4 h-4 text-green-400" />
                            <span className="text-xs font-semibold text-green-400">BUY</span>
                          </>
                        ) : (
                          <>
                            <TrendingDown className="w-4 h-4 text-red-400" />
                            <span className="text-xs font-semibold text-red-400">SELL</span>
                          </>
                        )}
                      </div>
                    </td>
                    <td className="p-3 text-right text-xs font-medium text-white">${signal.entry_price}</td>
                    <td className="p-3 text-right text-xs text-red-400">${signal.stop_loss_price}</td>
                    <td className="p-3 text-right text-xs text-green-400">${signal.take_profit_price}</td>
                    <td className="p-3 text-center">
                      <span 
                        className="px-2 py-1 rounded-full text-xs font-bold"
                        style={{
                          background: 'rgba(var(--theme-primary), 0.2)',
                          color: 'rgb(var(--theme-primary))'
                        }}
                      >
                        {signal.final_signal_strength}%
                      </span>
                    </td>
                    <td className="p-3 text-center text-xs font-bold text-green-400">
                      1:{signal.risk_reward_ratio}
                    </td>
                    <td className="p-3 text-center">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        signal.volatility_state === 'Low' ? 'bg-green-500/20 text-green-400' :
                        signal.volatility_state === 'High' ? 'bg-red-500/20 text-red-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {signal.volatility_state}
                      </span>
                    </td>
                    <td className="p-3 text-center">
                      <button
                        onClick={() => setExpandedSignal(expandedSignal === index ? null : index)}
                        className="p-1 hover:bg-white/10 rounded"
                      >
                        {expandedSignal === index ? (
                          <ChevronUp className="w-4 h-4 text-white/60" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-white/60" />
                        )}
                      </button>
                    </td>
                  </tr>
                  {expandedSignal === index && (
                    <tr>
                      <td colSpan="9" className="p-4" style={{ background: 'rgba(255, 255, 255, 0.03)' }}>
                        <div className="flex items-center gap-2 mb-3">
                          <BarChart3 className="w-4 h-4" style={{ color: 'rgb(var(--theme-primary))' }} />
                          <h4 className="text-xs font-bold text-white">Indicator Contributions</h4>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {Object.entries(signal.indicator_contributions || {}).map(([name, value]) => (
                            <div key={name} className="p-2 rounded-lg bg-white/5">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-white/70">{name}</span>
                                <span className="text-xs font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
                                  {value}%
                                </span>
                              </div>
                              <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                                <div 
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${value}%`,
                                    background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
                                  }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}