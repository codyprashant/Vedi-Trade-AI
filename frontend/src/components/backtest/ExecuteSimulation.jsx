import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Play, Loader2, TrendingUp, AlertTriangle, CheckCircle2, AlertCircle } from "lucide-react";
import { backtestApi } from "../services/backtestApi";

export default function ExecuteSimulation({ runId }) {
  const [formData, setFormData] = useState({
    manual_run_id: runId || '',
    initial_balance: 10000,
    risk_per_trade_percent: 2,
    commission_per_trade: 0.5,
    slippage_percent: 0.02
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!formData.manual_run_id) {
      setError('Please provide a valid run ID');
      return;
    }

    try {
      setLoading(true);
      const response = await backtestApi.executeSimulation(formData);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Form */}
      <div 
        className="rounded-2xl p-6 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <Play className="w-5 h-5" style={{ color: 'rgb(var(--theme-primary))' }} />
          <h2 className="text-lg font-bold text-white">Execute Trade Simulation</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Run ID</label>
            <Input
              type="text"
              value={formData.manual_run_id}
              onChange={(e) => setFormData({...formData, manual_run_id: e.target.value})}
              className="bg-white/5 border-white/10 text-white font-mono"
              placeholder="manual_run_20250101_120000"
              required
            />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-xs font-medium text-white/70 mb-2 block">Initial Balance ($)</label>
              <Input
                type="number"
                step="0.01"
                value={formData.initial_balance}
                onChange={(e) => setFormData({...formData, initial_balance: parseFloat(e.target.value)})}
                className="bg-white/5 border-white/10 text-white"
                required
              />
            </div>
            <div>
              <label className="text-xs font-medium text-white/70 mb-2 block">Risk Per Trade (%)</label>
              <Input
                type="number"
                step="0.1"
                value={formData.risk_per_trade_percent}
                onChange={(e) => setFormData({...formData, risk_per_trade_percent: parseFloat(e.target.value)})}
                className="bg-white/5 border-white/10 text-white"
                required
              />
            </div>
            <div>
              <label className="text-xs font-medium text-white/70 mb-2 block">Commission ($)</label>
              <Input
                type="number"
                step="0.01"
                value={formData.commission_per_trade}
                onChange={(e) => setFormData({...formData, commission_per_trade: parseFloat(e.target.value)})}
                className="bg-white/5 border-white/10 text-white"
                required
              />
            </div>
            <div>
              <label className="text-xs font-medium text-white/70 mb-2 block">Slippage (%)</label>
              <Input
                type="number"
                step="0.01"
                value={formData.slippage_percent}
                onChange={(e) => setFormData({...formData, slippage_percent: parseFloat(e.target.value)})}
                className="bg-white/5 border-white/10 text-white"
                required
              />
            </div>
          </div>

          <Button
            type="submit"
            disabled={loading}
            className="w-full"
            style={{
              background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
            }}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Running Simulation...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run Simulation
              </>
            )}
          </Button>
        </form>
      </div>

      {/* Results */}
      {result && (
        <div 
          className="rounded-2xl p-6 backdrop-blur-xl border"
          style={{ 
            background: result.account_blown 
              ? 'rgba(239, 68, 68, 0.1)' 
              : result.net_profit_percent > 0 
                ? 'rgba(34, 197, 94, 0.1)' 
                : 'rgba(59, 130, 246, 0.1)',
            borderColor: result.account_blown 
              ? 'rgba(239, 68, 68, 0.3)' 
              : result.net_profit_percent > 0 
                ? 'rgba(34, 197, 94, 0.3)' 
                : 'rgba(59, 130, 246, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
          }}
        >
          <div className="flex items-center gap-2 mb-6">
            {result.account_blown ? (
              <AlertTriangle className="w-6 h-6 text-red-400" />
            ) : result.net_profit_percent > 0 ? (
              <CheckCircle2 className="w-6 h-6 text-green-400" />
            ) : (
              <TrendingUp className="w-6 h-6 text-blue-400" />
            )}
            <h3 className="text-lg font-bold text-white">Simulation Results</h3>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div 
              className="p-4 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <div className="text-xs text-white/50 mb-1">Initial Balance</div>
              <div className="text-xl font-bold text-white">${result.initial_balance.toLocaleString()}</div>
            </div>
            <div 
              className="p-4 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <div className="text-xs text-white/50 mb-1">Final Balance</div>
              <div className={`text-xl font-bold ${result.net_profit_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${result.final_balance.toLocaleString()}
              </div>
            </div>
            <div 
              className="p-4 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <div className="text-xs text-white/50 mb-1">Net Profit</div>
              <div className={`text-xl font-bold ${result.net_profit_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {result.net_profit_percent >= 0 ? '+' : ''}{result.net_profit_percent.toFixed(2)}%
              </div>
            </div>
            <div 
              className="p-4 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.05)' }}
            >
              <div className="text-xs text-white/50 mb-1">Max Drawdown</div>
              <div className="text-xl font-bold text-red-400">
                {result.max_drawdown_percent.toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div 
              className="p-3 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.03)' }}
            >
              <div className="text-xs text-white/50 mb-1">Total Trades</div>
              <div className="text-lg font-bold text-white">{result.total_trades}</div>
            </div>
            <div 
              className="p-3 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.03)' }}
            >
              <div className="text-xs text-white/50 mb-1">Win Rate</div>
              <div className="text-lg font-bold text-green-400">{result.win_rate_percent.toFixed(1)}%</div>
            </div>
            <div 
              className="p-3 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.03)' }}
            >
              <div className="text-xs text-white/50 mb-1">Profit Factor</div>
              <div className="text-lg font-bold text-white">{result.profit_factor.toFixed(2)}</div>
            </div>
            <div 
              className="p-3 rounded-xl"
              style={{ background: 'rgba(255, 255, 255, 0.03)' }}
            >
              <div className="text-xs text-white/50 mb-1">Avg R:R</div>
              <div className="text-lg font-bold text-green-400">1:{result.average_rr_ratio.toFixed(2)}</div>
            </div>
          </div>

          <div 
            className="p-4 rounded-xl"
            style={{ background: 'rgba(255, 255, 255, 0.05)' }}
          >
            <div className="text-xs text-white/50 mb-2">Summary</div>
            <p className="text-sm text-white/80">{result.result_summary}</p>
            {result.critical_event && (
              <p className="text-sm text-red-400 mt-2">⚠️ {result.critical_event}</p>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div 
          className="rounded-2xl p-4 backdrop-blur-xl border"
          style={{
            background: 'rgba(239, 68, 68, 0.1)',
            borderColor: 'rgba(239, 68, 68, 0.3)'
          }}
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-red-400">{error}</span>
          </div>
        </div>
      )}
    </div>
  );
}