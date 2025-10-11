import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { backtestApi } from "../services/backtestApi";

export default function RunGenerator({ onRunGenerated }) {
  const [formData, setFormData] = useState({
    start_date: '',
    end_date: '',
    symbol: 'XAUUSD',
    timeframe: 'M15',
    initial_balance: 10000,
    commission_per_trade: 0.5,
    spread_adjustment: 0.0003
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const timeframes = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '30m', label: '30 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    // Validation
    if (!formData.start_date || !formData.end_date) {
      setError('Please select both start and end dates');
      return;
    }

    if (new Date(formData.start_date) >= new Date(formData.end_date)) {
      setError('Start date must be before end date');
      return;
    }

    try {
      setLoading(true);
      const response = await backtestApi.generateBacktest({
        ...formData,
        start_date: new Date(formData.start_date).toISOString(),
        end_date: new Date(formData.end_date).toISOString()
      });
      
      setResult(response);
      if (onRunGenerated) {
        onRunGenerated(response);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div 
      className="rounded-2xl p-6 backdrop-blur-xl"
      style={{ 
        background: 'rgba(0, 0, 0, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
      }}
    >
      <div className="flex items-center gap-2 mb-6">
        <Calendar className="w-5 h-5" style={{ color: 'rgb(var(--theme-primary))' }} />
        <h2 className="text-lg font-bold text-white">Generate Backtest Run</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Date Range */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Start Date</label>
            <Input
              type="datetime-local"
              value={formData.start_date}
              onChange={(e) => setFormData({...formData, start_date: e.target.value})}
              className="bg-white/5 border-white/10 text-white"
              required
            />
          </div>
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">End Date</label>
            <Input
              type="datetime-local"
              value={formData.end_date}
              onChange={(e) => setFormData({...formData, end_date: e.target.value})}
              className="bg-white/5 border-white/10 text-white"
              required
            />
          </div>
        </div>

        {/* Symbol and Timeframe */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Symbol</label>
            <Select
              value={formData.symbol}
              onValueChange={(value) => setFormData({...formData, symbol: value})}
            >
              <SelectTrigger className="bg-white/5 border-white/10 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="XAUUSD">XAU/USD (Gold)</SelectItem>
                <SelectItem value="EURUSD">EUR/USD</SelectItem>
                <SelectItem value="GBPUSD">GBP/USD</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Timeframe</label>
            <Select
              value={formData.timeframe}
              onValueChange={(value) => setFormData({...formData, timeframe: value})}
            >
              <SelectTrigger className="bg-white/5 border-white/10 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeframes.map(tf => (
                  <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Optional Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Initial Balance ($)</label>
            <Input
              type="number"
              step="0.01"
              value={formData.initial_balance}
              onChange={(e) => setFormData({...formData, initial_balance: parseFloat(e.target.value)})}
              className="bg-white/5 border-white/10 text-white"
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
            />
          </div>
          <div>
            <label className="text-xs font-medium text-white/70 mb-2 block">Spread Adjustment</label>
            <Input
              type="number"
              step="0.0001"
              value={formData.spread_adjustment}
              onChange={(e) => setFormData({...formData, spread_adjustment: parseFloat(e.target.value)})}
              className="bg-white/5 border-white/10 text-white"
            />
          </div>
        </div>

        {/* Submit Button */}
        <Button
          type="submit"
          disabled={loading}
          className="w-full"
          style={{
            background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
            color: 'white'
          }}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Backtest'
          )}
        </Button>

        {/* Success Result */}
        {result && (
          <div 
            className="p-4 rounded-xl border"
            style={{
              background: 'rgba(34, 197, 94, 0.1)',
              borderColor: 'rgba(34, 197, 94, 0.3)'
            }}
          >
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="w-5 h-5 text-green-400" />
              <span className="text-sm font-semibold text-green-400">Backtest Generated Successfully</span>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-white/50 mb-1">Run ID</div>
                <div className="text-sm font-mono text-white">{result.manual_run_id}</div>
              </div>
              <div>
                <div className="text-xs text-white/50 mb-1">Signals Generated</div>
                <div className="text-sm font-bold text-white">{result.signals_generated}</div>
              </div>
              <div>
                <div className="text-xs text-white/50 mb-1">Avg Confidence</div>
                <div className="text-sm font-bold text-white">{result.average_confidence}%</div>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div 
            className="p-4 rounded-xl border"
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
      </form>
    </div>
  );
}