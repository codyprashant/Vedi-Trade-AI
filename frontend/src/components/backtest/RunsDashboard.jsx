import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RefreshCw, Filter, Loader2, Eye } from "lucide-react";
import { backtestApi } from "../services/backtestApi";

export default function RunsDashboard({ onSelectRun }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    from_date: '',
    to_date: '',
    min_signal_strength: '',
    symbol: ''
  });

  const fetchRuns = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await backtestApi.listRuns(filters);
      setRuns(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const applyFilters = () => {
    fetchRuns();
  };

  const clearFilters = () => {
    setFilters({
      from_date: '',
      to_date: '',
      min_signal_strength: '',
      symbol: ''
    });
    setTimeout(() => fetchRuns(), 100);
  };

  if (loading && runs.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'rgb(var(--theme-primary))' }} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div 
        className="rounded-2xl p-4 backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-4 h-4" style={{ color: 'rgb(var(--theme-primary))' }} />
          <h3 className="text-sm font-bold text-white">Filters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <Input
            type="date"
            placeholder="From Date"
            value={filters.from_date}
            onChange={(e) => handleFilterChange('from_date', e.target.value)}
            className="bg-white/5 border-white/10 text-white text-sm"
          />
          <Input
            type="date"
            placeholder="To Date"
            value={filters.to_date}
            onChange={(e) => handleFilterChange('to_date', e.target.value)}
            className="bg-white/5 border-white/10 text-white text-sm"
          />
          <Input
            type="number"
            placeholder="Min Strength"
            value={filters.min_signal_strength}
            onChange={(e) => handleFilterChange('min_signal_strength', e.target.value)}
            className="bg-white/5 border-white/10 text-white text-sm"
          />
          <Select
            value={filters.symbol}
            onValueChange={(value) => handleFilterChange('symbol', value)}
          >
            <SelectTrigger className="bg-white/5 border-white/10 text-white text-sm">
              <SelectValue placeholder="Symbol" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={null}>All Symbols</SelectItem>
              <SelectItem value="XAUUSD">XAU/USD</SelectItem>
              <SelectItem value="EURUSD">EUR/USD</SelectItem>
              <SelectItem value="GBPUSD">GBP/USD</SelectItem>
            </SelectContent>
          </Select>
          <div className="flex gap-2">
            <Button
              onClick={applyFilters}
              size="sm"
              className="flex-1"
              style={{
                background: 'linear-gradient(to right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))'
              }}
            >
              Apply
            </Button>
            <Button
              onClick={clearFilters}
              size="sm"
              variant="outline"
              className="border-white/10"
            >
              Clear
            </Button>
          </div>
        </div>
      </div>

      {/* Runs Table */}
      <div 
        className="rounded-2xl backdrop-blur-xl overflow-hidden"
        style={{ 
          background: 'rgba(0, 0, 0, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <h2 className="text-sm font-bold text-white">Backtest Runs ({runs.length})</h2>
          <Button
            onClick={fetchRuns}
            size="sm"
            variant="ghost"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {error ? (
          <div className="p-8 text-center text-red-400">{error}</div>
        ) : runs.length === 0 ? (
          <div className="p-8 text-center text-white/50">No backtest runs found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
                <tr>
                  <th className="text-left p-3 text-xs font-semibold text-white/70">Run ID</th>
                  <th className="text-left p-3 text-xs font-semibold text-white/70">Symbol</th>
                  <th className="text-left p-3 text-xs font-semibold text-white/70">Timeframe</th>
                  <th className="text-left p-3 text-xs font-semibold text-white/70">Start Date</th>
                  <th className="text-left p-3 text-xs font-semibold text-white/70">End Date</th>
                  <th className="text-center p-3 text-xs font-semibold text-white/70">Signals</th>
                  <th className="text-center p-3 text-xs font-semibold text-white/70">Avg Confidence</th>
                  <th className="text-center p-3 text-xs font-semibold text-white/70">Avg R:R</th>
                  <th className="text-center p-3 text-xs font-semibold text-white/70">Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run, index) => (
                  <tr
                    key={run.manual_run_id}
                    className="hover:bg-white/5 transition-colors cursor-pointer"
                    style={{
                      background: index % 2 === 0 ? 'transparent' : 'rgba(255, 255, 255, 0.02)'
                    }}
                  >
                    <td className="p-3 text-xs font-mono text-white/80">{run.manual_run_id}</td>
                    <td className="p-3 text-xs text-white">{run.symbol}</td>
                    <td className="p-3 text-xs text-white">{run.timeframe}</td>
                    <td className="p-3 text-xs text-white/70">{new Date(run.start_date).toLocaleDateString()}</td>
                    <td className="p-3 text-xs text-white/70">{new Date(run.end_date).toLocaleDateString()}</td>
                    <td className="p-3 text-center text-sm font-bold text-white">{run.signals_generated}</td>
                    <td className="p-3 text-center">
                      <span 
                        className="px-2 py-1 rounded-full text-xs font-bold"
                        style={{
                          background: 'rgba(var(--theme-primary), 0.2)',
                          color: 'rgb(var(--theme-primary))'
                        }}
                      >
                        {run.average_confidence}%
                      </span>
                    </td>
                    <td className="p-3 text-center text-sm font-medium text-green-400">
                      {run.average_rr_ratio}
                    </td>
                    <td className="p-3 text-center">
                      <Button
                        onClick={() => onSelectRun && onSelectRun(run)}
                        size="sm"
                        variant="ghost"
                        className="h-7"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}