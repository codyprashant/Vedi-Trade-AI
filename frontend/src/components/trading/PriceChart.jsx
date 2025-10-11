
import React, { useState } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ComposedChart } from "recharts";
import { TrendingUp, RefreshCw, BarChart3, Activity } from "lucide-react";

// Mock data generator
const generateMockData = (timeframe, points = 50) => {
  const data = [];
  const basePrice = 2050;
  let currentPrice = basePrice;
  const now = Date.now();
  
  const timeframeMinutes = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1hr': 60,
    '4hr': 240
  };
  
  const interval = timeframeMinutes[timeframe] * 60 * 1000;
  
  for (let i = points - 1; i >= 0; i--) {
    const volatility = Math.random() * 5 - 2.5;
    currentPrice += volatility;
    
    const open = currentPrice;
    const close = currentPrice + (Math.random() * 4 - 2);
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    
    data.push({
      time: new Date(now - i * interval).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      price: close,
      open: open,
      high: high,
      low: low,
      close: close,
    });
  }
  
  return data;
};

const CustomAreaChart = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3}/>
            <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis 
          dataKey="time" 
          stroke="rgba(255,255,255,0.3)"
          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
        />
        <YAxis 
          stroke="rgba(255,255,255,0.3)"
          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
          domain={['dataMin - 5', 'dataMax + 5']}
        />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(0,0,0,0.8)', 
            border: '1px solid rgba(251,191,36,0.3)',
            borderRadius: '12px',
            backdropFilter: 'blur(10px)'
          }}
          labelStyle={{ color: '#FFF' }}
        />
        <Area 
          type="monotone" 
          dataKey="close" 
          stroke="#F59E0B" 
          strokeWidth={2}
          fill="url(#priceGradient)" 
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

const CustomCandlestickChart = ({ data }) => {
  const CandlestickBar = (props) => {
    const { x, y, width, payload } = props;
    const { open, close, high, low } = payload;
    
    const isGreen = close > open;
    const color = isGreen ? '#10b981' : '#ef4444';
    const bodyHeight = Math.abs(close - open);
    const bodyY = Math.min(close, open);
    
    return (
      <g>
        {/* Wick (high-low line) */}
        <line
          x1={x + width / 2}
          y1={y - (high - Math.max(open, close))}
          x2={x + width / 2}
          y2={y + bodyHeight + (Math.min(open, close) - low)}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x}
          y={y}
          width={width}
          height={bodyHeight || 1}
          fill={color}
          stroke={color}
          strokeWidth={1}
          opacity={0.8}
        />
      </g>
    );
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis 
          dataKey="time" 
          stroke="rgba(255,255,255,0.3)"
          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
        />
        <YAxis 
          stroke="rgba(255,255,255,0.3)"
          tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
          domain={['dataMin - 5', 'dataMax + 5']}
        />
        <Tooltip 
          contentStyle={{ 
            background: 'rgba(0,0,0,0.8)', 
            border: '1px solid rgba(251,191,36,0.3)',
            borderRadius: '12px',
            backdropFilter: 'blur(10px)'
          }}
          labelStyle={{ color: '#FFF' }}
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              const data = payload[0].payload;
              return (
                <div style={{ 
                  background: 'rgba(0,0,0,0.9)', 
                  border: '1px solid rgba(251,191,36,0.3)',
                  borderRadius: '12px',
                  padding: '12px',
                  backdropFilter: 'blur(10px)'
                }}>
                  <p style={{ color: '#FFF', marginBottom: '8px', fontSize: '12px' }}>{data.time}</p>
                  <p style={{ color: '#10b981', fontSize: '11px', marginBottom: '4px' }}>O: {data.open.toFixed(2)}</p>
                  <p style={{ color: '#3b82f6', fontSize: '11px', marginBottom: '4px' }}>H: {data.high.toFixed(2)}</p>
                  <p style={{ color: '#ef4444', fontSize: '11px', marginBottom: '4px' }}>L: {data.low.toFixed(2)}</p>
                  <p style={{ color: '#f59e0b', fontSize: '11px' }}>C: {data.close.toFixed(2)}</p>
                </div>
              );
            }
            return null;
          }}
        />
        <Bar 
          dataKey="close" 
          shape={<CandlestickBar />}
          maxBarSize={20}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default function PriceChart() {
  const [timeframe, setTimeframe] = useState('15min');
  const [chartType, setChartType] = useState('line');
  const [data, setData] = useState(generateMockData(timeframe));
  const [refreshing, setRefreshing] = useState(false);
  const [apiCallsLeft, setApiCallsLeft] = useState(5);
  const [countdown, setCountdown] = useState(0);

  const timeframes = ['1min', '5min', '15min', '30min', '1hr', '4hr'];

  const handleRefresh = () => {
    if (apiCallsLeft > 0) {
      setRefreshing(true);
      setApiCallsLeft(prev => prev - 1);
      
      setTimeout(() => {
        setData(generateMockData(timeframe));
        setRefreshing(false);
      }, 800);
      
      if (apiCallsLeft - 1 === 0) {
        setCountdown(60);
        const timer = setInterval(() => {
          setCountdown(prev => {
            if (prev <= 1) {
              clearInterval(timer);
              setApiCallsLeft(5);
              return 0;
            }
            return prev - 1;
          });
        }, 1000);
      }
    }
  };

  const handleTimeframeChange = (tf) => {
    setTimeframe(tf);
    setData(generateMockData(tf));
  };

  const currentPrice = data[data.length - 1]?.close || 0;
  const priceChange = data.length > 1 ? data[data.length - 1].close - data[0].close : 0;
  const priceChangePercent = data.length > 1 ? ((priceChange / data[0].close) * 100).toFixed(2) : 0;

  return (
    <div 
      className="rounded-xl p-3 backdrop-blur-xl border flex flex-col" 
      style={{ 
        background: 'rgba(0, 0, 0, 0.4)',
        borderColor: 'rgba(var(--theme-primary), 0.2)',
        height: '700px'
      }}
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <h2 className="text-sm font-bold text-white">XAU/USD</h2>
            <span 
              className="px-1.5 py-0.5 border rounded-full text-[9px] font-medium"
              style={{ 
                background: 'rgba(var(--theme-primary), 0.2)',
                borderColor: 'rgba(var(--theme-primary), 0.3)',
                color: 'rgb(var(--theme-primary))'
              }}
            >
              SPOT
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold text-white">${currentPrice.toFixed(2)}</span>
            <span className={`text-xs font-medium ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChange >= 0 ? '+' : ''}{priceChangePercent}%)
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Chart Type Toggle */}
          <div className="flex gap-1 p-0.5 bg-white/5 rounded-xl border border-white/10">
            <button
              onClick={() => setChartType('line')}
              className={`p-1.5 rounded-lg transition-all duration-300`}
              style={{
                background: chartType === 'line' ? 'rgba(var(--theme-primary), 0.2)' : 'transparent',
                border: chartType === 'line' ? '1px solid rgba(var(--theme-primary), 0.3)' : '1px solid transparent'
              }}
            >
              <Activity 
                className="w-3.5 h-3.5" 
                style={{ color: chartType === 'line' ? 'rgb(var(--theme-primary))' : 'rgba(255, 255, 255, 0.6)' }}
              />
            </button>
            <button
              onClick={() => setChartType('candlestick')}
              className={`p-1.5 rounded-lg transition-all duration-300`}
              style={{
                background: chartType === 'candlestick' ? 'rgba(var(--theme-primary), 0.2)' : 'transparent',
                border: chartType === 'candlestick' ? '1px solid rgba(var(--theme-primary), 0.3)' : '1px solid transparent'
              }}
            >
              <BarChart3 
                className="w-3.5 h-3.5" 
                style={{ color: chartType === 'candlestick' ? 'rgb(var(--theme-primary))' : 'rgba(255, 255, 255, 0.6)' }}
              />
            </button>
          </div>

          {/* Refresh Button */}
          <button
            onClick={handleRefresh}
            disabled={apiCallsLeft === 0 || refreshing}
            className={`p-2 rounded-xl transition-all duration-300 ${
              apiCallsLeft === 0 
                ? 'bg-red-500/20 border border-red-500/30 cursor-not-allowed' 
                : 'bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20'
            }`}
          >
            <RefreshCw 
              className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`}
              style={{ 
                color: refreshing 
                  ? 'rgb(var(--theme-primary))' 
                  : apiCallsLeft === 0 
                    ? '#f87171' 
                    : 'rgba(255, 255, 255, 0.6)' 
              }}
            />
          </button>
        </div>
      </div>

      {/* API Limit Warning */}
      {apiCallsLeft <= 2 && (
        <div 
          className={`mb-2 p-2 rounded-xl border ${
            apiCallsLeft === 0 
              ? 'bg-red-500/10 border-red-500/30' 
              : ''
          }`}
          style={{
            ...(apiCallsLeft > 0 && {
              background: 'rgba(var(--theme-primary), 0.1)',
              borderColor: 'rgba(var(--theme-primary), 0.3)'
            })
          }}
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-white/80">
              {apiCallsLeft === 0 
                ? `API limit reached. Refreshing in ${countdown}s...` 
                : `${apiCallsLeft} API calls remaining`}
            </span>
            {countdown > 0 && (
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full transition-all duration-1000"
                  style={{ 
                    width: `${((60 - countdown) / 60) * 100}%`,
                    background: 'rgb(var(--theme-primary))'
                  }}
                ></div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Timeframe Selector */}
      <div className="flex gap-1.5 mb-3 overflow-x-auto pb-1">
        {timeframes.map(tf => (
          <button
            key={tf}
            onClick={() => handleTimeframeChange(tf)}
            className="px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all duration-300 whitespace-nowrap border"
            style={{
              background: timeframe === tf 
                ? `linear-gradient(to right, rgba(var(--theme-primary), 0.2), rgba(var(--theme-secondary), 0.2))` 
                : 'rgba(255, 255, 255, 0.05)',
              borderColor: timeframe === tf 
                ? 'rgba(var(--theme-primary), 0.3)' 
                : 'rgba(255, 255, 255, 0.1)',
              color: timeframe === tf ? 'white' : 'rgba(255, 255, 255, 0.6)'
            }}
          >
            {tf}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="rounded-xl p-2 bg-black/40 border border-white/5 flex-1 min-h-0">
        {chartType === 'line' ? (
          <CustomAreaChart data={data} />
        ) : (
          <CustomCandlestickChart data={data} />
        )}
      </div>

      {/* Auto-refresh indicator */}
      <div className="mt-2 flex items-center justify-between text-[9px] text-white/50">
        <span>Auto-refresh: Every 30 seconds</span>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
          <span>Live</span>
        </div>
      </div>
    </div>
  );
}
