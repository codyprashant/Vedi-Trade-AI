
import React from "react";
import { TrendingUp, Activity, BarChart3, LineChart } from "lucide-react";

const generateMockRSI = () => ({
  rsi9: (Math.random() * 40 + 30).toFixed(2),
  rsi14: (Math.random() * 40 + 30).toFixed(2),
});

const generateMockMACD = () => ({
  macd: (Math.random() * 4 - 2).toFixed(2),
  signal: (Math.random() * 4 - 2).toFixed(2),
  histogram: (Math.random() * 2 - 1).toFixed(2),
});

const generateMockSMA = () => ({
  sma20: (2045 + Math.random() * 10).toFixed(2),
  sma50: (2040 + Math.random() * 10).toFixed(2),
  sma200: (2030 + Math.random() * 10).toFixed(2),
});

const generateMockEMA = () => ({
  ema9: (2050 + Math.random() * 5).toFixed(2),
  ema21: (2048 + Math.random() * 5).toFixed(2),
  ema55: (2045 + Math.random() * 5).toFixed(2),
});

export default function TechnicalIndicators() {
  const rsi = generateMockRSI();
  const macd = generateMockMACD();
  const sma = generateMockSMA();
  const ema = generateMockEMA();

  const indicators = [
    {
      title: "RSI",
      icon: Activity,
      color: "from-purple-500 to-pink-600",
      values: [
        { label: "RSI(9)", value: rsi.rsi9, status: rsi.rsi9 > 70 ? "Overbought" : rsi.rsi9 < 30 ? "Oversold" : "Neutral" },
        { label: "RSI(14)", value: rsi.rsi14, status: rsi.rsi14 > 70 ? "Overbought" : rsi.rsi14 < 30 ? "Oversold" : "Neutral" },
      ],
      settings: "OB: 70-75 | OS: 25-30"
    },
    {
      title: "MACD",
      icon: TrendingUp,
      color: "from-blue-500 to-cyan-600",
      values: [
        { label: "MACD", value: macd.macd, status: parseFloat(macd.macd) > 0 ? "Bullish" : "Bearish" },
        { label: "Signal", value: macd.signal },
        { label: "Histogram", value: macd.histogram },
      ],
      settings: "12, 26, 9"
    },
    {
      title: "SMA",
      icon: BarChart3,
      color: "from-green-500 to-emerald-600",
      values: [
        { label: "SMA(20)", value: sma.sma20 },
        { label: "SMA(50)", value: sma.sma50 },
        { label: "SMA(200)", value: sma.sma200 },
      ],
      settings: "20, 50, 200"
    },
    {
      title: "EMA",
      icon: LineChart,
      color: "from-amber-500 to-orange-600",
      values: [
        { label: "EMA(9)", value: ema.ema9 },
        { label: "EMA(21)", value: ema.ema21 },
        { label: "EMA(55)", value: ema.ema55 },
      ],
      settings: "9, 21, 55"
    },
  ];

  return (
    <div className="space-y-2">
      {indicators.map((indicator, index) => (
        <div 
          key={indicator.title}
          className="rounded-xl p-2 backdrop-blur-xl"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
            animation: `fadeInUp 0.5s ease-out ${index * 0.1}s backwards`
          }}
        >
          <div className="flex items-center gap-1.5 mb-1.5">
            <div className={`w-7 h-7 bg-gradient-to-br ${indicator.color} rounded-lg flex items-center justify-center shadow-lg`}>
              <indicator.icon className="w-3.5 h-3.5 text-white" />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-white">{indicator.title}</h3>
              <p className="text-[9px] text-white/50">{indicator.settings}</p>
            </div>
          </div>

          <div className="space-y-1">
            {indicator.values.map((item, idx) => (
              <div 
                key={idx} 
                className="flex items-center justify-between p-1.5 rounded-lg backdrop-blur-sm"
                style={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <span className="text-[10px] font-medium text-white/70">{item.label}</span>
                <div className="text-right">
                  <span className="text-[11px] font-bold text-white">{item.value}</span>
                  {item.status && (
                    <span 
                      className={`ml-1 text-[9px] px-1 py-0.5 rounded-full ${
                        item.status === "Overbought" || item.status === "Bearish" 
                          ? "text-red-400" 
                          : item.status === "Oversold" || item.status === "Bullish"
                          ? "text-green-400"
                          : "text-blue-400"
                      }`}
                      style={{
                        background: item.status === "Overbought" || item.status === "Bearish" 
                          ? 'rgba(239, 68, 68, 0.2)' 
                          : item.status === "Oversold" || item.status === "Bullish"
                          ? 'rgba(34, 197, 94, 0.2)'
                          : 'rgba(59, 130, 246, 0.2)'
                      }}
                    >
                      {item.status}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
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
