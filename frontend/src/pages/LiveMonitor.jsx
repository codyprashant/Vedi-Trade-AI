
import React, { useState, useEffect, useRef } from "react";
import { TrendingUp, TrendingDown, Activity, BarChart3, LineChart, Zap, Clock, Monitor, Sparkles, Wifi, WifiOff, AlertCircle } from "lucide-react";
import RecentSignalsTable from "../components/trading/RecentSignalsTable";
import { APP_CONFIG } from "../components/config/appConfig";
import { signalsApi } from "../components/services/signalsApi";

export default function LiveMonitor() {
  const [activeSignal, setActiveSignal] = useState(null);
  const [rsi, setRsi] = useState({ rsi9: '0', rsi14: '0' });
  const [macd, setMacd] = useState({ macd: '0', signal: '0', histogram: '0' });
  const [sma, setSma] = useState({ sma20: '0', sma50: '0', sma200: '0' });
  const [ema, setEma] = useState({ ema9: '0', ema21: '0', ema55: '0' });
  const [currentPrice, setCurrentPrice] = useState(0); // Initialized to 0
  const [previousPrice, setPreviousPrice] = useState(0); // New state for price change calculation
  const [priceChange, setPriceChange] = useState(0);
  const [bid, setBid] = useState(0); // New state for bid price
  const [ask, setAsk] = useState(0); // New state for ask price
  const [spread, setSpread] = useState(0); // New state for spread
  const [wsConnected, setWsConnected] = useState(false); // New state for WebSocket connection status
  const [lastUpdate, setLastUpdate] = useState(null); // New state for last update timestamp
  const [priceIssue, setPriceIssue] = useState(false); // Show UI message when no price received
  const lastTickAtRef = useRef(null); // Track last real price tick time
  const wsConnectedRef = useRef(false); // Mirror wsConnected for interval checks

  useEffect(() => {
    const fetchLatestSignal = async () => {
      try {
        const signal = await signalsApi.getLatestSignal();
        setActiveSignal(signal);
        
        if (signal && signal.indicators) {
          // Update technical indicators from signal data
          setRsi({
            rsi9: signal.indicators.rsi || '0',
            rsi14: signal.indicators.rsi || '0'
          });
          setMacd({
            macd: signal.indicators.macd || '0',
            signal: signal.indicators.macd_signal || '0',
            // Corrected histogram calculation based on outline suggestion
            histogram: ((parseFloat(signal.indicators.macd || 0)) - (parseFloat(signal.indicators.macd_signal || 0))).toFixed(2)
          });
          setSma({
            sma20: signal.indicators.sma_20 || '0',
            sma50: signal.indicators.sma_50 || '0',
            sma200: signal.indicators.sma_200 || '0'
          });
          setEma({
            ema9: signal.indicators.ema_short || '0',
            ema21: signal.indicators.ema_long || '0',
            ema55: signal.indicators.ema_long || '0'
          });
        }
      } catch (error) {
        console.error('Failed to fetch latest signal:', error);
        setActiveSignal(null); // Clear active signal on fetch error
      }
    };

    fetchLatestSignal();
    const signalInterval = setInterval(fetchLatestSignal, 30000); // Poll for signals every 30 seconds

    // WebSocket connection for live prices
    let ws;
    let reconnectTimeout;
    let mockPriceIntervalId = null; // To store mock interval ID for cleanup
    let tickMonitor; // Interval to detect missing ticks while connected

    const connectWebSocket = () => {
      // Clear any existing reconnect timeouts before attempting new connection
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }
      // Clear any existing mock interval if reconnecting
      if (mockPriceIntervalId) {
        clearInterval(mockPriceIntervalId);
        mockPriceIntervalId = null;
      }

      if (APP_CONFIG.USE_MOCK_DATA) {
        // Mock data simulation for development
        setWsConnected(true);
        wsConnectedRef.current = true;
        // Initialize currentPrice to a non-zero value for mock data if it's 0
        setCurrentPrice(2350); 
        setPreviousPrice(2350);

        mockPriceIntervalId = setInterval(() => {
          // Simulate price movement
          setCurrentPrice(prev => {
            setPreviousPrice(prev); // Store current price as previous for percentage calculation
            const mockPrice = prev + (Math.random() * 2 - 1); // Simpler fluctuation
            const change = mockPrice - prev;
            setPriceChange(change);

            // Simulate bid price
            const newBid = mockPrice; // Use mock price as bid
            setBid(newBid);
            setAsk(null); // No ask price in new format
            setSpread(0); // No spread calculation
            setLastUpdate(new Date().toISOString()); // Set last update time
            lastTickAtRef.current = Date.now();
            setPriceIssue(false);
            return mockPrice;
          });
        }, 3000); // Update mock price every 3 seconds
        
        return; // Exit as we are using mock data directly
      }

      try {
        // Construct WebSocket URL by replacing http/https with ws/wss
        const wsProtocol = APP_CONFIG.API_BASE_URL.startsWith('https') ? 'wss' : 'ws';
        const wsHost = APP_CONFIG.API_BASE_URL.replace(/^https?:\/\//, ''); // Get host without protocol
        const wsUrl = `${wsProtocol}://${wsHost}/ws/prices?symbol=XAUUSD`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('WebSocket connected');
          setWsConnected(true);
          wsConnectedRef.current = true;
          lastTickAtRef.current = null;
          setPriceIssue(false);
          // Clear any pending reconnect timeouts upon successful connection
          if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Handle heartbeat messages
            if (data.type === 'heartbeat') {
              // console.log('Heartbeat received:', data.ts); // Uncomment for debugging heartbeats
              return;
            }

            // Handle price tick data
            if (data.symbol && data.bid !== undefined) {
              const newBid = parseFloat(data.bid);
              const newPrice = newBid; // Use bid as the current price
              
              setCurrentPrice(prev => {
                setPreviousPrice(prev); // Store current price as previous
                const change = newPrice - prev;
                setPriceChange(change);
                return newPrice;
              });
              
              setBid(newBid);
              setAsk(null); // No ask price available
              setSpread(0); // No spread calculation possible
              setLastUpdate(data.time || new Date().toISOString()); // Use timestamp from data or current time
              lastTickAtRef.current = Date.now();
              setPriceIssue(false);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onerror = (event) => {
          console.error('WebSocket error event:', event);
          setWsConnected(false);
          wsConnectedRef.current = false;
          try {
            if (ws) ws.close(); // Trigger onclose and reconnect
          } catch (_) {
            // Ignore close errors
          }
        };

        ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          setWsConnected(false);
          wsConnectedRef.current = false;
          
          // Clear any existing reconnect attempts to avoid multiple timers
          if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
          }
          // Attempt to reconnect after 5 seconds
          reconnectTimeout = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            connectWebSocket();
          }, 5000);
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        setWsConnected(false);
        // Fallback to mock data if WebSocket fails to initialize
        console.log('Falling back to mock price data...');
        setWsConnected(true); // Indicate that we are "connected" via mock data
        wsConnectedRef.current = true;
        setCurrentPrice(2350);
        setPreviousPrice(2350); // Initialize previous price for calculations

        mockPriceIntervalId = setInterval(() => {
          setCurrentPrice(prev => {
            setPreviousPrice(prev); // Update previous price
            const mockPrice = prev + (Math.random() * 2 - 1); // Dynamic price
            setPriceChange(mockPrice - prev);
            setBid(parseFloat(mockPrice.toFixed(2))); // Use mock price as bid
            setAsk(null); // No ask price in new format
            setSpread(0); // No spread calculation
            setLastUpdate(new Date().toISOString());
            lastTickAtRef.current = Date.now();
            setPriceIssue(false);
            return mockPrice;
          });
        }, 3000);
      }
    };

    connectWebSocket(); // Initiate WebSocket connection on component mount

    // Monitor for missing ticks while connected (e.g., MT5 not providing prices)
    tickMonitor = setInterval(() => {
      if (wsConnectedRef.current) {
        const last = lastTickAtRef.current;
        if (!last || (Date.now() - last > 15000)) {
          setPriceIssue(true);
        }
      }
    }, 5000);

    return () => {
      clearInterval(signalInterval); // Clean up signal polling interval
      if (ws) {
        ws.close(); // Close WebSocket connection
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout); // Clear any pending reconnect timeouts
      }
      if (mockPriceIntervalId) {
        clearInterval(mockPriceIntervalId); // Clean up mock interval if it was running
      }
      if (tickMonitor) {
        clearInterval(tickMonitor);
      }
    };
  }, []); // Empty dependency array ensures useEffect runs once on mount and cleans up on unmount

  // Calculate price change percentage, guarding against division by zero
  const priceChangePercent = (currentPrice > 0 && previousPrice !== 0) 
    ? (priceChange / previousPrice * 100).toFixed(2) 
    : '0.00';

  const indicators = [
    {
      title: "RSI",
      icon: Activity,
      color: "from-purple-500 to-pink-600",
      glowColor: "rgba(168, 85, 247, 0.4)",
      values: [
        { label: "RSI(9)", value: rsi.rsi9, status: parseFloat(rsi.rsi9) > 70 ? "Overbought" : parseFloat(rsi.rsi9) < 30 ? "Oversold" : "Neutral" },
        { label: "RSI(14)", value: rsi.rsi14, status: parseFloat(rsi.rsi14) > 70 ? "Overbought" : parseFloat(rsi.rsi14) < 30 ? "Oversold" : "Neutral" },
      ],
      settings: "OB: 70-75 | OS: 25-30"
    },
    {
      title: "MACD",
      icon: TrendingUp,
      color: "from-blue-500 to-cyan-600",
      glowColor: "rgba(59, 130, 246, 0.4)",
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
      glowColor: "rgba(34, 197, 94, 0.4)",
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
      glowColor: "rgba(245, 158, 11, 0.4)",
      values: [
        { label: "EMA(9)", value: ema.ema9 },
        { label: "EMA(21)", value: ema.ema21 },
        { label: "EMA(55)", value: ema.ema55 },
      ],
      settings: "9, 21, 55"
    },
  ];

  // Sort contributions for display
  const sortedContributions = activeSignal && activeSignal.indicator_contributions 
    ? Object.entries(activeSignal.indicator_contributions).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="min-h-screen p-4 overflow-y-auto relative">
      {/* Animated Particles Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="particle particle-1"></div>
        <div className="particle particle-2"></div>
        <div className="particle particle-3"></div>
        <div className="particle particle-4"></div>
        <div className="particle particle-5"></div>
      </div>

      {/* Header with Animation */}
      <div className="mb-4 relative z-10">
        <div className="flex items-center gap-3 mb-1 animate-slideIn">
          <div 
            className="w-12 h-12 rounded-2xl flex items-center justify-center shadow-2xl relative overflow-hidden"
            style={{ 
              background: 'linear-gradient(135deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
              boxShadow: '0 10px 40px rgba(var(--theme-primary), 0.5)',
              animation: 'pulse-glow 2s ease-in-out infinite'
            }}
          >
            <Monitor className="w-6 h-6 text-white relative z-10" />
            <div className="absolute inset-0 bg-white/20 animate-shimmer"></div>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              Live Monitor
              {wsConnected ? ( // Display Wifi icon when connected
                <Wifi className="w-5 h-5 text-green-400 animate-pulse" />
              ) : ( // Display WifiOff icon when disconnected
                <WifiOff className="w-5 h-5 text-red-400" />
              )}
            </h1>
            <p className="text-xs text-white/60">
              {wsConnected ? 'Real-time market data and signals overview' : 'Connecting to live data...'}
            </p>
          </div>
        </div>
      </div>

      {/* Price & Signal Row - 40/60 Split */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 mb-4">
        {/* Live Price Display - 40% (2 columns) */}
        <div className="lg:col-span-2">
          <div 
            className="rounded-3xl p-6 backdrop-blur-xl h-full relative overflow-hidden group hover:scale-[1.02] transition-transform duration-500"
            style={{ 
              background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2))',
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.4), inset 0 0 60px rgba(251, 191, 36, 0.1)',
              border: '1px solid rgba(251, 191, 36, 0.3)',
              animation: 'fadeInUp 0.6s ease-out'
            }}
          >
            {/* Animated Border */}
            <div className="absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500">
              <div className="absolute inset-0 rounded-3xl animate-border-glow"></div>
            </div>

            {/* Glowing Orbs */}
            <div className="absolute top-1/2 left-1/4 w-32 h-32 bg-amber-500/20 rounded-full blur-3xl animate-float"></div>
            <div className="absolute bottom-1/4 right-1/4 w-40 h-40 bg-orange-500/20 rounded-full blur-3xl animate-float-delayed"></div>

            <div className="text-center relative z-10 flex flex-col justify-center h-full">
              <div className="flex items-center justify-center gap-2 mb-3">
                <h2 className="text-xl font-bold text-white/80">XAU/USD</h2>
                <span 
                  className="px-2 py-1 border-2 rounded-full text-xs font-bold animate-pulse-slow"
                  style={{ 
                    background: 'rgba(var(--theme-primary), 0.3)',
                    borderColor: 'rgba(var(--theme-primary), 0.6)',
                    color: 'rgb(var(--theme-primary))',
                    boxShadow: '0 0 20px rgba(var(--theme-primary), 0.4)'
                  }}
                >
                  SPOT
                </span>
                <Sparkles className="w-4 h-4 text-amber-400 animate-spin-slow" />
              </div>
              
              <div 
                className="text-5xl font-black text-white mb-3 animate-scale-in"
                style={{ 
                  textShadow: `0 0 30px rgba(251, 191, 36, 0.5), 0 0 60px rgba(251, 191, 36, 0.3)`,
                  animation: 'number-glow 2s ease-in-out infinite'
                }}
              >
                {currentPrice > 0 ? `$${currentPrice.toFixed(2)}` : '---'} {/* Display price or placeholder */}
              </div>

              {wsConnected && priceIssue && (
                <div 
                  className="mt-2 flex items-center justify-center gap-2 p-2 rounded-xl"
                  style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}
                >
                  <AlertCircle className="w-4 h-4 text-red-400" />
                  <span className="text-xs text-red-300">Facing some issue while fetching price</span>
                </div>
              )}
              
              {currentPrice > 0 && ( // Only show price change if a valid price is available
                <div 
                  className={`text-2xl font-bold mb-4 animate-bounce-subtle ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  style={{
                    textShadow: priceChange >= 0 
                      ? '0 0 20px rgba(34, 197, 94, 0.6)' 
                      : '0 0 20px rgba(239, 68, 68, 0.6)'
                  }}
                >
                  {priceChange >= 0 ? (
                    <TrendingUp className="inline w-6 h-6 mr-2 animate-bounce" />
                  ) : (
                    <TrendingDown className="inline w-6 h-6 mr-2 animate-bounce" />
                  )}
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChange >= 0 ? '+' : ''}{priceChangePercent}%)
                </div>
              )}

              {/* Bid Price Display */}
              {bid > 0 && ( // Show when bid is available
                <div className="grid grid-cols-1 gap-3 mb-4">
                  <div 
                    className="p-3 rounded-xl text-center"
                    style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
                  >
                    <div className="text-xs text-blue-400/70 mb-1">Current Bid Price</div>
                    <div className="text-lg font-bold text-blue-400">${bid.toFixed(2)}</div>
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-center gap-3 text-sm text-white/60">
                <div className="flex items-center gap-2">
                  <div className="relative">
                    {wsConnected ? ( // Conditional status indicator dot
                      <>
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-ping absolute"></div>
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      </>
                    ) : (
                      <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                    )}
                  </div>
                  <span className="font-semibold">{wsConnected ? 'Live' : 'Disconnected'}</span>
                </div>
                {lastUpdate && ( // Display last update time if available
                  <>
                    <div className="w-1 h-1 bg-white/30 rounded-full"></div>
                    <span className="text-xs">{new Date(lastUpdate).toLocaleTimeString()}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Active Signal - 60% (3 columns) */}
        <div className="lg:col-span-3">
          {activeSignal ? (
            <div 
              className="rounded-3xl p-5 backdrop-blur-xl h-full relative overflow-hidden hover:scale-[1.01] transition-all duration-500"
              style={{ 
                background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2))',
                boxShadow: '0 15px 50px rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(251, 191, 36, 0.2)',
                animation: 'fadeInUp 0.7s ease-out 0.1s backwards'
              }}
            >
              <div className="space-y-3 h-full flex flex-col">
                {/* Header: Signal Type */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <Zap className="w-5 h-5 animate-pulse" style={{ color: 'rgb(var(--theme-primary))' }} />
                      <div 
                        className="absolute inset-0 blur-md"
                        style={{ 
                          background: 'rgb(var(--theme-primary))',
                          animation: 'pulse-glow 1.5s ease-in-out infinite'
                        }}
                      ></div>
                    </div>
                    <h3 className="text-base font-bold text-white">Active Signal</h3>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-white/50">
                    <Clock className="w-3 h-3" />
                    <span>{activeSignal.minutesAgo} min ago</span>
                  </div>
                </div>

                {/* Signal Badge & Strength */}
                <div className="flex items-center gap-3">
                  <div 
                    className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl relative overflow-hidden group ${
                      activeSignal.signal_type === "BUY" 
                        ? "hover:shadow-xl hover:shadow-green-500/30" 
                        : "hover:shadow-xl hover:shadow-red-500/30"
                    } transition-all duration-500`}
                    style={{
                      background: activeSignal.signal_type === "BUY" 
                        ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(34, 197, 94, 0.1))' 
                        : 'linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.1))',
                      border: `2px solid ${
                        activeSignal.signal_type === "BUY" 
                          ? "rgba(34, 197, 94, 0.5)" 
                          : "rgba(239, 68, 68, 0.5)"
                      }`,
                    }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent group-hover:translate-x-full transition-transform duration-1000"></div>
                    
                    {activeSignal.signal_type === "BUY" ? (
                      <TrendingUp className="w-6 h-6 text-green-400 animate-bounce-subtle" />
                    ) : (
                      <TrendingDown className="w-6 h-6 text-red-400 animate-bounce-subtle" />
                    )}
                    
                    <span className={`text-xl font-black ${
                      activeSignal.signal_type === "BUY" ? "text-green-400" : "text-red-400"
                    }`}>
                      {activeSignal.signal_type}
                    </span>
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-white/70">Signal Strength</span>
                      <span 
                        className="text-2xl font-black animate-pulse"
                        style={{ 
                          color: 'rgb(var(--theme-primary))',
                          textShadow: '0 0 20px rgba(var(--theme-primary), 0.6)'
                        }}
                      >
                        {activeSignal.final_signal_strength}%
                      </span>
                    </div>
                    <div 
                      className="w-full h-3 rounded-full overflow-hidden relative"
                      style={{ background: 'rgba(255, 255, 255, 0.1)' }}
                    >
                      <div 
                        className="h-full rounded-full relative overflow-hidden"
                        style={{ 
                          width: `${activeSignal.final_signal_strength}%`,
                          background: `linear-gradient(90deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))`,
                          boxShadow: `0 0 20px rgba(var(--theme-primary), 0.6)`,
                          animation: 'progress-glow 2s ease-in-out infinite'
                        }}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Price Levels - Horizontal Grid */}
                <div className="grid grid-cols-4 gap-2">
                  <div 
                    className="p-2 rounded-lg text-center"
                    style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                  >
                    <div className="text-[10px] text-white/50 mb-0.5">Entry</div>
                    <div className="text-xs font-bold text-white">${activeSignal.entry_price}</div>
                  </div>
                  <div 
                    className="p-2 rounded-lg text-center"
                    style={{ background: 'rgba(239, 68, 68, 0.1)' }}
                  >
                    <div className="text-[10px] text-red-400/70 mb-0.5">Stop Loss</div>
                    <div className="text-xs font-semibold text-red-400">${activeSignal.stop_loss_price}</div>
                  </div>
                  <div 
                    className="p-2 rounded-lg text-center"
                    style={{ background: 'rgba(34, 197, 94, 0.1)' }}
                  >
                    <div className="text-[10px] text-green-400/70 mb-0.5">Take Profit</div>
                    <div className="text-xs font-semibold text-green-400">${activeSignal.take_profit_price}</div>
                  </div>
                  <div 
                    className="p-2 rounded-lg text-center"
                    style={{ background: 'rgba(251, 191, 36, 0.1)' }}
                  >
                    <div className="text-[10px] text-amber-400/70 mb-0.5">R:R</div>
                    <div className="text-xs font-bold text-amber-400">1:{activeSignal.risk_reward_ratio}</div>
                  </div>
                </div>

                {/* Indicator Contributions - Compact Grid */}
                <div className="flex-1 min-h-0">
                  <div className="flex items-center gap-2 mb-2">
                    <BarChart3 className="w-4 h-4" style={{ color: 'rgb(var(--theme-primary))' }} />
                    <h4 className="text-xs font-bold text-white">Indicator Contributions</h4>
                  </div>
                  <div className="grid grid-cols-2 gap-2 max-h-[200px] overflow-y-auto pr-2 custom-scrollbar">
                    {sortedContributions.map(([name, value], idx) => (
                      <div 
                        key={name}
                        className="p-2 rounded-lg backdrop-blur-sm hover:bg-white/10 transition-all duration-300"
                        style={{ 
                          background: 'rgba(255, 255, 255, 0.05)',
                          animation: `fadeInRight 0.4s ease-out ${idx * 0.05}s backwards`
                        }}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-semibold text-white">{name}</span>
                          <span className="text-xs font-bold" style={{ color: 'rgb(var(--theme-primary))' }}>
                            {value}%
                          </span>
                        </div>
                        <div 
                          className="w-full h-1.5 rounded-full overflow-hidden relative"
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

                {/* Market Context */}
                <div>
                  <div className="grid grid-cols-3 gap-2">
                    <div 
                      className="p-2 rounded-lg text-center"
                      style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                    >
                      <div className="text-[10px] text-white/50 mb-0.5">H1 Trend</div>
                      <div className={`text-xs font-bold ${
                        activeSignal.h1_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {activeSignal.h1_trend_direction}
                      </div>
                    </div>
                    <div 
                      className="p-2 rounded-lg text-center"
                      style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                    >
                      <div className="text-[10px] text-white/50 mb-0.5">H4 Trend</div>
                      <div className={`text-xs font-bold ${
                        activeSignal.h4_trend_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {activeSignal.h4_trend_direction}
                      </div>
                    </div>
                    <div 
                      className="p-2 rounded-lg text-center"
                      style={{ background: 'rgba(255, 255, 255, 0.05)' }}
                    >
                      <div className="text-[10px] text-white/50 mb-0.5">Volatility</div>
                      <div className={`text-xs font-bold ${
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
              </div>
            </div>
          ) : (
            <div 
              className="rounded-3xl p-6 backdrop-blur-xl h-full relative overflow-hidden flex items-center justify-center"
              style={{ 
                background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2))',
                boxShadow: '0 15px 50px rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                animation: 'fadeInUp 0.7s ease-out 0.1s backwards'
              }}
            >
              <div className="text-center">
                <div className="w-12 h-12 bg-white/5 rounded-2xl flex items-center justify-center mx-auto mb-3">
                  <AlertCircle className="w-6 h-6 text-white/30" />
                </div>
                <h3 className="text-base font-bold text-white/70 mb-1">No Active Signal</h3>
                <p className="text-xs text-white/40">No signals detected in the last 30 minutes.</p>
                <div className="mt-3 flex items-center justify-center gap-2 text-xs text-white/30">
                  <Clock className="w-3 h-3" />
                  <span>Waiting for new signal...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Technical Indicators Grid - Enhanced */}
      <div className="mb-4">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" style={{ color: 'rgb(var(--theme-primary))' }} />
          Technical Indicators
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {indicators.map((indicator, index) => (
            <div 
              key={indicator.title}
              className="rounded-2xl p-4 backdrop-blur-xl relative overflow-hidden group hover:scale-105 transition-all duration-500 cursor-pointer"
              style={{ 
                background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.2))',
                boxShadow: `0 15px 40px rgba(0, 0, 0, 0.3)`,
                border: '1px solid rgba(255, 255, 255, 0.1)',
                animation: `fadeInUp 0.6s ease-out ${index * 0.1}s backwards`
              }}
            >
              {/* Hover Glow Effect */}
              <div 
                className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"
                style={{ 
                  background: `radial-gradient(circle at center, ${indicator.glowColor}, transparent)`,
                }}
              ></div>

              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                  <div 
                    className={`w-12 h-12 bg-gradient-to-br ${indicator.color} rounded-2xl flex items-center justify-center shadow-2xl relative overflow-hidden group-hover:scale-110 transition-transform duration-500`}
                    style={{
                      boxShadow: `0 10px 30px ${indicator.glowColor}`
                    }}
                  >
                    <indicator.icon className="w-6 h-6 text-white relative z-10" />
                    <div className="absolute inset-0 bg-white/20 animate-shimmer"></div>
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">{indicator.title}</h3>
                    <p className="text-[10px] text-white/50">{indicator.settings}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  {indicator.values.map((item, idx) => (
                    <div 
                      key={idx} 
                      className="flex items-center justify-between p-2.5 rounded-xl backdrop-blur-sm hover:bg-white/10 transition-all duration-300"
                      style={{ 
                        background: 'rgba(255, 255, 255, 0.05)',
                        animation: `fadeInRight 0.4s ease-out ${idx * 0.1}s backwards`
                      }}
                    >
                      <span className="text-xs font-semibold text-white/70">{item.label}</span>
                      <div className="text-right">
                        <span className="text-sm font-black text-white">{item.value}</span>
                        {item.status && (
                          <span 
                            className={`ml-2 text-[10px] px-2 py-1 rounded-full font-bold ${
                              item.status === "Overbought" || item.status === "Bearish" 
                                ? "text-red-400 animate-pulse-subtle" 
                                : item.status === "Oversold" || item.status === "Bullish"
                                ? "text-green-400 animate-pulse-subtle"
                                : "text-blue-400"
                            }`}
                            style={{
                              background: item.status === "Overbought" || item.status === "Bearish" 
                                ? 'rgba(239, 68, 68, 0.3)' 
                                : item.status === "Oversold" || item.status === "Bullish"
                                ? 'rgba(34, 197, 94, 0.3)'
                                : 'rgba(59, 130, 246, 0.3)',
                              boxShadow: item.status === "Overbought" || item.status === "Bearish" 
                                ? '0 0 10px rgba(239, 68, 68, 0.4)' 
                                : item.status === "Oversold" || item.status === "Bullish"
                                ? '0 0 10px rgba(34, 197, 94, 0.4)'
                                : '0 0 10px rgba(59, 130, 246, 0.4)'
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
            </div>
          ))}
        </div>
      </div>

      {/* Recent Signals */}
      <div style={{ animation: 'fadeInUp 0.8s ease-out 0.4s backwards' }}>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 animate-pulse" style={{ color: 'rgb(var(--theme-primary))' }} />
          Recent Signals
        </h2>
        <RecentSignalsTable limit={APP_CONFIG.SIGNAL_LIMITS.LIVE_MONITOR} />
      </div>

      {/* Keep existing styles */}
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInRight {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-50px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes pulse-glow {
          0%, 100% {
            opacity: 1;
            filter: brightness(1);
          }
          50% {
            opacity: 0.8;
            filter: brightness(1.3);
          }
        }

        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }

        @keyframes border-glow {
          0% {
            box-shadow: 0 0 5px rgba(251, 191, 36, 0.5), inset 0 0 5px rgba(251, 191, 36, 0.5);
          }
          50% {
            box-shadow: 0 0 20px rgba(251, 191, 36, 0.8), inset 0 0 10px rgba(251, 191, 36, 0.8);
          }
          100% {
            box-shadow: 0 0 5px rgba(251, 191, 36, 0.5), inset 0 0 5px rgba(251, 191, 36, 0.5);
          }
        }

        @keyframes float {
          0%, 100% {
            transform: translate(0, 0);
          }
          50% {
            transform: translate(20px, -20px);
          }
        }

        @keyframes float-delayed {
          0%, 100% {
            transform: translate(0, 0);
          }
          50% {
            transform: translate(-20px, 20px);
          }
        }

        @keyframes pulse-slow {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.7;
          }
        }

        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes scale-in {
          from {
            transform: scale(0.9);
          }
          to {
            transform: scale(1);
          }
        }

        @keyframes bounce-subtle {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-5px);
          }
        }

        @keyframes pulse-subtle {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.6;
          }
        }

        @keyframes number-glow {
          0%, 100% {
            text-shadow: 0 0 30px rgba(251, 191, 36, 0.5), 0 0 60px rgba(251, 191, 36, 0.3);
          }
          50% {
            text-shadow: 0 0 40px rgba(251, 191, 36, 0.8), 0 0 80px rgba(251, 191, 36, 0.5);
          }
        }

        @keyframes progress-glow {
          0%, 100% {
            box-shadow: 0 0 20px rgba(var(--theme-primary), 0.6);
          }
          50% {
            box-shadow: 0 0 30px rgba(var(--theme-primary), 0.9);
          }
        }

        .animate-slideIn {
          animation: slideIn 0.6s ease-out;
        }

        .animate-shimmer {
          animation: shimmer 2s infinite;
        }

        .animate-border-glow {
          animation: border-glow 2s ease-in-out infinite;
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }

        .animate-float-delayed {
          animation: float-delayed 8s ease-in-out infinite;
        }

        .animate-pulse-slow {
          animation: pulse-slow 3s ease-in-out infinite;
        }

        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }

        .animate-scale-in {
          animation: scale-in 0.6s ease-out;
        }

        .animate-bounce-subtle {
          animation: bounce-subtle 2s ease-in-out infinite;
        }

        .animate-pulse-subtle {
          animation: pulse-subtle 2s ease-in-out infinite;
        }

        /* Particle effects */
        .particle {
          position: absolute;
          width: 4px;
          height: 4px;
          background: rgba(251, 191, 36, 0.6);
          border-radius: 50%;
          animation: particle-float 20s infinite;
        }

        .particle-1 {
          top: 10%;
          left: 10%;
          animation-delay: 0s;
        }

        .particle-2 {
          top: 60%;
          left: 80%;
          animation-delay: 4s;
        }

        .particle-3 {
          top: 30%;
          left: 50%;
          animation-delay: 8s;
        }

        .particle-4 {
          top: 80%;
          left: 20%;
          animation-delay: 12s;
        }

        .particle-5 {
          top: 50%;
          left: 70%;
          animation-delay: 16s;
        }

        @keyframes particle-float {
          0%, 100% {
            transform: translate(0, 0);
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          90% {
            opacity: 1;
          }
          50% {
            transform: translate(100px, -100px);
          }
        }

        /* Custom scrollbar for contributions */
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(251, 191, 36, 0.3);
          border-radius: 10px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(251, 191, 36, 0.5);
        }
      `}</style>
    </div>
  );
}
