import { APP_CONFIG } from '../config/appConfig';

// Mock data generator for development
const generateMockSignal = (minutesAgo = 0, id = 1) => {
  const side = Math.random() > 0.5 ? 'buy' : 'sell';
  const signal_type = side.toUpperCase();
  const strength = Math.floor(Math.random() * 20) + 60;
  const alignmentBoost = [10, 20, -10][Math.floor(Math.random() * 3)];
  const finalStrength = Math.min(100, strength + alignmentBoost);
  
  const timestamp = new Date(Date.now() - minutesAgo * 60 * 1000);
  
  const entryPrice = 2350 + Math.random() * 20 - 10;
  const stopLossDistance = Math.random() * 50 + 50;
  const takeProfitDistance = stopLossDistance * (1.5 + Math.random());
  
  return {
    id,
    timestamp: timestamp.toISOString(),
    symbol: 'XAUUSD',
    timeframe: '15m',
    side,
    strength,
    strategy: 'combined',
    indicators: {
      RSI: { direction: side },
      MACD: { direction: side },
      SMA: { direction: side },
      EMA: { direction: side },
      BBANDS: { direction: side },
      STOCH: { direction: side },
      // Additional raw values for display
      rsi: (Math.random() * 40 + 30).toFixed(2),
      macd: (Math.random() * 4 - 2).toFixed(2),
      macd_signal: (Math.random() * 4 - 2).toFixed(2),
      ema_short: (2050 + Math.random() * 10).toFixed(2),
      ema_long: (2045 + Math.random() * 10).toFixed(2),
      atr: (Math.random() * 10 + 10).toFixed(2),
      sma_20: (2045 + Math.random() * 10).toFixed(2),
      sma_50: (2040 + Math.random() * 10).toFixed(2),
      sma_200: (2030 + Math.random() * 10).toFixed(2),
    },
    contributions: {
      trend: Math.floor(Math.random() * 20) + 30,
      momentum: Math.floor(Math.random() * 20) + 25,
    },
    indicator_contributions: {
      RSI: Math.floor(Math.random() * 5) + 6,
      MACD: Math.floor(Math.random() * 5) + 10,
      STOCH: Math.floor(Math.random() * 5) + 4,
      BBANDS: Math.floor(Math.random() * 5) + 4,
      SMA_EMA: Math.floor(Math.random() * 5) + 8,
      MTF: Math.floor(Math.random() * 5) + 12,
      ATR_STABILITY: Math.floor(Math.random() * 5) + 6,
      PRICE_ACTION: Math.floor(Math.random() * 5) + 8,
    },
    signal_type,
    primary_timeframe: 'M15',
    confirmation_timeframe: 'H1',
    trend_timeframe: 'H4',
    h1_trend_direction: Math.random() > 0.5 ? 'Bullish' : 'Bearish',
    h4_trend_direction: Math.random() > 0.5 ? 'Bullish' : 'Bearish',
    alignment_boost: alignmentBoost,
    final_signal_strength: finalStrength,
    entry_price: entryPrice.toFixed(2),
    stop_loss_price: (entryPrice - (side === 'buy' ? stopLossDistance : -stopLossDistance) / 10).toFixed(2),
    take_profit_price: (entryPrice + (side === 'buy' ? takeProfitDistance : -takeProfitDistance) / 10).toFixed(2),
    stop_loss_distance_pips: stopLossDistance.toFixed(1),
    take_profit_distance_pips: takeProfitDistance.toFixed(1),
    risk_reward_ratio: (takeProfitDistance / stopLossDistance).toFixed(2),
    volatility_state: ['Low', 'Normal', 'High'][Math.floor(Math.random() * 3)],
    is_valid: true
  };
};

const generateMockSignals = (count) => {
  const signals = [];
  for (let i = 0; i < count; i++) {
    signals.push(generateMockSignal(i * 15, count - i)); // 15 minutes apart, descending IDs
  }
  return signals;
};

export const signalsApi = {
  /**
   * Fetch recent signals from backend or return mock data
   * @param {number} limit - Number of recent signals to fetch
   * @returns {Promise<{count: number, signals: Array}>}
   */
  async getRecentSignals(limit = 20) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      // Return mock data
      return new Promise((resolve) => {
        setTimeout(() => {
          const signals = generateMockSignals(limit);
          resolve({
            count: signals.length,
            signals
          });
        }, 300); // Simulate network delay
      });
    }

    // Fetch from live backend
    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/signals/recent?limit=${limit}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching signals from backend:', error);
      // Fallback to mock data on error
      console.log('Falling back to mock data...');
      const signals = generateMockSignals(limit);
      return {
        count: signals.length,
        signals
      };
    }
  },

  /**
   * Get the latest signal (within last 30 minutes)
   * @returns {Promise<Object|null>}
   */
  async getLatestSignal() {
    try {
      const data = await this.getRecentSignals(1);
      
      if (data.count === 0 || !data.signals || data.signals.length === 0) {
        return null;
      }

      const signal = data.signals[0];
      const signalTime = new Date(signal.timestamp);
      const now = new Date();
      const minutesAgo = Math.floor((now - signalTime) / (1000 * 60));

      // Only return if signal is within last 30 minutes
      if (minutesAgo <= 30) {
        return {
          ...signal,
          minutesAgo
        };
      }

      return null;
    } catch (error) {
      console.error('Error fetching latest signal:', error);
      return null;
    }
  }
};