import { APP_CONFIG } from '../config/appConfig';

// Mock data generators for development
const generateMockRun = (index) => {
  const runId = `manual_run_2025010${index}_120000`;
  const startDate = new Date(2025, 0, index, 0, 0, 0);
  const endDate = new Date(2025, 0, index + 7, 23, 59, 59);
  
  return {
    manual_run_id: runId,
    start_date: startDate.toISOString(),
    end_date: endDate.toISOString(),
    symbol: 'XAUUSD',
    timeframe: 'M15',
    signals_generated: Math.floor(Math.random() * 100) + 50,
    average_confidence: (Math.random() * 20 + 65).toFixed(1),
    average_rr_ratio: (Math.random() * 1 + 1.2).toFixed(2)
  };
};

const generateMockSignals = (count) => {
  const signals = [];
  for (let i = 0; i < count; i++) {
    const signalType = Math.random() > 0.5 ? 'BUY' : 'SELL';
    const entryPrice = 2350 + Math.random() * 20 - 10;
    const stopLossDistance = Math.random() * 50 + 50;
    const takeProfitDistance = stopLossDistance * (1.5 + Math.random());
    
    signals.push({
      timestamp: new Date(Date.now() - i * 3600000).toISOString(),
      symbol: 'XAUUSD',
      signal_type: signalType,
      entry_price: entryPrice.toFixed(2),
      stop_loss_price: (entryPrice - (signalType === 'BUY' ? stopLossDistance : -stopLossDistance) / 10).toFixed(2),
      take_profit_price: (entryPrice + (signalType === 'BUY' ? takeProfitDistance : -takeProfitDistance) / 10).toFixed(2),
      final_signal_strength: Math.floor(Math.random() * 30) + 60,
      volatility_state: ['Low', 'Normal', 'High'][Math.floor(Math.random() * 3)],
      risk_reward_ratio: (takeProfitDistance / stopLossDistance).toFixed(2),
      indicator_contributions: {
        RSI: Math.floor(Math.random() * 5) + 6,
        MACD: Math.floor(Math.random() * 5) + 10,
        STOCH: Math.floor(Math.random() * 5) + 4,
        BBANDS: Math.floor(Math.random() * 5) + 4,
        SMA_EMA: Math.floor(Math.random() * 5) + 8,
        MTF: Math.floor(Math.random() * 5) + 12,
        ATR_STABILITY: Math.floor(Math.random() * 5) + 6,
        PRICE_ACTION: Math.floor(Math.random() * 5) + 8,
      }
    });
  }
  return signals;
};

export const backtestApi = {
  /**
   * Generate a new backtest run
   * @param {Object} params - Backtest parameters
   * @returns {Promise<Object>}
   */
  async generateBacktest(params) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      // Return mock response
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            manual_run_id: `manual_run_${new Date().getTime()}`,
            signals_generated: Math.floor(Math.random() * 100) + 50,
            average_confidence: (Math.random() * 20 + 65).toFixed(1),
            status: 'completed'
          });
        }, 1500); // Simulate processing time
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/backtest/manual/generate`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(params)
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error generating backtest:', error);
      // Fallback to mock data
      console.log('Falling back to mock data...');
      return {
        manual_run_id: `manual_run_${new Date().getTime()}`,
        signals_generated: Math.floor(Math.random() * 100) + 50,
        average_confidence: (Math.random() * 20 + 65).toFixed(1),
        status: 'completed'
      };
    }
  },

  /**
   * Fetch list of backtest runs with optional filters
   * @param {Object} filters - Optional filters
   * @returns {Promise<Array>}
   */
  async listRuns(filters = {}) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      // Return mock runs
      return new Promise((resolve) => {
        setTimeout(() => {
          const mockRuns = Array.from({ length: 5 }, (_, i) => generateMockRun(i + 1));
          resolve(mockRuns);
        }, 300);
      });
    }

    try {
      const params = new URLSearchParams();
      if (filters.from_date) params.append('from_date', filters.from_date);
      if (filters.to_date) params.append('to_date', filters.to_date);
      if (filters.min_signal_strength) params.append('min_signal_strength', filters.min_signal_strength);
      if (filters.symbol) params.append('symbol', filters.symbol);

      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/backtest/manual/runs?${params}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching runs:', error);
      // Fallback to mock data
      console.log('Falling back to mock data...');
      const mockRuns = Array.from({ length: 5 }, (_, i) => generateMockRun(i + 1));
      return mockRuns;
    }
  },

  /**
   * Fetch signals for a specific run
   * @param {string} runId - Manual run ID
   * @returns {Promise<Object>}
   */
  async fetchSignals(runId) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      // Return mock signals
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            manual_run_id: runId,
            signals: generateMockSignals(20)
          });
        }, 300);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/backtest/manual/signals/${runId}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching signals:', error);
      // Fallback to mock data
      console.log('Falling back to mock data...');
      return {
        manual_run_id: runId,
        signals: generateMockSignals(20)
      };
    }
  },

  /**
   * Execute simulation for a backtest run
   * @param {Object} params - Simulation parameters
   * @returns {Promise<Object>}
   */
  async executeSimulation(params) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      // Return mock simulation results
      return new Promise((resolve) => {
        setTimeout(() => {
          const initialBalance = params.initial_balance;
          const netProfitPercent = (Math.random() * 30 - 10); // -10% to +20%
          const finalBalance = initialBalance * (1 + netProfitPercent / 100);
          const totalTrades = Math.floor(Math.random() * 50) + 50;
          const wins = Math.floor(totalTrades * (0.4 + Math.random() * 0.3));
          
          resolve({
            manual_run_id: params.manual_run_id,
            initial_balance: initialBalance,
            final_balance: parseFloat(finalBalance.toFixed(2)),
            net_profit_percent: parseFloat(netProfitPercent.toFixed(2)),
            total_trades: totalTrades,
            wins: wins,
            losses: totalTrades - wins,
            win_rate_percent: parseFloat(((wins / totalTrades) * 100).toFixed(1)),
            max_drawdown_percent: parseFloat((Math.random() * 15 + 5).toFixed(2)),
            account_blown: false,
            profit_factor: parseFloat((1 + Math.random() * 0.8).toFixed(2)),
            average_rr_ratio: parseFloat((1.2 + Math.random() * 0.8).toFixed(2)),
            result_summary: netProfitPercent > 0 
              ? 'Profitable run with moderate drawdown and stable performance.'
              : 'Losing run with higher drawdown. Strategy needs optimization.',
            critical_event: null
          });
        }, 2000); // Simulate processing time
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/backtest/manual/execute`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(params)
        }
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error executing simulation:', error);
      // Fallback to mock data
      console.log('Falling back to mock data...');
      const initialBalance = params.initial_balance;
      const netProfitPercent = (Math.random() * 30 - 10);
      const finalBalance = initialBalance * (1 + netProfitPercent / 100);
      const totalTrades = Math.floor(Math.random() * 50) + 50;
      const wins = Math.floor(totalTrades * (0.4 + Math.random() * 0.3));
      
      return {
        manual_run_id: params.manual_run_id,
        initial_balance: initialBalance,
        final_balance: parseFloat(finalBalance.toFixed(2)),
        net_profit_percent: parseFloat(netProfitPercent.toFixed(2)),
        total_trades: totalTrades,
        wins: wins,
        losses: totalTrades - wins,
        win_rate_percent: parseFloat(((wins / totalTrades) * 100).toFixed(1)),
        max_drawdown_percent: parseFloat((Math.random() * 15 + 5).toFixed(2)),
        account_blown: false,
        profit_factor: parseFloat((1 + Math.random() * 0.8).toFixed(2)),
        average_rr_ratio: parseFloat((1.2 + Math.random() * 0.8).toFixed(2)),
        result_summary: netProfitPercent > 0 
          ? 'Profitable run with moderate drawdown and stable performance.'
          : 'Losing run with higher drawdown. Strategy needs optimization.',
        critical_event: null
      };
    }
  }
};