import { APP_CONFIG } from '../config/appConfig';

// Mock data for development
const mockStrategies = [
  {
    id: 1,
    name: "Gold Strategy",
    is_active: true,
    signal_threshold: 0.9,
    run_interval_seconds: 60
  }
];

const mockStrategyDetail = {
  strategy: {
    id: 1,
    name: "Gold Strategy",
    description: "Default gold trading strategy with adaptive indicators",
    timeframes: ["M15", "H1", "H4"],
    signal_threshold: 0.9,
    run_interval_seconds: 60,
    is_active: true
  },
  indicator_params: {
    RSI: { period: 14, overbought: 70, oversold: 30 },
    MACD: { fast: 12, slow: 26, signal: 9 },
    SMA: { short: 20, long: 50 },
    EMA: { short: 20, long: 50 },
    BBANDS: { length: 20, stddev: 2 },
    STOCH: { k: 14, d: 3, overbought: 80, oversold: 20 },
    ATR: { length: 14 },
    PRICE_ACTION: { engulfing_lookback: 5, pinbar_lookback: 5 }
  },
  weights: {
    RSI: 0.15,
    MACD: 0.15,
    STOCH: 0.1,
    BBANDS: 0.1,
    SMA_EMA: 0.2,
    MTF: 0.15,
    ATR_STABILITY: 0.075,
    PRICE_ACTION: 0.075
  }
};

export const strategyApi = {
  /**
   * List all strategies
   */
  async listStrategies() {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ strategies: mockStrategies });
        }, 300);
      });
    }

    try {
      const response = await fetch(`${APP_CONFIG.API_BASE_URL}/api/config/strategies`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching strategies:', error);
      return { strategies: mockStrategies };
    }
  },

  /**
   * Get strategy details
   */
  async getStrategy(strategyId) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve(mockStrategyDetail);
        }, 300);
      });
    }

    try {
      const response = await fetch(`${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching strategy:', error);
      return mockStrategyDetail;
    }
  },

  /**
   * Update indicator parameters
   */
  async updateIndicator(strategyId, indicatorName, params) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ status: 'ok' });
        }, 500);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}/indicator/${indicatorName}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params })
        }
      );
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating indicator:', error);
      throw error;
    }
  },

  /**
   * Update contribution weights
   */
  async updateWeights(strategyId, weights) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ status: 'ok' });
        }, 500);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}/weights`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ weights })
        }
      );
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating weights:', error);
      throw error;
    }
  },

  /**
   * Update run schedule
   */
  async updateSchedule(strategyId, run_interval_seconds) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ status: 'ok' });
        }, 500);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}/schedule`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ run_interval_seconds })
        }
      );
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating schedule:', error);
      throw error;
    }
  },

  /**
   * Update signal threshold
   */
  async updateThreshold(strategyId, signal_threshold) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ status: 'ok' });
        }, 500);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}/threshold`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signal_threshold })
        }
      );
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating threshold:', error);
      throw error;
    }
  },

  /**
   * Activate strategy
   */
  async activateStrategy(strategyId) {
    if (APP_CONFIG.USE_MOCK_DATA) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({ status: 'ok' });
        }, 500);
      });
    }

    try {
      const response = await fetch(
        `${APP_CONFIG.API_BASE_URL}/api/config/strategies/${strategyId}/activate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        }
      );
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error activating strategy:', error);
      throw error;
    }
  }
};