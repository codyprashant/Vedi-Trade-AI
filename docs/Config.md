# VediTrading AI - Strategy Configuration Guide (Enhanced 2025)

Design a comprehensive configuration UI to manage multi-asset trading strategies and their technical indicator settings. This system supports dynamic strategy management for 12 asset classes including Gold (XAUUSD), Forex pairs, Indices, and Futures. All configurations are stored in the database and drive real-time signal generation, backtesting, and WebSocket streaming.

## Enhanced Goals (2025)

- **Multi-Strategy Management**: Create, edit, and manage multiple trading strategies
- **Real-time Configuration**: Dynamic parameter updates without system restart
- **21 Technical Indicators**: Configure comprehensive indicator suite with precision control
- **Multi-Asset Support**: Unified configuration for 12 supported symbols
- **Advanced Signal Control**: Configure signal thresholds, run frequency, and evaluation criteria
- **Live Integration**: Changes immediately affect WebSocket streaming and signal generation

## Enhanced Screens

### Strategy List Dashboard
  - **Display Fields**: `name`, `is_active`, `signal_threshold`, `run_interval_seconds`, `supported_assets`
  - **Actions**: View Details, Activate/Deactivate, Clone Strategy, Performance Metrics
  - **Filters**: Active/Inactive, Asset Type, Performance Range

### Strategy Detail (Multi-Asset Configuration)
  - **Overview Tab**: 
    - Basic Info: `name`, `description`, `is_active`, `supported_timeframes`
    - Asset Coverage: Supported symbols from ALLOWED_SYMBOLS list
    - Performance Metrics: Success rate, average strength, signal frequency
  - **Technical Indicators Tab**: 
    - 21 Indicator Groups: RSI, MACD, SMA, EMA, BBANDS, STOCH, ATR, PRICE_ACTION, etc.
    - Parameter Forms: Period settings, threshold values, calculation methods
    - Real-time Preview: Live indicator values with current parameters
  - **Strategy Weights Tab**: 
    - Contribution Editor: RSI (15%), MACD (20%), SMA_EMA (15%), BBANDS (10%), STOCH (10%), MTF (10%), ATR_STABILITY (10%), PRICE_ACTION (10%)
    - Weight Validation: Ensure total equals 100%
    - Impact Analysis: Show how weight changes affect signal strength
  - **Execution Schedule Tab**: 
    - `run_interval_seconds`: Configurable polling frequency (default: 120s)
    - Active Hours: Market session configuration
    - Asset-specific Timing: Different schedules per asset class
  - **Signal Threshold Tab**: 
    - `signal_threshold`: 0-1 float (UI displays as percentage, e.g., 90% = 0.9)
    - Minimum Strength: Required signal strength for database insertion
    - Evaluation Criteria: "buy", "sell", "neutral" threshold settings

## Field Guide (Common)

- `timeframes`: array of strings, e.g., `["M15","H1","H4"]`
- `signal_threshold`: float 0–1; 0.9 means 90% efficiency
- `run_interval_seconds`: integer seconds between engine iterations
- `is_active`: boolean marking the strategy used by the engine

## Enhanced Indicator Parameters (21 Technical Indicators)

### Core Oscillators
- **RSI (Relative Strength Index)**: `{ "periods": [9, 14], "overbought": 70, "oversold": 30 }`
  - Generates: `rsi_9`, `rsi_14` (2 decimal precision)
  - Evaluation: Oversold → "buy", Overbought → "sell", Middle → "neutral"

- **STOCH (Stochastic Oscillator)**: `{ "k": 14, "d": 3, "overbought": 80, "oversold": 20 }`
  - Generates: `stoch_k`, `stoch_d` (2 decimal precision)
  - Evaluation: %K/%D crossovers and overbought/oversold levels

### Trend Following Indicators
- **MACD (Moving Average Convergence Divergence)**: `{ "fast": 12, "slow": 26, "signal": 9 }`
  - Generates: `macd`, `macd_signal`, `macd_histogram` (2 decimal precision)
  - Evaluation: Signal line crossovers and histogram direction

- **SMA (Simple Moving Averages)**: `{ "periods": [20, 50, 200] }`
  - Generates: `sma_20`, `sma_50`, `sma_200` (2 decimal precision)
  - Evaluation: Price position relative to moving averages and crossovers

- **EMA (Exponential Moving Averages)**: `{ "periods": [9, 21, 55] }`
  - Generates: `ema_9`, `ema_21`, `ema_55` (2 decimal precision)
  - Evaluation: Fast/slow EMA crossovers and price position

### Volatility Indicators
- **BBANDS (Bollinger Bands)**: `{ "length": 20, "std": 2 }`
  - Generates: `bb_low`, `bb_mid`, `bb_high` (2 decimal precision)
  - Evaluation: Price position relative to bands and band squeeze

- **ATR (Average True Range)**: `{ "length": 14, "min_ratio": 0.002 }`
  - Generates: `atr` (2 decimal precision)
  - Evaluation: Volatility environment assessment for signal filtering

### Advanced Pattern Recognition
- **PRICE_ACTION**: `{ "engulfing_lookback": 5, "pinbar_lookback": 5 }`
  - Generates: Pattern-based signals (engulfing, pin bars, trend continuation)
  - Evaluation: Candlestick pattern strength and trend alignment

### Multi-Timeframe Analysis
- **MTF (Multi-Timeframe)**: Analyzes H1 and H4 trend alignment
  - Primary: M15, Confirmation: H1, Trend: H4
  - Generates alignment boost when timeframes agree

### Complete Indicator Suite (21 Total)
All indicators output values with 2 decimal precision and generate evaluations as "buy", "sell", or "neutral".

## Weights (Examples)

- `{ "RSI": 0.15, "MACD": 0.15, "STOCH": 0.1, "BBANDS": 0.1, "SMA_EMA": 0.2, "MTF": 0.15, "ATR_STABILITY": 0.075, "PRICE_ACTION": 0.075 }`

## Enhanced API Endpoints (2025)

### Strategy Management

#### List All Strategies
- **Endpoint**: `GET /api/config/strategies`
- **Description**: Retrieve all available trading strategies with basic information
- **Response Example**:
```json
{
  "strategies": [
    {
      "id": 1,
      "name": "VediTrading Multi-Asset Strategy",
      "is_active": true,
      "signal_threshold": 0.9,
      "run_interval_seconds": 120,
      "supported_assets": ["XAUUSD", "USDCAD", "USDJPY", "GBPUSD", "SPX", "NAS100"],
      "last_updated": "2025-01-01T12:00:00Z"
    }
  ]
}
```

#### Get Strategy Details
- **Endpoint**: `GET /api/config/strategies/{strategy_id}`
- **Description**: Retrieve complete strategy configuration including all 21 indicators and weights
- **Response Example**:
```json
{
  "strategy": {
    "id": 1,
    "name": "VediTrading Multi-Asset Strategy",
    "description": "Enhanced multi-asset strategy with 21 technical indicators",
    "timeframes": ["M15", "H1", "H4"],
    "signal_threshold": 0.9,
    "run_interval_seconds": 120,
    "is_active": true,
    "supported_assets": ["XAUUSD", "USDCAD", "USDJPY", "GBPUSD", "AUDUSD", "AUS200", "UK100", "DJ30", "SPX", "NAS100", "GER40", "FRA40"]
  },
  "indicator_params": {
    "RSI": { "periods": [9, 14], "overbought": 70, "oversold": 30 },
    "MACD": { "fast": 12, "slow": 26, "signal": 9 },
    "SMA": { "periods": [20, 50, 200] },
    "EMA": { "periods": [9, 21, 55] },
    "BBANDS": { "length": 20, "std": 2 },
    "STOCH": { "k": 14, "d": 3, "overbought": 80, "oversold": 20 },
    "ATR": { "length": 14, "min_ratio": 0.002 },
    "PRICE_ACTION": { "engulfing_lookback": 5, "pinbar_lookback": 5 }
  },
  "weights": {
    "RSI": 15,
    "MACD": 20,
    "SMA_EMA": 15,
    "BBANDS": 10,
    "STOCH": 10,
    "MTF": 10,
    "ATR_STABILITY": 10,
    "PRICE_ACTION": 10
  }
}
```

- Update indicator params
  - PATCH `/api/config/strategies/{strategy_id}/indicator/{indicator_name}`
  - Body: `{ "params": { /* indicator-specific params */ } }`
  - Response: `{ "status": "ok" }`

- Update weights
  - PATCH `/api/config/strategies/{strategy_id}/weights`
  - Body: `{ "weights": { /* contribution weights */ } }`
  - Response: `{ "status": "ok" }`

- Update schedule (run frequency)
  - PATCH `/api/config/strategies/{strategy_id}/schedule`
  - Body: `{ "run_interval_seconds": 60 }`
  - Response: `{ "status": "ok" }`

- Update signal threshold
  - PATCH `/api/config/strategies/{strategy_id}/threshold`
  - Body: `{ "signal_threshold": 0.9 }`
  - Response: `{ "status": "ok" }`

- Activate strategy
  - POST `/api/config/strategies/{strategy_id}/activate`
  - Response: `{ "status": "ok" }`

## Recent Enhancements (2025)

### WebSocket Integration
- **Real-time Configuration**: Strategy changes immediately affect live WebSocket streaming
- **Signal Integration**: Latest signals and signal history included in WebSocket responses
- **Precision Control**: All indicator values limited to 2 decimal places for consistency
- **Enhanced Evaluations**: "neutral" replaces "none" for better user experience

### Multi-Asset Support
- **12 Supported Assets**: XAUUSD, USDCAD, USDJPY, GBPUSD, AUDUSD, AUS200, UK100, DJ30, SPX, NAS100, GER40, FRA40
- **Asset-Specific Configuration**: Different parameters per asset class
- **Unified Strategy Management**: Single strategy can support multiple assets

### Advanced Technical Analysis
- **21 Technical Indicators**: Comprehensive indicator suite with real-time computation
- **Multi-Timeframe Analysis**: M15 primary, H1 confirmation, H4 trend analysis
- **Dynamic Weight Management**: Real-time weight adjustments with immediate effect
- **Pattern Recognition**: Advanced price action and candlestick pattern detection

## Enhanced UI Implementation Notes

### Configuration Management
- **Real-time Updates**: Changes take effect immediately without system restart
- **Validation Framework**: Comprehensive parameter validation with range checking
- **Preview Mode**: Live indicator values with current parameters before saving
- **Rollback Support**: Ability to revert changes if performance degrades

### User Experience Enhancements
- **Percentage Display**: `signal_threshold` shown as percentage slider (90% = 0.9)
- **Range Validation**: RSI (0-100), STOCH (0-100), BBANDS stddev (1-3), ATR min_ratio (0.001-0.01)
- **Optimistic Updates**: UI updates immediately with server confirmation
- **Error Handling**: Clear error messaging with suggested corrections
- **Performance Metrics**: Real-time strategy performance indicators

### Advanced Features
- **Strategy Cloning**: Duplicate existing strategies for testing variations
- **A/B Testing**: Compare multiple strategies with performance metrics
- **Asset-Specific Overrides**: Different parameters per asset within same strategy
- **Schedule Management**: Market session-aware execution timing

## Enhanced UI Checklist

### Core Functionality
- ✅ Strategy list with enhanced filtering and sorting
- ✅ Multi-tab strategy detail view with real-time preview
- ✅ 21 Indicator parameter forms with validation
- ✅ Dynamic weight editor with 100% total validation
- ✅ Schedule configuration with market session support
- ✅ Signal threshold management with percentage display

### Advanced Features
- ✅ Real-time performance metrics dashboard
- ✅ Strategy comparison and A/B testing tools
- ✅ Asset-specific parameter overrides
- ✅ Configuration history and rollback functionality
- ✅ Live indicator preview with current parameters
- ✅ WebSocket integration status monitoring