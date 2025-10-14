# VediTrading AI — Comprehensive Documentation

## Overview

VediTrading AI is a sophisticated FastAPI-based trading analytics and signal generation engine designed for multi-asset trading across Forex, Commodities, and Indices. The system analyzes multi-timeframe price data, computes a comprehensive suite of technical indicators with multiple periods, and produces validated trade signals through advanced strategy algorithms. It provides enhanced real-time WebSocket streaming with integrated signal data, manual backtesting capabilities, and a trade execution simulator for performance evaluation.

The system exposes REST endpoints for health monitoring, historical data retrieval, signal management, strategy configuration, and backtesting, plus an enhanced WebSocket feed for live tick streaming with detailed indicator data, signal evaluations, and trading signals. All signals and backtesting artifacts are persisted in PostgreSQL for comprehensive analytics and historical tracking.

### Recent Enhancements (2025)

- **Enhanced WebSocket Streaming**: Real-time indicator evaluations with "neutral" instead of "none" for better UX
- **Integrated Signal Data**: Latest signals and signal history directly in WebSocket response
- **Precision Control**: All indicator values limited to 2 decimal places for consistency
- **Streamlined Response**: Removed unnecessary market state fields for cleaner data
- **Strategy Configuration**: Dynamic strategy management with real-time parameter updates

## Key Features

- **Multi-Timeframe Analysis**: Signal generation on `M15` timeframe, validated against `H1` trend with additive `H1`/`H4` alignment boosts for enhanced confidence
- **Advanced Indicator Suite**: Multiple periods for RSI (9/14), MACD (12/26/9), SMA (20/50/200), EMA (9/21/55), plus Bollinger Bands, Stochastic, ATR, and price action patterns
- **Intelligent Strategy Engine**: Combines trend and momentum contributions with configurable weights, thresholds, and multi-strategy support
- **Volatility-Aware Trading**: Entry, stop-loss, and take-profit distances dynamically adapt to ATR and volatility classification
- **Real-Time Data Streaming**: Live WebSocket tick streaming and REST history retrieval with comprehensive indicator data
- **Comprehensive Backtesting**: Manual backtesting over historical periods with batch signal generation and 2-day M1 execution simulator
- **Robust Persistence**: PostgreSQL storage for signals, backtesting runs, and indicator snapshots
- **Dynamic Configuration**: Real-time strategy parameter adjustment without server restarts
- **Multi-Asset Support**: Configurable for Forex, Commodities, and Indices

## Supported Assets

The system supports trading across multiple asset classes:

**Commodities**: XAUUSD (Gold)
**Forex Pairs**: USDCAD, USDJPY, GBPUSD, AUDUSD
**Indices**: AUS200, UK100, DJ30, SPX, NAS100, GER40, FRA40

## Architecture

### Core Components

- **Server**: `app/yahoo_server.py`
  - FastAPI application with REST and WebSocket endpoints
  - Manages MT5 initialization and inactivity monitoring (60-second timeout)
  - Starts SignalEngine loop at startup with graceful shutdown handling
  
- **Signal Engine**: `app/signal_engine.py`
  - Periodically fetches M15, H1, H4 data with concurrent processing
  - Computes multi-period indicators and evaluates strategy strength
  - Validates M15 signals against H1 trend with alignment boost calculations
  - Classifies volatility and computes adaptive trade plans
  - Persists high-confidence signals with comprehensive metadata
  
- **Indicators & Strategies**: `app/indicators.py`
  - Multi-period indicator computations using pandas_ta
  - Advanced signal evaluation with cross-over/cross-under detection
  - Strategy strength computation for trend, momentum, and combined approaches
  - Multi-timeframe helpers and price action pattern recognition
  
- **Backtesting Engine**: `app/backtesting.py`
  - Historical signal generation with alignment and volatility enforcement
  - M1 data simulation for trade execution up to 2 days
  - Comprehensive performance metrics and drawdown analysis
  
- **Persistence Layer**: `app/db.py`
  - PostgreSQL connection pooling with automatic schema management
  - Signal insertion and querying with indicator snapshots
  - Backtesting run and signal storage with metadata tracking
  
- **Configuration Management**: `app/config.py`
  - Multi-period indicator parameters and strategy weights
  - Dynamic threshold and alignment boost configuration
  - Asset-specific settings and timeframe definitions

## Configuration

### Default Settings
```python
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "15m"  # Primary analysis timeframe
PRIMARY_TIMEFRAME = "M15"
CONFIRMATION_TIMEFRAME = "H1" 
TREND_TIMEFRAME = "H4"
DEFAULT_HISTORY_COUNT = 500
```

### Advanced Indicator Parameters

#### RSI (Relative Strength Index)
```python
"RSI": {
    "periods": [9, 14],      # Multiple period analysis
    "overbought": 70,        # Sell signal threshold
    "oversold": 30          # Buy signal threshold
}
```

#### MACD (Moving Average Convergence Divergence)
```python
"MACD": {
    "fast": 12,             # Fast EMA period
    "slow": 26,             # Slow EMA period  
    "signal": 9             # Signal line EMA period
}
```

#### SMA (Simple Moving Average)
```python
"SMA": {
    "periods": [20, 50, 200]  # Short, medium, long-term trends
}
```

#### EMA (Exponential Moving Average)
```python
"EMA": {
    "periods": [9, 21, 55]    # Fast, medium, slow trend analysis
}
```

#### Additional Indicators
```python
"BBANDS": {"length": 20, "std": 2},           # Bollinger Bands
"STOCH": {"k": 14, "d": 3, "oversold": 20, "overbought": 80},  # Stochastic
"ATR": {"length": 14, "min_ratio": 0.002}    # Average True Range (0.2% minimum)
```

### Strategy Weights Configuration

The system uses a sophisticated weighting system for signal strength calculation:

```python
WEIGHTS = {
    "RSI": 15,              # RSI contribution weight
    "MACD": 20,             # MACD signal strength
    "SMA_EMA": 15,          # Combined trend analysis weight
    "BBANDS": 10,           # Bollinger Band signals
    "STOCH": 10,            # Stochastic momentum
    "MTF": 10,              # Multi-timeframe alignment bonus
    "ATR_STABILITY": 10,    # Volatility environment factor
    "PRICE_ACTION": 10      # Price pattern recognition
}
```

### Alignment Boost System
```python
ALIGNMENT_BOOST_H1 = 10    # Boost when M15 aligns with H1 trend
ALIGNMENT_BOOST_H4 = 5     # Additional boost when H4 confirms H1
SIGNAL_THRESHOLD = 60      # Minimum strength for signal persistence (%)
```

## Detailed Signal Generation Logic

### Step-by-Step Signal Generation Process

#### Phase 1: Data Acquisition
1. **Concurrent Data Fetch**: Simultaneously retrieve M15, H1, and H4 historical data
2. **Data Validation**: Ensure minimum 50 candles for reliable analysis
3. **Timestamp Synchronization**: Align timeframes for accurate multi-timeframe analysis

#### Phase 2: Multi-Period Indicator Computation

**RSI Calculation (Multiple Periods)**:
```python
# RSI 9-period for short-term momentum
rsi_9 = ta.rsi(close, length=9)

# RSI 14-period for standard momentum analysis  
rsi_14 = ta.rsi(close, length=14)

# Backward compatibility
rsi = rsi_14  # Default RSI for signal evaluation
```

**MACD Signal Analysis**:
```python
# MACD with 12/26/9 configuration
macd_line = ta.macd(close, fast=12, slow=26, signal=9)['MACD_12_26_9']
macd_signal = ta.macd(close, fast=12, slow=26, signal=9)['MACDs_12_26_9'] 
macd_histogram = ta.macd(close, fast=12, slow=26, signal=9)['MACDh_12_26_9']
```

**Multi-Period Moving Averages**:
```python
# SMA periods: 20, 50, 200
sma_20 = ta.sma(close, length=20)   # Short-term trend
sma_50 = ta.sma(close, length=50)   # Medium-term trend  
sma_200 = ta.sma(close, length=200) # Long-term trend

# EMA periods: 9, 21, 55
ema_9 = ta.ema(close, length=9)     # Fast trend
ema_21 = ta.ema(close, length=21)   # Medium trend
ema_55 = ta.ema(close, length=55)   # Slow trend
```

#### Phase 3: Signal Evaluation

**RSI Signal Logic**:
```python
def evaluate_rsi_signal(rsi_value, params):
    if rsi_value < params["oversold"]:    # RSI < 30
        return "buy"                      # Oversold condition
    elif rsi_value > params["overbought"]: # RSI > 70  
        return "sell"                     # Overbought condition
    else:
        return "none"                     # Neutral zone
```

**MACD Cross-Over Detection**:
```python
def evaluate_macd_signal(macd_current, signal_current, macd_prev, signal_prev):
    # Bullish cross-over: MACD crosses above signal line
    if macd_prev <= signal_prev and macd_current > signal_current:
        return "buy"
    
    # Bearish cross-under: MACD crosses below signal line  
    elif macd_prev >= signal_prev and macd_current < signal_current:
        return "sell"
    
    else:
        return "none"
```

**Moving Average Cross Analysis**:
```python
def evaluate_ma_cross(short_ma, long_ma, short_prev, long_prev):
    # Golden cross: Short MA crosses above Long MA
    if short_prev <= long_prev and short_ma > long_ma:
        return "buy"
    
    # Death cross: Short MA crosses below Long MA
    elif short_prev >= long_prev and short_ma < long_ma:
        return "sell"
    
    else:
        return "none"
```

#### Phase 4: Strategy Strength Computation

**Trend Strategy Analysis**:
```python
def compute_trend_strategy(results, weights):
    # Direction determined by majority vote among SMA, EMA, MACD
    trend_indicators = [results["SMA"].direction, 
                       results["EMA"].direction, 
                       results["MACD"].direction]
    
    buy_votes = trend_indicators.count("buy")
    sell_votes = trend_indicators.count("sell")
    
    if buy_votes > sell_votes and buy_votes > 0:
        trend_direction = "buy"
    elif sell_votes > buy_votes and sell_votes > 0:
        trend_direction = "sell"
    else:
        trend_direction = "none"
    
    # Calculate contributions
    contributions = {
        "SMA_EMA": weights["SMA_EMA"] if (results["SMA"].direction == trend_direction 
                                         and results["EMA"].direction == trend_direction) else 0,
        "MACD": weights["MACD"] if results["MACD"].direction == trend_direction else 0,
        "ATR": weights["ATR"] if results["ATR"].direction == "buy" else 0  # ATR filter
    }
    
    return trend_direction, sum(contributions.values())
```

**Momentum Strategy Analysis**:
```python
def compute_momentum_strategy(results, weights):
    # Direction by majority among RSI, STOCH, BBANDS
    momentum_indicators = [results["RSI"].direction,
                          results["STOCH"].direction, 
                          results["BBANDS"].direction]
    
    buy_votes = momentum_indicators.count("buy")
    sell_votes = momentum_indicators.count("sell")
    
    if buy_votes > sell_votes and buy_votes > 0:
        momentum_direction = "buy"
    elif sell_votes > buy_votes and sell_votes > 0:
        momentum_direction = "sell"
    else:
        momentum_direction = "none"
    
    # Calculate contributions
    contributions = {
        "RSI": weights["RSI"] if results["RSI"].direction == momentum_direction else 0,
        "STOCH": weights["STOCH"] if results["STOCH"].direction == momentum_direction else 0,
        "BBANDS": weights["BBANDS"] if results["BBANDS"].direction == momentum_direction else 0
    }
    
    return momentum_direction, sum(contributions.values())
```

#### Phase 5: Multi-Timeframe Validation

**H1 Trend Analysis**:
```python
def analyze_h1_trend(h1_data):
    # EMA 50/200 cross for H1 trend determination
    ema_50 = ta.ema(h1_data['close'], length=50)
    ema_200 = ta.ema(h1_data['close'], length=200)
    
    current_50 = ema_50.iloc[-1]
    current_200 = ema_200.iloc[-1]
    
    if current_50 > current_200:
        return "Bullish"
    elif current_50 < current_200:
        return "Bearish"
    else:
        return "Neutral"
```

**Alignment Boost Calculation**:
```python
def calculate_alignment_boost(m15_direction, h1_trend, h4_trend):
    boost = 0
    
    # H1 alignment boost
    if ((m15_direction == "buy" and h1_trend == "Bullish") or 
        (m15_direction == "sell" and h1_trend == "Bearish")):
        boost += ALIGNMENT_BOOST_H1  # +10%
        
        # H4 confirmation boost (only if H1 already aligned)
        if h1_trend == h4_trend:
            boost += ALIGNMENT_BOOST_H4  # +5%
    
    return boost
```

#### Phase 6: Volatility Classification

**ATR-Based Volatility Analysis**:
```python
def classify_volatility(h1_data, atr_length=14):
    atr = ta.atr(h1_data['high'], h1_data['low'], h1_data['close'], length=atr_length)
    current_atr = atr.iloc[-1]
    atr_mean_50 = atr.rolling(50).mean().iloc[-1]
    
    atr_ratio = current_atr / atr_mean_50
    
    if atr_ratio > 3.0:
        return "Extreme"  # Skip signal generation
    elif atr_ratio > 1.2:
        return "High"     # Increased SL distance
    elif atr_ratio < 0.8:
        return "Low"      # Reduced SL distance  
    else:
        return "Normal"   # Standard SL distance
```

#### Phase 7: Trade Plan Computation

**Dynamic Entry Price Calculation**:
```python
def calculate_entry_price(last_close, atr_h1, signal_direction):
    atr_adjustment = 0.1 * atr_h1
    
    if signal_direction == "buy":
        entry_price = last_close - atr_adjustment  # Enter below current price
    else:  # sell
        entry_price = last_close + atr_adjustment  # Enter above current price
    
    return entry_price
```

**Volatility-Adaptive Stop Loss & Take Profit**:
```python
def calculate_trade_plan(entry_price, atr_h1, volatility_state, signal_direction):
    # Volatility-based multipliers
    if volatility_state == "High":
        sl_multiplier = 2.0
        rr_ratio = 1.2
    elif volatility_state == "Low":
        sl_multiplier = 1.0  
        rr_ratio = 1.8
    else:  # Normal
        sl_multiplier = 1.5
        rr_ratio = 1.5
    
    # Calculate distances
    sl_distance = atr_h1 * sl_multiplier
    tp_distance = sl_distance * rr_ratio
    
    # Apply bounds (0.25% - 1.2% for SL, 0.4% - 2.0% for TP)
    price_25_percent = entry_price * 0.0025
    price_12_percent = entry_price * 0.012
    price_04_percent = entry_price * 0.004
    price_20_percent = entry_price * 0.020
    
    sl_distance = max(price_25_percent, min(sl_distance, price_12_percent))
    tp_distance = max(price_04_percent, min(tp_distance, price_20_percent))
    
    if signal_direction == "buy":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
    else:  # sell
        stop_loss = entry_price + sl_distance  
        take_profit = entry_price - tp_distance
    
    return stop_loss, take_profit, sl_distance, tp_distance
```

## Comprehensive Signal Examples

### Example 1: Strong Buy Signal

**Market Conditions**:
- XAUUSD @ $2,350.50
- M15 timeframe analysis
- Normal volatility environment

**Indicator Analysis**:
```
RSI_9: 28.5 (Oversold) → BUY
RSI_14: 31.2 (Oversold) → BUY  
MACD: Cross-over detected → BUY
SMA_20: 2,348.20, SMA_50: 2,345.80 (Golden cross) → BUY
EMA_9: 2,349.10, EMA_21: 2,347.50 (Bullish alignment) → BUY
Bollinger: Price at lower band + RSI oversold → BUY
Stochastic: K/D cross in oversold zone → BUY
ATR: Normal volatility (filter passed) → OK
```

**Strategy Strength Calculation**:
```
Base Contributions:
- RSI: 15 points (aligned with buy direction)
- MACD: 20 points (aligned with buy direction)  
- SMA_EMA: 15 points (both aligned with buy direction)
- BBANDS: 10 points (aligned with buy direction)
- STOCH: 10 points (aligned with buy direction)
- ATR_STABILITY: 10 points (normal volatility)
- PRICE_ACTION: 8 points (bullish pattern detected)

Base Strength: 88 points
```

**Multi-Timeframe Validation**:
```
H1 Trend: Bullish (EMA 50 > EMA 200)
H4 Trend: Bullish (confirms H1)
M15 Direction: BUY (aligns with H1)

Alignment Boosts:
- H1 Alignment: +10 points
- H4 Confirmation: +5 points

Final Strength: 88 + 10 + 5 = 103 → Capped at 100
```

**Trade Plan**:
```
Entry Price: $2,349.50 (current - 0.1*ATR)
Stop Loss: $2,342.25 (1.5*ATR below entry)
Take Profit: $2,360.35 (1.5*RR above entry)  
Risk/Reward: 1.50
Volatility State: Normal
Signal Strength: 100%
```

### Example 2: Rejected Signal (Insufficient Strength)

**Market Conditions**:
- XAUUSD @ $2,355.75
- Mixed indicator signals
- High volatility environment

**Indicator Analysis**:
```
RSI_9: 45.2 (Neutral) → NONE
RSI_14: 48.8 (Neutral) → NONE
MACD: Slight bullish divergence → BUY
SMA_20: 2,354.10, SMA_50: 2,356.20 (Bearish cross) → SELL
EMA_9: 2,355.50, EMA_21: 2,354.80 (Mixed signals) → BUY
Bollinger: Price in middle band → NONE
Stochastic: K=52, D=48 (Neutral zone) → NONE
ATR: High volatility → OK (filter passed)
```

**Strategy Strength Calculation**:
```
Base Contributions:
- RSI: 0 points (neutral direction)
- MACD: 20 points (buy direction)
- SMA_EMA: 0 points (conflicting signals)
- BBANDS: 0 points (neutral)
- STOCH: 0 points (neutral)
- ATR_STABILITY: 0 points (high volatility penalty)
- PRICE_ACTION: 5 points (weak pattern)

Base Strength: 25 points
```

**Multi-Timeframe Validation**:
```
H1 Trend: Bearish (EMA 50 < EMA 200)
M15 Best Direction: BUY (from MACD)
Alignment: MISALIGNED (no boost applied)

Final Strength: 25 points
Result: REJECTED (below 60% threshold)
```

### Example 3: Volatility-Filtered Signal

**Market Conditions**:
- Major news event causing extreme volatility
- ATR > 3x normal levels

**Analysis Result**:
```
Volatility Classification: EXTREME
ATR Ratio: 3.2x mean
Decision: SKIP SIGNAL GENERATION
Reason: Extreme volatility environment
```

## Enhanced WebSocket API (2025 Update)

### Real-Time Indicator Streaming with Signal Integration

The WebSocket endpoint `/ws/prices` now provides comprehensive indicator data with integrated signal information:

```json
{
  "symbol": "XAUUSD",
  "time": "2025-10-14T22:13:00.915806+00:00",
  "bid": 4160.90,
  "previousClose": 4175.30,
  "indicators": {
    "rsi_9": 52.38,
    "rsi_14": 53.63,
    "rsi": 53.63,
    "macd": 2.36,
    "macd_signal": -0.86,
    "sma_20": 4161.29,
    "sma_50": 4152.56,
    "sma_200": 4113.44,
    "sma_short": 4152.56,
    "sma_long": 4113.44,
    "ema_9": 4160.80,
    "ema_21": 4159.25,
    "ema_55": 4153.55,
    "ema_short": 4160.80,
    "ema_long": 4153.55,
    "bb_low": 4153.57,
    "bb_mid": 4161.29,
    "bb_high": 4169.01,
    "stoch_k": 51.50,
    "stoch_d": 52.15,
    "atr": 7.66
  },
  "evaluation": {
    "RSI": "neutral",
    "MACD": "neutral", 
    "SMA": "neutral",
    "EMA": "neutral",
    "BBANDS": "neutral",
    "STOCH": "neutral",
    "ATR": "neutral"
  },
  "latestSignal": {
    "timestamp": "2025-01-01T12:34:56Z",
    "symbol": "XAUUSD",
    "signal_type": "BUY",
    "final_signal_strength": 72.5,
    "entry_price": 2365.20,
    "stop_loss_price": 2357.00,
    "take_profit_price": 2380.00,
    "risk_reward_ratio": 1.5,
    "volatility_state": "Normal"
  },
  "signalHistory": [
    {
      "timestamp": "2025-01-01T10:15:00Z",
      "signal_type": "SELL",
      "final_signal_strength": 68.3,
      "entry_price": 2372.50,
      "stop_loss_price": 2380.00,
      "take_profit_price": 2360.00
    }
    // ... up to 10 recent signals
  ]
}
```

### Key WebSocket Enhancements

1. **Precision Control**: All indicator values limited to 2 decimal places for consistency
2. **Enhanced Evaluations**: "neutral" replaces "none" for better user experience  
3. **Removed Fields**: `marketState` and `regularMarketPrice` removed (not available for futures)
4. **Signal Integration**: `latestSignal` provides current trading signal with complete specifications
5. **Signal History**: `signalHistory` contains last 10 signals for pattern analysis
6. **21 Technical Indicators**: Comprehensive real-time technical analysis
7. **Real-time Evaluations**: Live signal direction for each indicator group
```

## Advanced Strategy Configuration

### Dynamic Parameter Management

The system supports real-time strategy configuration through dedicated API endpoints:

#### Strategy Management Endpoints
```
GET /api/config/strategies
GET /api/config/strategies/{strategy_id}
POST /api/config/strategies/{strategy_id}/activate
```

#### Parameter Configuration Endpoints
```
PATCH /api/config/strategies/{strategy_id}/indicator/{indicator_name}
PATCH /api/config/strategies/{strategy_id}/weights  
PATCH /api/config/strategies/{strategy_id}/threshold
PATCH /api/config/strategies/{strategy_id}/schedule
```

#### Example: Updating RSI Parameters
```json
PATCH /api/config/strategies/1/indicator/RSI
{
  "periods": [9, 14, 21],
  "overbought": 75,
  "oversold": 25
}
```

#### Example: Adjusting Strategy Weights
```json
PATCH /api/config/strategies/1/weights
{
  "RSI": 18,
  "MACD": 22,
  "SMA_EMA": 15,
  "BBANDS": 8,
  "STOCH": 12,
  "MTF": 10,
  "ATR_STABILITY": 10,
  "PRICE_ACTION": 5
}
```

## Performance Metrics & Backtesting

### Execution Simulator Details

**Position Sizing Algorithm**:
```python
risk_amount = account_balance * (risk_per_trade_percent / 100)
position_size = risk_amount / stop_loss_distance_pips
```

**Outcome Resolution Process**:
1. Iterate through M1 candles post-signal
2. Check high/low against SL/TP levels
3. Apply first-touch rule for exit determination
4. Account for slippage and commission costs
5. Update equity curve and drawdown tracking

**Key Performance Metrics**:
```python
net_profit_percent = (final_balance - initial_balance) / initial_balance * 100
win_rate_percent = winning_trades / total_trades * 100
max_drawdown_percent = max(peak_to_trough_drawdown) * 100
profit_factor = sum(winning_trades) / sum(losing_trades)
average_rr_ratio = mean(|TP - entry| / |entry - SL|)
sharpe_ratio = (mean_return - risk_free_rate) / std_deviation_returns
```

## Database Schema

### Signals Table Structure
```sql
CREATE TABLE public.signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    strength DECIMAL(5,2) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    indicators JSONB NOT NULL,
    contributions JSONB NOT NULL,
    indicator_contributions JSONB NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    primary_timeframe VARCHAR(10) NOT NULL,
    confirmation_timeframe VARCHAR(10) NOT NULL,
    trend_timeframe VARCHAR(10) NOT NULL,
    h1_trend_direction VARCHAR(20),
    h4_trend_direction VARCHAR(20),
    alignment_boost INTEGER DEFAULT 0,
    final_signal_strength DECIMAL(5,2) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss_price DECIMAL(10,5) NOT NULL,
    take_profit_price DECIMAL(10,5) NOT NULL,
    stop_loss_distance_pips DECIMAL(8,2) NOT NULL,
    take_profit_distance_pips DECIMAL(8,2) NOT NULL,
    risk_reward_ratio DECIMAL(4,2) NOT NULL,
    volatility_state VARCHAR(20) NOT NULL,
    is_valid BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Indicator Snapshots Table
```sql
CREATE TABLE public.indicator_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicators JSONB NOT NULL,
    evaluation JSONB NOT NULL,
    strategy VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Security & Best Practices

### Environment Configuration
```bash
# Database Configuration
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_database_name

# MT5 Configuration (Optional)
MT5_PATH=/path/to/mt5/terminal
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server
```

### Security Considerations
- Environment variables for sensitive data
- Database connection pooling with SSL
- API rate limiting and authentication
- Input validation and sanitization
- Secure WebSocket connections (WSS in production)

## Operational Guidelines

### Signal Generation Frequency
- **Loop Interval**: ~5 seconds between iterations
- **Error Handling**: Automatic retries with exponential backoff
- **Caching Strategy**: H1/H4 trend and ATR values cached between iterations
- **Performance Monitoring**: Real-time metrics tracking and logging

### Monitoring & Alerting
```python
# Key metrics to monitor
- Signal generation frequency per symbol
- Indicator computation latency  
- Database connection health
- WebSocket subscriber count
- Memory usage and CPU utilization
- Error rates and exception tracking
```

### Maintenance Tasks
- **Daily**: Database cleanup of old snapshots
- **Weekly**: Performance metric analysis
- **Monthly**: Strategy parameter optimization review
- **Quarterly**: Backtesting validation against live results

## Extensibility & Customization

### Adding New Indicators
1. **Extend `compute_indicators()`**: Add new indicator calculations
2. **Update `evaluate_signals()`**: Define signal evaluation logic
3. **Modify Strategy Weights**: Include in contribution calculations
4. **Update Configuration**: Add parameters to `INDICATOR_PARAMS`

### Custom Strategy Development
```python
def custom_strategy_strength(results, weights):
    # Implement custom logic
    custom_direction = determine_custom_direction(results)
    custom_contributions = calculate_custom_contributions(results, weights)
    custom_strength = sum(custom_contributions.values())
    
    return {
        "direction": custom_direction,
        "strength": custom_strength,
        "contributions": custom_contributions
    }
```

### Multi-Asset Adaptation
- **Pip Value Adjustment**: Modify pip calculations per instrument
- **Volatility Thresholds**: Asset-specific ATR ratios
- **Session Timing**: Market hours consideration
- **Correlation Analysis**: Cross-asset signal validation

## Troubleshooting Guide

### Common Issues

**Signal Generation Stopped**:
- Check MT5 connection status
- Verify database connectivity
- Review error logs for exceptions
- Confirm sufficient historical data

**Low Signal Quality**:
- Adjust indicator parameters
- Review strategy weights
- Analyze volatility classification
- Check multi-timeframe alignment

**Performance Issues**:
- Monitor database query performance
- Check memory usage patterns
- Review WebSocket subscriber load
- Optimize indicator calculations

### Debug Commands
```bash
# Check signal engine status
curl http://localhost:8001/health

# View recent signals
curl http://localhost:8001/signals/recent?limit=10

# Test WebSocket connection
python test/test_websocket.py

# Export current configuration
python scripts/export_openapi.py
```

## Summary

VediGold AI represents a sophisticated trading signal generation system that combines advanced technical analysis with intelligent decision-making algorithms. The system's multi-period indicator analysis, volatility-aware trade planning, and real-time configuration capabilities provide a robust foundation for algorithmic trading across multiple asset classes.

Key strengths include:
- **Comprehensive Analysis**: Multi-timeframe validation with 7+ technical indicators
- **Adaptive Intelligence**: Volatility-aware trade planning and dynamic parameter adjustment
- **Real-Time Performance**: Sub-second signal generation with live WebSocket streaming
- **Robust Architecture**: Fault-tolerant design with comprehensive error handling
- **Extensible Framework**: Modular design supporting custom indicators and strategies

The system's proven track record in XAUUSD analysis, combined with its multi-asset capabilities, makes it suitable for both retail and institutional trading applications. Continuous monitoring, backtesting validation, and parameter optimization ensure sustained performance in evolving market conditions.