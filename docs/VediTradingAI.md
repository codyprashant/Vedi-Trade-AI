# VediTrading AI — Comprehensive Documentation

## Overview

VediTrading AI is a sophisticated FastAPI-based trading analytics and signal generation engine designed for multi-asset trading across Forex, Commodities, and Indices. The system analyzes multi-timeframe price data, computes a comprehensive suite of technical indicators with multiple periods, and produces validated trade signals through advanced strategy algorithms. It provides enhanced real-time WebSocket streaming with integrated signal data, unified backtesting capabilities, and a trade execution simulator for performance evaluation.

The system exposes REST endpoints for health monitoring, historical data retrieval, signal management, strategy configuration, and backtesting, plus an enhanced WebSocket feed for live tick streaming with detailed indicator data, signal evaluations, and trading signals. All signals and backtesting artifacts are persisted in PostgreSQL for comprehensive analytics and historical tracking.

### Recent Enhancements (2025)

#### Phase 1 & 2: Enhanced Signal Generation System
- **Partial Credit for Neutral Indicators**: Neutral signals now contribute weighted value instead of being ignored
- **Weighted Blend Strategy**: Replaced single-best strategy selection with intelligent multi-strategy blending
- **Multiplicative Alignment Boost**: Changed from additive to proportional enhancement for better scaling
- **Multi-Zone Confidence System**: Strong/Weak/Neutral confidence classification with dynamic thresholds
- **Enhanced MACD Logic**: Added histogram threshold analysis for improved signal quality
- **ATR Stability Integration**: Enhanced final score calculation with volatility-aware adjustments
- **Comprehensive Debug Logging**: Detailed signal processing insights for better transparency

#### WebSocket & Data Streaming
- **Enhanced WebSocket Streaming**: Real-time indicator evaluations with "neutral" instead of "none" for better UX
- **Integrated Signal Data**: Latest signals and signal history directly in WebSocket response
- **Precision Control**: All indicator values limited to 2 decimal places for consistency
- **Streamlined Response**: Removed unnecessary market state fields for cleaner data
- **Strategy Configuration**: Dynamic strategy management with real-time parameter updates

#### Phase 3: Enhanced Signal System (Latest)
- **Weighted Bias Logic**: Intelligent bias calculation using weighted indicator contributions for more accurate signal direction
- **Dynamic Confidence Thresholds**: Adaptive confidence zones (Strong: 75%+, Weak: 50-75%, Neutral: <50%) with enhanced filtering
- **Direction Confidence Tracking**: New database fields `direction_confidence` and `direction_reason` for signal transparency
- **Enhanced Error Handling**: Comprehensive scope management and variable initialization for robust signal processing
- **Improved Signal Quality**: Enhanced confidence filtering reduces false signals while maintaining high-quality signal generation
- **Database Compatibility**: Full backward compatibility with existing signal storage and backtesting systems

## Key Features

- **Multi-Timeframe Analysis**: Signal generation on `M15` timeframe, validated against `H1` trend with additive `H1`/`H4` alignment boosts for enhanced confidence
- **Advanced Indicator Suite**: Multiple periods for RSI (9/14), MACD (12/26/9), SMA (20/50/200), EMA (9/21/55), plus Bollinger Bands, Stochastic, ATR, and price action patterns
- **Intelligent Strategy Engine**: Combines trend and momentum contributions with configurable weights, thresholds, and multi-strategy support
- **Volatility-Aware Trading**: Entry, stop-loss, and take-profit distances dynamically adapt to ATR and volatility classification
- **Real-Time Data Streaming**: Live WebSocket tick streaming and REST history retrieval with comprehensive indicator data
- **Comprehensive Backtesting**: Unified BacktestEngine with strategy-based backtesting, ROI calculation, and detailed performance metrics
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
    "fast": 12,                    # Fast EMA period
    "slow": 26,                    # Slow EMA period  
    "signal": 9,                   # Signal line EMA period
    "histogram_threshold": 0.5     # Enhanced histogram threshold for signal quality
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

### Enhanced Alignment & Confidence System
```python
# Multiplicative Alignment Boosts (replaced additive approach)
ALIGNMENT_BOOST_H1_MULTIPLIER = 1.10    # 10% multiplicative boost for H1 alignment
ALIGNMENT_BOOST_H4_MULTIPLIER = 1.05    # 5% additional multiplicative boost for H4 confirmation
SIGNAL_THRESHOLD = 60                   # Minimum strength for signal persistence (%)

# Multi-Zone Confidence Classification
CONFIDENCE_ZONES = {
    "strong": {"min": 70, "label": "Strong Signal", "action": "trade"},
    "weak": {"min": 50, "label": "Weak Signal", "action": "consider"}, 
    "neutral": {"min": 0, "label": "Neutral Signal", "action": "monitor"}
}

# Enhanced Stability Bonuses
ATR_STABILITY_BONUS = 5     # Bonus for stable volatility environments
PRICE_ACTION_BONUS = 8      # Bonus for strong price action patterns
```

### Enhanced Signal System Configuration

#### Weighted Bias Logic
The enhanced signal system uses intelligent weighted bias calculation for more accurate signal direction:

```python
# Weighted Bias Calculation
BIAS_WEIGHTS = {
    "RSI": 0.25,           # RSI contribution to bias direction
    "MACD": 0.30,          # MACD signal line contribution  
    "SMA_EMA": 0.20,       # Moving average trend contribution
    "BBANDS": 0.15,        # Bollinger Band position weight
    "STOCH": 0.10          # Stochastic momentum contribution
}

# Direction Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    "STRONG": 0.75,        # 75%+ confidence for strong signals
    "WEAK": 0.50,          # 50-75% confidence for weak signals  
    "NEUTRAL": 0.00        # <50% confidence classified as neutral
}
```

#### Enhanced Filtering Parameters
```python
# Signal Quality Filters
MIN_CONFIDENCE_FOR_SIGNAL = 0.50      # Minimum 50% confidence required
ENHANCED_FILTERING_ENABLED = True      # Enable enhanced confidence filtering
NEUTRAL_SIGNAL_THRESHOLD = 0.25        # Target: <25% neutral signals

# Database Fields for Enhanced Tracking
ENHANCED_FIELDS = {
    "direction_confidence": "DECIMAL(5,4)",  # Confidence percentage (0.0000-1.0000)
    "direction_reason": "TEXT"               # Human-readable confidence explanation
}
```

#### Error Handling & Scope Management
```python
# Enhanced Error Handling Configuration
SCOPE_MANAGEMENT = {
    "initialize_variables": True,       # Initialize all variables at function start
    "comprehensive_logging": True,      # Enable detailed debug logging
    "fallback_values": {               # Default values for error scenarios
        "confidence": 0.0,
        "direction": "NEUTRAL", 
        "risk_reward": 1.5
    }
}
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

#### Phase 3: Enhanced Signal Evaluation

The enhanced signal system introduces weighted bias logic and confidence-based filtering:

**Enhanced Bias Calculation**:
```python
def calculate_weighted_bias(indicators):
    """Calculate weighted bias using multiple indicator contributions"""
    bias_score = 0.0
    total_weight = 0.0
    
    # RSI contribution (25% weight)
    if indicators.get('rsi_signal') == 'BUY':
        bias_score += 0.25
    elif indicators.get('rsi_signal') == 'SELL':
        bias_score -= 0.25
    total_weight += 0.25
    
    # MACD contribution (30% weight)  
    if indicators.get('macd_signal') == 'BUY':
        bias_score += 0.30
    elif indicators.get('macd_signal') == 'SELL':
        bias_score -= 0.30
    total_weight += 0.30
    
    # Additional indicator contributions...
    
    # Calculate final bias direction and confidence
    if total_weight > 0:
        normalized_bias = bias_score / total_weight
        confidence = abs(normalized_bias)
        direction = 'BUY' if normalized_bias > 0 else 'SELL'
    else:
        confidence = 0.0
        direction = 'NEUTRAL'
    
    return direction, confidence
```

**Confidence-Based Signal Filtering**:
```python
def enhanced_signal_filter(direction, confidence, signal_strength):
    """Enhanced filtering based on confidence thresholds"""
    
    # Confidence classification
    if confidence >= 0.75:
        confidence_status = "Strong"
    elif confidence >= 0.50:
        confidence_status = "Weak"  
    else:
        confidence_status = "Neutral"
        
    # Enhanced filtering logic
    if confidence < 0.50:  # Below minimum confidence threshold
        return None, "Confidence too low for signal generation"
        
    # Additional signal strength validation
    if signal_strength < 60:  # Below minimum signal strength
        return None, "Signal strength insufficient"
        
    return {
        'direction': direction,
        'confidence': confidence,
        'confidence_status': confidence_status,
        'signal_strength': signal_strength
    }, "Signal passed enhanced filtering"
```

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

#### Phase 4: Enhanced Strategy Strength Computation

**Enhanced Strategy Analysis with Partial Credit**:
The system now uses an advanced approach that gives partial credit to neutral indicators and employs weighted blending:

```python
def compute_enhanced_strategy(results, weights, strategy_type):
    """
    Enhanced strategy computation with partial credit for neutral indicators
    and weighted contribution blending.
    """
    total_strength = 0.0
    contributions = {}
    
    # Define indicator groups by strategy type
    if strategy_type == "trend":
        indicators = ["SMA_EMA", "MACD", "ATR_STABILITY"]
    elif strategy_type == "momentum": 
        indicators = ["RSI", "STOCH", "BBANDS"]
    else:  # combined
        indicators = ["RSI", "MACD", "SMA_EMA", "BBANDS", "STOCH", "ATR_STABILITY"]
    
    # Calculate weighted contributions
    for indicator in indicators:
        if indicator in results:
            direction = results[indicator].direction
            weight = weights.get(indicator, 0)
            
            if direction in ["buy", "sell"]:
                # Full weight for directional signals
                contributions[indicator] = weight
                total_strength += weight
            elif direction == "neutral":
                # Partial credit for neutral signals (50% weight)
                partial_weight = weight * 0.5
                contributions[indicator] = partial_weight
                total_strength += partial_weight
    
    # Determine overall direction by weighted vote
    buy_strength = sum(weights[ind] for ind in indicators 
                      if ind in results and results[ind].direction == "buy")
    sell_strength = sum(weights[ind] for ind in indicators 
                       if ind in results and results[ind].direction == "sell")
    
    if buy_strength > sell_strength:
        direction = "buy"
    elif sell_strength > buy_strength:
        direction = "sell"
    else:
        direction = "neutral"
    
    return {
        "direction": direction,
        "strength": total_strength,
        "contributions": contributions
    }
```

**Weighted Blend Strategy Selection**:
Instead of selecting a single "best" strategy, the system now uses intelligent weighted blending:

```python
def compute_weighted_blend_strategy(strategies, weights):
    """
    Creates a weighted blend of all strategies instead of selecting single best.
    This provides more stable and nuanced signal generation.
    """
    if not strategies:
        return None
    
    # Calculate weighted contributions from each strategy
    total_buy_strength = 0.0
    total_sell_strength = 0.0
    total_neutral_strength = 0.0
    
    combined_contributions = {}
    
    for strategy_name, strategy_data in strategies.items():
        direction = strategy_data["direction"]
        strength = strategy_data["strength"]
        contributions = strategy_data["contributions"]
        
        # Weight the strategy's influence
        strategy_weight = 1.0  # Equal weighting for now, can be configured
        weighted_strength = strength * strategy_weight
        
        # Accumulate directional strengths
        if direction == "buy":
            total_buy_strength += weighted_strength
        elif direction == "sell":
            total_sell_strength += weighted_strength
        else:  # neutral
            total_neutral_strength += weighted_strength
        
        # Merge contributions
        for indicator, contrib in contributions.items():
            combined_contributions[indicator] = combined_contributions.get(indicator, 0) + contrib
    
    # Determine final direction and strength
    if total_buy_strength > total_sell_strength and total_buy_strength > total_neutral_strength:
        final_direction = "buy"
        final_strength = total_buy_strength
    elif total_sell_strength > total_buy_strength and total_sell_strength > total_neutral_strength:
        final_direction = "sell" 
        final_strength = total_sell_strength
    else:
        final_direction = "neutral"
        final_strength = total_neutral_strength
    
    return {
        "strategy": "weighted_blend",
        "direction": final_direction,
        "strength": final_strength,
        "contributions": combined_contributions,
        "component_strategies": strategies
    }
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

**Enhanced Multiplicative Alignment Boost**:
```python
def calculate_multiplicative_alignment_boost(base_strength, m15_direction, h1_trend, h4_trend):
    """
    Enhanced alignment boost using multiplicative scaling instead of additive.
    This provides proportional enhancement based on signal strength.
    """
    boost_multiplier = 1.0
    
    # H1 alignment boost (10% multiplicative)
    if ((m15_direction == "buy" and h1_trend == "Bullish") or 
        (m15_direction == "sell" and h1_trend == "Bearish")):
        boost_multiplier *= 1.10  # 10% multiplicative boost
        
        # H4 confirmation boost (additional 5% if H1 already aligned)
        if h1_trend == h4_trend:
            boost_multiplier *= 1.05  # Additional 5% multiplicative boost
    
    # Apply multiplicative boost to base strength
    enhanced_strength = base_strength * boost_multiplier
    
    return enhanced_strength, boost_multiplier
```

**Multi-Zone Confidence System**:
```python
def get_signal_confidence_zone(signal_strength):
    """
    Classifies signals into confidence zones for better risk management.
    """
    # Sort zones by minimum threshold (descending order)
    sorted_zones = sorted(CONFIDENCE_ZONES.items(), 
                         key=lambda x: x[1]["min"], reverse=True)
    
    for zone_name, zone_config in sorted_zones:
        if signal_strength >= zone_config["min"]:
            return {
                "zone": zone_name,
                "label": zone_config["label"], 
                "action": zone_config["action"],
                "min_threshold": zone_config["min"]
            }
    
    # Default to lowest zone if no match
    return {
        "zone": "neutral",
        "label": "Neutral Signal",
        "action": "monitor",
        "min_threshold": 0
    }

# Configuration for confidence zones
CONFIDENCE_ZONES = {
    "strong": {
        "min": 70,
        "label": "Strong Signal",
        "action": "trade"
    },
    "weak": {
        "min": 50, 
        "label": "Weak Signal",
        "action": "consider"
    },
    "neutral": {
        "min": 0,
        "label": "Neutral Signal", 
        "action": "monitor"
    }
}
```

**Enhanced MACD Logic with Histogram Threshold**:
```python
def evaluate_enhanced_macd(macd_line, macd_signal, macd_histogram, histogram_threshold=0.5):
    """
    Enhanced MACD evaluation with histogram threshold for better signal quality.
    """
    current_macd = macd_line.iloc[-1]
    current_signal = macd_signal.iloc[-1]
    current_histogram = macd_histogram.iloc[-1]
    previous_histogram = macd_histogram.iloc[-2]
    
    # Standard MACD cross detection
    macd_cross_up = current_macd > current_signal and macd_line.iloc[-2] <= macd_signal.iloc[-2]
    macd_cross_down = current_macd < current_signal and macd_line.iloc[-2] >= macd_signal.iloc[-2]
    
    # Enhanced histogram analysis
    histogram_increasing = current_histogram > previous_histogram
    histogram_above_threshold = abs(current_histogram) > histogram_threshold
    
    if macd_cross_up and histogram_increasing and histogram_above_threshold:
        return "buy"
    elif macd_cross_down and not histogram_increasing and histogram_above_threshold:
        return "sell"
    else:
        return "neutral"
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

## Daily Signal Evaluation & ROI Tracker

### Overview

The Daily Signal Evaluation & ROI Tracker is an automated system that evaluates the performance of all trading signals generated by the VediTrading AI system. It runs daily at 05:00 AM server time to analyze the previous day's signals, determine their outcomes (profit/loss/still-open), and maintain comprehensive performance statistics.

### Key Features

- **Automated Daily Evaluation**: Runs at 05:00 AM daily to evaluate previous day's signals
- **Historical Outcome Analysis**: Determines signal results based on actual OHLC price data
- **Performance Metrics**: Tracks ROI, success rates, and efficiency scores per symbol/strategy/timeframe
- **Open Signal Monitoring**: Continuously re-evaluates open positions until they close
- **Backfill Capability**: Historical evaluation for any date range
- **Comprehensive Database**: Stores detailed results and performance summaries

### Schedule & Operation

**Daily Schedule**: 05:00 AM server time
**Evaluation Process**:
1. Fetch all signals from the previous trading day
2. Retrieve historical OHLC data for each signal's symbol
3. Simulate trade execution using entry, stop-loss, and take-profit levels
4. Determine outcome: profit (TP hit), loss (SL hit), or still open
5. Calculate percentage return and update database
6. Re-evaluate any previously open signals
7. Update performance summaries and daily statistics

### Signal Outcome Classification

#### Profit Signals
- **Buy Signal**: Price reaches or exceeds take-profit level
- **Sell Signal**: Price drops to or below take-profit level
- **Exit Price**: Actual take-profit price
- **Result**: Positive percentage return

#### Loss Signals
- **Buy Signal**: Price drops to or below stop-loss level
- **Sell Signal**: Price rises to or above stop-loss level
- **Exit Price**: Actual stop-loss price
- **Result**: Negative percentage return

#### Open Signals
- **Condition**: Neither take-profit nor stop-loss has been hit
- **Exit Price**: Current market price (last close)
- **Result**: Unrealized gain/loss percentage
- **Re-evaluation**: Checked daily until position closes

### Performance Metrics

#### Individual Signal Metrics
```python
profit_pct = ((exit_price - entry_price) / entry_price * 100) * direction_multiplier
direction_multiplier = 1 for buy signals, -1 for sell signals
```

#### Aggregate Performance Metrics
- **Total Signals**: Count of all evaluated signals
- **Win Count**: Number of profitable signals
- **Loss Count**: Number of losing signals
- **Open Count**: Number of still-open signals
- **Win Rate**: `(win_count / (win_count + loss_count)) * 100`
- **Average Profit %**: Mean percentage return across all signals
- **Total ROI %**: Cumulative percentage return
- **Efficiency Score**: Win rate percentage

### Database Schema

#### signal_results Table
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| signal_id | UUID | Foreign key to signals table |
| result | TEXT | "profit", "loss", or "open" |
| exit_price | DECIMAL(10,5) | Actual exit price |
| profit_pct | DECIMAL(8,4) | Percentage return |
| evaluated_at | TIMESTAMP | Evaluation timestamp |
| evaluation_notes | TEXT | Optional comments |

#### signal_performance_summary Table
| Column | Type | Description |
|--------|------|-------------|
| symbol | TEXT | Trading symbol |
| timeframe | TEXT | Signal timeframe |
| strategy | TEXT | Strategy name |
| total_signals | INTEGER | Total signal count |
| win_count | INTEGER | Profitable signals |
| loss_count | INTEGER | Losing signals |
| open_count | INTEGER | Open signals |
| avg_profit_pct | DECIMAL(8,4) | Average return % |
| total_roi_pct | DECIMAL(10,4) | Cumulative ROI % |
| efficiency_pct | DECIMAL(5,2) | Win rate percentage |
| last_updated | TIMESTAMP | Last update time |

#### signal_performance_daily Table
| Column | Type | Description |
|--------|------|-------------|
| evaluation_date | DATE | Evaluation date |
| symbol | TEXT | Trading symbol |
| signals_evaluated | INTEGER | Signals processed |
| new_profits | INTEGER | New profitable closures |
| new_losses | INTEGER | New losing closures |
| daily_roi_pct | DECIMAL(8,4) | Daily ROI contribution |
| created_at | TIMESTAMP | Record creation time |

### Management Commands

#### Daily Evaluation
```bash
# Run evaluation for yesterday (default)
python manage.py evaluate_signals

# Run evaluation for specific date
python manage.py evaluate_signals --date 2024-12-15

# Backfill historical evaluation
python manage.py evaluate_signals --from 2024-01-01 --to 2024-12-31
```

#### Open Signal Re-evaluation
```bash
# Re-evaluate all open signals
python manage.py reevaluate_open
```

#### Performance Statistics
```bash
# Show overall performance
python manage.py show_performance

# Filter by symbol
python manage.py show_performance --symbol XAUUSD

# Filter by timeframe
python manage.py show_performance --timeframe 15m
```

#### Database Initialization
```bash
# Initialize evaluation tables
python manage.py init_db
```

### Sample Performance Output

| Symbol | Direction | Entry | Exit | Result | P/L% | Status |
|--------|-----------|-------|------|--------|------|--------|
| XAUUSD | Buy | 2285.3 | 2298.2 | Profit | +0.57 | Closed |
| XAUUSD | Sell | 2290.5 | 2283.1 | Profit | +0.32 | Closed |
| EURUSD | Buy | 1.0850 | 1.0835 | Loss | -0.14 | Closed |
| GBPUSD | Sell | 1.2650 | - | Open | +0.08 | Open |

### Scheduler Configuration

#### Automated Scheduling
```python
# Daily signal evaluation at 5:00 AM
schedule.every().day.at("05:00").do(daily_signal_evaluation_job)

# Weekly cleanup on Sunday at 2:00 AM
schedule.every().sunday.at("02:00").do(weekly_cleanup_job)
```

#### Manual Scheduler Control
```bash
# Start scheduler daemon
python jobs/scheduler.py --mode start

# Run specific job once
python jobs/scheduler.py --mode run-once --job daily_evaluation

# Check scheduler status
python jobs/scheduler.py --mode status
```

### API Integration

#### Evaluation Endpoints
```
POST /api/evaluation/run-daily          # Trigger daily evaluation
POST /api/evaluation/reevaluate-open    # Re-evaluate open signals
GET  /api/evaluation/performance        # Get performance summary
GET  /api/evaluation/results/{signal_id} # Get specific signal result
```

#### Performance Query Examples
```bash
# Get overall performance
curl http://localhost:8001/api/evaluation/performance

# Get performance for specific symbol
curl http://localhost:8001/api/evaluation/performance?symbol=XAUUSD

# Get daily performance history
curl http://localhost:8001/api/evaluation/daily?from=2024-01-01&to=2024-12-31
```

### Testing & Validation

#### Test Coverage
- **test_signal_profit_hit_takeprofit()**: Validates profit detection when TP is hit
- **test_signal_loss_hit_stoploss()**: Validates loss detection when SL is hit
- **test_signal_still_open()**: Validates open signal handling
- **test_daily_summary_efficiency_calculation()**: Validates performance metrics
- **test_backfill_command()**: Validates historical evaluation

#### Running Tests
```bash
# Run evaluation system tests
python test/test_signal_performance_evaluator.py

# Run all tests including evaluation
python test/run_all_tests.py
```

### Monitoring & Alerts

#### Key Metrics to Monitor
- Daily evaluation completion status
- Number of signals evaluated per day
- Performance trend analysis
- Open signal count and duration
- Database performance and storage

#### Log Files
- **scheduler.log**: Scheduler operation logs
- **evaluation.log**: Signal evaluation details
- **performance.log**: Performance calculation logs

### Benefits & Use Cases

#### For Traders
- **Performance Tracking**: Real-time ROI and efficiency monitoring
- **Strategy Validation**: Historical performance analysis
- **Risk Management**: Open position monitoring and alerts

#### For System Administrators
- **Automated Monitoring**: Daily performance validation
- **Historical Analysis**: Backfill capability for any date range
- **Database Optimization**: Automated cleanup and maintenance

#### For Developers
- **API Integration**: RESTful endpoints for external systems
- **Extensible Design**: Easy addition of new metrics and evaluations
- **Comprehensive Testing**: Full test coverage for reliability

## Summary

VediTrading AI represents a next-generation trading signal generation system that combines advanced technical analysis with intelligent decision-making algorithms. The enhanced system features sophisticated weighted blending strategies, multi-zone confidence classification, and multiplicative alignment boosts that provide superior signal quality and stability.

### Enhanced Key Capabilities (2025)
- **Weighted Blend Strategy Engine**: Intelligent multi-strategy blending replaces single-best selection for more stable signals
- **Partial Credit System**: Neutral indicators now contribute weighted value, improving signal robustness
- **Multi-Zone Confidence Classification**: Strong/Weak/Neutral zones with dynamic thresholds for better risk management
- **Multiplicative Alignment Boosts**: Proportional enhancement scaling provides better signal strength distribution
- **Enhanced MACD Logic**: Histogram threshold analysis improves signal quality and reduces false positives
- **ATR Stability Integration**: Volatility-aware adjustments enhance final score calculations
- **Comprehensive Debug Logging**: Detailed signal processing insights for transparency and optimization

### Core Production Capabilities
- **Comprehensive Analysis**: Multi-timeframe validation with 7+ technical indicators
- **Adaptive Intelligence**: Volatility-aware trade planning and dynamic parameter adjustment
- **Real-Time Performance**: Sub-second signal generation with live WebSocket streaming
- **Robust Architecture**: Fault-tolerant design with comprehensive error handling
- **Extensible Framework**: Modular design supporting custom indicators and strategies
- **Dynamic Configuration**: Real-time parameter adjustment without server restarts
- **Comprehensive Testing**: Full test suite validates all enhanced functionality

The system's proven track record in XAUUSD analysis, combined with its enhanced multi-asset capabilities and sophisticated signal generation algorithms, makes it suitable for both retail and institutional trading applications. The new weighted blend approach and confidence zone system provide superior signal stability and risk management in evolving market conditions.