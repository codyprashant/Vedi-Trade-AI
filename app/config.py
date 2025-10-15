DEFAULT_SYMBOL = "XAUUSD"

# Canonical allowed symbols (use these everywhere across app/backend/frontend)
ALLOWED_SYMBOLS = [
    "XAUUSD",
    "USDCAD",
    "USDJPY",
    "GBPUSD",
    "AUDUSD",
    "AUS200",
    "UK100",
    "DJ30",
    "SPX",
    "NAS100",
    "GER40",
    "FRA40",
]

# Backward-compatible alias for legacy imports
DEFAULT_SYMBOLS = ALLOWED_SYMBOLS
DEFAULT_TIMEFRAME = "15m"  # primary analysis timeframe for signals (M15)
PRIMARY_TIMEFRAME = "M15"
CONFIRMATION_TIMEFRAME = "H1"
TREND_TIMEFRAME = "H4"
DEFAULT_HISTORY_COUNT = 500

# Indicator parameters
INDICATOR_PARAMS = {
    "RSI": {"periods": [9, 14], "overbought": 70, "oversold": 30},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "SMA": {"periods": [20, 50, 200]},
    "EMA": {"periods": [9, 21, 55]},
    "BBANDS": {"length": 20, "std": 2},
    "STOCH": {"k": 14, "d": 3, "oversold": 20, "overbought": 80},
    "ATR": {"length": 14, "min_ratio": 0.002},  # 0.2% of price as minimal volatility
}

# Indicator weights (percent totals) — aligned to new spec
# Base indicators on M15
WEIGHTS = {
    "RSI": 15,
    "MACD": 20,
    "SMA_EMA": 15,  # combined weight for trend crosses (EMA/SMA)
    "BBANDS": 10,
    "STOCH": 10,
    # Additional categories
    "MTF": 10,             # H1 alignment credit
    "ATR_STABILITY": 10,   # Normal volatility environment
    "PRICE_ACTION": 10,    # Recent 5-candle momentum pattern
}

# Alignment boost configuration (applies additively on top of base strength)
# H1 boost is applied when M15 aligns with H1; H4 boost is applied when H4 agrees with H1
# and M15 aligns with H1. No penalties are applied on misalignment.
ALIGNMENT_BOOST_H1 = 10
ALIGNMENT_BOOST_H4 = 5

SIGNAL_THRESHOLD = 60  # percent

# Enhanced Decision Tree Parameters
NEUTRAL_WEIGHT_FACTOR = 0.5  # Partial credit for neutral indicators aligned with majority
TREND_WEIGHT_RATIO = 0.6     # Weight for trend strategy in blended approach
MOMENTUM_WEIGHT_RATIO = 0.4  # Weight for momentum strategy in blended approach
MACD_HIST_MIN = 0.5         # Minimum MACD histogram magnitude for signal

# Signal Confidence Zones
CONFIDENCE_ZONES = {
    "strong": {"min": 70, "label": "Strong Buy/Sell", "action": "Confirmed signal"},
    "weak": {"min": 50, "label": "Weak Buy/Sell", "action": "Conditional/Small-size trade"},
    "neutral": {"min": 0, "label": "Neutral", "action": "No trade"}
}

# ATR and Price Action Bonuses
ATR_STABILITY_BONUS = 10    # Bonus when ATR stability is "normal"
PRICE_ACTION_BONUS = 5      # Bonus when price action aligns with final direction

# Logging Configuration
DEBUG_WEBSOCKET = False  # Set to True to enable detailed WebSocket debug logs
DEBUG_SIGNALS = False    # Set to True to enable detailed signal processing logs

# Adaptive Threshold Manager Configuration
THRESHOLD_MANAGER_CONFIG = {
    "base_threshold": 60.0,
    "atr_factor": 1.5,
    "min_threshold": 45.0,
    "max_threshold": 85.0,
    "volatility_weight": 0.4,
    "momentum_weight": 0.3,
    "trend_weight": 0.3,
    # Enhanced volatility regime classification
    "volatility_regime_thresholds": {
        "low": 0.8,      # ATR ratio < 0.8 = low volatility
        "normal": 1.5,   # 0.8 <= ATR ratio < 1.5 = normal volatility
        "high": 2.0,     # 1.5 <= ATR ratio < 2.0 = high volatility
        # ATR ratio >= 2.0 = extreme volatility
    },
    "stress_detection_enabled": True,
    "adaptive_parameters": True  # Enable dynamic weight adjustment based on market regime
}

# Enhanced Sanity Filter Configuration
SANITY_FILTER_CONFIG = {
    "strict": {
        "min_volatility": 0.002,
        "min_body_ratio": 0.6,
        "min_confidence": 65.0,
        "min_volume_ratio": 1.0,
        "max_spread_ratio": 0.03,
        "enable_candle_pattern_filter": True,
        "enable_volatility_filter": True,
        "enable_confidence_filter": True,
        # Enhanced ATR validation
        "atr_volatility_thresholds": {
            "very_low": 0.0005,  # 0.05%
            "low": 0.7,
            "normal": 1.8,
            "high": 3.0
        },
        "enable_atr_regime_validation": True,
        # Enhanced candle body validation
        "body_ratio_thresholds": {
            "doji": 0.1,
            "weak": 0.3,
            "moderate": 0.5,
            "strong": 0.7
        },
        "enable_dynamic_body_validation": True,
        "max_wick_to_body_ratio": 4.0,
        "min_directional_body_ratio": 0.4
    },
    "permissive": {
        "min_volatility": 0.001,
        "min_body_ratio": 0.3,
        "min_confidence": 50.0,
        "min_volume_ratio": 0.7,
        "max_spread_ratio": 0.06,
        "enable_candle_pattern_filter": True,
        "enable_volatility_filter": True,
        "enable_confidence_filter": True,
        # Enhanced ATR validation
        "atr_volatility_thresholds": {
            "very_low": 0.0003,  # 0.03%
            "low": 0.5,
            "normal": 2.0,
            "high": 4.0
        },
        "enable_atr_regime_validation": True,
        # Enhanced candle body validation
        "body_ratio_thresholds": {
            "doji": 0.08,
            "weak": 0.2,
            "moderate": 0.4,
            "strong": 0.6
        },
        "enable_dynamic_body_validation": True,
        "max_wick_to_body_ratio": 6.0,
        "min_directional_body_ratio": 0.25
    }
}

# Weighted Voting System Configuration
WEIGHTED_VOTING_CONFIG = {
    "enable_normalized_scoring": True,
    "confidence_weight": 0.3,
    "strength_weight": 0.4,
    "consensus_weight": 0.3,
    "threshold_adjustment_factor": 0.15,  # How much vote quality affects threshold
    "min_consensus_ratio": 0.6,  # Minimum ratio for strong consensus
    "weak_signal_penalty": 0.5   # Penalty factor for weak signals
}

# Supabase
SUPABASE_TABLE = "signals"