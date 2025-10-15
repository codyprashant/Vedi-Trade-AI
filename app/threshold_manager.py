"""
Dynamic Adaptive Threshold Manager

This module provides intelligent threshold adjustment based on market conditions,
volatility, momentum, and price deviation from key moving averages.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ThresholdManager:
    """
    Computes adaptive thresholds per timeframe and symbol based on:
    - Volatility (ATR ratio)
    - Momentum spread (RSI deviation from 50, MACD histogram magnitude)
    - Price deviation from key MAs (EMA55, SMA200)
    """
    
    def __init__(self, 
                 base_threshold: float = 60.0,
                 atr_factor: float = 0.002,
                 min_threshold: float = 45.0,
                 max_threshold: float = 75.0,
                 volatility_weight: float = 1.0,
                 momentum_weight: float = 1.0,
                 trend_weight: float = 0.8):
        """
        Initialize ThresholdManager with configurable parameters.
        
        Args:
            base_threshold: Base signal strength threshold (default: 60%)
            atr_factor: ATR sensitivity factor for volatility adjustment
            min_threshold: Minimum allowed threshold (prevents over-tightening)
            max_threshold: Maximum allowed threshold (prevents over-loosening)
            volatility_weight: Weight for volatility-based adjustments
            momentum_weight: Weight for momentum-based adjustments
            trend_weight: Weight for trend-based adjustments
        """
        self.base_threshold = base_threshold
        self.atr_factor = atr_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.volatility_weight = volatility_weight
        self.momentum_weight = momentum_weight
        self.trend_weight = trend_weight
        
        logger.info(f"ThresholdManager initialized: base={base_threshold}, "
                   f"range=[{min_threshold}, {max_threshold}], atr_factor={atr_factor}")
    
    def compute_adaptive_threshold(self, 
                                 atr_ratio: float,
                                 rsi_deviation: float,
                                 macd_histogram: float,
                                 price_ma_deviation: float = 0.0,
                                 symbol: str = "UNKNOWN",
                                 timeframe: str = "M15") -> Tuple[float, Dict[str, Any]]:
        """
        Compute adaptive threshold based on market conditions.
        
        Args:
            atr_ratio: Current ATR as ratio of price (e.g., 0.002 = 0.2%)
            rsi_deviation: Absolute deviation of RSI from 50 (0-50 range)
            macd_histogram: MACD histogram magnitude (absolute value)
            price_ma_deviation: Price deviation from key MAs (percentage)
            symbol: Trading symbol for logging
            timeframe: Timeframe for context
            
        Returns:
            Tuple of (adaptive_threshold, metadata_dict)
        """
        try:
            # Calculate individual adjustment components
            volatility_adj = self._calculate_volatility_adjustment(atr_ratio)
            momentum_adj = self._calculate_momentum_adjustment(rsi_deviation, macd_histogram)
            trend_adj = self._calculate_trend_adjustment(price_ma_deviation)
            
            # Combine adjustments with weights
            total_adjustment = (
                volatility_adj * self.volatility_weight +
                momentum_adj * self.momentum_weight +
                trend_adj * self.trend_weight
            )
            
            # Apply adjustment to base threshold
            adaptive_threshold = self.base_threshold + total_adjustment
            
            # Clamp to min/max bounds
            clamped_threshold = max(self.min_threshold, 
                                  min(self.max_threshold, adaptive_threshold))
            
            # Create metadata for transparency
            metadata = {
                "base_threshold": self.base_threshold,
                "volatility_adjustment": volatility_adj,
                "momentum_adjustment": momentum_adj,
                "trend_adjustment": trend_adj,
                "total_adjustment": total_adjustment,
                "raw_threshold": adaptive_threshold,
                "final_threshold": clamped_threshold,
                "was_clamped": clamped_threshold != adaptive_threshold,
                "atr_ratio": atr_ratio,
                "rsi_deviation": rsi_deviation,
                "macd_histogram": macd_histogram,
                "price_ma_deviation": price_ma_deviation,
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            logger.debug(f"Adaptive threshold for {symbol} {timeframe}: "
                        f"{clamped_threshold:.1f}% (base: {self.base_threshold}%, "
                        f"adj: {total_adjustment:+.1f}%)")
            
            return clamped_threshold, metadata
            
        except Exception as e:
            logger.error(f"Error computing adaptive threshold: {e}")
            # Return base threshold as fallback
            fallback_metadata = {
                "base_threshold": self.base_threshold,
                "final_threshold": self.base_threshold,
                "error": str(e),
                "fallback_used": True
            }
            return self.base_threshold, fallback_metadata
    
    def _calculate_volatility_adjustment(self, atr_ratio: float) -> float:
        """
        Calculate threshold adjustment based on volatility.
        
        During low volatility → tighten thresholds (positive adjustment)
        During high volatility → loosen thresholds (negative adjustment)
        """
        # Normalize ATR ratio (typical range: 0.001-0.010)
        normalized_atr = atr_ratio / self.atr_factor
        
        # Low volatility (< 1.0) → tighten by up to +5%
        # High volatility (> 1.0) → loosen by up to -5%
        if normalized_atr < 1.0:
            # Tighten threshold for low volatility
            adjustment = (1.0 - normalized_atr) * 5.0
        else:
            # Loosen threshold for high volatility
            adjustment = -(normalized_atr - 1.0) * 3.0
        
        # Cap adjustment at ±8%
        return max(-8.0, min(8.0, adjustment))
    
    def _calculate_momentum_adjustment(self, rsi_deviation: float, macd_histogram: float) -> float:
        """
        Calculate threshold adjustment based on momentum indicators.
        
        Strong momentum → loosen thresholds (easier to trigger signals)
        Weak momentum → tighten thresholds (require stronger confirmation)
        """
        # RSI deviation component (0-50 range)
        rsi_component = (rsi_deviation / 50.0) * 3.0  # Up to ±3%
        
        # MACD histogram component (normalize to reasonable range)
        macd_normalized = min(abs(macd_histogram) * 1000, 10.0)  # Cap at 10
        macd_component = (macd_normalized / 10.0) * 2.0  # Up to ±2%
        
        # Strong momentum → negative adjustment (loosen threshold)
        momentum_strength = rsi_component + macd_component
        adjustment = -momentum_strength  # Negative = loosen
        
        # Cap adjustment at ±5%
        return max(-5.0, min(5.0, adjustment))
    
    def _calculate_trend_adjustment(self, price_ma_deviation: float) -> float:
        """
        Calculate threshold adjustment based on price deviation from key MAs.
        
        Price far from MAs → tighten thresholds (require stronger signals)
        Price near MAs → neutral adjustment
        """
        # Normalize deviation (percentage)
        normalized_deviation = abs(price_ma_deviation)
        
        # Large deviation → tighten threshold
        if normalized_deviation > 2.0:  # > 2% deviation
            adjustment = min(normalized_deviation, 5.0)  # Up to +5%
        else:
            adjustment = 0.0  # No adjustment for small deviations
        
        return adjustment
    
    def get_threshold_for_conditions(self, market_conditions: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Convenience method to compute threshold from market conditions dict.
        
        Args:
            market_conditions: Dict containing market data
            
        Returns:
            Tuple of (threshold, metadata)
        """
        atr_ratio = market_conditions.get('atr_ratio', 0.002)
        rsi = market_conditions.get('rsi', 50.0)
        rsi_deviation = abs(rsi - 50.0)
        macd_histogram = abs(market_conditions.get('macd_histogram', 0.0))
        price_ma_deviation = market_conditions.get('price_ma_deviation', 0.0)
        symbol = market_conditions.get('symbol', 'UNKNOWN')
        timeframe = market_conditions.get('timeframe', 'M15')
        
        return self.compute_adaptive_threshold(
            atr_ratio=atr_ratio,
            rsi_deviation=rsi_deviation,
            macd_histogram=macd_histogram,
            price_ma_deviation=price_ma_deviation,
            symbol=symbol,
            timeframe=timeframe
        )
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update threshold manager configuration dynamically.
        
        Args:
            config_updates: Dict of configuration parameters to update
        """
        for key, value in config_updates.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated {key}: {old_value} → {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")


class ThresholdManagerFactory:
    """Factory for creating ThresholdManager instances with different configurations."""
    
    @staticmethod
    def create_conservative() -> ThresholdManager:
        """Create a conservative threshold manager (higher thresholds)."""
        return ThresholdManager(
            base_threshold=65.0,
            min_threshold=55.0,
            max_threshold=80.0,
            volatility_weight=1.2,
            momentum_weight=0.8
        )
    
    @staticmethod
    def create_aggressive() -> ThresholdManager:
        """Create an aggressive threshold manager (lower thresholds)."""
        return ThresholdManager(
            base_threshold=55.0,
            min_threshold=40.0,
            max_threshold=70.0,
            volatility_weight=0.8,
            momentum_weight=1.2
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> ThresholdManager:
        """Create threshold manager from configuration dict."""
        return ThresholdManager(
            base_threshold=config.get('base_threshold', 60.0),
            atr_factor=config.get('atr_factor', 0.002),
            min_threshold=config.get('min_threshold', 45.0),
            max_threshold=config.get('max_threshold', 75.0),
            volatility_weight=config.get('volatility_weight', 1.0),
            momentum_weight=config.get('momentum_weight', 1.0),
            trend_weight=config.get('trend_weight', 0.8)
        )