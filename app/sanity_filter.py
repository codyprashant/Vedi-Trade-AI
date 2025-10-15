"""
Signal Sanity Filter Module

This module provides intelligent noise suppression and signal validation
to filter out low-quality or noisy trade setups while preserving valid
signal opportunities during strong market conditions.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalSanityFilter:
    """
    Validates generated signals before database insertion to prevent
    false positives and noisy trade setups.
    """
    
    def __init__(self,
                 min_volatility: float = 0.001,
                 min_body_ratio: float = 0.5,
                 min_confidence: float = 50.0,
                 min_volume_ratio: float = 0.8,
                 max_spread_ratio: float = 0.05,
                 enable_candle_pattern_filter: bool = True,
                 enable_volatility_filter: bool = True,
                 enable_confidence_filter: bool = True):
        """
        Initialize SignalSanityFilter with configurable parameters.
        
        Args:
            min_volatility: Minimum ATR ratio required (e.g., 0.001 = 0.1%)
            min_body_ratio: Minimum candle body to range ratio (0.0-1.0)
            min_confidence: Minimum direction confidence percentage
            min_volume_ratio: Minimum volume compared to average (if available)
            max_spread_ratio: Maximum spread as ratio of price
            enable_candle_pattern_filter: Enable candle pattern validation
            enable_volatility_filter: Enable volatility-based filtering
            enable_confidence_filter: Enable confidence-based filtering
        """
        self.min_volatility = min_volatility
        self.min_body_ratio = min_body_ratio
        self.min_confidence = min_confidence
        self.min_volume_ratio = min_volume_ratio
        self.max_spread_ratio = max_spread_ratio
        self.enable_candle_pattern_filter = enable_candle_pattern_filter
        self.enable_volatility_filter = enable_volatility_filter
        self.enable_confidence_filter = enable_confidence_filter
        
        # Statistics tracking
        self.filter_stats = {
            "total_signals": 0,
            "passed_signals": 0,
            "rejected_low_volatility": 0,
            "rejected_weak_candle": 0,
            "rejected_low_confidence": 0,
            "rejected_high_spread": 0,
            "rejected_low_volume": 0,
            "rejected_pattern_invalid": 0
        }
        
        logger.info(f"SignalSanityFilter initialized: min_volatility={min_volatility}, "
                   f"min_body_ratio={min_body_ratio}, min_confidence={min_confidence}")
    
    def validate_signal(self,
                       candle: Dict[str, float],
                       atr: float,
                       direction_confidence: float,
                       signal_strength: float,
                       symbol: str = "UNKNOWN",
                       additional_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a generated signal against multiple sanity checks.
        
        Args:
            candle: Current candle data with OHLC values
            atr: Average True Range value
            direction_confidence: Signal direction confidence (0-100)
            signal_strength: Overall signal strength (0-100)
            symbol: Trading symbol for context
            additional_data: Optional additional market data
            
        Returns:
            Tuple of (is_valid, rejection_reason, validation_metadata)
        """
        self.filter_stats["total_signals"] += 1
        
        try:
            validation_results = []
            rejection_reasons = []
            
            # Extract candle data
            open_price = candle.get('open', 0)
            high_price = candle.get('high', 0)
            low_price = candle.get('low', 0)
            close_price = candle.get('close', 0)
            volume = candle.get('volume', 0)
            
            # Calculate candle metrics
            body = abs(close_price - open_price)
            range_size = high_price - low_price
            body_ratio = body / range_size if range_size > 0 else 0
            atr_ratio = atr / close_price if close_price > 0 else 0
            
            # 1. Volatility Filter
            if self.enable_volatility_filter:
                volatility_valid, volatility_reason = self._check_volatility(atr_ratio)
                validation_results.append(volatility_valid)
                if not volatility_valid:
                    rejection_reasons.append(volatility_reason)
                    self.filter_stats["rejected_low_volatility"] += 1
            
            # 2. Candle Pattern Filter
            if self.enable_candle_pattern_filter:
                candle_valid, candle_reason = self._check_candle_pattern(
                    body_ratio, open_price, close_price, high_price, low_price
                )
                validation_results.append(candle_valid)
                if not candle_valid:
                    rejection_reasons.append(candle_reason)
                    self.filter_stats["rejected_weak_candle"] += 1
            
            # 3. Confidence Filter
            if self.enable_confidence_filter:
                confidence_valid, confidence_reason = self._check_confidence(direction_confidence)
                validation_results.append(confidence_valid)
                if not confidence_valid:
                    rejection_reasons.append(confidence_reason)
                    self.filter_stats["rejected_low_confidence"] += 1
            
            # 4. Spread Filter (if spread data available)
            if additional_data and 'spread' in additional_data:
                spread_valid, spread_reason = self._check_spread(
                    additional_data['spread'], close_price
                )
                validation_results.append(spread_valid)
                if not spread_valid:
                    rejection_reasons.append(spread_reason)
                    self.filter_stats["rejected_high_spread"] += 1
            
            # 5. Volume Filter (if volume data available)
            if volume > 0 and additional_data and 'avg_volume' in additional_data:
                volume_valid, volume_reason = self._check_volume(
                    volume, additional_data['avg_volume']
                )
                validation_results.append(volume_valid)
                if not volume_valid:
                    rejection_reasons.append(volume_reason)
                    self.filter_stats["rejected_low_volume"] += 1
            
            # 6. Pattern Consistency Check
            pattern_valid, pattern_reason = self._check_pattern_consistency(
                candle, signal_strength, additional_data
            )
            validation_results.append(pattern_valid)
            if not pattern_valid:
                rejection_reasons.append(pattern_reason)
                self.filter_stats["rejected_pattern_invalid"] += 1
            
            # Overall validation result
            is_valid = all(validation_results) if validation_results else False
            
            if is_valid:
                self.filter_stats["passed_signals"] += 1
                final_reason = "passed_all_filters"
            else:
                final_reason = "; ".join(rejection_reasons) if rejection_reasons else "unknown_rejection"
            
            # Create validation metadata
            metadata = {
                "symbol": symbol,
                "validation_timestamp": datetime.now().isoformat(),
                "is_valid": is_valid,
                "rejection_reason": final_reason,
                "candle_metrics": {
                    "body_ratio": round(body_ratio, 4),
                    "atr_ratio": round(atr_ratio, 6),
                    "range_size": round(range_size, 5),
                    "body_size": round(body, 5)
                },
                "filter_results": {
                    "volatility_check": validation_results[0] if len(validation_results) > 0 else None,
                    "candle_check": validation_results[1] if len(validation_results) > 1 else None,
                    "confidence_check": validation_results[2] if len(validation_results) > 2 else None
                },
                "signal_metrics": {
                    "direction_confidence": direction_confidence,
                    "signal_strength": signal_strength
                }
            }
            
            # Log validation result
            if is_valid:
                logger.debug(f"Signal PASSED sanity filter for {symbol}: "
                           f"body_ratio={body_ratio:.3f}, atr_ratio={atr_ratio:.6f}, "
                           f"confidence={direction_confidence:.1f}%")
            else:
                logger.info(f"Signal REJECTED for {symbol}: {final_reason}")
            
            return is_valid, final_reason, metadata
            
        except Exception as e:
            logger.error(f"Error in signal validation: {e}")
            error_metadata = {
                "error": str(e),
                "validation_failed": True,
                "symbol": symbol
            }
            return False, f"validation_error: {str(e)}", error_metadata
    
    def _check_volatility(self, atr_ratio: float) -> Tuple[bool, str]:
        """Check if volatility meets minimum requirements."""
        if atr_ratio < self.min_volatility:
            return False, f"low_volatility (atr_ratio={atr_ratio:.6f} < {self.min_volatility})"
        return True, "volatility_ok"
    
    def _check_candle_pattern(self, body_ratio: float, open_price: float, 
                            close_price: float, high_price: float, 
                            low_price: float) -> Tuple[bool, str]:
        """Check candle pattern validity."""
        # Body ratio check
        if body_ratio < self.min_body_ratio:
            return False, f"weak_candle_body (ratio={body_ratio:.3f} < {self.min_body_ratio})"
        
        # Doji filter (very small body relative to range)
        if body_ratio < 0.1:
            return False, "doji_candle_rejected"
        
        # Extreme wick filter (body should not be too small compared to wicks)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        body_size = abs(close_price - open_price)
        
        if body_size > 0:
            wick_to_body_ratio = (upper_wick + lower_wick) / body_size
            if wick_to_body_ratio > 5.0:  # Wicks more than 5x body size
                return False, f"excessive_wicks (ratio={wick_to_body_ratio:.2f})"
        
        return True, "candle_pattern_ok"
    
    def _check_confidence(self, direction_confidence: float) -> Tuple[bool, str]:
        """Check if direction confidence meets minimum requirements."""
        if direction_confidence < self.min_confidence:
            return False, f"low_confidence ({direction_confidence:.1f}% < {self.min_confidence}%)"
        return True, "confidence_ok"
    
    def _check_spread(self, spread: float, price: float) -> Tuple[bool, str]:
        """Check if spread is within acceptable limits."""
        spread_ratio = spread / price if price > 0 else 1.0
        if spread_ratio > self.max_spread_ratio:
            return False, f"high_spread (ratio={spread_ratio:.6f} > {self.max_spread_ratio})"
        return True, "spread_ok"
    
    def _check_volume(self, current_volume: float, avg_volume: float) -> Tuple[bool, str]:
        """Check if volume meets minimum requirements."""
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        if volume_ratio < self.min_volume_ratio:
            return False, f"low_volume (ratio={volume_ratio:.2f} < {self.min_volume_ratio})"
        return True, "volume_ok"
    
    def _check_pattern_consistency(self, candle: Dict[str, float], 
                                 signal_strength: float,
                                 additional_data: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Check for pattern consistency and logical coherence."""
        # Basic sanity checks
        open_price = candle.get('open', 0)
        high_price = candle.get('high', 0)
        low_price = candle.get('low', 0)
        close_price = candle.get('close', 0)
        
        # Price validation
        if not (low_price <= open_price <= high_price and 
                low_price <= close_price <= high_price):
            return False, "invalid_ohlc_relationship"
        
        # Signal strength consistency
        if signal_strength < 30:  # Very weak signals
            return False, f"signal_too_weak ({signal_strength:.1f}% < 30%)"
        
        return True, "pattern_consistent"
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics for analysis."""
        total = self.filter_stats["total_signals"]
        passed = self.filter_stats["passed_signals"]
        
        stats = self.filter_stats.copy()
        stats["pass_rate"] = (passed / total * 100) if total > 0 else 0
        stats["rejection_rate"] = ((total - passed) / total * 100) if total > 0 else 0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        for key in self.filter_stats:
            self.filter_stats[key] = 0
        logger.info("Filter statistics reset")
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update filter configuration dynamically."""
        for key, value in config_updates.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated filter config {key}: {old_value} → {value}")
            else:
                logger.warning(f"Unknown filter config parameter: {key}")


class MultiTimeframeNoiseFilter:
    """
    Lightweight temporal filter for multi-timeframe noise confirmation.
    Prevents rapid flip-flops in choppy markets.
    """
    
    def __init__(self, 
                 consistency_window: int = 2,
                 alignment_boost: float = 1.1,
                 flip_penalty: float = 0.8,
                 max_history: int = 10):
        """
        Initialize multi-timeframe noise filter.
        
        Args:
            consistency_window: Number of consecutive candles to check
            alignment_boost: Confidence multiplier for consistent direction
            flip_penalty: Confidence multiplier for direction changes
            max_history: Maximum signal history to maintain
        """
        self.consistency_window = consistency_window
        self.alignment_boost = alignment_boost
        self.flip_penalty = flip_penalty
        self.max_history = max_history
        
        # Signal history tracking
        self.signal_history: List[Dict[str, Any]] = []
        
        logger.info(f"MultiTimeframeNoiseFilter initialized: window={consistency_window}")
    
    def apply_temporal_filter(self, 
                            current_direction: str,
                            current_confidence: float,
                            timeframe: str = "M15",
                            h1_alignment: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Apply temporal filtering to adjust confidence based on signal history.
        
        Args:
            current_direction: Current signal direction (BUY/SELL)
            current_confidence: Current confidence level
            timeframe: Current timeframe
            h1_alignment: Whether H1 timeframe is aligned
            
        Returns:
            Tuple of (adjusted_confidence, filter_metadata)
        """
        try:
            # Add current signal to history
            current_signal = {
                "direction": current_direction,
                "confidence": current_confidence,
                "timestamp": datetime.now(),
                "timeframe": timeframe,
                "h1_alignment": h1_alignment
            }
            
            self.signal_history.append(current_signal)
            
            # Maintain history size
            if len(self.signal_history) > self.max_history:
                self.signal_history = self.signal_history[-self.max_history:]
            
            # Check for consistency
            adjusted_confidence = current_confidence
            consistency_factor = 1.0
            
            if len(self.signal_history) >= self.consistency_window:
                recent_signals = self.signal_history[-self.consistency_window:]
                
                # Check direction consistency
                directions = [sig["direction"] for sig in recent_signals]
                consistent_direction = all(d == current_direction for d in directions)
                
                if consistent_direction:
                    # Boost confidence for consistent direction
                    consistency_factor = self.alignment_boost
                    adjusted_confidence *= consistency_factor
                else:
                    # Penalize confidence for direction flip
                    consistency_factor = self.flip_penalty
                    adjusted_confidence *= consistency_factor
            
            # H1 alignment bonus
            h1_factor = 1.0
            if h1_alignment:
                h1_factor = 1.05  # 5% bonus for H1 alignment
                adjusted_confidence *= h1_factor
            
            # Cap confidence at 100%
            adjusted_confidence = min(100.0, adjusted_confidence)
            
            # Create filter metadata
            metadata = {
                "original_confidence": current_confidence,
                "adjusted_confidence": adjusted_confidence,
                "consistency_factor": consistency_factor,
                "h1_alignment_factor": h1_factor,
                "signal_history_length": len(self.signal_history),
                "recent_directions": [sig["direction"] for sig in self.signal_history[-3:]] if len(self.signal_history) >= 3 else [],
                "filter_applied": True
            }
            
            logger.debug(f"Temporal filter applied: {current_confidence:.1f}% → "
                        f"{adjusted_confidence:.1f}% (factor: {consistency_factor:.2f})")
            
            return adjusted_confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in temporal filtering: {e}")
            return current_confidence, {"error": str(e), "filter_applied": False}
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()
        logger.info("Signal history cleared")


class SignalSanityFilterFactory:
    """Factory for creating SignalSanityFilter instances with different configurations."""
    
    @staticmethod
    def create_strict() -> SignalSanityFilter:
        """Create a strict filter for high-quality signals only."""
        return SignalSanityFilter(
            min_volatility=0.0015,
            min_body_ratio=0.6,
            min_confidence=60.0,
            min_volume_ratio=1.0,
            max_spread_ratio=0.03
        )
    
    @staticmethod
    def create_permissive() -> SignalSanityFilter:
        """Create a permissive filter for more signal opportunities."""
        return SignalSanityFilter(
            min_volatility=0.0008,
            min_body_ratio=0.4,
            min_confidence=40.0,
            min_volume_ratio=0.6,
            max_spread_ratio=0.08
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> SignalSanityFilter:
        """Create filter from configuration dict."""
        return SignalSanityFilter(
            min_volatility=config.get('min_volatility', 0.001),
            min_body_ratio=config.get('min_body_ratio', 0.5),
            min_confidence=config.get('min_confidence', 50.0),
            min_volume_ratio=config.get('min_volume_ratio', 0.8),
            max_spread_ratio=config.get('max_spread_ratio', 0.05),
            enable_candle_pattern_filter=config.get('enable_candle_pattern_filter', True),
            enable_volatility_filter=config.get('enable_volatility_filter', True),
            enable_confidence_filter=config.get('enable_confidence_filter', True)
        )