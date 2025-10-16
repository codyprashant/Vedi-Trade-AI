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
                 enable_confidence_filter: bool = True,
                 # Enhanced ATR validation parameters
                 atr_volatility_thresholds: Dict[str, float] = None,
                 enable_atr_regime_validation: bool = True,
                 # Enhanced candle body ratio parameters
                 body_ratio_thresholds: Dict[str, float] = None,
                 enable_dynamic_body_validation: bool = True,
                 max_wick_to_body_ratio: float = 5.0,
                 min_directional_body_ratio: float = 0.3,
                 # Volatility adaptation parameters
                 volatility_adaptation: Dict[str, Any] = None):
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
            atr_volatility_thresholds: ATR regime thresholds for enhanced validation
            enable_atr_regime_validation: Enable ATR regime-based validation
            body_ratio_thresholds: Body ratio thresholds for different market conditions
            enable_dynamic_body_validation: Enable dynamic body ratio validation
            max_wick_to_body_ratio: Maximum allowed wick-to-body ratio
            min_directional_body_ratio: Minimum body ratio for directional signals
        """
        self.min_volatility = min_volatility
        self.min_body_ratio = min_body_ratio
        self.min_confidence = min_confidence
        self.min_volume_ratio = min_volume_ratio
        self.max_spread_ratio = max_spread_ratio
        self.enable_candle_pattern_filter = enable_candle_pattern_filter
        self.enable_volatility_filter = enable_volatility_filter
        self.enable_confidence_filter = enable_confidence_filter
        
        # Enhanced ATR validation setup
        self.enable_atr_regime_validation = enable_atr_regime_validation
        self.atr_volatility_thresholds = atr_volatility_thresholds or {
            'very_low': 0.0005,    # 0.05%
            'low': 0.001,          # 0.1%
            'normal': 0.002,       # 0.2%
            'high': 0.004,         # 0.4%
            'extreme': 0.008       # 0.8%
        }
        
        # Enhanced candle body ratio validation setup
        self.enable_dynamic_body_validation = enable_dynamic_body_validation
        self.body_ratio_thresholds = body_ratio_thresholds or {
            'doji': 0.1,           # Very small body
            'weak': 0.3,           # Weak directional signal
            'moderate': 0.5,       # Moderate directional signal
            'strong': 0.7,         # Strong directional signal
            'very_strong': 0.85    # Very strong directional signal
        }
        self.max_wick_to_body_ratio = max_wick_to_body_ratio
        self.min_directional_body_ratio = min_directional_body_ratio
        
        # Volatility adaptation setup
        self.volatility_adaptation = volatility_adaptation or {
            "enabled": False,
            "base_atr_lookback": 20,
            "adaptation_factors": {
                "very_low": {"body_ratio_factor": 1.0, "confidence_factor": 1.0},
                "low": {"body_ratio_factor": 1.0, "confidence_factor": 1.0},
                "normal": {"body_ratio_factor": 1.0, "confidence_factor": 1.0},
                "high": {"body_ratio_factor": 1.0, "confidence_factor": 1.0},
                "extreme": {"body_ratio_factor": 1.0, "confidence_factor": 1.0}
            },
            "volatility_ratio_thresholds": {
                "very_low": 0.5, "low": 0.8, "normal": 1.5, "high": 2.5, "extreme": 3.5
            }
        }
        
        # Store original thresholds for adaptation
        self.base_min_body_ratio = min_body_ratio
        self.base_min_confidence = min_confidence
        
        # Statistics tracking
        self.filter_stats = {
            "total_signals": 0,
            "passed_signals": 0,
            "rejected_low_volatility": 0,
            "rejected_weak_candle": 0,
            "rejected_low_confidence": 0,
            "rejected_high_spread": 0,
            "rejected_low_volume": 0,
            "rejected_pattern_invalid": 0,
            "volatility_adaptations": 0
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
            
            # Apply volatility adaptation to adjust thresholds
            adapted_body_ratio, adapted_confidence = self._adapt_thresholds_for_volatility(atr_ratio)
            
            # Extract signal direction from additional data if available
            signal_direction = additional_data.get('signal_direction') if additional_data else None
            
            # 1. Enhanced ATR/Volatility Filter
            if self.enable_volatility_filter:
                if self.enable_atr_regime_validation:
                    # Use enhanced ATR validation with regime classification
                    volatility_valid, volatility_reason, atr_metadata = self._enhanced_atr_validation(
                        atr_ratio, signal_strength
                    )
                else:
                    # Use basic volatility validation
                    volatility_valid, volatility_reason = self._check_volatility(atr_ratio)
                    atr_metadata = {'basic_validation': True, 'atr_ratio': atr_ratio}
                
                validation_results.append(volatility_valid)
                if not volatility_valid:
                    rejection_reasons.append(volatility_reason)
                    self.filter_stats["rejected_low_volatility"] += 1
            else:
                atr_metadata = {'validation_disabled': True}
            
            # 2. Enhanced Candle Pattern Filter (with volatility adaptation)
            if self.enable_candle_pattern_filter:
                if self.enable_dynamic_body_validation:
                    # Use enhanced candle validation with volatility-adapted dynamic thresholds
                    candle_valid, candle_reason, candle_metadata = self._enhanced_candle_validation_adapted(
                        candle, signal_strength, adapted_body_ratio, signal_direction
                    )
                else:
                    # Use basic candle pattern validation with adapted threshold
                    candle_valid, candle_reason = self._check_candle_pattern_adapted(
                        body_ratio, adapted_body_ratio, open_price, close_price, high_price, low_price
                    )
                    candle_metadata = {'basic_validation': True, 'body_ratio': body_ratio, 'adapted_body_ratio': adapted_body_ratio}
                
                validation_results.append(candle_valid)
                if not candle_valid:
                    rejection_reasons.append(candle_reason)
                    self.filter_stats["rejected_weak_candle"] += 1
            else:
                candle_metadata = {'validation_disabled': True}
            
            # 3. Confidence Filter (with volatility adaptation)
            if self.enable_confidence_filter:
                confidence_valid, confidence_reason = self._check_confidence_adapted(
                    direction_confidence, adapted_confidence
                )
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
            
            # Create comprehensive validation metadata
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
                    "signal_strength": signal_strength,
                    "signal_direction": signal_direction
                },
                # Enhanced validation metadata
                "enhanced_atr_analysis": atr_metadata,
                "enhanced_candle_analysis": candle_metadata,
                "validation_features": {
                    "atr_regime_validation": self.enable_atr_regime_validation,
                    "dynamic_body_validation": self.enable_dynamic_body_validation,
                    "enhanced_filters_used": self.enable_atr_regime_validation or self.enable_dynamic_body_validation,
                    "volatility_adaptation_enabled": self.volatility_adaptation["enabled"]
                },
                "volatility_adaptation": {
                    "enabled": self.volatility_adaptation["enabled"],
                    "volatility_regime": self._classify_volatility_regime(atr_ratio) if self.volatility_adaptation["enabled"] else None,
                    "adapted_body_ratio": adapted_body_ratio,
                    "adapted_confidence": adapted_confidence,
                    "base_body_ratio": self.base_min_body_ratio,
                    "base_confidence": self.base_min_confidence,
                    "body_adaptation_factor": adapted_body_ratio / self.base_min_body_ratio,
                    "confidence_adaptation_factor": adapted_confidence / self.base_min_confidence
                }
            }
            
            # Enhanced logging for validation result
            if is_valid:
                logger.info(f"Sanity Filter PASSED for {symbol}: "
                           f"body_ratio={body_ratio:.3f}, atr_ratio={atr_ratio:.6f}, "
                           f"confidence={direction_confidence:.1f}%, strength={signal_strength:.1f}%")
                logger.debug(f"Sanity Filter Details for {symbol}: "
                           f"volatility_check={validation_results[0] if len(validation_results) > 0 else 'N/A'}, "
                           f"candle_check={validation_results[1] if len(validation_results) > 1 else 'N/A'}, "
                           f"confidence_check={validation_results[2] if len(validation_results) > 2 else 'N/A'}")
            else:
                logger.warning(f"Sanity Filter REJECTED for {symbol}: {final_reason}")
                logger.debug(f"Sanity Filter Rejection Details for {symbol}: "
                           f"body_ratio={body_ratio:.3f}, atr_ratio={atr_ratio:.6f}, "
                           f"confidence={direction_confidence:.1f}%, strength={signal_strength:.1f}%, "
                           f"failed_checks={len([r for r in validation_results if not r])}/{len(validation_results)}")
            
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
    
    def _classify_volatility_regime(self, atr_ratio: float) -> str:
        """
        Classify the current volatility regime based on ATR ratio.
        
        Args:
            atr_ratio: Current ATR ratio (current ATR / average ATR)
            
        Returns:
            str: Volatility regime classification
        """
        thresholds = self.volatility_adaptation["volatility_ratio_thresholds"]
        
        if atr_ratio < thresholds["very_low"]:
            return "very_low"
        elif atr_ratio < thresholds["low"]:
            return "low"
        elif atr_ratio < thresholds["normal"]:
            return "normal"
        elif atr_ratio < thresholds["high"]:
            return "high"
        else:
            return "extreme"
    
    def _adapt_thresholds_for_volatility(self, atr_ratio: float) -> Tuple[float, float]:
        """
        Adapt body ratio and confidence thresholds based on current volatility regime.
        
        Args:
            atr_ratio: Current ATR ratio (current ATR / average ATR)
            
        Returns:
            Tuple[float, float]: Adapted (min_body_ratio, min_confidence)
        """
        if not self.volatility_adaptation["enabled"]:
            return self.base_min_body_ratio, self.base_min_confidence
        
        regime = self._classify_volatility_regime(atr_ratio)
        factors = self.volatility_adaptation["adaptation_factors"][regime]
        
        adapted_body_ratio = self.base_min_body_ratio * factors["body_ratio_factor"]
        adapted_confidence = self.base_min_confidence * factors["confidence_factor"]
        
        self.filter_stats["volatility_adaptations"] += 1
        
        return adapted_body_ratio, adapted_confidence
    
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
    
    def _check_candle_pattern_adapted(self, body_ratio: float, adapted_body_ratio: float, 
                                    open_price: float, close_price: float, 
                                    high_price: float, low_price: float) -> Tuple[bool, str]:
        """Check candle pattern validity with volatility-adapted thresholds."""
        # Body ratio check with adapted threshold
        if body_ratio < adapted_body_ratio:
            return False, f"weak_candle_body_adapted (ratio={body_ratio:.3f} < {adapted_body_ratio:.3f})"
        
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
        
        return True, "candle_pattern_ok_adapted"
    
    def _check_confidence(self, direction_confidence: float) -> Tuple[bool, str]:
        """Check if direction confidence meets minimum threshold."""
        if direction_confidence < self.min_confidence:
            return False, f"low_confidence_{direction_confidence:.1f}_min_{self.min_confidence}"
        return True, "confidence_ok"
    
    def _check_confidence_adapted(self, direction_confidence: float, adapted_min_confidence: float) -> Tuple[bool, str]:
        """Check if direction confidence meets volatility-adapted minimum threshold."""
        if direction_confidence < adapted_min_confidence:
            return False, f"low_confidence_{direction_confidence:.1f}_adapted_min_{adapted_min_confidence:.1f}"
        return True, "confidence_ok_adapted"
    
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
    
    def _classify_atr_regime(self, atr_ratio: float) -> str:
        """Classify ATR ratio into volatility regime."""
        if atr_ratio <= self.atr_volatility_thresholds['very_low']:
            return 'very_low'
        elif atr_ratio <= self.atr_volatility_thresholds['low']:
            return 'low'
        elif atr_ratio <= self.atr_volatility_thresholds['normal']:
            return 'normal'
        elif atr_ratio <= self.atr_volatility_thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def _enhanced_atr_validation(self, atr_ratio: float, signal_strength: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced ATR validation with regime-based thresholds."""
        regime = self._classify_atr_regime(atr_ratio)
        
        # Regime-specific validation logic
        validation_passed = True
        rejection_reason = "atr_validation_passed"
        
        if regime == 'very_low':
            # Very low volatility - require higher signal strength
            if signal_strength < 70:
                validation_passed = False
                rejection_reason = f"very_low_volatility_insufficient_strength (atr={atr_ratio:.6f}, strength={signal_strength:.1f}% < 70%)"
        elif regime == 'low':
            # Low volatility - require moderate signal strength
            if signal_strength < 60:
                validation_passed = False
                rejection_reason = f"low_volatility_insufficient_strength (atr={atr_ratio:.6f}, strength={signal_strength:.1f}% < 60%)"
        elif regime == 'extreme':
            # Extreme volatility - be more cautious
            if signal_strength < 55:
                validation_passed = False
                rejection_reason = f"extreme_volatility_insufficient_strength (atr={atr_ratio:.6f}, strength={signal_strength:.1f}% < 55%)"
        
        # Basic minimum volatility check
        if atr_ratio < self.min_volatility:
            validation_passed = False
            rejection_reason = f"below_minimum_volatility (atr={atr_ratio:.6f} < {self.min_volatility})"
        
        metadata = {
            'atr_ratio': atr_ratio,
            'volatility_regime': regime,
            'regime_thresholds': self.atr_volatility_thresholds,
            'validation_passed': validation_passed,
            'rejection_reason': rejection_reason
        }
        
        return validation_passed, rejection_reason, metadata
    
    def _classify_candle_strength(self, body_ratio: float) -> str:
        """Classify candle strength based on body ratio."""
        if body_ratio <= self.body_ratio_thresholds['doji']:
            return 'doji'
        elif body_ratio <= self.body_ratio_thresholds['weak']:
            return 'weak'
        elif body_ratio <= self.body_ratio_thresholds['moderate']:
            return 'moderate'
        elif body_ratio <= self.body_ratio_thresholds['strong']:
            return 'strong'
        else:
            return 'very_strong'
    
    def _enhanced_candle_validation(self, candle: Dict[str, float], signal_strength: float, 
                                  signal_direction: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced candle body ratio validation with dynamic thresholds."""
        open_price = candle.get('open', 0)
        high_price = candle.get('high', 0)
        low_price = candle.get('low', 0)
        close_price = candle.get('close', 0)
        
        # Calculate candle metrics
        body = abs(close_price - open_price)
        range_size = high_price - low_price
        body_ratio = body / range_size if range_size > 0 else 0
        
        # Calculate wick metrics
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_wick = upper_wick + lower_wick
        wick_to_body_ratio = total_wick / body if body > 0 else float('inf')
        
        # Classify candle strength
        candle_strength = self._classify_candle_strength(body_ratio)
        
        # Validation logic
        validation_passed = True
        rejection_reasons = []
        
        # 1. Basic body ratio check
        if body_ratio < self.min_body_ratio:
            validation_passed = False
            rejection_reasons.append(f"body_ratio_too_small ({body_ratio:.3f} < {self.min_body_ratio})")
        
        # 2. Doji filter
        if candle_strength == 'doji':
            validation_passed = False
            rejection_reasons.append(f"doji_candle_rejected (body_ratio={body_ratio:.3f})")
        
        # 3. Directional signal validation
        if signal_direction and body_ratio < self.min_directional_body_ratio:
            validation_passed = False
            rejection_reasons.append(f"insufficient_directional_body ({body_ratio:.3f} < {self.min_directional_body_ratio})")
        
        # 4. Wick-to-body ratio check
        if wick_to_body_ratio > self.max_wick_to_body_ratio:
            validation_passed = False
            rejection_reasons.append(f"excessive_wicks (ratio={wick_to_body_ratio:.2f} > {self.max_wick_to_body_ratio})")
        
        # 5. Signal strength consistency with candle strength
        if candle_strength == 'weak' and signal_strength > 80:
            # Weak candle with very strong signal - potential false signal
            validation_passed = False
            rejection_reasons.append(f"candle_signal_mismatch (weak_candle={body_ratio:.3f}, strong_signal={signal_strength:.1f}%)")
        elif candle_strength == 'very_strong' and signal_strength < 40:
            # Very strong candle with weak signal - potential noise
            validation_passed = False
            rejection_reasons.append(f"candle_signal_mismatch (strong_candle={body_ratio:.3f}, weak_signal={signal_strength:.1f}%)")
        
        # 6. Price action validation
        is_bullish_candle = close_price > open_price
        is_bearish_candle = close_price < open_price
        
        if signal_direction == 'BUY' and is_bearish_candle and body_ratio > 0.6:
            # Strong bearish candle with buy signal
            validation_passed = False
            rejection_reasons.append(f"bearish_candle_buy_signal (body_ratio={body_ratio:.3f})")
        elif signal_direction == 'SELL' and is_bullish_candle and body_ratio > 0.6:
            # Strong bullish candle with sell signal
            validation_passed = False
            rejection_reasons.append(f"bullish_candle_sell_signal (body_ratio={body_ratio:.3f})")
        
        rejection_reason = "; ".join(rejection_reasons) if rejection_reasons else "candle_validation_passed"
        
        metadata = {
            'body_ratio': body_ratio,
            'candle_strength': candle_strength,
            'wick_to_body_ratio': wick_to_body_ratio,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'is_bullish': is_bullish_candle,
            'is_bearish': is_bearish_candle,
            'validation_passed': validation_passed,
            'rejection_reasons': rejection_reasons,
            'strength_thresholds': self.body_ratio_thresholds
        }
        
        return validation_passed, rejection_reason, metadata
    
    def _enhanced_candle_validation_adapted(self, candle: Dict[str, float], signal_strength: float, 
                                          adapted_body_ratio: float, signal_direction: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """Enhanced candle body ratio validation with volatility-adapted dynamic thresholds."""
        open_price = candle.get('open', 0)
        high_price = candle.get('high', 0)
        low_price = candle.get('low', 0)
        close_price = candle.get('close', 0)
        
        # Calculate candle metrics
        body = abs(close_price - open_price)
        range_size = high_price - low_price
        body_ratio = body / range_size if range_size > 0 else 0
        
        # Calculate wick metrics
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_wick = upper_wick + lower_wick
        wick_to_body_ratio = total_wick / body if body > 0 else float('inf')
        
        # Classify candle strength
        candle_strength = self._classify_candle_strength(body_ratio)
        
        # Validation logic with adapted thresholds
        validation_passed = True
        rejection_reasons = []
        
        # 1. Adapted body ratio check
        if body_ratio < adapted_body_ratio:
            validation_passed = False
            rejection_reasons.append(f"body_ratio_too_small_adapted ({body_ratio:.3f} < {adapted_body_ratio:.3f})")
        
        # 2. Doji filter (still use base threshold for doji detection)
        if candle_strength == 'doji':
            validation_passed = False
            rejection_reasons.append(f"doji_candle_rejected (body_ratio={body_ratio:.3f})")
        
        # 3. Directional signal validation (use adapted threshold)
        adapted_directional_ratio = self.min_directional_body_ratio * (adapted_body_ratio / self.base_min_body_ratio)
        if signal_direction and body_ratio < adapted_directional_ratio:
            validation_passed = False
            rejection_reasons.append(f"insufficient_directional_body_adapted ({body_ratio:.3f} < {adapted_directional_ratio:.3f})")
        
        # 4. Wick-to-body ratio check (unchanged)
        if wick_to_body_ratio > self.max_wick_to_body_ratio:
            validation_passed = False
            rejection_reasons.append(f"excessive_wicks (ratio={wick_to_body_ratio:.2f} > {self.max_wick_to_body_ratio})")
        
        # 5. Signal strength consistency with candle strength (unchanged)
        if candle_strength == 'weak' and signal_strength > 80:
            validation_passed = False
            rejection_reasons.append(f"candle_signal_mismatch (weak_candle={body_ratio:.3f}, strong_signal={signal_strength:.1f}%)")
        
        # Create detailed metadata
        metadata = {
            "candle_strength": candle_strength,
            "body_ratio": round(body_ratio, 4),
            "adapted_body_ratio": round(adapted_body_ratio, 4),
            "adaptation_factor": round(adapted_body_ratio / self.base_min_body_ratio, 3),
            "wick_to_body_ratio": round(wick_to_body_ratio, 2),
            "upper_wick": round(upper_wick, 5),
            "lower_wick": round(lower_wick, 5),
            "validation_passed": validation_passed,
            "rejection_reasons": rejection_reasons,
            "volatility_adapted": True
        }
        
        return validation_passed, "; ".join(rejection_reasons) if rejection_reasons else "candle_valid_adapted", metadata
    
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
            max_spread_ratio=0.03,
            enable_atr_regime_validation=True,
            enable_dynamic_body_validation=True,
            min_directional_body_ratio=0.4,
            max_wick_to_body_ratio=4.0
        )
    
    @staticmethod
    def create_permissive() -> SignalSanityFilter:
        """Create a permissive filter for more signal opportunities."""
        return SignalSanityFilter(
            min_volatility=0.0008,
            min_body_ratio=0.4,
            min_confidence=40.0,
            min_volume_ratio=0.6,
            max_spread_ratio=0.08,
            enable_atr_regime_validation=True,
            enable_dynamic_body_validation=True,
            min_directional_body_ratio=0.25,
            max_wick_to_body_ratio=6.0
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
            enable_confidence_filter=config.get('enable_confidence_filter', True),
            # Enhanced parameters
            atr_volatility_thresholds=config.get('atr_volatility_thresholds'),
            enable_atr_regime_validation=config.get('enable_atr_regime_validation', True),
            body_ratio_thresholds=config.get('body_ratio_thresholds'),
            enable_dynamic_body_validation=config.get('enable_dynamic_body_validation', True),
            max_wick_to_body_ratio=config.get('max_wick_to_body_ratio', 5.0),
            min_directional_body_ratio=config.get('min_directional_body_ratio', 0.3),
            # Volatility adaptation
            volatility_adaptation=config.get('volatility_adaptation')
        )