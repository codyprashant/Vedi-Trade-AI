"""
Test suite for volatility adaptation in SignalSanityFilter.

This module tests the new volatility adaptation feature that dynamically adjusts
body ratio and confidence thresholds based on market volatility (ATR ratio).
"""

import pytest
from unittest.mock import Mock
from app.sanity_filter import SignalSanityFilter


class TestVolatilityAdaptation:
    """Test volatility adaptation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Volatility adaptation configuration for testing
        self.volatility_adaptation_config = {
            "enabled": True,
            "base_atr_lookback": 20,
            "adaptation_factors": {
                "very_low": {"body_ratio_factor": 0.7, "confidence_factor": 1.2},
                "low": {"body_ratio_factor": 0.85, "confidence_factor": 1.1},
                "normal": {"body_ratio_factor": 1.0, "confidence_factor": 1.0},
                "high": {"body_ratio_factor": 1.2, "confidence_factor": 0.9},
                "extreme": {"body_ratio_factor": 1.4, "confidence_factor": 0.8}
            },
            "volatility_ratio_thresholds": {
                "very_low": 0.005,   # ATR ratio < 0.005 = very low volatility
                "low": 0.008,        # 0.005 <= ATR ratio < 0.008 = low volatility  
                "normal": 0.015,     # 0.008 <= ATR ratio < 0.015 = normal volatility
                "high": 0.025,       # 0.015 <= ATR ratio < 0.025 = high volatility
                "extreme": 0.035     # ATR ratio >= 0.025 = extreme volatility
            }
        }
        
        # Base filter configuration
        self.base_config = {
            "min_volatility": 0.001,
            "min_body_ratio": 0.5,
            "min_confidence": 60.0,
            "enable_candle_pattern_filter": True,
            "enable_volatility_filter": True,
            "enable_confidence_filter": True,
            "enable_dynamic_body_validation": True,
            "volatility_adaptation": self.volatility_adaptation_config
        }
        
        # Test candle data
        self.test_candle = {
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.5,
            "volume": 1000
        }
    
    def test_volatility_regime_classification(self):
        """Test volatility regime classification based on ATR ratio."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Test very low volatility
        assert filter_instance._classify_volatility_regime(0.003) == "very_low"
        
        # Test low volatility
        assert filter_instance._classify_volatility_regime(0.007) == "low"
        
        # Test normal volatility
        assert filter_instance._classify_volatility_regime(0.012) == "normal"
        
        # Test high volatility
        assert filter_instance._classify_volatility_regime(0.020) == "high"
        
        # Test extreme volatility
        assert filter_instance._classify_volatility_regime(0.040) == "extreme"
    
    def test_threshold_adaptation_very_low_volatility(self):
        """Test threshold adaptation for very low volatility regime."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Very low volatility (ATR ratio = 0.003)
        adapted_body, adapted_confidence = filter_instance._adapt_thresholds_for_volatility(0.003)
        
        # Should relax body ratio and boost confidence
        expected_body = 0.5 * 0.7  # 0.35
        expected_confidence = 60.0 * 1.2  # 72.0
        
        assert abs(adapted_body - expected_body) < 0.001
        assert abs(adapted_confidence - expected_confidence) < 0.001
    
    def test_threshold_adaptation_high_volatility(self):
        """Test threshold adaptation for high volatility regime."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # High volatility (ATR ratio = 0.020)
        adapted_body, adapted_confidence = filter_instance._adapt_thresholds_for_volatility(0.020)
        
        # Should tighten body ratio and reduce confidence requirement
        expected_body = 0.5 * 1.2  # 0.6
        expected_confidence = 60.0 * 0.9  # 54.0
        
        assert abs(adapted_body - expected_body) < 0.001
        assert abs(adapted_confidence - expected_confidence) < 0.001
    
    def test_threshold_adaptation_normal_volatility(self):
        """Test threshold adaptation for normal volatility regime."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Normal volatility (ATR ratio = 0.012)
        adapted_body, adapted_confidence = filter_instance._adapt_thresholds_for_volatility(0.012)
        
        # Should keep original thresholds
        assert abs(adapted_body - 0.5) < 0.001
        assert abs(adapted_confidence - 60.0) < 0.001
    
    def test_adaptation_disabled(self):
        """Test behavior when volatility adaptation is disabled."""
        config_disabled = self.base_config.copy()
        config_disabled["volatility_adaptation"]["enabled"] = False
        
        filter_instance = SignalSanityFilter(**config_disabled)
        
        # Should return base thresholds regardless of ATR ratio
        adapted_body, adapted_confidence = filter_instance._adapt_thresholds_for_volatility(0.040)
        
        assert abs(adapted_body - 0.5) < 0.001
        assert abs(adapted_confidence - 60.0) < 0.001
    
    def test_signal_validation_with_adaptation_low_volatility(self):
        """Test signal validation with adaptation in low volatility environment."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Create a candle with high body ratio to avoid body ratio rejections
        candle = {
            "open": 100.0,
            "high": 103.0,
            "low": 99.5,
            "close": 102.8,  # Strong body ratio = 2.8/3.5 = 0.8
            "volume": 1000
        }
        
        # Low volatility ATR (ATR ratio = 0.4/102.8 = 0.0039, which is very low)
        atr = 0.4  # Very low ATR value
        
        is_valid, reason, metadata = filter_instance.validate_signal(
            candle=candle,
            atr=atr,
            direction_confidence=70.0,
            signal_strength=75.0,
            symbol="TEST"
        )
        
        # Should fail due to stricter confidence thresholds in very low volatility
        assert not is_valid
        assert "low_confidence" in reason
        assert metadata["volatility_adaptation"]["volatility_regime"] == "very_low"
        assert metadata["volatility_adaptation"]["confidence_adaptation_factor"] > 1.0
    
    def test_signal_validation_with_adaptation_high_volatility(self):
        """Test signal validation with adaptation in high volatility environment."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # High volatility candle with moderate body
        high_vol_candle = {
            "open": 100.0,
            "high": 103.0,
            "low": 97.0,
            "close": 102.0,  # Body ratio ~0.33
            "volume": 1000
        }
        
        # High volatility ATR (ATR ratio = 2.5/102 = 0.0245, which is high)
        atr = 2.5  # High ATR value
        
        is_valid, reason, metadata = filter_instance.validate_signal(
            candle=high_vol_candle,
            atr=atr,
            direction_confidence=55.0,  # Lower confidence
            signal_strength=75.0,
            symbol="TEST"
        )
        
        # Should fail due to tightened body ratio requirements
        assert not is_valid
        assert "body_ratio_too_small_adapted" in reason
        assert metadata["volatility_adaptation"]["volatility_regime"] == "high"
        assert metadata["volatility_adaptation"]["body_adaptation_factor"] > 1.0
    
    def test_confidence_adaptation_in_validation(self):
        """Test confidence threshold adaptation during validation."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Create a candle with high body ratio to avoid body ratio rejections
        strong_candle = {
            "open": 100.0,
            "high": 103.0,
            "low": 99.5,
            "close": 102.8,  # Strong body ratio = 2.8/3.5 = 0.8
            "volume": 1000
        }
        
        # Test with low volatility (should boost confidence requirement)
        # ATR ratio = 0.4/102.8 = 0.0039, which is very low
        atr_low = 0.4  # Very low volatility
        
        is_valid_low, reason_low, metadata_low = filter_instance.validate_signal(
            candle=strong_candle,
            atr=atr_low,
            direction_confidence=65.0,  # Would normally pass
            signal_strength=75.0,
            symbol="TEST"
        )
        
        # Should fail due to boosted confidence requirement (60 * 1.2 = 72)
        assert not is_valid_low
        assert "low_confidence" in reason_low
        
        # Test with high volatility (should reduce confidence requirement)
        # ATR ratio = 2.5/102.8 = 0.0243, which is high
        atr_high = 2.5  # High volatility
        
        is_valid_high, reason_high, metadata_high = filter_instance.validate_signal(
            candle=strong_candle,
            atr=atr_high,
            direction_confidence=55.0,  # Would normally fail
            signal_strength=75.0,
            symbol="TEST"
        )
        
        # Should pass due to reduced confidence requirement (60 * 0.9 = 54)
        assert is_valid_high
    
    def test_metadata_includes_adaptation_info(self):
        """Test that validation metadata includes volatility adaptation information."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # ATR ratio = 2.5/101.5 = 0.0246, which is high volatility
        is_valid, reason, metadata = filter_instance.validate_signal(
            candle=self.test_candle,
            atr=2.5,  # High volatility
            direction_confidence=70.0,
            signal_strength=80.0,
            symbol="TEST"
        )
        
        # Check adaptation metadata
        adaptation_meta = metadata["volatility_adaptation"]
        assert adaptation_meta["enabled"] is True
        assert adaptation_meta["volatility_regime"] == "high"
        assert "adapted_body_ratio" in adaptation_meta
        assert "adapted_confidence" in adaptation_meta
        assert "body_adaptation_factor" in adaptation_meta
        assert "confidence_adaptation_factor" in adaptation_meta
        
        # Check that factors are correctly calculated
        assert adaptation_meta["body_adaptation_factor"] == 1.2  # High volatility factor
        assert adaptation_meta["confidence_adaptation_factor"] == 0.9  # High volatility factor
    
    def test_statistics_tracking_adaptations(self):
        """Test that filter statistics track volatility adaptations."""
        filter_instance = SignalSanityFilter(**self.base_config)
        
        # Perform several validations with different ATR values
        # ATR ratios will be: 0.3/101.5=0.003 (very_low), 1.0/101.5=0.01 (low), 2.5/101.5=0.025 (high)
        for atr_value in [0.3, 1.0, 2.5]:
            filter_instance.validate_signal(
                candle=self.test_candle,
                atr=atr_value,
                direction_confidence=70.0,
                signal_strength=80.0,
                symbol="TEST"
            )
        
        stats = filter_instance.get_filter_statistics()
        assert "volatility_adaptations" in stats
        assert stats["volatility_adaptations"] == 3  # One for each validation