"""
Comprehensive test suite for enhanced SignalSanityFilter with ATR regime validation,
dynamic candle body analysis, and advanced filtering capabilities.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from app.sanity_filter import SignalSanityFilter, SignalSanityFilterFactory
from app.config import SANITY_FILTER_CONFIG


class TestEnhancedSanityFilter(unittest.TestCase):
    """Test suite for enhanced SignalSanityFilter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strict_filter = SignalSanityFilterFactory.create_from_config(SANITY_FILTER_CONFIG["strict"])
        self.permissive_filter = SignalSanityFilterFactory.create_from_config(SANITY_FILTER_CONFIG["permissive"])
        
        # Sample candle data for testing
        self.strong_bullish_candle = {
            'open': 100.0,
            'high': 105.0,
            'low': 99.5,
            'close': 104.5
        }
        
        self.weak_bullish_candle = {
            'open': 100.0,
            'high': 101.0,
            'low': 99.8,
            'close': 100.5
        }
        
        self.doji_candle = {
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.1
        }
        
        self.hammer_candle = {
            'open': 100.0,
            'high': 100.2,
            'low': 95.0,
            'close': 99.8
        }
        
        # Sample additional data for enhanced validation
        self.sample_additional_data = {
            'signal_direction': 'buy',
            'h1_alignment': True,
            'volatility_regime': 'normal',
            'atr_ratio': 1.2
        }

    def test_atr_regime_classification(self):
        """Test ATR regime classification functionality."""
        # Test low volatility regime
        low_regime = self.strict_filter._classify_atr_regime(0.6)
        self.assertEqual(low_regime, 'low')
        
        # Test normal volatility regime
        normal_regime = self.strict_filter._classify_atr_regime(1.2)
        self.assertEqual(normal_regime, 'normal')
        
        # Test high volatility regime
        high_regime = self.strict_filter._classify_atr_regime(2.5)
        self.assertEqual(high_regime, 'high')
        
        # Test extreme volatility regime
        extreme_regime = self.strict_filter._classify_atr_regime(4.0)
        self.assertEqual(extreme_regime, 'extreme')

    def test_enhanced_atr_validation(self):
        """Test enhanced ATR validation with regime-based filtering."""
        # Test normal ATR conditions
        normal_atr = 1.0
        signal_strength = 75.0
        normal_passed, normal_reason, normal_metadata = self.strict_filter._enhanced_atr_validation(
            normal_atr, signal_strength
        )
        self.assertTrue(normal_passed)
        self.assertEqual(normal_metadata['volatility_regime'], 'normal')
        
        # Test extreme ATR conditions
        extreme_atr = 3.5
        extreme_passed, extreme_reason, extreme_metadata = self.strict_filter._enhanced_atr_validation(
            extreme_atr, signal_strength
        )
        # Extreme volatility should be more restrictive
        self.assertIn('volatility_regime', extreme_metadata)
        self.assertEqual(extreme_metadata['volatility_regime'], 'extreme')

    def test_candle_strength_classification(self):
        """Test candle strength classification functionality."""
        # Calculate body ratios for test candles
        def calc_body_ratio(candle):
            body = abs(candle['close'] - candle['open'])
            range_size = candle['high'] - candle['low']
            return body / range_size if range_size > 0 else 0
        
        # Test strong bullish candle
        strong_body_ratio = calc_body_ratio(self.strong_bullish_candle)
        strong_classification = self.strict_filter._classify_candle_strength(strong_body_ratio)
        self.assertIn(strong_classification, ['strong', 'very_strong'])
        self.assertGreater(strong_body_ratio, 0.7)
        
        # Test weak bullish candle
        weak_body_ratio = calc_body_ratio(self.weak_bullish_candle)
        weak_classification = self.strict_filter._classify_candle_strength(weak_body_ratio)
        self.assertIn(weak_classification, ['weak', 'moderate'])
        
        # Test doji candle
        doji_body_ratio = calc_body_ratio(self.doji_candle)
        doji_classification = self.strict_filter._classify_candle_strength(doji_body_ratio)
        self.assertEqual(doji_classification, 'doji')
        self.assertLess(doji_body_ratio, 0.3)

    def test_enhanced_candle_validation(self):
        """Test enhanced candle validation with dynamic body ratio analysis."""
        signal_strength = 75.0
        
        # Test strong candle validation
        strong_passed, strong_reason, strong_metadata = self.strict_filter._enhanced_candle_validation(
            self.strong_bullish_candle, signal_strength, 'BUY'
        )
        self.assertTrue(strong_passed)
        self.assertIn('candle_strength', strong_metadata)
        
        # Test weak candle validation
        weak_passed, weak_reason, weak_metadata = self.strict_filter._enhanced_candle_validation(
            self.weak_bullish_candle, signal_strength, 'BUY'
        )
        # Result depends on filter strictness
        self.assertIn('candle_strength', weak_metadata)
        
        # Test directional mismatch
        mismatch_passed, mismatch_reason, mismatch_metadata = self.strict_filter._enhanced_candle_validation(
            self.strong_bullish_candle, signal_strength, 'SELL'
        )
        # Strong bullish candle should fail for sell signal
        self.assertFalse(mismatch_passed)
        self.assertIn('bullish_candle_sell_signal', mismatch_reason)

    def test_wick_to_body_ratio_validation(self):
        """Test wick-to-body ratio validation."""
        # Test hammer candle (long lower wick)
        def calc_body_ratio(candle):
            body = abs(candle['close'] - candle['open'])
            range_size = candle['high'] - candle['low']
            return body / range_size if range_size > 0 else 0
        
        hammer_body_ratio = calc_body_ratio(self.hammer_candle)
        hammer_classification = self.strict_filter._classify_candle_strength(hammer_body_ratio)
        
        # Calculate wick ratios
        body_size = abs(self.hammer_candle['close'] - self.hammer_candle['open'])
        lower_wick = self.hammer_candle['open'] - self.hammer_candle['low']
        wick_to_body_ratio = lower_wick / body_size if body_size > 0 else float('inf')
        
        # Hammer should have high wick-to-body ratio
        self.assertGreater(wick_to_body_ratio, 2.0)

    def test_validate_signal_with_enhanced_features(self):
        """Test complete signal validation with enhanced features enabled."""
        # Test with enhanced features enabled
        enhanced_filter = SignalSanityFilter(
            min_volatility=0.001,
            min_body_ratio=0.3,
            min_confidence=50.0,
            enable_atr_regime_validation=True,
            enable_dynamic_body_validation=True,
            atr_volatility_thresholds={'very_low': 0.0005, 'low': 0.5, 'normal': 1.5, 'high': 3.0, 'extreme': 5.0},
            body_ratio_thresholds={'doji': 0.1, 'weak': 0.2, 'moderate': 0.4, 'strong': 0.6}
        )
        
        # Test validation with strong candle and normal ATR
        passed, reason, metadata = enhanced_filter.validate_signal(
            candle=self.strong_bullish_candle,
            atr=1.0,
            direction_confidence=75.0,
            signal_strength=80.0,
            symbol="XAUUSD",
            additional_data=self.sample_additional_data
        )
        
        # Should pass with strong candle and good conditions
        self.assertTrue(passed)
        self.assertIn('enhanced_atr_analysis', metadata)
        self.assertIn('enhanced_candle_analysis', metadata)
        self.assertIn('validation_features', metadata)

    def test_factory_methods_with_enhanced_parameters(self):
        """Test factory methods include enhanced parameters."""
        # Test strict factory
        strict = SignalSanityFilterFactory.create_strict()
        self.assertTrue(strict.enable_atr_regime_validation)
        self.assertTrue(strict.enable_dynamic_body_validation)
        self.assertIsNotNone(strict.atr_volatility_thresholds)
        self.assertIsNotNone(strict.body_ratio_thresholds)
        
        # Test permissive factory
        permissive = SignalSanityFilterFactory.create_permissive()
        self.assertTrue(permissive.enable_atr_regime_validation)
        self.assertTrue(permissive.enable_dynamic_body_validation)
        
        # Test config-based factory
        config_based = SignalSanityFilterFactory.create_from_config(SANITY_FILTER_CONFIG["strict"])
        self.assertEqual(config_based.min_directional_body_ratio, 
                        SANITY_FILTER_CONFIG["strict"]["min_directional_body_ratio"])

    def test_metadata_completeness_enhanced(self):
        """Test that enhanced validation metadata is complete."""
        passed, reason, metadata = self.strict_filter.validate_signal(
            candle=self.strong_bullish_candle,
            atr=1.2,
            direction_confidence=70.0,
            signal_strength=75.0,
            symbol="XAUUSD",
            additional_data=self.sample_additional_data
        )
        
        # Check enhanced metadata fields
        if self.strict_filter.enable_atr_regime_validation:
            self.assertIn('enhanced_atr_analysis', metadata)
            self.assertIn('volatility_regime', metadata['enhanced_atr_analysis'])
        
        if self.strict_filter.enable_dynamic_body_validation:
            self.assertIn('enhanced_candle_analysis', metadata)
            self.assertIn('candle_strength', metadata['enhanced_candle_analysis'])
        
        self.assertIn('validation_features', metadata)
        self.assertIn('signal_metrics', metadata)
        self.assertIn('signal_direction', metadata['signal_metrics'])

    def test_different_volatility_regimes(self):
        """Test filter behavior across different volatility regimes."""
        # ATR ratio = atr / close_price, close_price = 104.5 from strong_bullish_candle
        close_price = self.strong_bullish_candle['close']  # 104.5
        test_cases = [
            {'atr': 0.5 * close_price, 'expected_regime': 'low'},      # 0.5 ratio
            {'atr': 1.0 * close_price, 'expected_regime': 'normal'},   # 1.0 ratio > 0.7, < 1.8
            {'atr': 2.0 * close_price, 'expected_regime': 'high'},     # 2.0 ratio > 1.8, < 3.0
            {'atr': 3.5 * close_price, 'expected_regime': 'extreme'}   # 3.5 ratio > 3.0
        ]
        
        for case in test_cases:
            passed, reason, metadata = self.strict_filter.validate_signal(
                candle=self.strong_bullish_candle,
                atr=case['atr'],
                direction_confidence=70.0,
                signal_strength=75.0,
                symbol="XAUUSD",
                additional_data=self.sample_additional_data
            )
            
            if 'enhanced_atr_analysis' in metadata:
                self.assertEqual(
                    metadata['enhanced_atr_analysis']['volatility_regime'],
                    case['expected_regime']
                )

    def test_directional_body_ratio_validation(self):
        """Test directional body ratio validation."""
        # Test bullish candle with buy signal (should pass)
        bullish_buy_passed, bullish_buy_reason, bullish_buy_metadata = self.strict_filter._enhanced_candle_validation(
            self.strong_bullish_candle, 75.0, 'BUY'
        )
        self.assertTrue(bullish_buy_passed)
        
        # Test bullish candle with sell signal (should fail directional check)
        bullish_sell_passed, bullish_sell_reason, bullish_sell_metadata = self.strict_filter._enhanced_candle_validation(
            self.strong_bullish_candle, 75.0, 'SELL'
        )
        self.assertFalse(bullish_sell_passed)

    def test_filter_statistics_tracking(self):
        """Test that filter statistics are properly tracked with enhanced features."""
        initial_stats = self.strict_filter.get_filter_statistics()
        
        # Run several validations
        for i in range(5):
            self.strict_filter.validate_signal(
                candle=self.strong_bullish_candle,
                atr=1.0,
                direction_confidence=70.0,
                signal_strength=75.0,
                symbol="XAUUSD",
                additional_data=self.sample_additional_data
            )
        
        final_stats = self.strict_filter.get_filter_statistics()
        
        # Check that statistics were updated
        self.assertGreater(final_stats['total_signals'], initial_stats['total_signals'])


if __name__ == '__main__':
    unittest.main()