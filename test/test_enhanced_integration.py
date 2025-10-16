"""
Integration test suite for enhanced signal processing pipeline including
weighted voting, adaptive thresholds, and enhanced sanity filtering.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import asyncio

from app.signal_engine import SignalEngine
from app.threshold_manager import ThresholdManagerFactory
from app.sanity_filter import SignalSanityFilterFactory
from app.indicators import compute_weighted_vote_aggregation
from app.config import THRESHOLD_MANAGER_CONFIG, SANITY_FILTER_CONFIG, WEIGHTS


class TestEnhancedIntegration(unittest.TestCase):
    """Integration test suite for enhanced signal processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock fetch history function
        self.mock_fetch_history = Mock()
        
        # Create signal engine with mocked dependencies
        self.signal_engine = SignalEngine(self.mock_fetch_history)
        
        # Sample market data for testing
        self.sample_m15_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01 10:00', periods=100, freq='15min'),
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2050, 2150, 100),
            'low': np.random.uniform(1950, 2050, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Ensure high > low and proper OHLC relationships
        for i in range(len(self.sample_m15_data)):
            row = self.sample_m15_data.iloc[i]
            high = max(row['open'], row['close']) + np.random.uniform(0, 10)
            low = min(row['open'], row['close']) - np.random.uniform(0, 10)
            self.sample_m15_data.at[i, 'high'] = high
            self.sample_m15_data.at[i, 'low'] = low
        
        self.sample_h1_data = self.sample_m15_data.copy()
        self.sample_h4_data = self.sample_m15_data.copy()

    def test_weighted_voting_integration(self):
        """Test weighted voting system integration."""
        # Create mock indicator results with proper structure
        from app.indicators import IndicatorResult
        
        sample_results = {
            'RSI': IndicatorResult(
                direction='buy', 
                value={'rsi': 75.0}, 
                contribution=15.0,
                vote=1, 
                strength=75.0, 
                label='strong'
            ),
            'MACD': IndicatorResult(
                direction='buy', 
                value={'macd': 0.5}, 
                contribution=20.0,
                vote=1, 
                strength=70.0, 
                label='strong'
            ),
            'SMA_EMA': IndicatorResult(
                direction='weak_buy', 
                value={'sma': 2050.0}, 
                contribution=7.5,
                vote=1, 
                strength=60.0, 
                label='weak'
            ),
            'BBANDS': IndicatorResult(
                direction='neutral', 
                value={'bb_position': 0.5}, 
                contribution=0.0,
                vote=0, 
                strength=50.0, 
                label='neutral'
            ),
            'STOCH': IndicatorResult(
                direction='sell', 
                value={'stoch': 25.0}, 
                contribution=0.0,
                vote=-1, 
                strength=65.0, 
                label='strong'
            )
        }
        
        # Test weighted vote aggregation
        vote_result = compute_weighted_vote_aggregation(
            sample_results, 
            WEIGHTS, 
            threshold=self.signal_engine.threshold_manager.base_threshold
        )
        
        # Verify vote result structure (using correct field names)
        self.assertIn('final_direction', vote_result)
        self.assertIn('confidence', vote_result)
        self.assertIn('strong_signals', vote_result)
        self.assertIn('weak_signals', vote_result)
        self.assertIn('normalized_score', vote_result)
        self.assertIn('indicator_count', vote_result)
        
        # Verify confidence is within valid range
        self.assertGreaterEqual(vote_result['confidence'], 0)
        self.assertLessEqual(vote_result['confidence'], 1.0)

    def test_adaptive_threshold_integration(self):
        """Test adaptive threshold manager integration."""
        # Test adaptive threshold computation with correct parameters
        dynamic_threshold, metadata = self.signal_engine.threshold_manager.compute_adaptive_threshold(
            atr_ratio=0.015,  # 1.5% ATR ratio
            rsi_deviation=15.0,  # RSI deviation from 50
            macd_histogram=0.8,
            price_ma_deviation=0.05,
            symbol="XAUUSD",
            timeframe="M15"
        )
        
        # Verify threshold is within bounds
        self.assertGreaterEqual(dynamic_threshold, self.signal_engine.threshold_manager.min_threshold)
        self.assertLessEqual(dynamic_threshold, self.signal_engine.threshold_manager.max_threshold)
        
        # Verify metadata completeness
        self.assertIn('volatility_adjustment', metadata)
        self.assertIn('momentum_adjustment', metadata)
        self.assertIn('trend_adjustment', metadata)
        self.assertIn('total_adjustment', metadata)
        
        # Test dynamic threshold with votes
        from app.indicators import IndicatorResult
        sample_results = {
            'RSI': IndicatorResult(
                direction='buy', 
                value={'rsi': 75.0}, 
                contribution=15.0,
                vote=1, 
                strength=75.0, 
                label='strong'
            ),
            'MACD': IndicatorResult(
                direction='buy', 
                value={'macd': 0.5}, 
                contribution=20.0,
                vote=1, 
                strength=70.0, 
                label='strong'
            )
        }
        
        vote_result = compute_weighted_vote_aggregation(
            sample_results, 
            WEIGHTS, 
            threshold=self.signal_engine.threshold_manager.base_threshold
        )
        
        market_conditions = {
            'atr_ratio': 0.015,
            'rsi': 65.0,
            'macd_histogram': 0.8,
            'price_ma_deviation': 0.05
        }
        
        # Test dynamic threshold with votes
        dynamic_threshold_votes, metadata_votes = self.signal_engine.threshold_manager.compute_dynamic_threshold_with_votes(
            vote_result=vote_result,
            market_conditions=market_conditions,
            symbol="XAUUSD",
            timeframe="M15"
        )
        
        # Verify threshold is within bounds
        self.assertGreaterEqual(dynamic_threshold_votes, self.signal_engine.threshold_manager.min_threshold)
        self.assertLessEqual(dynamic_threshold_votes, self.signal_engine.threshold_manager.max_threshold)
        
        # Verify enhanced metadata
        self.assertIn('vote_adjustments', metadata_votes)
        self.assertIn('enhanced_volatility_analysis', metadata_votes)
        self.assertIn('final_threshold', metadata_votes)

    def test_enhanced_sanity_filter_integration(self):
        """Test enhanced sanity filter integration."""
        # Sample candle data with proper OHLC structure
        sample_candle = {
            'open': 2050.0,
            'high': 2065.0,
            'low': 2048.0,
            'close': 2062.0
        }
        
        # Sample additional data for enhanced validation with realistic ATR ratio
        additional_data = {
            'signal_direction': 'buy',
            'h1_alignment': True,
            'volatility_regime': 'normal',
            'atr_ratio': 0.012  # 1.2% ATR ratio (realistic value)
        }
        
        # Test enhanced validation with realistic parameters
        passed, reason, metadata = self.signal_engine.sanity_filter.validate_signal(
            candle=sample_candle,
            atr=25.0,  # Realistic ATR value for gold
            direction_confidence=75.0,
            signal_strength=80.0,
            symbol="XAUUSD",
            additional_data=additional_data
        )
        
        # Verify validation result structure
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(reason, (str, type(None)))
        self.assertIsInstance(metadata, dict)
        
        # If validation passed, check for enhanced metadata fields
        if passed and self.signal_engine.sanity_filter.enable_atr_regime_validation:
            self.assertIn('enhanced_atr_analysis', metadata)
        
        if passed and self.signal_engine.sanity_filter.enable_dynamic_body_validation:
            self.assertIn('enhanced_candle_analysis', metadata)
        
        # Test with a scenario that should pass validation
        good_candle = {
            'open': 2050.0,
            'high': 2055.0,
            'low': 2048.0,
            'close': 2054.5  # Body ratio = 4.5/7 = 0.643 > 0.6 threshold
        }
        
        good_additional_data = {
            'signal_direction': 'buy',
            'h1_alignment': True,
            'volatility_regime': 'normal',
            'atr_ratio': 0.008  # Normal volatility
        }
        
        passed_good, reason_good, metadata_good = self.signal_engine.sanity_filter.validate_signal(
            candle=good_candle,
            atr=16.0,
            direction_confidence=80.0,
            signal_strength=85.0,
            symbol="XAUUSD",
            additional_data=good_additional_data
        )
        
        # This should pass validation
        self.assertTrue(passed_good, f"Good signal should pass validation, but got: {reason_good}")
        
        # Verify enhanced metadata is present for successful validation
        if self.signal_engine.sanity_filter.enable_atr_regime_validation:
            self.assertIn('enhanced_atr_analysis', metadata_good)
        
        if self.signal_engine.sanity_filter.enable_dynamic_body_validation:
            self.assertIn('enhanced_candle_analysis', metadata_good)

    @patch('app.signal_engine.insert_signal')
    @patch('app.signal_engine.insert_indicator_snapshot')
    @patch('app.signal_engine.get_active_strategy_config')
    def test_full_pipeline_integration(self, mock_strategy, mock_insert_snapshot, mock_insert_signal):
        """Test full signal processing pipeline integration."""
        # Mock strategy configuration
        mock_strategy.return_value = {
            'indicator_params': {
                'RSI': {'periods': [9, 14], 'overbought': 70, 'oversold': 30},
                'MACD': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'weights': WEIGHTS,
            'primary_timeframe': '15m'
        }
        
        # Mock fetch history to return sample data
        self.mock_fetch_history.return_value = self.sample_m15_data
        
        # Mock database operations
        mock_insert_signal.return_value = None
        mock_insert_snapshot.return_value = None
        
        # Test signal processing for a single symbol
        try:
            # This would normally be called in the async run loop
            # We'll test the core logic without the async wrapper
            symbol = "XAUUSD"
            
            # Verify that the signal engine components are properly initialized
            self.assertIsNotNone(self.signal_engine.threshold_manager)
            self.assertIsNotNone(self.signal_engine.sanity_filter)
            self.assertIsNotNone(self.signal_engine.mtf_confirmation)
            
            # Verify configuration integration
            self.assertEqual(
                self.signal_engine.threshold_manager.base_threshold,
                THRESHOLD_MANAGER_CONFIG['base_threshold']
            )
            
            # Test that enhanced features are enabled
            self.assertTrue(self.signal_engine.sanity_filter.enable_atr_regime_validation)
            self.assertTrue(self.signal_engine.sanity_filter.enable_dynamic_body_validation)
            
        except Exception as e:
            self.fail(f"Full pipeline integration test failed: {e}")

    def test_configuration_consistency(self):
        """Test that all components use consistent configuration."""
        # Verify threshold manager configuration
        tm_config = THRESHOLD_MANAGER_CONFIG
        self.assertIn('base_threshold', tm_config)
        self.assertIn('volatility_regime_thresholds', tm_config)
        self.assertIn('stress_detection_enabled', tm_config)
        self.assertIn('adaptive_parameters', tm_config)
        
        # Verify sanity filter configuration
        sf_config = SANITY_FILTER_CONFIG
        self.assertIn('strict', sf_config)
        self.assertIn('permissive', sf_config)
        
        for mode in ['strict', 'permissive']:
            config = sf_config[mode]
            self.assertIn('atr_volatility_thresholds', config)
            self.assertIn('body_ratio_thresholds', config)
            self.assertIn('enable_atr_regime_validation', config)
            self.assertIn('enable_dynamic_body_validation', config)

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Threshold manager should handle invalid conditions gracefully
        try:
            threshold, metadata = self.signal_engine.threshold_manager.compute_adaptive_threshold(
                atr_ratio=-1.0,  # Invalid negative ATR ratio
                rsi_deviation=150.0,  # Invalid RSI deviation
                macd_histogram=float('inf'),  # Infinite value
                price_ma_deviation=float('nan'),  # NaN value
                symbol="XAUUSD",
                timeframe="M15"
            )
            # Should return a valid threshold within bounds
            self.assertGreaterEqual(threshold, self.signal_engine.threshold_manager.min_threshold)
            self.assertLessEqual(threshold, self.signal_engine.threshold_manager.max_threshold)
        except Exception as e:
            self.fail(f"Threshold manager failed to handle invalid conditions: {e}")
        
        # Test sanity filter with invalid candle data
        invalid_candle = {
            'open': None,
            'high': -100.0,
            'low': 200.0,  # Low > High (invalid)
            'close': float('nan')
        }
        
        try:
            passed, reason, metadata = self.signal_engine.sanity_filter.validate_signal(
                candle=invalid_candle,
                atr=10.0,
                direction_confidence=75.0,
                signal_strength=80.0,
                symbol="XAUUSD"
            )
            # Should handle gracefully and return False
            self.assertFalse(passed)
            self.assertIsNotNone(reason)
        except Exception as e:
            self.fail(f"Sanity filter failed to handle invalid candle data: {e}")

    def test_performance_metrics_integration(self):
        """Test that performance metrics are properly tracked across components."""
        # Test threshold manager metadata tracking
        threshold, metadata = self.signal_engine.threshold_manager.compute_adaptive_threshold(
            atr_ratio=0.012,  # 1.2% ATR ratio
            rsi_deviation=5.0,  # RSI deviation from 50
            macd_histogram=0.3,
            price_ma_deviation=0.02,
            symbol="XAUUSD",
            timeframe="M15"
        )
        
        # Verify performance metadata
        self.assertIn('volatility_adjustment', metadata)
        self.assertIn('momentum_adjustment', metadata)
        self.assertIn('trend_adjustment', metadata)
        self.assertIn('total_adjustment', metadata)
        
        # Test sanity filter statistics
        initial_stats = self.signal_engine.sanity_filter.get_filter_statistics()
        
        # Run a validation
        sample_candle = {
            'open': 2050.0,
            'high': 2055.0,
            'low': 2048.0,
            'close': 2053.0
        }
        
        self.signal_engine.sanity_filter.validate_signal(
            candle=sample_candle,
            atr=12.0,
            direction_confidence=70.0,
            signal_strength=75.0,
            symbol="XAUUSD"
        )
        
        final_stats = self.signal_engine.sanity_filter.get_filter_statistics()
        
        # Verify statistics were updated
        self.assertGreater(final_stats['total_signals'], initial_stats['total_signals'])


if __name__ == '__main__':
    unittest.main()