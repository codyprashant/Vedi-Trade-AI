"""
Comprehensive test suite for enhanced ThresholdManager with volatility regime classification,
market stress detection, and adaptive parameter adjustment.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from app.threshold_manager import ThresholdManager, ThresholdManagerFactory
from app.config import THRESHOLD_MANAGER_CONFIG


class TestEnhancedThresholdManager(unittest.TestCase):
    """Test suite for enhanced ThresholdManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.threshold_manager = ThresholdManagerFactory.create_from_config(THRESHOLD_MANAGER_CONFIG)
        
        # Sample market conditions for testing
        self.normal_conditions = {
            'atr_ratio': 1.0,
            'rsi': 50.0,
            'macd_histogram': 0.5,
            'price_ma_deviation': 0.02
        }
        
        self.high_volatility_conditions = {
            'atr_ratio': 1.8,
            'rsi': 75.0,
            'macd_histogram': -1.2,
            'price_ma_deviation': 0.08
        }
        
        self.extreme_volatility_conditions = {
            'atr_ratio': 2.5,
            'rsi': 85.0,
            'macd_histogram': -2.0,
            'price_ma_deviation': 0.15
        }
        
        # Sample vote result for testing
        self.sample_vote_result = {
            'final_decision': 'buy',
            'confidence': 75.0,
            'strong_signals': 3,
            'weak_signals': 1,
            'vote_score': 0.8,
            'indicator_count': 5
        }

    def test_volatility_regime_classification(self):
        """Test volatility regime classification functionality."""
        # Test low volatility
        low_vol_regime, low_vol_meta = self.threshold_manager.classify_volatility_regime(0.6)
        self.assertEqual(low_vol_regime, 'low')
        
        # Test normal volatility
        normal_vol_regime, normal_vol_meta = self.threshold_manager.classify_volatility_regime(1.2)
        self.assertEqual(normal_vol_regime, 'normal_high')
        
        # Test high volatility
        high_vol_regime, high_vol_meta = self.threshold_manager.classify_volatility_regime(1.7)
        self.assertEqual(high_vol_regime, 'high')
        
        # Test extreme volatility
        extreme_vol_regime, extreme_vol_meta = self.threshold_manager.classify_volatility_regime(2.3)
        self.assertEqual(extreme_vol_regime, 'extreme')

    def test_market_stress_detection(self):
        """Test market stress detection functionality."""
        # Test normal market conditions
        normal_stress = self.threshold_manager.detect_market_stress(self.normal_conditions)
        self.assertFalse(normal_stress['is_stressed'])
        self.assertEqual(normal_stress['stress_level'], 'none')
        
        # Test high volatility conditions
        high_vol_stress = self.threshold_manager.detect_market_stress(self.high_volatility_conditions)
        self.assertTrue(high_vol_stress['is_stressed'])
        self.assertIn(high_vol_stress['stress_level'], ['medium', 'high'])
        
        # Test extreme volatility conditions
        extreme_stress = self.threshold_manager.detect_market_stress(self.extreme_volatility_conditions)
        self.assertTrue(extreme_stress['is_stressed'])
        self.assertEqual(extreme_stress['stress_level'], 'extreme')

    def test_enhanced_volatility_adjustment(self):
        """Test enhanced volatility adjustment with regime classification."""
        # Test normal volatility adjustment
        normal_adj, normal_meta = self.threshold_manager._calculate_volatility_adjustment(
            1.0, self.normal_conditions
        )
        self.assertIsInstance(normal_adj, float)
        self.assertIn('volatility_regime', normal_meta)
        self.assertEqual(normal_meta['volatility_regime']['regime'], 'normal_high')
        
        # Test high volatility adjustment
        high_adj, high_meta = self.threshold_manager._calculate_volatility_adjustment(
            1.8, self.high_volatility_conditions
        )
        self.assertIsInstance(high_adj, float)
        self.assertEqual(high_meta['volatility_regime']['regime'], 'high')
        self.assertGreater(abs(high_adj), abs(normal_adj))  # Higher volatility should have larger adjustment
        
        # Test extreme volatility adjustment
        extreme_adj, extreme_meta = self.threshold_manager._calculate_volatility_adjustment(
            2.5, self.extreme_volatility_conditions
        )
        self.assertEqual(extreme_meta['volatility_regime']['regime'], 'extreme')
        self.assertGreater(abs(extreme_adj), abs(high_adj))  # Extreme volatility should have largest adjustment

    def test_adaptive_parameter_adjustment(self):
        """Test adaptive parameter adjustment based on market regime."""
        # Test with adaptive parameters enabled
        adaptive_manager = ThresholdManager(
            base_threshold=60.0,
            adaptive_parameters=True,
            volatility_regime_thresholds=THRESHOLD_MANAGER_CONFIG['volatility_regime_thresholds']
        )
        
        # Test normal conditions
        normal_threshold, normal_meta = adaptive_manager.get_threshold_for_conditions(self.normal_conditions)
        self.assertIn('adaptive_weights', normal_meta)
        
        # Test extreme conditions - should adjust weights
        extreme_threshold, extreme_meta = adaptive_manager.get_threshold_for_conditions(self.extreme_volatility_conditions)
        self.assertIn('adaptive_weights', extreme_meta)
        
        # Volatility weight should be higher in extreme conditions
        normal_vol_weight = normal_meta['adaptive_weights']['volatility']
        extreme_vol_weight = extreme_meta['adaptive_weights']['volatility']
        self.assertGreater(extreme_vol_weight, normal_vol_weight)

    def test_compute_dynamic_threshold_with_votes(self):
        """Test dynamic threshold computation with vote integration."""
        dynamic_threshold, metadata = self.threshold_manager.compute_dynamic_threshold_with_votes(
            vote_result=self.sample_vote_result,
            market_conditions=self.normal_conditions,
            symbol="XAUUSD",
            timeframe="M15"
        )
        
        # Verify threshold is within bounds
        self.assertGreaterEqual(dynamic_threshold, self.threshold_manager.min_threshold)
        self.assertLessEqual(dynamic_threshold, self.threshold_manager.max_threshold)
        
        # Verify metadata contains expected fields
        self.assertIn('vote_adjustments', metadata)
        self.assertIn('market_conditions', metadata)
        self.assertIn('final_threshold', metadata)
        self.assertIn('enhanced_volatility_analysis', metadata)

    def test_threshold_bounds_enforcement(self):
        """Test that thresholds are always within configured bounds."""
        # Test with extreme conditions that might push threshold out of bounds
        extreme_conditions = {
            'atr_ratio': 5.0,  # Extremely high volatility
            'rsi': 95.0,
            'macd_histogram': -5.0,
            'price_ma_deviation': 0.25
        }
        
        threshold, _ = self.threshold_manager.get_threshold_for_conditions(extreme_conditions)
        
        # Threshold should be clamped within bounds
        self.assertGreaterEqual(threshold, self.threshold_manager.min_threshold)
        self.assertLessEqual(threshold, self.threshold_manager.max_threshold)

    def test_factory_methods(self):
        """Test ThresholdManager factory methods."""
        # Test conservative factory
        conservative = ThresholdManagerFactory.create_conservative()
        self.assertIsInstance(conservative, ThresholdManager)
        self.assertEqual(conservative.base_threshold, 65.0)
        
        # Test aggressive factory
        aggressive = ThresholdManagerFactory.create_aggressive()
        self.assertIsInstance(aggressive, ThresholdManager)
        self.assertEqual(aggressive.base_threshold, 55.0)
        
        # Test config-based factory
        config_based = ThresholdManagerFactory.create_from_config(THRESHOLD_MANAGER_CONFIG)
        self.assertIsInstance(config_based, ThresholdManager)
        self.assertEqual(config_based.base_threshold, THRESHOLD_MANAGER_CONFIG['base_threshold'])

    def test_metadata_completeness(self):
        """Test that all metadata fields are properly populated."""
        threshold, metadata = self.threshold_manager.get_threshold_for_conditions(self.normal_conditions)
        
        # Check required metadata fields
        required_fields = [
            'volatility_adjustment', 'momentum_adjustment', 'trend_adjustment',
            'total_adjustment', 'base_threshold', 'final_threshold',
            'enhanced_volatility_analysis'
        ]
        
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing metadata field: {field}")

    def test_stress_detection_recommendations(self):
        """Test that market stress detection provides appropriate threshold recommendations."""
        stress_result = self.threshold_manager.detect_market_stress(self.extreme_volatility_conditions)
        
        self.assertIn('threshold_recommendation', stress_result)
        self.assertIn('recommended_adjustment', stress_result)
        
        # High stress should recommend increasing threshold
        if stress_result['stress_level'] == 'high':
            self.assertGreater(stress_result['recommended_adjustment'], 0)


if __name__ == '__main__':
    unittest.main()