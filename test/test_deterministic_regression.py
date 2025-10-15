"""
Deterministic regression test suite with synthetic data for signal processing pipeline.
This test ensures consistent behavior across different runs and environments.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta

from app.signal_engine import SignalEngine
from app.indicators import compute_weighted_vote_aggregation
from app.utils_time import last_closed


class TestDeterministicRegression(unittest.TestCase):
    """Deterministic regression tests with synthetic data."""
    
    def setUp(self):
        """Set up test fixtures with deterministic synthetic data."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Mock fetch history function
        self.mock_fetch_history = Mock()
        
        # Create signal engine with mocked dependencies
        self.signal_engine = SignalEngine(self.mock_fetch_history)
        
        # Generate deterministic synthetic data
        self.base_time = datetime(2024, 1, 1, 10, 0)
        self.synthetic_data = self._generate_synthetic_ohlcv_data()
        
    def _generate_synthetic_ohlcv_data(self):
        """Generate deterministic synthetic OHLCV data with known patterns."""
        periods = 200
        base_price = 2000.0
        
        # Create time series with 15-minute intervals
        times = [self.base_time + timedelta(minutes=15*i) for i in range(periods)]
        
        # Generate price trend with sine wave pattern for predictable signals
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 50
        noise = np.random.normal(0, 5, periods)  # Small random noise
        
        closes = base_price + trend + noise
        
        # Generate OHLC data with realistic relationships
        data = []
        for i in range(periods):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['close'] + np.random.normal(0, 2)
            
            close_price = closes[i]
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 3))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 3))
            volume = 1000 + abs(np.random.normal(0, 200))
            
            data.append({
                'time': times[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_synthetic_data_consistency(self):
        """Test that synthetic data generation is deterministic."""
        # Reset seed and regenerate data
        np.random.seed(42)
        data1 = self._generate_synthetic_ohlcv_data()
        
        np.random.seed(42)
        data2 = self._generate_synthetic_ohlcv_data()
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)
        
        # Verify data quality
        self.assertEqual(len(data1), 200)
        self.assertTrue(all(data1['high'] >= data1['low']))
        self.assertTrue(all(data1['high'] >= data1['open']))
        self.assertTrue(all(data1['high'] >= data1['close']))
        self.assertTrue(all(data1['low'] <= data1['open']))
        self.assertTrue(all(data1['low'] <= data1['close']))
    
    def test_indicator_calculation_determinism(self):
        """Test that indicator calculations are deterministic with synthetic data."""
        from app.indicators import compute_indicators
        
        # Calculate indicators multiple times
        indicators1 = compute_indicators(self.synthetic_data.copy())
        indicators2 = compute_indicators(self.synthetic_data.copy())
        
        # Results should be identical
        for key in indicators1:
            if isinstance(indicators1[key], (list, np.ndarray)):
                np.testing.assert_array_equal(indicators1[key], indicators2[key], 
                                            err_msg=f"Indicator {key} not deterministic")
            elif hasattr(indicators1[key], 'equals'):  # pandas Series
                self.assertTrue(indicators1[key].equals(indicators2[key]), 
                               f"Indicator {key} not deterministic")
            else:
                self.assertEqual(indicators1[key], indicators2[key], 
                               f"Indicator {key} not deterministic")
    
    def test_weighted_voting_determinism(self):
        """Test that weighted voting produces deterministic results."""
        from app.indicators import compute_indicators, evaluate_signals
        
        indicators = compute_indicators(self.synthetic_data)
        signal_results = evaluate_signals(self.synthetic_data, indicators)
        
        # Run weighted voting multiple times
        vote1 = compute_weighted_vote_aggregation(signal_results)
        vote2 = compute_weighted_vote_aggregation(signal_results)
        
        # Results should be identical
        self.assertEqual(vote1['final_direction'], vote2['final_direction'])
        self.assertEqual(vote1['confidence'], vote2['confidence'])
        self.assertEqual(vote1['normalized_score'], vote2['normalized_score'])
    
    @patch('app.signal_engine.insert_signal')
    @patch('app.signal_engine.insert_indicator_snapshot')
    @patch('app.signal_engine.get_active_strategy_config')
    def test_signal_engine_determinism(self, mock_strategy, mock_insert_snapshot, mock_insert_signal):
        """Test that signal engine produces deterministic results."""
        # Mock strategy configuration
        mock_strategy.return_value = {
            'id': 1,
            'name': 'test_strategy',
            'timeframes': ['15m', '1h', '4h'],
            'is_active': True
        }
        
        # Mock fetch history to return synthetic data
        self.mock_fetch_history.return_value = self.synthetic_data
        
        # Run signal computation multiple times
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result1 = loop.run_until_complete(
                self.signal_engine.compute_once(['AAPL'])
            )
            result2 = loop.run_until_complete(
                self.signal_engine.compute_once(['AAPL'])
            )
            
            # Results should be identical
            self.assertEqual(len(result1), len(result2))
            for r1, r2 in zip(result1, result2):
                self.assertEqual(r1['symbol'], r2['symbol'])
                # Check available fields and compare what exists
                if 'had_signal' in r1 and 'had_signal' in r2:
                    self.assertEqual(r1['had_signal'], r2['had_signal'])
                if 'final_strength' in r1 and 'final_strength' in r2:
                    self.assertAlmostEqual(r1['final_strength'], r2['final_strength'], places=6)
        finally:
            loop.close()
    
    def test_partial_bar_cropping_determinism(self):
        """Test that partial bar cropping produces consistent results."""
        # Add a partial bar at the end
        partial_data = self.synthetic_data.copy()
        last_time = partial_data.iloc[-1]['time']
        partial_time = last_time + timedelta(minutes=7)  # Partial 15-min bar
        
        partial_bar = {
            'time': partial_time,
            'open': 2050.0,
            'high': 2055.0,
            'low': 2048.0,
            'close': 2052.0,
            'volume': 500
        }
        partial_data = pd.concat([partial_data, pd.DataFrame([partial_bar])], ignore_index=True)
        
        # Test cropping
        anchor = last_closed(partial_time, '15min')
        
        # Convert time column to timezone-aware for comparison
        partial_data_tz = partial_data.copy()
        partial_data_tz['time'] = pd.to_datetime(partial_data_tz['time'], utc=True)
        
        cropped_data = partial_data_tz[partial_data_tz['time'] <= anchor]
        
        # Should exclude the partial bar
        self.assertEqual(len(cropped_data), len(self.synthetic_data))
        
        # Compare timestamps (convert original to UTC for comparison)
        original_last_time = pd.to_datetime(self.synthetic_data.iloc[-1]['time'], utc=True)
        self.assertEqual(cropped_data.iloc[-1]['time'], original_last_time)
    
    def test_regression_baseline_signals(self):
        """Test against known baseline signal patterns in synthetic data."""
        from app.indicators import compute_indicators
        
        # Use specific synthetic data that should generate known signals
        np.random.seed(123)  # Different seed for specific test case
        
        # Generate data with clear trend reversal pattern
        periods = 50
        times = [self.base_time + timedelta(minutes=15*i) for i in range(periods)]
        
        # Create downtrend followed by uptrend
        prices = []
        for i in range(periods):
            if i < 25:
                # Downtrend
                price = 2100 - (i * 2) + np.random.normal(0, 1)
            else:
                # Uptrend
                price = 2050 + ((i - 25) * 3) + np.random.normal(0, 1)
            prices.append(price)
        
        trend_data = pd.DataFrame({
            'time': times,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 2)) for p in prices],
            'close': prices,
            'volume': [1000 + abs(np.random.normal(0, 100)) for _ in range(periods)]
        })
        
        # Calculate indicators
        indicators = compute_indicators(trend_data)
        
        # Verify expected signal characteristics
        self.assertIn('rsi', indicators)
        self.assertIn('macd', indicators)
        self.assertIn('bb_low', indicators)
        self.assertIn('bb_mid', indicators)
        self.assertIn('bb_high', indicators)
        
        # RSI should show oversold then recovery pattern
        rsi_values = indicators['rsi']
        if hasattr(rsi_values, 'values'):  # pandas Series
            rsi_array = rsi_values.values
        else:
            rsi_array = rsi_values
        
        # Check for oversold and recovery patterns
        self.assertTrue(any(rsi < 30 for rsi in rsi_array[:30] if not np.isnan(rsi)))  # Oversold in downtrend
        self.assertTrue(any(rsi > 50 for rsi in rsi_array[35:] if not np.isnan(rsi)))  # Recovery in uptrend


if __name__ == '__main__':
    unittest.main()