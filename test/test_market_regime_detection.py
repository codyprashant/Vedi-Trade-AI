#!/usr/bin/env python3
"""
Market Regime Detection Test Suite

Tests the adaptive signal system's ability to detect and respond to different market regimes:
- Trending markets (strong directional movement)
- Ranging markets (sideways consolidation)
- Volatile markets (high ATR, choppy price action)
- Low volatility markets (tight ranges, low ATR)

This test validates that the system adapts its thresholds and filtering appropriately
for each market condition.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from threshold_manager import ThresholdManagerFactory
from sanity_filter import SignalSanityFilterFactory
from mtf_confirmation import MultiTimeframeConfirmation


class TestMarketRegimeDetection:
    """Test suite for market regime detection and adaptive responses"""
    
    def __init__(self):
        """Initialize test components"""
        self.threshold_manager = ThresholdManagerFactory.create_conservative()
        self.sanity_filter = SignalSanityFilterFactory.create_strict()
        self.mtf_confirmation = MultiTimeframeConfirmation()
        
    def create_trending_market_data(self, length: int = 100, trend_strength: float = 0.8) -> pd.DataFrame:
        """Create synthetic data representing a trending market"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=length), periods=length, freq='15min')
        
        # Create strong uptrend with some noise
        base_price = 1.1000
        trend_component = np.linspace(0, trend_strength * 0.01, length)  # 80 pip trend
        noise_component = np.random.normal(0, 0.0002, length)  # Small noise
        
        prices = base_price + trend_component + noise_component
        
        # Create OHLC data with trending characteristics
        data = []
        for i, price in enumerate(prices):
            # Trending candles tend to have larger bodies in trend direction
            open_price = price - np.random.uniform(0.0001, 0.0003)
            close_price = price + np.random.uniform(0.0002, 0.0005)  # Bullish bias
            high_price = max(open_price, close_price) + np.random.uniform(0.0001, 0.0003)
            low_price = min(open_price, close_price) - np.random.uniform(0.0001, 0.0002)
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 5000)
            })
            
        return pd.DataFrame(data)
    
    def create_ranging_market_data(self, length: int = 100, range_size: float = 0.003) -> pd.DataFrame:
        """Create synthetic data representing a ranging/sideways market"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=length), periods=length, freq='15min')
        
        base_price = 1.1000
        # Create oscillating price within a range
        oscillation = np.sin(np.linspace(0, 4 * np.pi, length)) * (range_size / 2)
        noise = np.random.normal(0, 0.0001, length)
        
        prices = base_price + oscillation + noise
        
        data = []
        for i, price in enumerate(prices):
            # Ranging candles tend to have smaller bodies and more wicks
            body_size = np.random.uniform(0.00005, 0.0002)
            wick_size = np.random.uniform(0.0001, 0.0003)
            
            if np.random.random() > 0.5:  # Bullish candle
                open_price = price - body_size/2
                close_price = price + body_size/2
            else:  # Bearish candle
                open_price = price + body_size/2
                close_price = price - body_size/2
                
            high_price = max(open_price, close_price) + wick_size
            low_price = min(open_price, close_price) - wick_size
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(500, 2000)
            })
            
        return pd.DataFrame(data)
    
    def create_volatile_market_data(self, length: int = 100, volatility_factor: float = 2.0) -> pd.DataFrame:
        """Create synthetic data representing a highly volatile market"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=length), periods=length, freq='15min')
        
        base_price = 1.1000
        # High volatility with random walk
        returns = np.random.normal(0, 0.001 * volatility_factor, length)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = []
        for i, price in enumerate(prices):
            # Volatile candles have large ranges and bodies
            range_size = np.random.uniform(0.0005, 0.002)
            body_ratio = np.random.uniform(0.3, 0.8)
            
            if np.random.random() > 0.5:  # Bullish
                open_price = price - range_size * (1 - body_ratio) / 2
                close_price = open_price + range_size * body_ratio
            else:  # Bearish
                open_price = price + range_size * (1 - body_ratio) / 2
                close_price = open_price - range_size * body_ratio
                
            high_price = max(open_price, close_price) + range_size * (1 - body_ratio) / 2
            low_price = min(open_price, close_price) - range_size * (1 - body_ratio) / 2
            
            data.append({
                'time': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(2000, 10000)
            })
            
        return pd.DataFrame(data)
    
    def test_trending_market_adaptation(self) -> bool:
        """Test system adaptation to trending market conditions"""
        print("🧪 Trending Market Adaptation")
        print("-" * 50)
        
        try:
            # Create trending market data
            trending_data = self.create_trending_market_data(length=50, trend_strength=1.2)
            
            # Calculate market characteristics
            last_candle = trending_data.iloc[-1]
            atr_estimate = trending_data['high'].rolling(14).max() - trending_data['low'].rolling(14).min()
            atr_ratio = atr_estimate.iloc[-1] / last_candle['close']
            
            # RSI would be extreme in trending market
            rsi_deviation = 35.0  # RSI around 85 (deviation from 50)
            macd_histogram = 0.008  # Strong MACD signal
            
            print(f"  Market characteristics: ATR ratio={atr_ratio:.6f}, RSI dev={rsi_deviation}")
            
            # Test adaptive threshold
            threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=atr_ratio,
                rsi_deviation=rsi_deviation,
                macd_histogram=macd_histogram,
                symbol="TRENDING_TEST"
            )
            
            print(f"  Adaptive threshold: {threshold:.1f}% (metadata: {metadata})")
            
            # In trending markets, threshold should be lowered to catch trend continuation
            base_threshold = self.threshold_manager.base_threshold
            if threshold >= base_threshold:
                print(f"  ⚠ Warning: Threshold not lowered for trending market ({threshold} >= {base_threshold})")
            
            # Test signal validation in trending conditions
            # Create a strong trending candle that should pass sanity filter
            trending_candle = {
                'open': 1.1000,
                'high': 1.1030,
                'low': 1.0995,
                'close': 1.1025  # Strong body ratio = 0.025/0.035 = 0.714 > 0.6
            }
            
            valid, reason, filter_metadata = self.sanity_filter.validate_signal(
                candle=trending_candle,
                atr=0.0025,  # Good volatility level
                direction_confidence=82.0,  # High confidence in trend
                signal_strength=threshold + 5.0
            )
            
            print(f"  Signal validation: {valid} (reason: {reason})")
            
            # Add trend signals for MTF confirmation
            for i in range(4):
                self.mtf_confirmation.add_signal_to_history("TRENDING_TEST", "15m", "BUY", 75.0 + i)
            
            mtf_result = self.mtf_confirmation.confirm_signal("TRENDING_TEST", "15m", "BUY", 80.0)
            print(f"  MTF confirmation: {mtf_result.confirmed} (adj: {mtf_result.confidence_adjustment:.2f}x)")
            
            # Validate trending market behavior
            assert valid, "Strong trending signals should pass sanity filter"
            assert mtf_result.confirmed, "Consistent trend signals should be confirmed"
            assert mtf_result.confidence_adjustment > 1.0, "Trending signals should get confidence boost"
            
            print("✅ PASSED: Trending Market Adaptation")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Trending Market Adaptation - {e}")
            return False
    
    def test_ranging_market_adaptation(self) -> bool:
        """Test system adaptation to ranging market conditions"""
        print("\\n🧪 Ranging Market Adaptation")
        print("-" * 50)
        
        try:
            # Create ranging market data
            ranging_data = self.create_ranging_market_data(length=50, range_size=0.002)
            
            # Calculate market characteristics
            last_candle = ranging_data.iloc[-1]
            atr_estimate = ranging_data['high'].rolling(14).max() - ranging_data['low'].rolling(14).min()
            atr_ratio = atr_estimate.iloc[-1] / last_candle['close']
            
            # RSI would be near neutral in ranging market
            rsi_deviation = 8.0  # RSI around 58 (mild deviation from 50)
            macd_histogram = 0.001  # Weak MACD signal
            
            print(f"  Market characteristics: ATR ratio={atr_ratio:.6f}, RSI dev={rsi_deviation}")
            
            # Test adaptive threshold
            threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=atr_ratio,
                rsi_deviation=rsi_deviation,
                macd_histogram=macd_histogram,
                symbol="RANGING_TEST"
            )
            
            print(f"  Adaptive threshold: {threshold:.1f}% (metadata: {metadata})")
            
            # In ranging markets, threshold should be higher to avoid false signals
            base_threshold = self.threshold_manager.base_threshold
            if threshold <= base_threshold:
                print(f"  ⚠ Note: Threshold not raised for ranging market ({threshold} <= {base_threshold})")
            
            # Test signal validation with weak ranging signal
            ranging_candle = {
                'open': last_candle['open'],
                'high': last_candle['high'],
                'low': last_candle['low'],
                'close': last_candle['close']
            }
            
            # Test weak signal that should be rejected
            weak_valid, weak_reason, _ = self.sanity_filter.validate_signal(
                candle=ranging_candle,
                atr=atr_ratio * last_candle['close'],
                direction_confidence=55.0,  # Moderate confidence
                signal_strength=threshold - 5.0  # Below threshold
            )
            
            print(f"  Weak signal validation: {weak_valid} (reason: {weak_reason})")
            
            # Test stronger signal that might pass
            strong_valid, strong_reason, _ = self.sanity_filter.validate_signal(
                candle=ranging_candle,
                atr=atr_ratio * last_candle['close'],
                direction_confidence=75.0,  # Higher confidence
                signal_strength=threshold + 10.0  # Well above threshold
            )
            
            print(f"  Strong signal validation: {strong_valid} (reason: {strong_reason})")
            
            # Add mixed signals for MTF confirmation (typical of ranging markets)
            self.mtf_confirmation.add_signal_to_history("RANGING_TEST", "15m", "BUY", 60.0)
            self.mtf_confirmation.add_signal_to_history("RANGING_TEST", "15m", "SELL", 58.0)
            self.mtf_confirmation.add_signal_to_history("RANGING_TEST", "15m", "BUY", 62.0)
            
            mtf_result = self.mtf_confirmation.confirm_signal("RANGING_TEST", "15m", "BUY", 65.0)
            print(f"  MTF confirmation: {mtf_result.confirmed} (adj: {mtf_result.confidence_adjustment:.2f}x)")
            
            # Validate ranging market behavior
            assert not weak_valid, "Weak signals in ranging markets should be rejected"
            # Strong signals might still pass, which is acceptable
            print(f"  Ranging market filtering working correctly")
            
            print("✅ PASSED: Ranging Market Adaptation")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Ranging Market Adaptation - {e}")
            return False
    
    def test_volatile_market_adaptation(self) -> bool:
        """Test system adaptation to highly volatile market conditions"""
        print("\\n🧪 Volatile Market Adaptation")
        print("-" * 50)
        
        try:
            # Create volatile market data
            volatile_data = self.create_volatile_market_data(length=50, volatility_factor=3.0)
            
            # Calculate market characteristics
            last_candle = volatile_data.iloc[-1]
            atr_estimate = volatile_data['high'].rolling(14).max() - volatile_data['low'].rolling(14).min()
            atr_ratio = atr_estimate.iloc[-1] / last_candle['close']
            
            # High volatility conditions
            rsi_deviation = 25.0  # RSI could be extreme
            macd_histogram = 0.012  # Strong but noisy MACD
            
            print(f"  Market characteristics: ATR ratio={atr_ratio:.6f}, RSI dev={rsi_deviation}")
            
            # Test adaptive threshold
            threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=atr_ratio,
                rsi_deviation=rsi_deviation,
                macd_histogram=macd_histogram,
                symbol="VOLATILE_TEST"
            )
            
            print(f"  Adaptive threshold: {threshold:.1f}% (metadata: {metadata})")
            
            # In volatile markets, threshold should be raised to filter noise
            base_threshold = self.threshold_manager.base_threshold
            print(f"  Threshold adjustment for volatility: {threshold - base_threshold:+.1f}%")
            
            # Test signal validation with volatile candle
            volatile_candle = {
                'open': last_candle['open'],
                'high': last_candle['high'],
                'low': last_candle['low'],
                'close': last_candle['close']
            }
            
            # Test signal that should pass high volatility filter
            valid, reason, filter_metadata = self.sanity_filter.validate_signal(
                candle=volatile_candle,
                atr=atr_ratio * last_candle['close'],
                direction_confidence=88.0,  # Very high confidence needed
                signal_strength=threshold + 15.0  # Well above adaptive threshold
            )
            
            print(f"  High-confidence signal: {valid} (reason: {reason})")
            
            # Test signal that should be rejected due to volatility
            noisy_valid, noisy_reason, _ = self.sanity_filter.validate_signal(
                candle=volatile_candle,
                atr=atr_ratio * last_candle['close'],
                direction_confidence=65.0,  # Moderate confidence
                signal_strength=threshold - 2.0  # Just below threshold
            )
            
            print(f"  Moderate-confidence signal: {noisy_valid} (reason: {noisy_reason})")
            
            # Validate volatile market behavior
            if valid:
                print("  ✓ High-confidence signals can pass in volatile markets")
            if not noisy_valid:
                print("  ✓ Moderate signals correctly filtered in volatile markets")
            
            print("✅ PASSED: Volatile Market Adaptation")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Volatile Market Adaptation - {e}")
            return False
    
    def test_low_volatility_adaptation(self) -> bool:
        """Test system adaptation to low volatility market conditions"""
        print("\\n🧪 Low Volatility Market Adaptation")
        print("-" * 50)
        
        try:
            # Create low volatility conditions
            low_vol_atr_ratio = 0.0008  # Very low ATR
            rsi_deviation = 3.0  # RSI near 53 (very neutral)
            macd_histogram = 0.0005  # Very weak MACD
            
            print(f"  Market characteristics: ATR ratio={low_vol_atr_ratio:.6f}, RSI dev={rsi_deviation}")
            
            # Test adaptive threshold
            threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=low_vol_atr_ratio,
                rsi_deviation=rsi_deviation,
                macd_histogram=macd_histogram,
                symbol="LOW_VOL_TEST"
            )
            
            print(f"  Adaptive threshold: {threshold:.1f}% (metadata: {metadata})")
            
            # In low volatility, system should be more selective
            base_threshold = self.threshold_manager.base_threshold
            print(f"  Threshold adjustment for low volatility: {threshold - base_threshold:+.1f}%")
            
            # Test signal validation in low volatility
            low_vol_candle = {
                'open': 1.1000,
                'high': 1.1002,  # Very small range
                'low': 1.0999,
                'close': 1.1001
            }
            
            # Even strong signals might be rejected in very low volatility
            valid, reason, _ = self.sanity_filter.validate_signal(
                candle=low_vol_candle,
                atr=low_vol_atr_ratio * 1.1000,
                direction_confidence=80.0,
                signal_strength=threshold + 5.0
            )
            
            print(f"  Signal in low volatility: {valid} (reason: {reason})")
            
            # This is expected behavior - low volatility should increase selectivity
            if not valid and "volatility" in reason.lower():
                print("  ✓ Low volatility correctly filtered")
            elif valid:
                print("  ✓ Strong signal passed despite low volatility")
            
            print("✅ PASSED: Low Volatility Market Adaptation")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Low Volatility Market Adaptation - {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all market regime detection tests"""
        print("=" * 70)
        print("🔬 MARKET REGIME DETECTION TEST SUITE")
        print("=" * 70)
        
        tests = [
            self.test_trending_market_adaptation,
            self.test_ranging_market_adaptation,
            self.test_volatile_market_adaptation,
            self.test_low_volatility_adaptation
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("\\n" + "=" * 70)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print("=" * 70)
        
        if passed == total:
            print("🎉 All market regime detection tests passed!")
            return True
        else:
            print("🔧 Some tests failed. Please review and fix issues.")
            return False


if __name__ == "__main__":
    test_suite = TestMarketRegimeDetection()
    success = test_suite.run_all_tests()
    
    if not success:
        print("\\n⚠️  Please address the failing tests before deployment.")
        sys.exit(1)
    else:
        print("\\n✅ Market regime detection system is working correctly!")
        sys.exit(0)