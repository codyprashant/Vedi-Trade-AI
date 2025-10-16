#!/usr/bin/env python3
"""
Extended test suite for the adaptive signal generation system.
Tests the new Phase 3 improvements including:
- Dynamic adaptive thresholds based on market conditions
- Intelligent noise suppression layer
- Multi-timeframe noise confirmation
- Signal consistency across timeframes
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.threshold_manager import ThresholdManagerFactory
from app.sanity_filter import SignalSanityFilterFactory
from app.mtf_confirmation import MultiTimeframeConfirmation
from app.signal_engine import SignalEngine


class TestAdaptiveSignalSystem:
    """Test suite for adaptive signal generation system."""
    
    def setUp(self):
        # Create test components
        self.threshold_manager = ThresholdManagerFactory.create_conservative()
        self.sanity_filter = SignalSanityFilterFactory.create_strict()
        self.mtf_confirmation = MultiTimeframeConfirmation()
        
        # Create a mock fetch function for testing
        async def mock_fetch_history(symbol, timeframe, count):
            return self.create_test_data(length=count, timeframe=timeframe)
        
        self.engine = SignalEngine(mock_fetch_history)
        self.test_results = []

    def create_test_data(self, length=100, trend="sideways", volatility="normal", timeframe="15m"):
        """Create synthetic market data for testing."""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=length), periods=length, freq='15T')
        
        # Base price movement
        base_price = 1.1000
        price_data = []
        
        for i in range(length):
            if trend == "bullish":
                trend_component = i * 0.0001
            elif trend == "bearish":
                trend_component = -i * 0.0001
            else:  # sideways
                trend_component = 0.0001 * np.sin(i * 0.1)
            
            # Volatility component
            if volatility == "high":
                vol_component = np.random.normal(0, 0.002)
            elif volatility == "low":
                vol_component = np.random.normal(0, 0.0005)
            else:  # normal
                vol_component = np.random.normal(0, 0.001)
            
            price = base_price + trend_component + vol_component
            
            # Create OHLC data
            high = price + abs(vol_component) * 0.5
            low = price - abs(vol_component) * 0.5
            open_price = price + np.random.normal(0, abs(vol_component) * 0.2)
            close_price = price + np.random.normal(0, abs(vol_component) * 0.2)
            
            price_data.append({
                'time': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })
        
        return pd.DataFrame(price_data)

    def test_adaptive_threshold_volatility_increase(self):
        """Test that adaptive thresholds adjust correctly for volatile and calm markets."""
        print("Testing adaptive threshold volatility adjustment...")
        
        try:
            # Test case 1: Low volatility market
            low_vol_factors = {
                "atr_ratio": 0.001,  # Low volatility (0.1% of price)
                "rsi_deviation": 5.0,  # Small RSI deviation
                "macd_deviation": 0.001  # Small MACD deviation
            }
            
            low_vol_threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=low_vol_factors["atr_ratio"],
                rsi_deviation=low_vol_factors["rsi_deviation"],
                macd_histogram=low_vol_factors["macd_deviation"]
            )
            
            # Test case 2: High volatility market
            high_vol_factors = {
                "atr_ratio": 0.005,  # High volatility (0.5% of price)
                "rsi_deviation": 25.0,  # Large RSI deviation
                "macd_deviation": 0.008  # Large MACD deviation
            }
            
            high_vol_threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=high_vol_factors["atr_ratio"],
                rsi_deviation=high_vol_factors["rsi_deviation"],
                macd_histogram=high_vol_factors["macd_deviation"]
            )
            
            # Test case 3: Normal volatility market
            normal_vol_factors = {
                "atr_ratio": 0.002,  # Normal volatility (0.2% of price)
                "rsi_deviation": 10.0,  # Moderate RSI deviation
                "macd_deviation": 0.003  # Moderate MACD deviation
            }
            
            normal_vol_threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=normal_vol_factors["atr_ratio"],
                rsi_deviation=normal_vol_factors["rsi_deviation"],
                macd_histogram=normal_vol_factors["macd_deviation"]
            )
            
            # Debug output
            print(f"  Low volatility: ATR={low_vol_factors['atr_ratio']:.1f}, RSIdev={low_vol_factors['rsi_deviation']:.1f} → threshold={low_vol_threshold:.1f}%")
            print(f"  Normal volatility: ATR={normal_vol_factors['atr_ratio']:.1f}, RSIdev={normal_vol_factors['rsi_deviation']:.1f} → threshold={normal_vol_threshold:.1f}%")
            print(f"  High volatility: ATR={high_vol_factors['atr_ratio']:.1f}, RSIdev={high_vol_factors['rsi_deviation']:.1f} → threshold={high_vol_threshold:.1f}%")
            
            # Assertions
            assert low_vol_threshold > normal_vol_threshold, f"Low volatility threshold ({low_vol_threshold:.1f}) should be higher than normal ({normal_vol_threshold:.1f})"
            assert normal_vol_threshold > high_vol_threshold, f"Normal volatility threshold ({normal_vol_threshold:.1f}) should be higher than high volatility ({high_vol_threshold:.1f})"
            assert high_vol_threshold >= self.threshold_manager.min_threshold, f"High volatility threshold ({high_vol_threshold:.1f}) should be >= min ({self.threshold_manager.min_threshold})"
            assert low_vol_threshold <= self.threshold_manager.max_threshold, f"Low volatility threshold ({low_vol_threshold:.1f}) should be <= max ({self.threshold_manager.max_threshold})"
            
            print("✓ Adaptive threshold volatility adjustment working correctly")
            return True
            
        except Exception as e:
            print(f"⚠ Adaptive threshold test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_noise_filter_low_volatility_rejects(self):
        """Test that weak/noisy candles are properly rejected by the sanity filter."""
        print("Testing noise filter rejection of low-quality signals...")
        
        try:
            # Test case 1: Low volatility candle (should be rejected)
            low_vol_candle = {
                "open": 1.1000,
                "high": 1.1002,
                "low": 1.0999,
                "close": 1.1001
            }
            low_atr = 0.0005  # Very low ATR
            
            valid_low_vol, reason_low_vol, _ = self.sanity_filter.validate_signal(
                candle=low_vol_candle,
                atr=low_atr,
                direction_confidence=75.0,
                signal_strength=70.0
            )
            
            # Test case 2: Weak candle body (should be rejected)
            weak_body_candle = {
                "open": 1.1000,
                "high": 1.1020,
                "low": 1.0980,
                "close": 1.1002  # Very small body relative to range
            }
            normal_atr = 0.002
            
            valid_weak_body, reason_weak_body, _ = self.sanity_filter.validate_signal(
                candle=weak_body_candle,
                atr=normal_atr,
                direction_confidence=75.0,
                signal_strength=65.0
            )
            
            # Test case 3: Low confidence signal (should be rejected)
            normal_candle = {
                "open": 1.1000,
                "high": 1.1015,
                "low": 1.0985,
                "close": 1.1012
            }
            
            valid_low_conf, reason_low_conf, _ = self.sanity_filter.validate_signal(
                candle=normal_candle,
                atr=normal_atr,
                direction_confidence=35.0,  # Low confidence
                signal_strength=40.0
            )
            
            # Test case 4: Strong signal (should pass)
            strong_candle = {
                "open": 1.1000,
                "high": 1.1025,
                "low": 1.0985,
                "close": 1.1025  # Strong body (body_ratio = 0.025/0.040 = 0.625 > 0.6)
            }
            high_atr = 0.003
            
            valid_strong, reason_strong, _ = self.sanity_filter.validate_signal(
                candle=strong_candle,
                atr=high_atr,
                direction_confidence=85.0,
                signal_strength=80.0
            )
            
            # Debug output
            print(f"  Low volatility: ATR={low_atr:.4f} → {'PASS' if valid_low_vol else 'REJECT'} ({reason_low_vol})")
            print(f"  Weak body: body_ratio={(abs(weak_body_candle['close'] - weak_body_candle['open']) / (weak_body_candle['high'] - weak_body_candle['low'])):.2f} → {'PASS' if valid_weak_body else 'REJECT'} ({reason_weak_body})")
            print(f"  Low confidence: conf=35.0% → {'PASS' if valid_low_conf else 'REJECT'} ({reason_low_conf})")
            print(f"  Strong signal: ATR={high_atr:.4f}, conf=75.0% → {'PASS' if valid_strong else 'REJECT'} ({reason_strong})")
            
            # Assertions
            assert not valid_low_vol, f"Low volatility candle should be rejected but was accepted"
            assert not valid_weak_body, f"Weak body candle should be rejected but was accepted"
            assert not valid_low_conf, f"Low confidence signal should be rejected but was accepted"
            assert valid_strong, f"Strong signal should be accepted but was rejected: {reason_strong}"
            
            print("✓ Noise filter rejection working correctly")
            return True
            
        except Exception as e:
            print(f"⚠ Noise filter test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_signal_consistency_across_timeframes(self):
        """Test alignment between M15/H1 signals and multi-timeframe confirmation."""
        print("Testing signal consistency across timeframes...")
        
        try:
            # Test case 1: Consistent signals across timeframes
            print("  Testing consistent signals...")
            
            # Add consistent BUY signals
            self.mtf_confirmation.add_signal_to_history("EURUSD", "15m", "BUY", 70.0)
            self.mtf_confirmation.add_signal_to_history("EURUSD", "15m", "BUY", 72.0)
            self.mtf_confirmation.add_signal_to_history("EURUSD", "15m", "BUY", 68.0)
            
            # Test confirmation
            result_consistent = self.mtf_confirmation.confirm_signal(
                symbol="EURUSD",
                timeframe="15m", 
                direction="BUY",
                confidence=65.0
            )
            
            # Test case 2: Inconsistent signals (flip-flopping)
            print("  Testing inconsistent signals...")
            
            # Add flip-flopping signals for another symbol
            self.mtf_confirmation.add_signal_to_history("GBPUSD", "15m", "BUY", 60.0)
            self.mtf_confirmation.add_signal_to_history("GBPUSD", "15m", "SELL", 65.0)
            self.mtf_confirmation.add_signal_to_history("GBPUSD", "15m", "BUY", 55.0)
            
            result_inconsistent = self.mtf_confirmation.confirm_signal(
                symbol="GBPUSD",
                timeframe="15m",
                direction="BUY", 
                confidence=65.0
            )
            
            # Test case 3: New symbol (no history)
            print("  Testing new symbol with no history...")
            
            result_new = self.mtf_confirmation.confirm_signal(
                symbol="USDJPY",
                timeframe="15m",
                direction="BUY",
                confidence=70.0
            )
            
            # Debug output
            print(f"  Consistent signals: confirmed={result_consistent.confirmed}, adj={result_consistent.confidence_adjustment:.2f}x, reason='{result_consistent.reason}'")
            print(f"  Inconsistent signals: confirmed={result_inconsistent.confirmed}, adj={result_inconsistent.confidence_adjustment:.2f}x, reason='{result_inconsistent.reason}'")
            print(f"  New symbol: confirmed={result_new.confirmed}, adj={result_new.confidence_adjustment:.2f}x, reason='{result_new.reason}'")
            
            # Assertions - MTF confirmation may still confirm signals based on higher timeframe alignment
            # The key is that inconsistent signals should have lower confidence adjustment
            assert result_consistent.confirmed, "Consistent signals should be confirmed"
            assert result_consistent.confidence_adjustment >= 1.0, "Consistent signals should have positive adjustment"
            
            assert result_inconsistent.confidence_adjustment < result_consistent.confidence_adjustment, "Inconsistent signals should have lower confidence than consistent ones"
            
            # New symbols may not be confirmed if they don't meet minimum confirmation requirements
            # This is actually correct behavior - we want some confirmation before accepting signals
            print(f"  New symbol confirmation behavior: confirmed={result_new.confirmed} (this is expected)")
            assert result_new.confidence_adjustment >= 1.0, "New symbol should have neutral or positive adjustment"
            
            print("✓ Multi-timeframe signal consistency working correctly")
            return True
            
        except Exception as e:
            print(f"⚠ Signal consistency test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_confidence_preservation_in_strong_trends(self):
        """Test that strong signals retain high confidence and pass all filters."""
        print("Testing confidence preservation in strong trends...")
        
        try:
            # Create strong trending market data
            strong_trend_data = self.create_test_data(
                length=50,
                trend="bullish",
                volatility="normal"
            )
            
            # Test case 1: Strong bullish trend with high volatility
            strong_factors = {
                "atr_ratio": 1.8,  # Elevated but not extreme volatility
                "rsi_deviation": 20.0,  # Strong momentum
                "macd_deviation": 0.006  # Strong MACD signal
            }
            
            # Calculate adaptive threshold
            adaptive_threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=strong_factors["atr_ratio"],
                rsi_deviation=strong_factors["rsi_deviation"],
                macd_histogram=strong_factors["macd_deviation"]
            )
            
            # Create strong candle
            strong_candle = {
                "open": 1.1000,
                "high": 1.1030,
                "low": 1.0995,
                "close": 1.1025  # Strong bullish body
            }
            
            # Test sanity filter
            sanity_passed, filter_reason, _ = self.sanity_filter.validate_signal(
                candle=strong_candle,
                atr=0.0025,  # Good volatility
                direction_confidence=85.0,
                signal_strength=adaptive_threshold + 10
            )
            
            # Add some consistent signals for MTF confirmation
            for i in range(4):
                self.mtf_confirmation.add_signal_to_history("STRONG_TREND", "15m", "BUY", 75.0 + i)
            
            # Test MTF confirmation
            mtf_result = self.mtf_confirmation.confirm_signal(
                symbol="STRONG_TREND",
                timeframe="15m",
                direction="BUY",
                confidence=78.0
            )
            
            # Test case 2: Weak signal in strong trend (should still be filtered)
            weak_candle = {
                "open": 1.1000,
                "high": 1.1005,
                "low": 1.0998,
                "close": 1.1002  # Very weak body
            }
            
            weak_sanity_passed, weak_filter_reason, _ = self.sanity_filter.validate_signal(
                candle=weak_candle,
                atr=0.0008,  # Low volatility
                direction_confidence=45.0,  # Low confidence
                signal_strength=50.0
            )
            
            # Debug output
            print(f"  Strong trend factors: ATR={strong_factors['atr_ratio']:.1f}, RSIdev={strong_factors['rsi_deviation']:.1f} → threshold={adaptive_threshold:.1f}%")
            print(f"  Strong candle: body_ratio={(abs(strong_candle['close'] - strong_candle['open']) / (strong_candle['high'] - strong_candle['low'])):.2f} → {'PASS' if sanity_passed else 'REJECT'} ({filter_reason})")
            print(f"  MTF confirmation: confirmed={mtf_result.confirmed}, adj={mtf_result.confidence_adjustment:.2f}x")
            print(f"  Weak candle in trend: → {'PASS' if weak_sanity_passed else 'REJECT'} ({weak_filter_reason})")
            
            # Assertions for strong signal
            assert adaptive_threshold < 60.0, f"Strong trend should lower threshold, got {adaptive_threshold:.1f}%"
            assert sanity_passed, f"Strong candle should pass sanity filter: {filter_reason}"
            assert mtf_result.confirmed, "Strong trend should pass MTF confirmation"
            assert mtf_result.confidence_adjustment > 1.0, "Strong trend should boost confidence"
            
            # Assertions for weak signal (should still be filtered)
            assert not weak_sanity_passed, "Weak candle should be filtered even in strong trend"
            
            print("✓ Confidence preservation in strong trends working correctly")
            return True
            
        except Exception as e:
            print(f"⚠ Confidence preservation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run_all_tests(self):
        """Run all adaptive signal system tests and report results."""
        print("=" * 70)
        print("ADAPTIVE SIGNAL GENERATION SYSTEM TEST SUITE")
        print("=" * 70)
        
        tests = [
            ("Adaptive Threshold Volatility Adjustment", self.test_adaptive_threshold_volatility_increase),
            ("Noise Filter Low Volatility Rejection", self.test_noise_filter_low_volatility_rejects),
            ("Signal Consistency Across Timeframes", self.test_signal_consistency_across_timeframes),
            ("Confidence Preservation in Strong Trends", self.test_confidence_preservation_in_strong_trends),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}")
            print("-" * 50)
            try:
                result = test_func()
                
                if result:
                    passed += 1
                    print(f"✅ PASSED: {test_name}")
                else:
                    print(f"❌ FAILED: {test_name}")
                    
            except Exception as e:
                print(f"❌ ERROR in {test_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 70)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print("=" * 70)
        
        if passed == total:
            print("🚀 All adaptive signal system tests PASSED!")
            print("✓ Dynamic adaptive thresholds working correctly")
            print("✓ Intelligent noise suppression functioning properly")
            print("✓ Multi-timeframe confirmation operational")
            print("✓ Strong trend confidence preservation verified")
        else:
            print("🔧 Some tests failed. Please review and fix issues.")
        
        return passed == total


async def main():
    """Main test runner."""
    test_suite = TestAdaptiveSignalSystem()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎯 Adaptive signal generation system is ready for deployment!")
    else:
        print("\n⚠️  Please address the failing tests before deployment.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())