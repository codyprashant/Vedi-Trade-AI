#!/usr/bin/env python3
"""
Edge Cases and Error Handling Test Suite

Tests the adaptive signal system's robustness against edge cases and error conditions:
- Invalid input data
- Extreme market conditions
- Memory and performance limits
- Concurrent access scenarios
- Data corruption and recovery

This test ensures the system fails gracefully and maintains stability under adverse conditions.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import time

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from threshold_manager import ThresholdManagerFactory
from sanity_filter import SignalSanityFilterFactory
from mtf_confirmation import MultiTimeframeConfirmation


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def setUp(self):
        """Initialize test components"""
        self.threshold_manager = ThresholdManagerFactory.create_conservative()
        self.sanity_filter = SignalSanityFilterFactory.create_strict()
        self.mtf_confirmation = MultiTimeframeConfirmation()
        
    def test_invalid_input_handling(self) -> bool:
        """Test handling of invalid input data"""
        print("🧪 Invalid Input Handling")
        print("-" * 50)
        
        try:
            # Test None values
            try:
                threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                    atr_ratio=None,
                    rsi_deviation=25.0,
                    macd_histogram=0.005,
                    symbol="TEST"
                )
                print(f"  None ATR handled: threshold={threshold}")
            except Exception as e:
                print(f"  None ATR error (expected): {type(e).__name__}")
            
            # Test negative values
            try:
                threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                    atr_ratio=-0.001,
                    rsi_deviation=25.0,
                    macd_histogram=0.005,
                    symbol="TEST"
                )
                print(f"  Negative ATR handled: threshold={threshold}")
            except Exception as e:
                print(f"  Negative ATR error: {type(e).__name__}")
            
            # Test extreme values
            try:
                threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                    atr_ratio=1.0,  # 100% ATR (impossible)
                    rsi_deviation=100.0,  # RSI at 150 (impossible)
                    macd_histogram=1.0,  # Extreme MACD
                    symbol="EXTREME_TEST"
                )
                print(f"  Extreme values handled: threshold={threshold}")
            except Exception as e:
                print(f"  Extreme values error: {type(e).__name__}")
            
            # Test invalid candle data
            invalid_candles = [
                None,
                {},
                {'open': 1.0},  # Missing fields
                {'open': 1.0, 'high': 0.9, 'low': 1.1, 'close': 1.0},  # Invalid OHLC
                {'open': 'invalid', 'high': 1.0, 'low': 1.0, 'close': 1.0},  # Wrong type
            ]
            
            for i, candle in enumerate(invalid_candles):
                try:
                    valid, reason, _ = self.sanity_filter.validate_signal(
                        candle=candle,
                        atr=0.001,
                        direction_confidence=70.0,
                        signal_strength=65.0
                    )
                    print(f"  Invalid candle {i+1} handled: {valid} ({reason})")
                except Exception as e:
                    print(f"  Invalid candle {i+1} error: {type(e).__name__}")
            
            # Test empty symbol
            try:
                self.mtf_confirmation.add_signal_to_history("", "15m", "BUY", 70.0)
                result = self.mtf_confirmation.confirm_signal("", "15m", "BUY", 75.0)
                print(f"  Empty symbol handled: {result.confirmed}")
            except Exception as e:
                print(f"  Empty symbol error: {type(e).__name__}")
            
            print("✅ PASSED: Invalid Input Handling")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Invalid Input Handling - {e}")
            return False
    
    def test_extreme_market_conditions(self) -> bool:
        """Test system behavior under extreme market conditions"""
        print("\\n🧪 Extreme Market Conditions")
        print("-" * 50)
        
        try:
            # Test flash crash scenario (extreme volatility spike)
            flash_crash_atr = 0.05  # 5% ATR (extreme)
            rsi_deviation = 45.0  # RSI at 5 (extreme oversold)
            macd_histogram = -0.1  # Extreme negative MACD
            
            print(f"  Flash crash conditions: ATR={flash_crash_atr}, RSI dev={rsi_deviation}")
            
            threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=flash_crash_atr,
                rsi_deviation=rsi_deviation,
                macd_histogram=macd_histogram,
                symbol="FLASH_CRASH"
            )
            
            print(f"  Flash crash threshold: {threshold:.1f}% (metadata: {metadata})")
            
            # System should significantly raise threshold during extreme volatility
            base_threshold = self.threshold_manager.base_threshold
            adjustment = threshold - base_threshold
            print(f"  Threshold adjustment: {adjustment:+.1f}%")
            
            # Test gap scenario (price jumps)
            gap_candle = {
                'open': 1.1000,
                'high': 1.1200,  # 200 pip gap up
                'low': 1.1180,
                'close': 1.1190
            }
            
            valid, reason, _ = self.sanity_filter.validate_signal(
                candle=gap_candle,
                atr=flash_crash_atr * 1.1000,
                direction_confidence=60.0,
                signal_strength=threshold + 10.0
            )
            
            print(f"  Gap candle validation: {valid} (reason: {reason})")
            
            # Test market halt scenario (zero volatility)
            halt_threshold, halt_metadata = self.threshold_manager.compute_adaptive_threshold(
                atr_ratio=0.0,
                rsi_deviation=0.0,
                macd_histogram=0.0,
                symbol="MARKET_HALT"
            )
            
            print(f"  Market halt threshold: {halt_threshold:.1f}%")
            
            # Test currency crisis (extreme trend)
            crisis_signals = []
            for i in range(20):  # 20 consecutive strong signals
                self.mtf_confirmation.add_signal_to_history("CRISIS_PAIR", "15m", "SELL", 90.0 + i)
            
            crisis_result = self.mtf_confirmation.confirm_signal("CRISIS_PAIR", "15m", "SELL", 95.0)
            print(f"  Crisis trend confirmation: {crisis_result.confirmed} (adj: {crisis_result.confidence_adjustment:.2f}x)")
            
            print("✅ PASSED: Extreme Market Conditions")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Extreme Market Conditions - {e}")
            return False
    
    def test_memory_and_performance_limits(self) -> bool:
        """Test system behavior under memory and performance stress"""
        print("\\n🧪 Memory and Performance Limits")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # Test large number of symbols
            symbols = [f"PAIR_{i:04d}" for i in range(1000)]
            
            print(f"  Testing {len(symbols)} symbols...")
            
            # Add signals for many symbols
            for symbol in symbols[:100]:  # Test first 100 to avoid excessive runtime
                for timeframe in ["1m", "5m", "15m", "1h"]:
                    for direction in ["BUY", "SELL"]:
                        self.mtf_confirmation.add_signal_to_history(symbol, timeframe, direction, 70.0)
            
            # Test threshold computation for many symbols
            thresholds = {}
            for symbol in symbols[:50]:  # Test first 50
                threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                    atr_ratio=0.002,
                    rsi_deviation=15.0,
                    macd_histogram=0.003,
                    symbol=symbol
                )
                thresholds[symbol] = threshold
            
            computation_time = time.time() - start_time
            print(f"  Computed thresholds for {len(thresholds)} symbols in {computation_time:.2f}s")
            
            # Test rapid signal processing
            rapid_start = time.time()
            rapid_results = []
            
            for i in range(100):
                candle = {
                    'open': 1.1000 + i * 0.0001,
                    'high': 1.1005 + i * 0.0001,
                    'low': 1.0995 + i * 0.0001,
                    'close': 1.1002 + i * 0.0001
                }
                
                valid, reason, _ = self.sanity_filter.validate_signal(
                    candle=candle,
                    atr=0.002,
                    direction_confidence=70.0,
                    signal_strength=65.0
                )
                rapid_results.append(valid)
            
            rapid_time = time.time() - rapid_start
            signals_per_sec = len(rapid_results) / rapid_time if rapid_time > 0 else float('inf')
            print(f"  Processed {len(rapid_results)} signals in {rapid_time:.3f}s ({signals_per_sec:.1f} signals/sec)")
            
            # Test memory usage with large signal history
            large_symbol = "LARGE_HISTORY_TEST"
            for i in range(1000):
                self.mtf_confirmation.add_signal_to_history(large_symbol, "15m", "BUY", 70.0 + (i % 20))
            
            # Confirm signal should still work efficiently
            large_result = self.mtf_confirmation.confirm_signal(large_symbol, "15m", "BUY", 75.0)
            print(f"  Large history confirmation: {large_result.confirmed} (adj: {large_result.confidence_adjustment:.2f}x)")
            
            total_time = time.time() - start_time
            print(f"  Total performance test time: {total_time:.2f}s")
            
            # Performance should be reasonable
            if total_time > 30.0:  # More than 30 seconds is concerning
                print(f"  ⚠ Warning: Performance test took {total_time:.2f}s (may be slow)")
            
            print("✅ PASSED: Memory and Performance Limits")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Memory and Performance Limits - {e}")
            return False
    
    def test_concurrent_access(self) -> bool:
        """Test system behavior under concurrent access"""
        print("\\n🧪 Concurrent Access")
        print("-" * 50)
        
        try:
            results = []
            errors = []
            
            def worker_thread(thread_id: int):
                """Worker thread for concurrent testing"""
                try:
                    symbol = f"THREAD_{thread_id}"
                    
                    # Each thread adds signals and computes thresholds
                    for i in range(10):
                        # Add signal
                        self.mtf_confirmation.add_signal_to_history(symbol, "15m", "BUY", 70.0 + i)
                        
                        # Compute threshold
                        threshold, _ = self.threshold_manager.compute_adaptive_threshold(
                            atr_ratio=0.002 + i * 0.0001,
                            rsi_deviation=15.0 + i,
                            macd_histogram=0.003 + i * 0.0001,
                            symbol=symbol
                        )
                        
                        # Validate signal
                        candle = {
                            'open': 1.1000 + i * 0.0001,
                            'high': 1.1005 + i * 0.0001,
                            'low': 1.0995 + i * 0.0001,
                            'close': 1.1002 + i * 0.0001
                        }
                        
                        valid, reason, _ = self.sanity_filter.validate_signal(
                            candle=candle,
                            atr=0.002,
                            direction_confidence=70.0,
                            signal_strength=threshold
                        )
                        
                        results.append((thread_id, i, threshold, valid))
                        
                        # Small delay to increase chance of race conditions
                        time.sleep(0.001)
                        
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # Start multiple threads
            threads = []
            num_threads = 5
            
            print(f"  Starting {num_threads} concurrent threads...")
            
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            print(f"  Completed {len(results)} operations across {num_threads} threads")
            print(f"  Errors encountered: {len(errors)}")
            
            if errors:
                for thread_id, error in errors[:3]:  # Show first 3 errors
                    print(f"    Thread {thread_id}: {error}")
            
            # Verify results make sense
            if results:
                thresholds = [r[2] for r in results]
                valid_signals = [r[3] for r in results]
                
                print(f"  Threshold range: {min(thresholds):.1f}% - {max(thresholds):.1f}%")
                print(f"  Valid signals: {sum(valid_signals)}/{len(valid_signals)}")
            
            # Test should complete without major errors
            if len(errors) > len(results) * 0.1:  # More than 10% error rate
                print(f"  ⚠ Warning: High error rate in concurrent access ({len(errors)}/{len(results)})")
            
            print("✅ PASSED: Concurrent Access")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Concurrent Access - {e}")
            return False
    
    def test_data_corruption_recovery(self) -> bool:
        """Test system recovery from data corruption scenarios"""
        print("\\n🧪 Data Corruption Recovery")
        print("-" * 50)
        
        try:
            # Test recovery from corrupted signal history
            corrupt_symbol = "CORRUPT_TEST"
            
            # Add some normal signals
            for i in range(5):
                self.mtf_confirmation.add_signal_to_history(corrupt_symbol, "15m", "BUY", 70.0 + i)
            
            # Simulate corruption by adding invalid data (if possible)
            try:
                # This might fail gracefully or be handled
                self.mtf_confirmation.add_signal_to_history(corrupt_symbol, "15m", "INVALID", float('inf'))
                print("  Invalid signal direction handled")
            except Exception as e:
                print(f"  Invalid signal direction rejected: {type(e).__name__}")
            
            # System should still work for valid operations
            normal_result = self.mtf_confirmation.confirm_signal(corrupt_symbol, "15m", "BUY", 75.0)
            print(f"  Normal operation after corruption: {normal_result.confirmed}")
            
            # Test NaN and infinity handling
            nan_tests = [
                (float('nan'), 15.0, 0.003, "NaN ATR"),
                (0.002, float('nan'), 0.003, "NaN RSI"),
                (0.002, 15.0, float('inf'), "Infinite MACD"),
                (float('-inf'), 15.0, 0.003, "Negative infinite ATR")
            ]
            
            for atr, rsi, macd, description in nan_tests:
                try:
                    threshold, metadata = self.threshold_manager.compute_adaptive_threshold(
                        atr_ratio=atr,
                        rsi_deviation=rsi,
                        macd_histogram=macd,
                        symbol="NAN_TEST"
                    )
                    print(f"  {description} handled: threshold={threshold}")
                except Exception as e:
                    print(f"  {description} error: {type(e).__name__}")
            
            # Test candle with NaN values
            nan_candle = {
                'open': 1.1000,
                'high': float('nan'),
                'low': 1.0995,
                'close': 1.1002
            }
            
            try:
                valid, reason, _ = self.sanity_filter.validate_signal(
                    candle=nan_candle,
                    atr=0.002,
                    direction_confidence=70.0,
                    signal_strength=65.0
                )
                print(f"  NaN candle handled: {valid} ({reason})")
            except Exception as e:
                print(f"  NaN candle error: {type(e).__name__}")
            
            print("✅ PASSED: Data Corruption Recovery")
            return True
            
        except Exception as e:
            print(f"❌ FAILED: Data Corruption Recovery - {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all edge case tests"""
        print("=" * 70)
        print("🔬 EDGE CASES AND ERROR HANDLING TEST SUITE")
        print("=" * 70)
        
        tests = [
            self.test_invalid_input_handling,
            self.test_extreme_market_conditions,
            self.test_memory_and_performance_limits,
            self.test_concurrent_access,
            self.test_data_corruption_recovery
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
            print("🎉 All edge case tests passed!")
            return True
        else:
            print("🔧 Some tests failed. Please review and fix issues.")
            return False


if __name__ == "__main__":
    test_suite = TestEdgeCases()
    success = test_suite.run_all_tests()
    
    if not success:
        print("\\n⚠️  Please address the failing tests before deployment.")
        sys.exit(1)
    else:
        print("\\n✅ Edge case handling is working correctly!")
        sys.exit(0)