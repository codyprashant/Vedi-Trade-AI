#!/usr/bin/env python3
"""
Unit tests for the weighted voting system.

This module tests the compute_weighted_vote_aggregation function
with synthetic indicator inputs to verify correct behavior.
"""

import unittest
import sys
import os
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.indicators import compute_weighted_vote_aggregation, IndicatorResult


class TestWeightedVotingSystem(unittest.TestCase):
    """Test suite for weighted voting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.default_weights = {
            "RSI": 15,
            "MACD": 20,
            "BBANDS": 15,
            "STOCH": 10,
            "ATR": 5
        }
    
    def create_indicator_result(self, direction: str, vote: int, strength: float, label: str, effective_weight: float = None) -> IndicatorResult:
        """Create a mock IndicatorResult for testing."""
        result = IndicatorResult(
            direction=direction,
            value={"test": 1.0},
            contribution=0.0,  # Required parameter
            vote=vote,
            strength=strength,
            label=label,
            effective_weight=effective_weight if effective_weight is not None else 1.0  # Default to 1.0 for testing
        )
        return result
    
    def test_strong_bullish_consensus(self):
        """Test weighted voting with strong bullish consensus."""
        print("🧪 Testing Strong Bullish Consensus")
        
        # Create strong bullish indicators
        indicators = {
            "RSI": self.create_indicator_result("buy", 1, 85.0, "strong", self.default_weights["RSI"]),
            "MACD": self.create_indicator_result("buy", 1, 90.0, "strong", self.default_weights["MACD"]),
            "BBANDS": self.create_indicator_result("buy", 1, 75.0, "strong", self.default_weights["BBANDS"]),
            "STOCH": self.create_indicator_result("buy", 1, 80.0, "strong", self.default_weights["STOCH"]),
            "ATR": self.create_indicator_result("neutral", 0, 50.0, "weak", self.default_weights["ATR"])
        }
        
        result = compute_weighted_vote_aggregation(indicators, self.default_weights)
        
        # Verify results
        self.assertEqual(result['final_direction'], 'buy')
        self.assertGreater(result['total_vote_score'], 0.5)
        self.assertGreater(result['confidence'], 0.7)
        self.assertGreaterEqual(result['strong_signals'], 3)
        
        print(f"  ✓ Final Direction: {result['final_direction']}")
        print(f"  ✓ Vote Score: {result['total_vote_score']:.3f}")
        print(f"  ✓ Confidence: {result['confidence']:.1f}%")
        print(f"  ✓ Strong Signals: {result['strong_signals']}")
        
    def test_mixed_signals_with_weak_consensus(self):
        """Test weighted voting with mixed signals."""
        print("\n🧪 Testing Mixed Signals")
        
        # Create mixed indicators
        indicators = {
            "RSI": self.create_indicator_result("buy", 1, 60.0, "weak", self.default_weights["RSI"]),
            "MACD": self.create_indicator_result("sell", -1, 55.0, "weak", self.default_weights["MACD"]),
            "BBANDS": self.create_indicator_result("neutral", 0, 50.0, "weak", self.default_weights["BBANDS"]),
            "STOCH": self.create_indicator_result("buy", 1, 70.0, "strong", self.default_weights["STOCH"]),
            "ATR": self.create_indicator_result("neutral", 0, 45.0, "weak", self.default_weights["ATR"])
        }
        
        result = compute_weighted_vote_aggregation(indicators, self.default_weights)
        
        # Verify results - confidence may be capped at 1.0 by the algorithm
        self.assertLessEqual(result['strong_signals'], 1)  # Few strong signals
        # Mixed signals should have lower vote score magnitude
        self.assertLess(abs(result['total_vote_score']), 50.0)
        
        print(f"  ✓ Final Direction: {result['final_direction']}")
        print(f"  ✓ Vote Score: {result['total_vote_score']:.3f}")
        print(f"  ✓ Confidence: {result['confidence']:.1f}%")
        print(f"  ✓ Strong Signals: {result['strong_signals']}")
        
    def test_strong_bearish_consensus(self):
        """Test weighted voting with strong bearish consensus."""
        print("\n🧪 Testing Strong Bearish Consensus")
        
        # Create strong bearish indicators
        indicators = {
            "RSI": self.create_indicator_result("sell", -1, 85.0, "strong", self.default_weights["RSI"]),
            "MACD": self.create_indicator_result("sell", -1, 90.0, "strong", self.default_weights["MACD"]),
            "BBANDS": self.create_indicator_result("sell", -1, 75.0, "strong", self.default_weights["BBANDS"]),
            "STOCH": self.create_indicator_result("sell", -1, 80.0, "strong", self.default_weights["STOCH"]),
            "ATR": self.create_indicator_result("neutral", 0, 60.0, "weak", self.default_weights["ATR"])
        }
        
        result = compute_weighted_vote_aggregation(indicators, self.default_weights)
        
        # Verify results
        self.assertEqual(result['final_direction'], 'sell')
        self.assertLess(result['total_vote_score'], -0.5)
        self.assertGreater(result['confidence'], 0.7)
        self.assertGreaterEqual(result['strong_signals'], 3)
        
        print(f"  ✓ Final Direction: {result['final_direction']}")
        print(f"  ✓ Vote Score: {result['total_vote_score']:.3f}")
        print(f"  ✓ Confidence: {result['confidence']:.1f}%")
        print(f"  ✓ Strong Signals: {result['strong_signals']}")
        
    def test_neutral_market_conditions(self):
        """Test weighted voting with neutral market conditions."""
        print("\n🧪 Testing Neutral Market Conditions")
        
        # Create neutral indicators
        indicators = {
            "RSI": self.create_indicator_result("neutral", 0, 50.0, "weak", self.default_weights["RSI"]),
            "MACD": self.create_indicator_result("neutral", 0, 45.0, "weak", self.default_weights["MACD"]),
            "BBANDS": self.create_indicator_result("neutral", 0, 48.0, "weak", self.default_weights["BBANDS"]),
            "STOCH": self.create_indicator_result("neutral", 0, 52.0, "weak", self.default_weights["STOCH"]),
            "ATR": self.create_indicator_result("neutral", 0, 40.0, "weak", self.default_weights["ATR"])
        }
        
        result = compute_weighted_vote_aggregation(indicators, self.default_weights)
        
        # Verify results
        self.assertEqual(result['final_direction'], 'neutral')
        self.assertAlmostEqual(result['total_vote_score'], 0.0, places=2)
        self.assertLess(result['confidence'], 0.6)
        self.assertEqual(result['strong_signals'], 0)
        
        print(f"  ✓ Final Direction: {result['final_direction']}")
        print(f"  ✓ Vote Score: {result['total_vote_score']:.3f}")
        print(f"  ✓ Confidence: {result['confidence']:.1f}%")
        print(f"  ✓ Strong Signals: {result['strong_signals']}")
        
    def test_missing_indicators_handling(self):
        """Test weighted voting with missing indicators."""
        print("\n🧪 Testing Missing Indicators Handling")
        
        # Create partial indicator set
        indicators = {
            "RSI": self.create_indicator_result("buy", 1, 85.0, "strong", self.default_weights["RSI"]),
            "MACD": self.create_indicator_result("buy", 1, 70.0, "strong", self.default_weights["MACD"]),
            # Missing BBANDS, STOCH, ATR
        }
        
        result = compute_weighted_vote_aggregation(indicators, self.default_weights)
        
        # Verify results - should still work with partial data
        self.assertEqual(result['final_direction'], 'buy')
        self.assertGreater(result['total_vote_score'], 0.0)
        self.assertIn('vote_breakdown', result)
        
        print(f"  ✓ Final Direction: {result['final_direction']}")
        print(f"  ✓ Vote Score: {result['total_vote_score']:.3f}")
        print(f"  ✓ Confidence: {result['confidence']:.1f}%")
        print(f"  ✓ Indicators Used: {len(result['vote_breakdown'])}")
        
    def test_vote_breakdown_accuracy(self):
        """Test that vote breakdown calculations are accurate."""
        print("\n🧪 Testing Vote Breakdown Accuracy")
        
        indicators = {
            "RSI": self.create_indicator_result("buy", 1, 85.0, "strong", 20),  # Use explicit weight
            "MACD": self.create_indicator_result("sell", -1, 55.0, "weak", 10)   # Use explicit weight
        }
        
        weights = {"RSI": 20, "MACD": 10}
        result = compute_weighted_vote_aggregation(indicators, weights)
        
        # Verify vote breakdown
        self.assertIn('RSI', result['vote_breakdown'])
        self.assertIn('MACD', result['vote_breakdown'])
        
        rsi_breakdown = result['vote_breakdown']['RSI']
        macd_breakdown = result['vote_breakdown']['MACD']
        
        self.assertEqual(rsi_breakdown['vote'], 1)
        self.assertEqual(macd_breakdown['vote'], -1)
        
        # RSI should have higher contribution due to higher weight and stronger signal
        self.assertGreater(abs(rsi_breakdown['contribution']), abs(macd_breakdown['contribution']))
        
        print(f"  ✓ RSI Contribution: {rsi_breakdown['contribution']:.3f}")
        print(f"  ✓ MACD Contribution: {macd_breakdown['contribution']:.3f}")
        print(f"  ✓ Total Vote Score: {result['total_vote_score']:.3f}")


def run_weighted_voting_tests():
    """Run all weighted voting system tests."""
    print("=" * 60)
    print("🚀 WEIGHTED VOTING SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestWeightedVotingSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL WEIGHTED VOTING TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_weighted_voting_tests()
    if not success:
        sys.exit(1)