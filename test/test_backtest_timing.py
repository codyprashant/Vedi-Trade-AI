"""
Test backtest trade timing to ensure realistic next-bar-open execution.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from backtest.backtest_engine import BacktestEngine


class TestBacktestTiming:
    """Test realistic trade timing in backtest engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create backtest engine with correct constructor
        self.engine = BacktestEngine(
            strategy_id=1,
            symbol="EURUSD",
            start_date="2024-01-01",
            end_date="2024-01-02",
            investment=10000.0,
            timeframe="15m"
        )
        
        # Create test market data with predictable prices
        dates = pd.date_range(start='2024-01-01', periods=150, freq='15min')
        self.market_data = pd.DataFrame({
            'open': [100.0 + i * 0.1 for i in range(150)],
            'high': [100.5 + i * 0.1 for i in range(150)],
            'low': [99.5 + i * 0.1 for i in range(150)],
            'close': [100.2 + i * 0.1 for i in range(150)],
            'volume': [1000] * 150
        }, index=dates)
    
    def test_entry_price_uses_next_bar_open(self):
        """Test that entry prices use next bar open, not current bar close."""
        # Mock the _evaluate_signal_for_backtest method to return a signal at bar 100
        def mock_evaluate_signal(historical_data, current_timestamp):
            if len(historical_data) == 101:  # Signal at bar 100 (0-indexed)
                return {
                    'direction': 'buy',
                    'entry_price': 110.2,  # Current bar close (bar 100)
                    'stop_loss_price': 109.0,
                    'take_profit_price': 112.0,
                    'strength': 0.8,
                    'confidence': 0.9
                }
            return None
        
        with patch.object(self.engine, '_evaluate_signal_for_backtest', side_effect=mock_evaluate_signal):
            # Run signal generation
            self.engine._generate_signals(self.market_data)
            
            # Verify one signal was generated
            assert len(self.engine.signals) == 1
            
            signal = self.engine.signals[0]
            
            # Entry price should be next bar's open (bar 101), not current bar's close
            expected_entry_price = self.market_data.iloc[101]['open']  # 110.1
            assert signal['entry_price'] == expected_entry_price
            
            # Original entry price should be preserved
            assert signal['original_entry_price'] == 110.2
            
            # Entry adjustment should be calculated
            expected_adjustment = expected_entry_price - 110.2  # 110.1 - 110.2 = -0.1
            assert signal['entry_adjustment'] == expected_adjustment

    def test_multiple_signals_timing(self):
        """Test that multiple signals all use correct next bar open prices."""
        # Mock to return signals at bars 100 and 120
        def mock_evaluate_signal(historical_data, current_timestamp):
            data_length = len(historical_data)
            if data_length == 101:  # Signal at bar 100
                return {
                    'direction': 'buy',
                    'entry_price': 110.2,  # Current bar close
                    'stop_loss_price': 109.0,
                    'take_profit_price': 112.0,
                    'strength': 0.8,
                    'confidence': 0.9
                }
            elif data_length == 121:  # Signal at bar 120
                return {
                    'direction': 'sell',
                    'entry_price': 112.2,  # Current bar close
                    'stop_loss_price': 113.5,
                    'take_profit_price': 110.5,
                    'strength': 0.7,
                    'confidence': 0.8
                }
            return None
        
        with patch.object(self.engine, '_evaluate_signal_for_backtest', side_effect=mock_evaluate_signal):
            # Run signal generation
            self.engine._generate_signals(self.market_data)
            
            # Verify two signals were generated
            assert len(self.engine.signals) == 2
            
            # Check first signal (bar 100 -> entry at bar 101)
            signal1 = self.engine.signals[0]
            expected_entry1 = self.market_data.iloc[101]['open']  # 110.1
            assert signal1['entry_price'] == expected_entry1
            assert signal1['original_entry_price'] == 110.2
            
            # Check second signal (bar 120 -> entry at bar 121)
            signal2 = self.engine.signals[1]
            expected_entry2 = self.market_data.iloc[121]['open']  # 112.1
            assert signal2['entry_price'] == expected_entry2
            assert signal2['original_entry_price'] == 112.2

    def test_no_signal_at_last_bar(self):
        """Test that no signal is generated at the last bar (no next bar available)."""
        # Mock to return a signal at the last bar (should be ignored)
        def mock_evaluate_signal(historical_data, current_timestamp):
            if len(historical_data) == len(self.market_data):  # Last bar
                return {
                    'direction': 'buy',
                    'entry_price': 114.9,
                    'stop_loss_price': 113.5,
                    'take_profit_price': 116.5,
                    'strength': 0.8,
                    'confidence': 0.9
                }
            return None
        
        with patch.object(self.engine, '_evaluate_signal_for_backtest', side_effect=mock_evaluate_signal):
            # Run signal generation
            self.engine._generate_signals(self.market_data)
            
            # Verify no signals were generated (last bar excluded)
            assert len(self.engine.signals) == 0

    def test_signal_timestamp_vs_entry_timing(self):
        """Test that signal timestamp is current bar but entry price is next bar."""
        # Mock to return a signal at bar 100
        def mock_evaluate_signal(historical_data, current_timestamp):
            if len(historical_data) == 101:  # Signal at bar 100
                return {
                    'direction': 'buy',
                    'entry_price': 110.2,  # Current bar close
                    'stop_loss_price': 109.0,
                    'take_profit_price': 112.0,
                    'strength': 0.8,
                    'confidence': 0.9
                }
            return None
        
        with patch.object(self.engine, '_evaluate_signal_for_backtest', side_effect=mock_evaluate_signal):
            # Run signal generation
            self.engine._generate_signals(self.market_data)
            
            # Verify one signal was generated
            assert len(self.engine.signals) == 1
            
            signal = self.engine.signals[0]
            
            # Signal timestamp should be current bar (bar 100)
            expected_timestamp = self.market_data.index[100]
            assert signal['timestamp'] == expected_timestamp
            
            # Entry price should be next bar's open (bar 101)
            expected_entry_price = self.market_data.iloc[101]['open']
            assert signal['entry_price'] == expected_entry_price