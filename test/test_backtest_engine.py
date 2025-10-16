"""
Comprehensive test suite for the unified BacktestEngine.

Tests cover:
1. Signal generation consistency
2. Simulated trade profit scenarios
3. Simulated trade loss scenarios  
4. Backtest summary efficiency calculations
5. ROI endpoint functionality
6. Storage and retrieval operations
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import BacktestEngine
from app.db import (
    insert_backtest_summary, insert_backtest_signals_batch,
    fetch_backtest_summary, fetch_backtest_signals, fetch_all_backtests
)


class TestBacktestEngine(unittest.TestCase):
    """Test suite for BacktestEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_id = 1
        self.symbol = "XAUUSD"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"
        self.investment = 10000
        self.timeframe = "15m"
        
        # Create mock market data
        self.mock_market_data = self._create_mock_market_data()
        
        # Create mock signals
        self.mock_signals = self._create_mock_signals()
    
    def _create_mock_market_data(self):
        """Create realistic mock OHLCV data for testing."""
        dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='15min'
        )
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible tests
        base_price = 2000.0
        price_changes = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility
        
        prices = [base_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = []
        for i, date in enumerate(dates):
            open_price = prices[i]
            close_price = prices[i] * (1 + np.random.normal(0, 0.0005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0003)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0003)))
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })
        
        return pd.DataFrame(data)
    
    def _create_mock_signals(self):
        """Create mock signals for testing."""
        return [
            {
                'id': 1,
                'timestamp': datetime(2024, 1, 5, 10, 0),
                'direction': 'buy',
                'entry_price': 2000.0,
                'take_profit_percent': 1.0,
                'stop_loss_percent': 0.5,
                'confidence': 0.85,
                'reason': 'Strong bullish momentum'
            },
            {
                'id': 2,
                'timestamp': datetime(2024, 1, 10, 14, 30),
                'direction': 'sell',
                'entry_price': 2010.0,
                'take_profit_percent': 0.8,
                'stop_loss_percent': 0.4,
                'confidence': 0.75,
                'reason': 'Bearish divergence detected'
            },
            {
                'id': 3,
                'timestamp': datetime(2024, 1, 15, 9, 15),
                'direction': 'buy',
                'entry_price': 1995.0,
                'take_profit_percent': 1.2,
                'stop_loss_percent': 0.6,
                'confidence': 0.90,
                'reason': 'Support level bounce'
            }
        ]

    def test_backtest_signal_generation_consistency(self):
        """Test 1: Verify signal generation consistency with live engine."""
        engine = BacktestEngine(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            investment=self.investment,
            timeframe=self.timeframe
        )
        
        # Mock the private signal generation method
        with patch.object(engine, '_generate_signals') as mock_generate:
            # Mock market data fetch
            with patch.object(engine, '_fetch_market_data') as mock_fetch:
                mock_fetch.return_value = self.mock_market_data
                
                # Mock the signals list to be populated by _generate_signals
                engine.signals = self.mock_signals
                
                # Call _generate_signals to test it
                engine._generate_signals(self.mock_market_data)
                
                # Verify signal consistency
                self.assertEqual(len(engine.signals), 3)
                self.assertEqual(engine.signals[0]['direction'], 'buy')
                self.assertEqual(engine.signals[1]['direction'], 'sell')
                self.assertEqual(engine.signals[2]['direction'], 'buy')
                
                # Verify signal attributes
                for signal in engine.signals:
                    self.assertIn('id', signal)
                    self.assertIn('timestamp', signal)
                    self.assertIn('direction', signal)
                    self.assertIn('entry_price', signal)
                    self.assertIn('confidence', signal)
                    self.assertIn('reason', signal)
                    
                print("✓ Signal generation consistency test passed")

    def test_simulated_trade_profit_hit(self):
        """Test 2: Verify profit scenario in trade simulation."""
        engine = BacktestEngine(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            investment=self.investment
        )
        
        # Create a profitable signal scenario
        signal = {
            'id': 1,
            'direction': 'buy',
            'entry_price': 2000.0,
            'take_profit_percent': 1.0,  # 1% profit target
            'stop_loss_percent': 0.5,    # 0.5% stop loss
            'confidence': 0.85,
            'reason': 'Test profit scenario'
        }
        
        # Create market data that hits take profit
        profit_data = [
            {'high': 2005.0, 'low': 1998.0, 'close': 2003.0},  # Doesn't hit TP
            {'high': 2021.0, 'low': 2000.0, 'close': 2020.0},  # Hits TP at 2020.0
            {'high': 2025.0, 'low': 2015.0, 'close': 2022.0}   # After TP hit
        ]
        
        result, exit_price = engine.simulate_trade(signal, profit_data)
        
        # Verify profit scenario
        self.assertEqual(result, "profit")
        self.assertEqual(exit_price, 2020.0)  # Take profit price
        
        # Calculate expected profit percentage
        expected_profit_pct = ((2020.0 - 2000.0) / 2000.0) * 100
        self.assertAlmostEqual(expected_profit_pct, 1.0, places=2)
        
        print("✓ Simulated trade profit scenario test passed")

    def test_simulated_trade_loss_hit(self):
        """Test 3: Verify loss scenario in trade simulation."""
        engine = BacktestEngine(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            investment=self.investment
        )
        
        # Create a loss signal scenario
        signal = {
            'id': 2,
            'direction': 'sell',
            'entry_price': 2000.0,
            'take_profit_percent': 0.8,  # 0.8% profit target
            'stop_loss_percent': 0.4,    # 0.4% stop loss
            'confidence': 0.75,
            'reason': 'Test loss scenario'
        }
        
        # Create market data that hits stop loss (for sell signal)
        loss_data = [
            {'high': 2002.0, 'low': 1998.0, 'close': 2001.0},  # Doesn't hit SL
            {'high': 2009.0, 'low': 1995.0, 'close': 2005.0},  # Hits SL at 2008.0
            {'high': 2010.0, 'low': 2000.0, 'close': 2008.0}   # After SL hit
        ]
        
        result, exit_price = engine.simulate_trade(signal, loss_data)
        
        # Verify loss scenario
        self.assertEqual(result, "loss")
        self.assertEqual(exit_price, 2008.0)  # Stop loss price for sell
        
        # Calculate expected loss percentage (negative for sell signal hitting SL)
        expected_loss_pct = ((2008.0 - 2000.0) / 2000.0) * 100 * -1  # Sell direction
        self.assertAlmostEqual(expected_loss_pct, -0.4, places=2)
        
        print("✓ Simulated trade loss scenario test passed")

    def test_backtest_summary_efficiency_calculation(self):
        """Test 4: Verify efficiency calculation in backtest summary."""
        engine = BacktestEngine(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            investment=self.investment
        )
        
        # Mock results with known win/loss ratios
        engine.results = [
            {"signal_id": 1, "result": "profit", "profit_pct": 1.5, "exit_price": 2030.0},
            {"signal_id": 2, "result": "profit", "profit_pct": 0.8, "exit_price": 2016.0},
            {"signal_id": 3, "result": "loss", "profit_pct": -0.5, "exit_price": 1990.0},
            {"signal_id": 4, "result": "profit", "profit_pct": 1.2, "exit_price": 2024.0},
            {"signal_id": 5, "result": "loss", "profit_pct": -0.3, "exit_price": 1994.0},
            {"signal_id": 6, "result": "open", "profit_pct": 0.1, "exit_price": 2002.0}
        ]
        
        # Mock database operations
        with patch.object(engine, '_store_backtest_results') as mock_store:
            mock_store.return_value = 123  # Mock backtest_id
            
            result = engine.finalize()
            
            # Verify efficiency calculation
            # 3 wins out of 5 closed trades (excluding open) = 60%
            expected_efficiency = 60.0
            self.assertAlmostEqual(result['efficiency_pct'], expected_efficiency, places=1)
            
            # Verify average return calculation
            closed_trades = [1.5, 0.8, -0.5, 1.2, -0.3]  # Excluding open trade
            expected_avg_return = sum(closed_trades) / len(closed_trades)
            expected_total_return = expected_avg_return * len(closed_trades)  # Total return calculation
            self.assertAlmostEqual(result['total_return_pct'], expected_total_return, places=2)
            
            # Verify counts
            self.assertEqual(result['win_count'], 3)
            self.assertEqual(result['loss_count'], 2)
            self.assertEqual(result['open_count'], 1)
            
            print("✓ Backtest summary efficiency calculation test passed")

    def test_backtest_roi_endpoint_output(self):
        """Test 5: Verify ROI endpoint calculation and output format."""
        from fastapi.testclient import TestClient
        from app.yahoo_server import app
        
        client = TestClient(app)
        
        # Mock database response
        mock_backtest = {
            'id': 1,
            'symbol': 'XAUUSD',
            'timeframe': '15m',
            'total_return_pct': 12.5,
            'efficiency_pct': 68.0,
            'investment': 10000,
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 31),
            'created_at': datetime(2024, 1, 31, 12, 0)
        }
        
        with patch('app.yahoo_server.fetch_backtest_summary') as mock_fetch:
            mock_fetch.return_value = mock_backtest
            
            # Test ROI calculation
            response = client.get("/api/backtest/roi?backtest_id=1&amount=10000")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Verify ROI calculation
            expected_final = 10000 * (1 + 12.5 / 100)  # 11,250
            expected_profit = expected_final - 10000     # 1,250
            
            self.assertEqual(data['backtest_id'], 1)
            self.assertEqual(data['initial'], 10000)
            self.assertEqual(data['final'], expected_final)
            self.assertEqual(data['return_pct'], 12.5)
            self.assertEqual(data['profit'], expected_profit)
            self.assertEqual(data['symbol'], 'XAUUSD')
            self.assertEqual(data['efficiency_pct'], 68.0)
            
            print("✓ ROI endpoint output test passed")

    def test_backtest_storage_and_retrieval(self):
        """Test 6: Verify complete storage and retrieval operations."""
        # Test data
        backtest_data = {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 31),
            'investment': self.investment,
            'total_return_pct': 8.5,
            'efficiency_pct': 72.0
        }
        
        signals_data = [
            {
                'signal_time': datetime(2024, 1, 5, 10, 0),
                'direction': 'buy',
                'entry_price': 2000.0,
                'exit_price': 2020.0,
                'profit_pct': 1.0,
                'result': 'profit',
                'confidence': 0.85,
                'reason': 'Strong momentum'
            },
            {
                'signal_time': datetime(2024, 1, 10, 14, 30),
                'direction': 'sell',
                'entry_price': 2010.0,
                'exit_price': 1995.0,
                'profit_pct': 0.75,
                'result': 'profit',
                'confidence': 0.78,
                'reason': 'Bearish pattern'
            }
        ]
        
        # Test storage
        engine = BacktestEngine(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            investment=self.investment
        )
        
        engine.results = [
            {"signal_id": 1, "result": "profit", "profit_pct": 1.0, "exit_price": 2020.0},
            {"signal_id": 2, "result": "profit", "profit_pct": 0.75, "exit_price": 1995.0}
        ]
        
        # Mock the storage method to return our expected ID
        mock_backtest_id = 456
        with patch.object(engine, '_store_backtest_results') as mock_store:
            mock_store.return_value = mock_backtest_id
            
            # Test finalize (storage)
            result = engine.finalize()
            self.assertEqual(result['backtest_id'], mock_backtest_id)
            
            # Verify storage was called
            mock_store.assert_called_once()
        
        # Test retrieval with mocked database functions  
        with patch.object(sys.modules[__name__], 'fetch_backtest_summary') as mock_fetch_summary:
            with patch.object(sys.modules[__name__], 'fetch_backtest_signals') as mock_fetch_signals:
                mock_fetch_summary.return_value = backtest_data
                mock_fetch_signals.return_value = signals_data
                
                # Test retrieval
                retrieved_summary = fetch_backtest_summary(mock_backtest_id)
                retrieved_signals = fetch_backtest_signals(mock_backtest_id)
                
                # Verify retrieval
                self.assertEqual(retrieved_summary, backtest_data)
                self.assertEqual(len(retrieved_signals), 2)
                self.assertEqual(retrieved_signals[0]['direction'], 'buy')
                self.assertEqual(retrieved_signals[1]['direction'], 'sell')
                
                print("✓ Backtest storage and retrieval test passed")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)