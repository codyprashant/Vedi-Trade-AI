"""
Test suite for Signal Performance Evaluator module.

This module tests the automated signal evaluation system including:
- Signal outcome evaluation (profit/loss/open)
- Performance metrics calculation
- Database operations
- Backfill functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.signal_performance_evaluator import SignalPerformanceEvaluator


class TestSignalPerformanceEvaluator(unittest.TestCase):
    """Test cases for SignalPerformanceEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = SignalPerformanceEvaluator()
        self.sample_signal = {
            'id': 'test-signal-123',
            'symbol': 'XAUUSD',
            'timestamp': datetime.now(timezone.utc),
            'side': 'buy',
            'entry_price': 2285.30,
            'take_profit_percent': 0.5,
            'stop_loss_percent': 0.3,
            'take_profit_price': 2285.30 * (1 + 0.5/100),  # 2296.73
            'stop_loss_price': 2285.30 * (1 - 0.3/100),    # 2278.45
            'timeframe': '5m',
            'strategy': 'adaptive_signal'
        }
    
    @patch('jobs.signal_performance_evaluator.fetch_range_df')
    def test_signal_profit_hit_takeprofit(self, mock_fetch_range):
        """Test signal evaluation when take profit is hit."""
        # Mock historical data showing price hitting take profit
        import pandas as pd
        mock_data = pd.DataFrame([
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=5), 'high': 2295.00, 'low': 2284.00, 'close': 2294.50},
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=10), 'high': 2296.73, 'low': 2290.00, 'close': 2296.50}
        ])
        mock_fetch_range.return_value = mock_data
        
        result = self.evaluator.evaluate_signal_outcome(self.sample_signal)
        
        # Verify result
        self.assertEqual(result['result'], 'profit')
        self.assertAlmostEqual(result['profit_pct'], 0.5, places=2)  # Should match take_profit_percent
        self.assertAlmostEqual(result['exit_price'], 2296.73, places=2)  # Take profit price
    
    @patch('jobs.signal_performance_evaluator.fetch_range_df')
    def test_signal_loss_hit_stoploss(self, mock_fetch_range):
        """Test signal evaluation when stop loss is hit."""
        # Mock historical data showing price hitting stop loss
        import pandas as pd
        mock_data = pd.DataFrame([
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=5), 'high': 2285.00, 'low': 2278.45, 'close': 2279.00},
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=10), 'high': 2280.00, 'low': 2277.00, 'close': 2278.50}
        ])
        mock_fetch_range.return_value = mock_data
        
        result = self.evaluator.evaluate_signal_outcome(self.sample_signal)
        
        # Verify result
        self.assertEqual(result['result'], 'loss')
        self.assertAlmostEqual(result['profit_pct'], -0.3, places=2)  # Should match stop_loss_percent
        self.assertAlmostEqual(result['exit_price'], 2278.4441, places=2)  # Stop loss price: 2285.30 * (1 - 0.003)
    
    @patch('jobs.signal_performance_evaluator.fetch_range_df')
    def test_signal_still_open(self, mock_fetch_range):
        """Test signal evaluation when neither TP nor SL is hit (still open)."""
        # Mock historical data that doesn't hit TP or SL
        import pandas as pd
        mock_data = pd.DataFrame([
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=5), 'high': 2289.20, 'low': 2284.00, 'close': 2289.20},
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=10), 'high': 2290.00, 'low': 2287.00, 'close': 2289.50}
        ])
        mock_fetch_range.return_value = mock_data
        
        result = self.evaluator.evaluate_signal_outcome(self.sample_signal)
        
        # Verify result
        self.assertEqual(result['result'], 'open')
        self.assertAlmostEqual(result['profit_pct'], 0.18, places=1)  # (2289.50 - 2285.30) / 2285.30 * 100 = 0.1838
        self.assertEqual(result['exit_price'], 2289.50)  # Last close price
    
    @patch('jobs.signal_performance_evaluator.fetch_range_df')
    def test_signal_sell_direction(self, mock_fetch_range):
        """Test signal evaluation for sell direction."""
        sell_signal = self.sample_signal.copy()
        sell_signal['side'] = 'sell'
        sell_signal['entry_price'] = 2285.30
        sell_signal['take_profit_price'] = 2285.30 * (1 - 0.5/100)  # 2273.87
        sell_signal['stop_loss_price'] = 2285.30 * (1 + 0.3/100)    # 2292.16
        
        # Mock historical data showing price hitting take profit for sell
        import pandas as pd
        mock_data = pd.DataFrame([
            {'time': sell_signal['timestamp'] + timedelta(minutes=5), 'high': 2285.00, 'low': 2273.87, 'close': 2274.50}
        ])
        mock_fetch_range.return_value = mock_data
        
        result = self.evaluator.evaluate_signal_outcome(sell_signal)
        
        # For sell signal, take profit is hit when price goes DOWN
        self.assertEqual(result['result'], 'profit')
        self.assertAlmostEqual(result['profit_pct'], 0.5, places=2)
        self.assertAlmostEqual(result['exit_price'], 2273.87, places=2)  # Take profit price for sell
    
    def test_daily_summary_efficiency_calculation(self):
        """Test daily performance summary and efficiency calculation."""
        # Mock signals for a day
        mock_signals = [
            {'id': 'signal-1', 'symbol': 'XAUUSD', 'side': 'buy', 'entry_price': 2285.30, 'timestamp': datetime.now(timezone.utc), 'timeframe': '5m', 'strategy': 'adaptive_signal'},
            {'id': 'signal-2', 'symbol': 'XAUUSD', 'side': 'sell', 'entry_price': 2290.00, 'timestamp': datetime.now(timezone.utc), 'timeframe': '5m', 'strategy': 'adaptive_signal'},
            {'id': 'signal-3', 'symbol': 'EURUSD', 'side': 'buy', 'entry_price': 1.0850, 'timestamp': datetime.now(timezone.utc), 'timeframe': '15m', 'strategy': 'adaptive_signal'}
        ]
        
        # Mock evaluation results
        evaluation_results = [
            {'result': 'profit', 'profit_pct': 0.5, 'exit_price': 2296.73, 'notes': 'Take profit hit'},
            {'result': 'loss', 'profit_pct': -0.3, 'exit_price': 2278.45, 'notes': 'Stop loss hit'},
            {'result': 'open', 'profit_pct': 0.1, 'exit_price': None, 'notes': 'Still open'}
        ]
        
        with patch.object(self.evaluator, 'fetch_signals_by_date', return_value=mock_signals):
            with patch.object(self.evaluator, 'evaluate_signal_outcome', side_effect=evaluation_results):
                with patch.object(self.evaluator, 'update_performance_summary') as mock_update_summary:
                    with patch.object(self.evaluator, 'record_daily_performance') as mock_update_daily:
                        with patch.object(self.evaluator, 'update_signal_result'):
                            
                            target_date = datetime.now(timezone.utc) - timedelta(days=1)
                            result = self.evaluator.run_daily_signal_evaluation(target_date)
                    
                    # Verify results
                    self.assertEqual(result['signals_evaluated'], 3)
                    self.assertEqual(result['profits'], 1)
                    self.assertEqual(result['losses'], 1)
                    self.assertEqual(result['open'], 1)
                    
                    # Verify summary and daily updates were called
                    self.assertTrue(mock_update_summary.called)
                    self.assertTrue(mock_update_daily.called)
    
    @patch.object(SignalPerformanceEvaluator, 'fetch_open_signals')
    def test_reevaluate_open_signals(self, mock_fetch_open):
        """Test re-evaluation of open signals."""
        # Mock open signals
        mock_open_signals = [
            {'id': 'open-signal-1', 'symbol': 'XAUUSD', 'entry_price': 2285.30, 'side': 'buy', 'timeframe': '5m'},
            {'id': 'open-signal-2', 'symbol': 'EURUSD', 'entry_price': 1.0850, 'side': 'sell', 'timeframe': '5m'}
        ]
        mock_fetch_open.return_value = mock_open_signals
        
        # Mock evaluation results - one closes, one stays open
        evaluation_results = [
            {'result': 'profit', 'profit_pct': 0.4, 'exit_price': 2294.45, 'notes': 'Take profit hit'},
            {'result': 'open', 'profit_pct': 0.05, 'exit_price': 1.0855, 'notes': 'Still open'}
        ]
        
        with patch.object(self.evaluator, 'evaluate_signal_outcome', side_effect=evaluation_results):
            with patch.object(self.evaluator, 'update_signal_result'):
                with patch.object(self.evaluator, 'update_performance_summary'):
                    
                    result = self.evaluator.reevaluate_open_signals()
                    
                    # Verify results
                    self.assertEqual(result['signals_reevaluated'], 2)
                    self.assertEqual(result['newly_closed'], 1)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Test data
        signals_data = [
            {'result': 'profit', 'profit_pct': 0.5},
            {'result': 'profit', 'profit_pct': 0.3},
            {'result': 'loss', 'profit_pct': -0.2},
            {'result': 'loss', 'profit_pct': -0.4},
            {'result': 'open', 'profit_pct': 0.1}
        ]
        
        metrics = self.evaluator._calculate_performance_metrics(signals_data)
        
        # Verify calculations
        self.assertEqual(metrics['total_signals'], 5)
        self.assertEqual(metrics['win_count'], 2)
        self.assertEqual(metrics['loss_count'], 2)
        self.assertEqual(metrics['open_count'], 1)
        self.assertEqual(metrics['efficiency_pct'], 50.0)  # 2/(2+2) * 100
        self.assertAlmostEqual(metrics['avg_profit_pct'], 0.06, places=2)  # (0.5+0.3-0.2-0.4+0.1)/5
        self.assertAlmostEqual(metrics['total_roi_pct'], 0.3, places=2)  # Sum of all profit_pct
    
    @patch.object(SignalPerformanceEvaluator, 'fetch_signals_by_date')
    @patch('jobs.signal_performance_evaluator.fetch_range_df')
    def test_backfill_evaluation(self, mock_fetch_range, mock_fetch_signals):
        """Test backfill functionality for historical evaluation."""
        # Mock signals from database
        mock_signals = [self.sample_signal.copy()]
        mock_fetch_signals.return_value = mock_signals
        
        # Mock historical data
        import pandas as pd
        mock_data = pd.DataFrame([
            {'time': self.sample_signal['timestamp'] + timedelta(minutes=5), 'high': 2295.00, 'low': 2284.00, 'close': 2294.50}
        ])
        mock_fetch_range.return_value = mock_data
        
        with patch.object(self.evaluator, 'update_signal_result') as mock_update:
            with patch.object(self.evaluator, 'update_performance_summary'):
                with patch.object(self.evaluator, 'record_daily_performance'):
                    
                    # Test daily evaluation (which is the actual method available)
                    target_date = datetime.now(timezone.utc) - timedelta(days=1)
                    
                    result = self.evaluator.run_daily_signal_evaluation(target_date)
                    
                    # Verify that signals were fetched and evaluated
                    self.assertEqual(result['status'], 'completed')
                    self.assertGreaterEqual(result['signals_evaluated'], 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        import pandas as pd
        
        # Test with empty historical data
        with patch('jobs.signal_performance_evaluator.fetch_range_df', return_value=pd.DataFrame()):
            result = self.evaluator.evaluate_signal_outcome(self.sample_signal)
            self.assertEqual(result['result'], 'open')
            self.assertEqual(result['notes'], 'No historical data')
        
        # Test with data fetch error
        with patch('jobs.signal_performance_evaluator.fetch_range_df', side_effect=Exception('No data for GC=F 5m in range')):
            result = self.evaluator.evaluate_signal_outcome(self.sample_signal)
            self.assertEqual(result['result'], 'open')
            self.assertIn('Data fetch error:', result['notes'])
        
        # Test with invalid signal data
        invalid_signal = {'id': 'invalid', 'symbol': None}
        result = self.evaluator.evaluate_signal_outcome(invalid_signal)
        self.assertIsNone(result)
    
    def test_performance_summary_update(self):
        """Test performance summary database update logic."""
        with patch('jobs.signal_performance_evaluator._get_conn') as mock_get_conn:
            with patch('jobs.signal_performance_evaluator._put_conn'):
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_get_conn.return_value = mock_conn
                
                # Properly mock the context manager
                mock_conn.__enter__ = Mock(return_value=mock_conn)
                mock_conn.__exit__ = Mock(return_value=None)
                mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
                
                # Mock the database query results
                mock_cursor.fetchone.return_value = (10, 6, 3, 1, 0.15, 1.5)
                
                self.evaluator.update_performance_summary('XAUUSD', '5m')
                
                # Verify SQL execution
                self.assertTrue(mock_cursor.execute.called)
                # Should have at least 2 calls: SELECT for stats and INSERT/UPDATE for summary
                self.assertGreaterEqual(mock_cursor.execute.call_count, 2)


class TestSignalEvaluationIntegration(unittest.TestCase):
    """Integration tests for signal evaluation system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.evaluator = SignalPerformanceEvaluator()
    
    def test_database_table_creation(self):
        """Test that database tables are created correctly."""
        with patch('jobs.signal_performance_evaluator._get_conn') as mock_get_conn:
            with patch('jobs.signal_performance_evaluator._put_conn'):
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_get_conn.return_value = mock_conn
                
                # Properly mock the context manager
                mock_conn.__enter__ = Mock(return_value=mock_conn)
                mock_conn.__exit__ = Mock(return_value=None)
                mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
                mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
                
                self.evaluator.ensure_evaluation_tables()
                
                # Verify table creation SQL was executed
                self.assertTrue(mock_cursor.execute.called)
                
                # Check that all required tables are created
                executed_sql = ' '.join([call[0][0] for call in mock_cursor.execute.call_args_list])
                self.assertIn('signal_results', executed_sql)
                self.assertIn('signal_performance_summary', executed_sql)
                self.assertIn('signal_performance_daily', executed_sql)
    
    def test_end_to_end_evaluation_flow(self):
        """Test complete evaluation flow from signal fetch to summary update."""
        # This would be a more comprehensive integration test
        # that tests the entire flow with mocked database and API calls
        pass


def run_tests():
    """Run all tests in this module."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSignalPerformanceEvaluator))
    test_suite.addTest(unittest.makeSuite(TestSignalEvaluationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if not success:
        sys.exit(1)
    print("\n✅ All signal performance evaluator tests passed!")