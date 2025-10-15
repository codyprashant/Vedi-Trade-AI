"""
Signal Performance Evaluator

This module evaluates the performance of trading signals by analyzing their outcomes
based on historical OHLC data. It runs daily at 5 AM to assess previous day's signals.
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import schedule
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import _get_conn, _put_conn
from app.yahoo_server import fetch_range_df
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SignalPerformanceEvaluator:
    """Evaluates trading signal performance and maintains statistics."""
    
    def __init__(self):
        self.ensure_evaluation_tables()
    
    def ensure_evaluation_tables(self) -> None:
        """Create the signal evaluation tables if they don't exist."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Create signal_results table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.signal_results (
                            id SERIAL PRIMARY KEY,
                            signal_id BIGINT NOT NULL REFERENCES public.signals(id),
                            result TEXT NOT NULL CHECK (result IN ('profit', 'loss', 'open')),
                            exit_price FLOAT,
                            profit_pct FLOAT,
                            evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            evaluation_notes TEXT,
                            UNIQUE(signal_id)
                        );
                    """)
                    
                    # Create signal_performance_summary table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.signal_performance_summary (
                            id SERIAL PRIMARY KEY,
                            strategy_id INT,
                            symbol TEXT NOT NULL,
                            timeframe TEXT NOT NULL,
                            total_signals INT DEFAULT 0,
                            win_count INT DEFAULT 0,
                            loss_count INT DEFAULT 0,
                            open_count INT DEFAULT 0,
                            avg_profit_pct FLOAT DEFAULT 0.0,
                            total_roi_pct FLOAT DEFAULT 0.0,
                            efficiency_pct FLOAT DEFAULT 0.0,
                            last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(symbol, timeframe)
                        );
                    """)
                    
                    # Create signal_performance_daily table for analytics
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.signal_performance_daily (
                            id SERIAL PRIMARY KEY,
                            evaluation_date DATE NOT NULL,
                            symbol TEXT NOT NULL,
                            timeframe TEXT NOT NULL,
                            signals_evaluated INT DEFAULT 0,
                            new_profits INT DEFAULT 0,
                            new_losses INT DEFAULT 0,
                            daily_roi_pct FLOAT DEFAULT 0.0,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(evaluation_date, symbol, timeframe)
                        );
                    """)
                    
                    logger.info("Signal evaluation tables ensured")
        finally:
            _put_conn(conn)
    
    def fetch_signals_by_date(self, target_date: datetime) -> List[Dict[str, Any]]:
        """Fetch all signals from a specific date."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM public.signals 
                        WHERE DATE(timestamp) = %s
                        ORDER BY timestamp ASC
                    """, (target_date.date(),))
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        finally:
            _put_conn(conn)
    
    def fetch_open_signals(self) -> List[Dict[str, Any]]:
        """Fetch all signals that are still marked as 'open'."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT s.*, sr.result, sr.evaluated_at
                        FROM public.signals s
                        LEFT JOIN public.signal_results sr ON s.id = sr.signal_id
                        WHERE sr.result = 'open' OR sr.result IS NULL
                        ORDER BY s.timestamp ASC
                    """)
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        finally:
            _put_conn(conn)
    
    def evaluate_signal_outcome(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single signal's outcome based on historical OHLC data.
        
        Args:
            signal: Signal record from database
            
        Returns:
            Dict containing evaluation results or None for invalid signals
        """
        # Validate required signal fields
        required_fields = ['id', 'symbol', 'side', 'entry_price', 'timestamp']
        if not all(field in signal and signal[field] is not None for field in required_fields):
            logger.warning(f"Invalid signal: missing required fields {required_fields}")
            return None
            
        try:
            # Get signal details
            signal_id = signal['id']
            symbol = signal['symbol']
            entry_price = signal.get('entry_price')
            direction = signal['side']  # 'buy' or 'sell'
            timestamp = signal['timestamp']
            
            # Calculate stop loss and take profit levels
            stop_loss_price = signal.get('stop_loss_price')
            take_profit_price = signal.get('take_profit_price')
            
            if not all([entry_price, stop_loss_price, take_profit_price]):
                logger.warning(f"Signal {signal_id} missing price levels")
                return {'result': 'open', 'exit_price': None, 'profit_pct': 0.0, 'notes': 'Missing price levels'}
            
            # Fetch historical data from signal time to now
            start_time = timestamp
            end_time = datetime.now(timezone.utc)
            
            try:
                # Get 5-minute interval data for precise exit detection
                data = fetch_range_df(symbol, "5m", start_time, end_time)
                
                if data.empty:
                    logger.warning(f"No historical data available for {symbol}")
                    return {'result': 'open', 'exit_price': None, 'profit_pct': 0.0, 'notes': 'No historical data'}
                
                # Analyze each candle to find exit condition
                result = 'open'
                exit_price = None
                exit_time = None
                
                for idx, candle in data.iterrows():
                    candle_time = candle['time']
                    
                    # Skip candles before signal time
                    if candle_time <= timestamp:
                        continue
                    
                    high = candle['high']
                    low = candle['low']
                    close = candle['close']
                    
                    if direction == 'buy':
                        # Check if take profit hit
                        if high >= take_profit_price:
                            result = 'profit'
                            exit_price = take_profit_price
                            exit_time = candle_time
                            break
                        # Check if stop loss hit
                        elif low <= stop_loss_price:
                            result = 'loss'
                            exit_price = stop_loss_price
                            exit_time = candle_time
                            break
                    
                    else:  # sell
                        # Check if take profit hit (price goes down)
                        if low <= take_profit_price:
                            result = 'profit'
                            exit_price = take_profit_price
                            exit_time = candle_time
                            break
                        # Check if stop loss hit (price goes up)
                        elif high >= stop_loss_price:
                            result = 'loss'
                            exit_price = stop_loss_price
                            exit_time = candle_time
                            break
                
                # If no exit condition met, use current price
                if result == 'open':
                    exit_price = data.iloc[-1]['close']
                    exit_time = data.iloc[-1]['time']
                
                # Calculate profit percentage
                if direction == 'buy':
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100
                else:  # sell
                    profit_pct = ((entry_price - exit_price) / entry_price) * 100
                
                notes = f"Exit at {exit_time}" if exit_time else "Still open"
                
                return {
                    'result': result,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'notes': notes
                }
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return {'result': 'open', 'exit_price': None, 'profit_pct': 0.0, 'notes': f'Data fetch error: {e}'}
                
        except Exception as e:
            logger.error(f"Error evaluating signal {signal.get('id', 'unknown')}: {e}")
            return {'result': 'open', 'exit_price': None, 'profit_pct': 0.0, 'notes': f'Evaluation error: {e}'}
    
    def update_signal_result(self, signal_id: int, result: str, profit_pct: float, exit_price: float, notes: str = None) -> None:
        """Update or insert signal result in database."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO public.signal_results 
                        (signal_id, result, exit_price, profit_pct, evaluation_notes)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (signal_id) 
                        DO UPDATE SET 
                            result = EXCLUDED.result,
                            exit_price = EXCLUDED.exit_price,
                            profit_pct = EXCLUDED.profit_pct,
                            evaluation_notes = EXCLUDED.evaluation_notes,
                            evaluated_at = NOW()
                    """, (signal_id, result, exit_price, profit_pct, notes))
        finally:
            _put_conn(conn)
    
    def update_performance_summary(self, symbol: str, timeframe: str) -> None:
        """Update performance summary statistics for a symbol/timeframe."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Calculate aggregated statistics
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_signals,
                            COUNT(CASE WHEN sr.result = 'profit' THEN 1 END) as win_count,
                            COUNT(CASE WHEN sr.result = 'loss' THEN 1 END) as loss_count,
                            COUNT(CASE WHEN sr.result = 'open' THEN 1 END) as open_count,
                            AVG(CASE WHEN sr.result IN ('profit', 'loss') THEN sr.profit_pct END) as avg_profit_pct,
                            SUM(CASE WHEN sr.result IN ('profit', 'loss') THEN sr.profit_pct ELSE 0 END) as total_roi_pct
                        FROM public.signals s
                        LEFT JOIN public.signal_results sr ON s.id = sr.signal_id
                        WHERE s.symbol = %s AND s.timeframe = %s
                    """, (symbol, timeframe))
                    
                    stats = cur.fetchone()
                    total_signals, win_count, loss_count, open_count, avg_profit_pct, total_roi_pct = stats
                    
                    # Calculate efficiency
                    closed_signals = (win_count or 0) + (loss_count or 0)
                    efficiency_pct = (win_count / closed_signals * 100) if closed_signals > 0 else 0.0
                    
                    # Update summary table
                    cur.execute("""
                        INSERT INTO public.signal_performance_summary 
                        (symbol, timeframe, total_signals, win_count, loss_count, open_count, 
                         avg_profit_pct, total_roi_pct, efficiency_pct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timeframe)
                        DO UPDATE SET
                            total_signals = EXCLUDED.total_signals,
                            win_count = EXCLUDED.win_count,
                            loss_count = EXCLUDED.loss_count,
                            open_count = EXCLUDED.open_count,
                            avg_profit_pct = EXCLUDED.avg_profit_pct,
                            total_roi_pct = EXCLUDED.total_roi_pct,
                            efficiency_pct = EXCLUDED.efficiency_pct,
                            last_updated = NOW()
                    """, (symbol, timeframe, total_signals or 0, win_count or 0, loss_count or 0, 
                          open_count or 0, avg_profit_pct or 0.0, total_roi_pct or 0.0, efficiency_pct))
        finally:
            _put_conn(conn)
    
    def record_daily_performance(self, evaluation_date: datetime, symbol: str, timeframe: str, 
                               signals_evaluated: int, new_profits: int, new_losses: int, daily_roi_pct: float) -> None:
        """Record daily performance metrics."""
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO public.signal_performance_daily 
                        (evaluation_date, symbol, timeframe, signals_evaluated, new_profits, new_losses, daily_roi_pct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (evaluation_date, symbol, timeframe)
                        DO UPDATE SET
                            signals_evaluated = EXCLUDED.signals_evaluated,
                            new_profits = EXCLUDED.new_profits,
                            new_losses = EXCLUDED.new_losses,
                            daily_roi_pct = EXCLUDED.daily_roi_pct
                    """, (evaluation_date.date(), symbol, timeframe, signals_evaluated, new_profits, new_losses, daily_roi_pct))
        finally:
            _put_conn(conn)
    
    def run_daily_signal_evaluation(self, target_date: datetime = None) -> Dict[str, Any]:
        """
        Main function to evaluate signals from a specific date.
        
        Args:
            target_date: Date to evaluate signals for. Defaults to yesterday.
            
        Returns:
            Dict with evaluation summary
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc) - timedelta(days=1)
        
        logger.info(f"Starting signal evaluation for {target_date.date()}")
        
        # Get signals from target date
        signals = self.fetch_signals_by_date(target_date)
        
        if not signals:
            logger.info(f"No signals found for {target_date.date()}")
            return {'status': 'completed', 'signals_evaluated': 0}
        
        # Track daily metrics per symbol/timeframe
        daily_metrics = {}
        total_evaluated = 0
        total_profits = 0
        total_losses = 0
        total_open = 0
        
        for signal in signals:
            try:
                # Evaluate signal outcome
                evaluation = self.evaluate_signal_outcome(signal)
                
                # Update signal result
                self.update_signal_result(
                    signal['id'],
                    evaluation['result'],
                    evaluation['profit_pct'],
                    evaluation['exit_price'],
                    evaluation['notes']
                )
                
                # Track daily metrics
                symbol = signal['symbol']
                timeframe = signal['timeframe']
                key = f"{symbol}_{timeframe}"
                
                if key not in daily_metrics:
                    daily_metrics[key] = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'signals_evaluated': 0,
                        'new_profits': 0,
                        'new_losses': 0,
                        'daily_roi': 0.0
                    }
                
                daily_metrics[key]['signals_evaluated'] += 1
                daily_metrics[key]['daily_roi'] += evaluation['profit_pct']
                
                if evaluation['result'] == 'profit':
                    daily_metrics[key]['new_profits'] += 1
                    total_profits += 1
                elif evaluation['result'] == 'loss':
                    daily_metrics[key]['new_losses'] += 1
                    total_losses += 1
                elif evaluation['result'] == 'open':
                    total_open += 1
                
                total_evaluated += 1
                
                logger.info(f"Evaluated signal {signal['id']}: {evaluation['result']} ({evaluation['profit_pct']:.2f}%)")
                
            except Exception as e:
                logger.error(f"Failed to evaluate signal {signal['id']}: {e}")
        
        # Update performance summaries and daily records
        for metrics in daily_metrics.values():
            self.update_performance_summary(metrics['symbol'], metrics['timeframe'])
            self.record_daily_performance(
                target_date,
                metrics['symbol'],
                metrics['timeframe'],
                metrics['signals_evaluated'],
                metrics['new_profits'],
                metrics['new_losses'],
                metrics['daily_roi']
            )
        
        logger.info(f"Completed evaluation of {total_evaluated} signals for {target_date.date()}")
        
        return {
            'status': 'completed',
            'evaluation_date': target_date.date().isoformat(),
            'signals_evaluated': total_evaluated,
            'profits': total_profits,
            'losses': total_losses,
            'open': total_open,
            'symbols_processed': len(daily_metrics)
        }
    
    def reevaluate_open_signals(self) -> Dict[str, Any]:
        """Re-evaluate all signals that are still marked as 'open'."""
        logger.info("Re-evaluating open signals")
        
        open_signals = self.fetch_open_signals()
        
        if not open_signals:
            logger.info("No open signals to re-evaluate")
            return {'status': 'completed', 'signals_reevaluated': 0}
        
        updated_count = 0
        closed_count = 0
        
        for signal in open_signals:
            try:
                evaluation = self.evaluate_signal_outcome(signal)
                
                # Only update if status changed from open
                if evaluation['result'] != 'open':
                    self.update_signal_result(
                        signal['id'],
                        evaluation['result'],
                        evaluation['profit_pct'],
                        evaluation['exit_price'],
                        evaluation['notes']
                    )
                    closed_count += 1
                    logger.info(f"Signal {signal['id']} closed: {evaluation['result']} ({evaluation['profit_pct']:.2f}%)")
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to re-evaluate signal {signal['id']}: {e}")
        
        # Update performance summaries for affected symbols
        symbols_timeframes = set((s['symbol'], s['timeframe']) for s in open_signals)
        for symbol, timeframe in symbols_timeframes:
            self.update_performance_summary(symbol, timeframe)
        
        logger.info(f"Re-evaluated {updated_count} open signals, {closed_count} newly closed")
        
        return {
            'status': 'completed',
            'signals_reevaluated': updated_count,
            'newly_closed': closed_count
        }
    
    def _calculate_performance_metrics(self, signals_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from signals data."""
        if not signals_data:
            return {
                'total_signals': 0,
                'win_count': 0,
                'loss_count': 0,
                'open_count': 0,
                'efficiency_pct': 0.0,
                'avg_profit_pct': 0.0,
                'total_roi_pct': 0.0
            }
        
        total_signals = len(signals_data)
        win_count = sum(1 for s in signals_data if s['result'] == 'profit')
        loss_count = sum(1 for s in signals_data if s['result'] == 'loss')
        open_count = sum(1 for s in signals_data if s['result'] == 'open')
        
        # Calculate efficiency (win rate excluding open signals)
        closed_signals = win_count + loss_count
        efficiency_pct = (win_count / closed_signals * 100) if closed_signals > 0 else 0.0
        
        # Calculate average profit percentage
        total_profit = sum(s['profit_pct'] for s in signals_data)
        avg_profit_pct = total_profit / total_signals
        
        return {
            'total_signals': total_signals,
            'win_count': win_count,
            'loss_count': loss_count,
            'open_count': open_count,
            'efficiency_pct': efficiency_pct,
            'avg_profit_pct': avg_profit_pct,
            'total_roi_pct': total_profit
        }


def run_scheduled_evaluation():
    """Function to be called by the scheduler."""
    try:
        evaluator = SignalPerformanceEvaluator()
        
        # First, re-evaluate any open signals
        evaluator.reevaluate_open_signals()
        
        # Then evaluate yesterday's signals
        result = evaluator.run_daily_signal_evaluation()
        
        logger.info(f"Scheduled evaluation completed: {result}")
        
    except Exception as e:
        logger.error(f"Scheduled evaluation failed: {e}")


def start_scheduler():
    """Start the daily scheduler."""
    logger.info("Starting signal performance evaluator scheduler")
    
    # Schedule daily evaluation at 5:00 AM
    schedule.every().day.at("05:00").do(run_scheduled_evaluation)
    
    logger.info("Scheduler configured to run daily at 05:00")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Signal Performance Evaluator')
    parser.add_argument('--mode', choices=['schedule', 'evaluate', 'backfill'], default='schedule',
                       help='Run mode: schedule (default), evaluate (one-time), or backfill')
    parser.add_argument('--date', type=str, help='Date to evaluate (YYYY-MM-DD format)')
    parser.add_argument('--from-date', type=str, help='Start date for backfill (YYYY-MM-DD format)')
    parser.add_argument('--to-date', type=str, help='End date for backfill (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    evaluator = SignalPerformanceEvaluator()
    
    if args.mode == 'schedule':
        start_scheduler()
    elif args.mode == 'evaluate':
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            target_date = datetime.now(timezone.utc) - timedelta(days=1)
        
        result = evaluator.run_daily_signal_evaluation(target_date)
        print(f"Evaluation result: {result}")
    elif args.mode == 'backfill':
        if not args.from_date or not args.to_date:
            print("Backfill mode requires --from-date and --to-date arguments")
            sys.exit(1)
        
        from_date = datetime.strptime(args.from_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        to_date = datetime.strptime(args.to_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        current_date = from_date
        while current_date <= to_date:
            print(f"Evaluating signals for {current_date.date()}")
            result = evaluator.run_daily_signal_evaluation(current_date)
            print(f"Result: {result}")
            current_date += timedelta(days=1)
        
        print("Backfill completed")