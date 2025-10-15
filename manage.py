#!/usr/bin/env python3
"""
Management script for trading system operations.

This script provides command-line interface for various system operations
including signal evaluation, database management, and system maintenance.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jobs.signal_performance_evaluator import SignalPerformanceEvaluator
from app.db import ensure_signal_evaluation_tables, ensure_signals_table, ensure_new_backtesting_tables


def evaluate_signals_command(args):
    """Handle the evaluate_signals command."""
    evaluator = SignalPerformanceEvaluator()
    
    if args.from_date and args.to_date:
        # Backfill mode
        from_date = datetime.strptime(args.from_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        to_date = datetime.strptime(args.to_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        print(f"Starting backfill evaluation from {from_date.date()} to {to_date.date()}")
        
        current_date = from_date
        total_signals = 0
        
        while current_date <= to_date:
            print(f"Evaluating signals for {current_date.date()}")
            result = evaluator.run_daily_signal_evaluation(current_date)
            total_signals += result.get('signals_evaluated', 0)
            print(f"  - Evaluated {result.get('signals_evaluated', 0)} signals")
            current_date += timedelta(days=1)
        
        print(f"Backfill completed. Total signals evaluated: {total_signals}")
        
    elif args.date:
        # Single date evaluation
        target_date = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        print(f"Evaluating signals for {target_date.date()}")
        result = evaluator.run_daily_signal_evaluation(target_date)
        print(f"Evaluation result: {result}")
        
    else:
        # Default to yesterday
        target_date = datetime.now(timezone.utc) - timedelta(days=1)
        print(f"Evaluating signals for {target_date.date()}")
        result = evaluator.run_daily_signal_evaluation(target_date)
        print(f"Evaluation result: {result}")


def reevaluate_open_command(args):
    """Handle the reevaluate_open command."""
    evaluator = SignalPerformanceEvaluator()
    print("Re-evaluating open signals...")
    result = evaluator.reevaluate_open_signals()
    print(f"Re-evaluation result: {result}")


def init_db_command(args):
    """Initialize database tables."""
    print("Initializing database tables...")
    
    try:
        ensure_signals_table()
        print("✓ Signals table ensured")
        
        ensure_new_backtesting_tables()
        print("✓ New unified backtesting tables ensured")
        
        ensure_signal_evaluation_tables()
        print("✓ Signal evaluation tables ensured")
        
        print("Database initialization completed successfully")
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        sys.exit(1)


def show_performance_command(args):
    """Show performance statistics."""
    from app.db import fetch_signal_performance_summary
    
    try:
        summaries = fetch_signal_performance_summary(args.symbol, args.timeframe)
        
        if not summaries:
            print("No performance data found")
            return
        
        print("\nSignal Performance Summary")
        print("=" * 80)
        print(f"{'Symbol':<10} {'Timeframe':<10} {'Total':<8} {'Wins':<6} {'Losses':<7} {'Open':<6} {'Efficiency':<10} {'Avg P/L%':<10} {'Total ROI%':<12}")
        print("-" * 80)
        
        for summary in summaries:
            print(f"{summary['symbol']:<10} {summary['timeframe']:<10} {summary['total_signals']:<8} "
                  f"{summary['win_count']:<6} {summary['loss_count']:<7} {summary['open_count']:<6} "
                  f"{summary['efficiency_pct']:<10.1f} {summary['avg_profit_pct']:<10.2f} {summary['total_roi_pct']:<12.2f}")
        
        print("-" * 80)
        
        # Calculate overall statistics
        total_signals = sum(s['total_signals'] for s in summaries)
        total_wins = sum(s['win_count'] for s in summaries)
        total_losses = sum(s['loss_count'] for s in summaries)
        total_open = sum(s['open_count'] for s in summaries)
        overall_efficiency = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        total_roi = sum(s['total_roi_pct'] for s in summaries)
        
        print(f"{'OVERALL':<10} {'ALL':<10} {total_signals:<8} {total_wins:<6} {total_losses:<7} {total_open:<6} "
              f"{overall_efficiency:<10.1f} {'-':<10} {total_roi:<12.2f}")
        
    except Exception as e:
        print(f"Error fetching performance data: {e}")


def main():
    """Main entry point for the management script."""
    parser = argparse.ArgumentParser(description='Trading System Management')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # evaluate_signals command
    eval_parser = subparsers.add_parser('evaluate_signals', help='Evaluate signal performance')
    eval_parser.add_argument('--date', type=str, help='Date to evaluate (YYYY-MM-DD format)')
    eval_parser.add_argument('--from', dest='from_date', type=str, help='Start date for backfill (YYYY-MM-DD format)')
    eval_parser.add_argument('--to', dest='to_date', type=str, help='End date for backfill (YYYY-MM-DD format)')
    eval_parser.set_defaults(func=evaluate_signals_command)
    
    # reevaluate_open command
    reeval_parser = subparsers.add_parser('reevaluate_open', help='Re-evaluate open signals')
    reeval_parser.set_defaults(func=reevaluate_open_command)
    
    # init_db command
    init_parser = subparsers.add_parser('init_db', help='Initialize database tables')
    init_parser.set_defaults(func=init_db_command)
    
    # show_performance command
    perf_parser = subparsers.add_parser('show_performance', help='Show performance statistics')
    perf_parser.add_argument('--symbol', type=str, help='Filter by symbol')
    perf_parser.add_argument('--timeframe', type=str, help='Filter by timeframe')
    perf_parser.set_defaults(func=show_performance_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Call the appropriate function
    args.func(args)


if __name__ == '__main__':
    main()