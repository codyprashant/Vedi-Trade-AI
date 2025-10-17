"""
Job Scheduler for Trading System

This module handles scheduling of automated tasks including:
- Daily signal performance evaluation at 5:00 AM
- Re-evaluation of open signals
- Structured signal trace retention cleanup
- System maintenance tasks
"""

import schedule
import time
import logging
import threading
from datetime import datetime, timezone
from signal_performance_evaluator import SignalPerformanceEvaluator
from app.db_trace import cleanup_signal_traces

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingSystemScheduler:
    """Scheduler for trading system automated tasks."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.evaluator = SignalPerformanceEvaluator()
        self.running = False
        self.scheduler_thread = None
    
    def daily_signal_evaluation_job(self):
        """Daily job to evaluate previous day's signals."""
        try:
            logger.info("Starting daily signal evaluation job")
            
            # Run daily evaluation for yesterday
            result = self.evaluator.run_daily_signal_evaluation()
            
            logger.info(f"Daily evaluation completed: {result}")
            
            # Also re-evaluate open signals
            open_result = self.evaluator.reevaluate_open_signals()
            logger.info(f"Open signals re-evaluation completed: {open_result}")
            
        except Exception as e:
            logger.error(f"Daily signal evaluation job failed: {e}", exc_info=True)
    
    def weekly_cleanup_job(self):
        """Weekly cleanup and maintenance job."""
        try:
            logger.info("Starting weekly cleanup job")

            # Add any weekly maintenance tasks here
            # For example: cleanup old logs, optimize database, etc.

            logger.info("Weekly cleanup completed")

        except Exception as e:
            logger.error(f"Weekly cleanup job failed: {e}", exc_info=True)

    def scheduled_trace_cleanup(self):
        """Daily cleanup for signal trace retention."""
        try:
            logger.info("Starting signal trace cleanup job")
            cleanup_signal_traces(days=7)
            logger.info("Signal trace cleanup job completed")
        except Exception as e:
            logger.error(f"Signal trace cleanup job failed: {e}", exc_info=True)
    
    def setup_schedule(self):
        """Set up the job schedule."""
        # Daily signal evaluation at 5:00 AM
        schedule.every().day.at("05:00").do(self.daily_signal_evaluation_job)
        
        # Daily signal trace cleanup at 3:00 AM
        schedule.every().day.at("03:00").do(self.scheduled_trace_cleanup)

        # Weekly cleanup on Sunday at 2:00 AM
        schedule.every().sunday.at("02:00").do(self.weekly_cleanup_job)
        
        logger.info("Job schedule configured:")
        logger.info("- Daily signal evaluation: 05:00 AM")
        logger.info("- Signal trace cleanup: Daily 03:00 AM")
        logger.info("- Weekly cleanup: Sunday 02:00 AM")
    
    def run_scheduler(self):
        """Run the scheduler in a loop."""
        logger.info("Starting job scheduler...")
        self.running = True
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                time.sleep(60)  # Continue after error
        
        logger.info("Job scheduler stopped")
    
    def start(self):
        """Start the scheduler in a separate thread."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.setup_schedule()
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Scheduler started in background thread")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def run_job_now(self, job_name):
        """Run a specific job immediately for testing."""
        if job_name == "daily_evaluation":
            self.daily_signal_evaluation_job()
        elif job_name == "weekly_cleanup":
            self.weekly_cleanup_job()
        else:
            logger.error(f"Unknown job: {job_name}")
    
    def get_next_run_times(self):
        """Get the next scheduled run times."""
        jobs_info = []
        for job in schedule.jobs:
            jobs_info.append({
                'job': str(job.job_func.__name__),
                'next_run': job.next_run.strftime('%Y-%m-%d %H:%M:%S') if job.next_run else 'Not scheduled'
            })
        return jobs_info


def main():
    """Main entry point for the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading System Job Scheduler')
    parser.add_argument('--mode', choices=['start', 'run-once', 'status'], default='start',
                       help='Scheduler mode')
    parser.add_argument('--job', choices=['daily_evaluation', 'weekly_cleanup'],
                       help='Job to run immediately (for run-once mode)')
    
    args = parser.parse_args()
    
    scheduler = TradingSystemScheduler()
    
    if args.mode == 'start':
        # Start the scheduler and keep it running
        try:
            scheduler.start()
            logger.info("Scheduler is running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while scheduler.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping scheduler...")
            scheduler.stop()
    
    elif args.mode == 'run-once':
        # Run a specific job once
        if not args.job:
            logger.error("--job parameter required for run-once mode")
            return
        
        logger.info(f"Running job '{args.job}' once...")
        scheduler.run_job_now(args.job)
    
    elif args.mode == 'status':
        # Show scheduler status
        scheduler.setup_schedule()
        next_runs = scheduler.get_next_run_times()
        
        print("Scheduled Jobs:")
        print("-" * 50)
        for job_info in next_runs:
            print(f"Job: {job_info['job']}")
            print(f"Next Run: {job_info['next_run']}")
            print("-" * 50)


if __name__ == '__main__':
    main()