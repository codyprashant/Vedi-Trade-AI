"""
Comprehensive logging configuration for the trading application.
Provides structured logging with proper formatting, levels, and handlers.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
            
        return json.dumps(log_entry, default=str)


class SafeNaNFormatter(logging.Formatter):
    """Formatter that safely handles NaN and Inf values."""
    
    def format(self, record):
        """Format record with NaN safety."""
        # Replace NaN/Inf in message
        if hasattr(record, 'args') and record.args:
            safe_args = []
            for arg in record.args:
                if isinstance(arg, (int, float)):
                    if str(arg).lower() in ['nan', 'inf', '-inf']:
                        safe_args.append(f"<{arg}>")
                    else:
                        safe_args.append(arg)
                else:
                    safe_args.append(arg)
            record.args = tuple(safe_args)
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to logs/ in project root)
        enable_json: Whether to use JSON formatting for file logs
        enable_console: Whether to enable console logging
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    # Create log directory
    if log_dir is None:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = SafeNaNFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # File handlers
    handlers_config = [
        {
            'name': 'main',
            'filename': log_dir / 'trading_app.log',
            'level': log_level.upper(),
            'format_type': 'json' if enable_json else 'standard'
        },
        {
            'name': 'error',
            'filename': log_dir / 'errors.log',
            'level': 'ERROR',
            'format_type': 'json' if enable_json else 'standard'
        },
        {
            'name': 'database',
            'filename': log_dir / 'database.log',
            'level': 'DEBUG',
            'format_type': 'json' if enable_json else 'standard'
        },
        {
            'name': 'signals',
            'filename': log_dir / 'signals.log',
            'level': 'INFO',
            'format_type': 'json' if enable_json else 'standard'
        },
        {
            'name': 'performance',
            'filename': log_dir / 'performance.log',
            'level': 'INFO',
            'format_type': 'json' if enable_json else 'standard'
        }
    ]
    
    # Create file handlers
    for handler_config in handlers_config:
        file_handler = logging.handlers.RotatingFileHandler(
            handler_config['filename'],
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if handler_config['format_type'] == 'json':
            formatter = JSONFormatter()
        else:
            formatter = SafeNaNFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, handler_config['level']))
        root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {
        'main': logging.getLogger('trading_app'),
        'database': logging.getLogger('trading_app.database'),
        'signals': logging.getLogger('trading_app.signals'),
        'indicators': logging.getLogger('trading_app.indicators'),
        'backtest': logging.getLogger('trading_app.backtest'),
        'websocket': logging.getLogger('trading_app.websocket'),
        'api': logging.getLogger('trading_app.api'),
        'performance': logging.getLogger('trading_app.performance'),
        'tests': logging.getLogger('trading_app.tests')
    }
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f'trading_app.{name}')


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    extra_data = {
        'operation': operation,
        'duration_ms': round(duration * 1000, 2),
        **kwargs
    }
    
    # Create a custom log record with extra data
    record = logger.makeRecord(
        logger.name, logging.INFO, '', 0,
        f"Performance: {operation} completed in {duration:.3f}s",
        (), None
    )
    record.extra_data = extra_data
    logger.handle(record)


def log_database_operation(logger: logging.Logger, operation: str, table: str, 
                          rows_affected: int = None, duration: float = None, **kwargs):
    """Log database operations with structured data."""
    extra_data = {
        'operation_type': 'database',
        'operation': operation,
        'table': table,
        'rows_affected': rows_affected,
        'duration_ms': round(duration * 1000, 2) if duration else None,
        **kwargs
    }
    
    message = f"Database {operation} on {table}"
    if rows_affected is not None:
        message += f" ({rows_affected} rows)"
    if duration:
        message += f" in {duration:.3f}s"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, '', 0, message, (), None
    )
    record.extra_data = extra_data
    logger.handle(record)


def log_signal_event(logger: logging.Logger, event_type: str, symbol: str, 
                    signal_data: Dict[str, Any], **kwargs):
    """Log signal-related events with structured data."""
    extra_data = {
        'event_type': 'signal',
        'signal_event': event_type,
        'symbol': symbol,
        'signal_data': signal_data,
        **kwargs
    }
    
    message = f"Signal {event_type} for {symbol}: {signal_data.get('side', 'unknown')} @ {signal_data.get('strength', 'N/A')}"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, '', 0, message, (), None
    )
    record.extra_data = extra_data
    logger.handle(record)


def safe_log_value(value: Any) -> str:
    """Safely convert value to string for logging, handling NaN/Inf."""
    if isinstance(value, (int, float)):
        if str(value).lower() in ['nan', 'inf', '-inf']:
            return f"<{value}>"
    return str(value)


def create_test_logger(test_name: str) -> logging.Logger:
    """Create a logger specifically for test output."""
    logger = logging.getLogger(f'trading_app.tests.{test_name}')
    
    # Remove console handler for tests to avoid print-like output
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    return logger


# Initialize default logging configuration
_default_loggers = None

def get_default_loggers() -> Dict[str, logging.Logger]:
    """Get default configured loggers."""
    global _default_loggers
    if _default_loggers is None:
        _default_loggers = setup_logging()
    return _default_loggers


# Convenience functions for common logging patterns
def log_info(message: str, logger_name: str = 'main', **kwargs):
    """Log info message with optional extra data."""
    logger = get_logger(logger_name)
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.extra_data = kwargs
        logger.handle(record)
    else:
        logger.info(message)


def log_error(message: str, exception: Exception = None, logger_name: str = 'main', **kwargs):
    """Log error message with optional exception and extra data."""
    logger = get_logger(logger_name)
    if exception:
        logger.error(message, exc_info=exception)
    else:
        if kwargs:
            record = logger.makeRecord(
                logger.name, logging.ERROR, '', 0, message, (), None
            )
            record.extra_data = kwargs
            logger.handle(record)
        else:
            logger.error(message)


def log_warning(message: str, logger_name: str = 'main', **kwargs):
    """Log warning message with optional extra data."""
    logger = get_logger(logger_name)
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.WARNING, '', 0, message, (), None
        )
        record.extra_data = kwargs
        logger.handle(record)
    else:
        logger.warning(message)


def log_debug(message: str, logger_name: str = 'main', **kwargs):
    """Log debug message with optional extra data."""
    logger = get_logger(logger_name)
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.DEBUG, '', 0, message, (), None
        )
        record.extra_data = kwargs
        logger.handle(record)
    else:
        logger.debug(message)