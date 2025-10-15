"""
Time utilities for UTC normalization and closed bar calculations.
Provides consistent timestamp handling across the trading system.
"""

import pandas as pd
import pytz
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def as_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Localize or convert DataFrame index to UTC timezone.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with UTC-aware index
    """
    if df.empty:
        return df
    
    try:
        if df.index.tz is None:
            # Assume naive timestamps are in UTC and localize them
            df.index = df.index.tz_localize('UTC')
            logger.debug("Localized naive datetime index to UTC")
        elif df.index.tz != pytz.UTC:
            # Convert to UTC if in different timezone
            df.index = df.index.tz_convert('UTC')
            logger.debug(f"Converted datetime index from {df.index.tz} to UTC")
        
        return df
    except Exception as e:
        logger.error(f"Error converting index to UTC: {e}")
        # Return original DataFrame if conversion fails
        return df


def last_closed(ts: Union[pd.Timestamp, str], freq: str) -> pd.Timestamp:
    """
    Returns the floor of timestamp to the specified frequency in UTC.
    This represents the last closed bar for the given frequency.
    
    Args:
        ts: Timestamp to floor
        freq: Frequency string (e.g., '15min', '1h', '4h', '1d')
        
    Returns:
        UTC timestamp representing the last closed bar
    """
    try:
        # Convert to pandas Timestamp if string
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, utc=True)
        elif isinstance(ts, pd.Timestamp):
            # Ensure UTC timezone
            if ts.tz is None:
                ts = ts.tz_localize('UTC')
            elif ts.tz != pytz.UTC:
                ts = ts.tz_convert('UTC')
        
        # Floor to the specified frequency
        closed_ts = ts.floor(freq)
        
        logger.debug(f"Floored {ts} to {closed_ts} for frequency {freq}")
        return closed_ts
        
    except Exception as e:
        logger.error(f"Error calculating last closed bar: {e}")
        # Return current UTC time floored to frequency as fallback
        now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
        return now_utc.floor(freq)


def safe_float(value: Union[float, int, str, None], default: float = 0.0) -> float:
    """
    Safely convert value to float, returning default if conversion fails or value is NaN.
    
    Args:
        value: Value to convert
        default: Default value to return if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        
        result = float(value)
        
        # Check for NaN
        if pd.isna(result):
            return default
            
        return result
    except (ValueError, TypeError):
        return default


def normalize_timestamp(ts: Union[pd.Timestamp, str, None]) -> Optional[pd.Timestamp]:
    """
    Normalize timestamp to UTC timezone-aware format.
    
    Args:
        ts: Timestamp to normalize
        
    Returns:
        UTC timezone-aware timestamp or None if conversion fails
    """
    if ts is None:
        return None
    
    try:
        if isinstance(ts, str):
            result = pd.to_datetime(ts, utc=True)
        elif isinstance(ts, pd.Timestamp):
            if ts.tz is None:
                result = ts.tz_localize('UTC')
            else:
                result = ts.tz_convert('UTC')
        else:
            # Try to convert other types
            result = pd.to_datetime(ts, utc=True)
        
        return result
    except Exception as e:
        logger.error(f"Error normalizing timestamp {ts}: {e}")
        return None


def get_market_hours_utc(date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get market open and close times in UTC for a given date.
    Assumes US market hours (9:30 AM - 4:00 PM ET).
    
    Args:
        date: Date to get market hours for
        
    Returns:
        Tuple of (market_open_utc, market_close_utc)
    """
    try:
        # Ensure date is timezone-aware
        if date.tz is None:
            date = date.tz_localize('UTC')
        
        # Convert to Eastern timezone for market hours calculation
        et_tz = pytz.timezone('US/Eastern')
        date_et = date.tz_convert(et_tz).normalize()
        
        # Market hours in Eastern time
        market_open_et = date_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_et = date_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Convert back to UTC
        market_open_utc = market_open_et.tz_convert('UTC')
        market_close_utc = market_close_et.tz_convert('UTC')
        
        return market_open_utc, market_close_utc
    except Exception as e:
        logger.error(f"Error calculating market hours for {date}: {e}")
        # Return current day with default hours as fallback
        now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
        fallback_open = now_utc.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 AM ET in UTC (approx)
        fallback_close = now_utc.replace(hour=21, minute=0, second=0, microsecond=0)  # 4:00 PM ET in UTC (approx)
        return fallback_open, fallback_close


def retry(fn, tries=3, base=0.5, cap=4.0):
    """
    Retry a function with exponential backoff and jitter.
    
    Args:
        fn: Function to retry
        tries: Maximum number of attempts (default: 3)
        base: Base delay in seconds (default: 0.5)
        cap: Maximum delay in seconds (default: 4.0)
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception if all retries fail
    """
    import time
    import random
    
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:  # Last attempt
                raise e
            # Exponential backoff with jitter
            delay = min(cap, base * (2 ** i) + random.random() * 0.1)
            time.sleep(delay)


def compute_bounded_alignment_boost(
    base_strength: float,
    h1_aligned: bool,
    h4_aligned: bool,
    atr_ratio: float = 1.0,
    boost_config: dict = None
) -> tuple[float, dict]:
    """
    Compute bounded alignment boost with caps to prevent excessive signal amplification.
    
    Args:
        base_strength: Base signal strength (0-100)
        h1_aligned: Whether H1 timeframe is aligned
        h4_aligned: Whether H4 timeframe is aligned (requires h1_aligned)
        atr_ratio: Current ATR ratio for volatility scaling
        boost_config: Configuration dict with boost parameters
        
    Returns:
        tuple: (alignment_multiplier, boost_details)
    """
    if boost_config is None:
        # Default configuration if not provided
        boost_config = {
            "max_total_boost": 20.0,
            "max_individual_boost": 15.0,
            "volatility_scaling": True,
            "high_volatility_cap": 0.7,
            "extreme_volatility_cap": 0.5,
            "strength_based_scaling": True,
            "high_strength_threshold": 80.0,
            "diminishing_factor": 0.6
        }
    
    # Base boost percentages (from config)
    h1_boost_pct = 10.0  # ALIGNMENT_BOOST_H1
    h4_boost_pct = 5.0   # ALIGNMENT_BOOST_H4
    
    # Calculate raw boosts
    h1_boost = h1_boost_pct if h1_aligned else 0.0
    h4_boost = h4_boost_pct if (h1_aligned and h4_aligned) else 0.0
    total_raw_boost = h1_boost + h4_boost
    
    # Apply individual boost caps
    max_individual = boost_config.get("max_individual_boost", 15.0)
    h1_boost = min(h1_boost, max_individual)
    h4_boost = min(h4_boost, max_individual)
    
    # Apply total boost cap
    max_total = boost_config.get("max_total_boost", 20.0)
    total_boost = min(h1_boost + h4_boost, max_total)
    
    # Apply volatility scaling if enabled
    volatility_factor = 1.0
    if boost_config.get("volatility_scaling", True):
        if atr_ratio > 3.0:  # Extreme volatility
            volatility_factor = boost_config.get("extreme_volatility_cap", 0.5)
        elif atr_ratio > 2.0:  # High volatility
            volatility_factor = boost_config.get("high_volatility_cap", 0.7)
    
    # Apply strength-based scaling if enabled
    strength_factor = 1.0
    if boost_config.get("strength_based_scaling", True):
        high_strength_threshold = boost_config.get("high_strength_threshold", 80.0)
        if base_strength > high_strength_threshold:
            # Apply diminishing returns for high-strength signals
            strength_factor = boost_config.get("diminishing_factor", 0.6)
    
    # Apply all scaling factors
    final_boost = total_boost * volatility_factor * strength_factor
    
    # Convert to multiplier (boost percentage to multiplier)
    alignment_multiplier = 1.0 + (final_boost / 100.0)
    
    # Prepare detailed breakdown for logging
    boost_details = {
        "h1_aligned": h1_aligned,
        "h4_aligned": h4_aligned,
        "raw_h1_boost": h1_boost_pct if h1_aligned else 0.0,
        "raw_h4_boost": h4_boost_pct if (h1_aligned and h4_aligned) else 0.0,
        "total_raw_boost": total_raw_boost,
        "capped_boost": total_boost,
        "volatility_factor": volatility_factor,
        "strength_factor": strength_factor,
        "final_boost": final_boost,
        "alignment_multiplier": alignment_multiplier,
        "atr_ratio": atr_ratio,
        "base_strength": base_strength
    }
    
    return alignment_multiplier, boost_details