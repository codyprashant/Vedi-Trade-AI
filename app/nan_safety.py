"""
NaN safety utilities for handling NaN and Inf values throughout the application.
Provides comprehensive protection against invalid numerical values.
"""

import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List, Union, Optional
import json
from decimal import Decimal, InvalidOperation


def is_nan_or_inf(value: Any) -> bool:
    """
    Check if a value is NaN or Inf.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is NaN or Inf, False otherwise
    """
    if value is None:
        return False
    
    try:
        if isinstance(value, (int, float)):
            return math.isnan(value) or math.isinf(value)
        elif isinstance(value, np.number):
            return np.isnan(value) or np.isinf(value)
        elif isinstance(value, pd.Series):
            return value.isna().any() or np.isinf(value).any()
        elif isinstance(value, pd.DataFrame):
            return value.isna().any().any() or np.isinf(value.select_dtypes(include=[np.number])).any().any()
        elif isinstance(value, (list, tuple)):
            return any(is_nan_or_inf(item) for item in value)
        elif isinstance(value, dict):
            return any(is_nan_or_inf(v) for v in value.values())
        else:
            # Try to convert to float and check
            float_val = float(value)
            return math.isnan(float_val) or math.isinf(float_val)
    except (ValueError, TypeError, OverflowError):
        return False


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float, replacing NaN/Inf with default.
    
    Args:
        value: Value to convert
        default: Default value to use if conversion fails or result is NaN/Inf
        
    Returns:
        Safe float value
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            # Handle string representations
            if value.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
                return default
            result = float(value)
        elif isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, np.number):
            result = float(value)
        elif isinstance(value, Decimal):
            result = float(value)
        else:
            result = float(value)
        
        # Check if result is valid
        if math.isnan(result) or math.isinf(result):
            return default
        
        return result
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int, replacing NaN/Inf with default.
    
    Args:
        value: Value to convert
        default: Default value to use if conversion fails
        
    Returns:
        Safe int value
    """
    try:
        float_val = safe_float(value, default)
        if float_val == default and is_nan_or_inf(value):
            return default
        return int(float_val)
    except (ValueError, TypeError, OverflowError):
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero and NaN/Inf.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for invalid operations
        
    Returns:
        Safe division result
    """
    try:
        num = safe_float(numerator, default)
        den = safe_float(denominator, 1.0 if default == 0.0 else default)
        
        if den == 0.0:
            return default
        
        result = num / den
        
        if math.isnan(result) or math.isinf(result):
            return default
        
        return result
    except (ZeroDivisionError, OverflowError):
        return default


def safe_percentage(value: float, total: float, default: float = 0.0) -> float:
    """
    Safely calculate percentage, handling edge cases.
    
    Args:
        value: Value to calculate percentage for
        total: Total value
        default: Default percentage if calculation fails
        
    Returns:
        Safe percentage value
    """
    return safe_divide(value * 100, total, default)


def clean_dataframe(df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
    """
    Clean DataFrame by handling NaN and Inf values.
    
    Args:
        df: DataFrame to clean
        fill_method: Method to fill NaN values ('forward', 'backward', 'zero', 'drop')
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Replace Inf values with NaN first
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values based on method
    if fill_method == 'forward':
        cleaned_df = cleaned_df.fillna(method='ffill')
    elif fill_method == 'backward':
        cleaned_df = cleaned_df.fillna(method='bfill')
    elif fill_method == 'zero':
        cleaned_df = cleaned_df.fillna(0)
    elif fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Final check - if still NaN, fill with 0
    cleaned_df = cleaned_df.fillna(0)
    
    return cleaned_df


def clean_series(series: pd.Series, fill_method: str = 'forward') -> pd.Series:
    """
    Clean Series by handling NaN and Inf values.
    
    Args:
        series: Series to clean
        fill_method: Method to fill NaN values
        
    Returns:
        Cleaned Series
    """
    if series is None or series.empty:
        return series
    
    # Create a copy
    cleaned_series = series.copy()
    
    # Replace Inf values with NaN
    cleaned_series = cleaned_series.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values
    if fill_method == 'forward':
        cleaned_series = cleaned_series.fillna(method='ffill')
    elif fill_method == 'backward':
        cleaned_series = cleaned_series.fillna(method='bfill')
    elif fill_method == 'zero':
        cleaned_series = cleaned_series.fillna(0)
    elif fill_method == 'drop':
        cleaned_series = cleaned_series.dropna()
    
    # Final fallback
    cleaned_series = cleaned_series.fillna(0)
    
    return cleaned_series


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON, handling NaN/Inf values.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string with NaN/Inf values replaced
    """
    def clean_for_json(item):
        if isinstance(item, dict):
            return {k: clean_for_json(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return [clean_for_json(i) for i in item]
        elif isinstance(item, (int, float)):
            if math.isnan(item) or math.isinf(item):
                return None
            return item
        elif isinstance(item, np.number):
            if np.isnan(item) or np.isinf(item):
                return None
            return float(item)
        elif isinstance(item, pd.Series):
            return clean_for_json(item.tolist())
        elif isinstance(item, pd.DataFrame):
            return clean_for_json(item.to_dict())
        else:
            return item
    
    cleaned_obj = clean_for_json(obj)
    return json.dumps(cleaned_obj, default=str)


def validate_numerical_dict(data: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
    """
    Validate and clean numerical values in a dictionary.
    
    Args:
        data: Dictionary to validate
        required_keys: Keys that must be present and valid
        
    Returns:
        Validated and cleaned dictionary
        
    Raises:
        ValueError: If required keys are missing or invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    
    cleaned_data = {}
    
    for key, value in data.items():
        if isinstance(value, (int, float)):
            cleaned_data[key] = safe_float(value)
        elif isinstance(value, dict):
            cleaned_data[key] = validate_numerical_dict(value)
        elif isinstance(value, (list, tuple)):
            cleaned_data[key] = [safe_float(item) if isinstance(item, (int, float)) else item for item in value]
        else:
            cleaned_data[key] = value
    
    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in cleaned_data:
                raise ValueError(f"Required key '{key}' is missing")
            if is_nan_or_inf(cleaned_data[key]):
                raise ValueError(f"Required key '{key}' has invalid value: {cleaned_data[key]}")
    
    return cleaned_data


def safe_calculation(func, *args, default=0.0, **kwargs):
    """
    Safely execute a calculation function, handling exceptions and NaN/Inf results.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        default: Default value if calculation fails
        **kwargs: Keyword arguments for the function
        
    Returns:
        Safe calculation result
    """
    try:
        result = func(*args, **kwargs)
        
        if is_nan_or_inf(result):
            return default
        
        return result
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, RuntimeError):
        return default


def ensure_finite_dataframe(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Ensure DataFrame contains only finite values in specified columns.
    
    Args:
        df: DataFrame to check
        columns: Specific columns to check (if None, check all numeric columns)
        
    Returns:
        DataFrame with finite values only
        
    Raises:
        ValueError: If DataFrame contains non-finite values after cleaning
    """
    if df is None or df.empty:
        return df
    
    # Determine columns to check
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Clean the DataFrame
    cleaned_df = clean_dataframe(df)
    
    # Check for remaining non-finite values
    for col in columns:
        if col in cleaned_df.columns:
            if not np.isfinite(cleaned_df[col]).all():
                non_finite_count = (~np.isfinite(cleaned_df[col])).sum()
                raise ValueError(f"Column '{col}' contains {non_finite_count} non-finite values after cleaning")
    
    return cleaned_df


def safe_technical_indicator(indicator_func, data: pd.DataFrame, *args, **kwargs) -> pd.Series:
    """
    Safely calculate technical indicator, handling NaN/Inf values.
    
    Args:
        indicator_func: Technical indicator function
        data: Input data
        *args: Arguments for indicator function
        **kwargs: Keyword arguments for indicator function
        
    Returns:
        Safe indicator series
    """
    try:
        # Clean input data
        clean_data = clean_dataframe(data)
        
        # Calculate indicator
        result = indicator_func(clean_data, *args, **kwargs)
        
        # Clean result
        if isinstance(result, pd.Series):
            return clean_series(result)
        elif isinstance(result, pd.DataFrame):
            return clean_dataframe(result)
        else:
            return safe_float(result)
    except Exception:
        # Return empty series with same index as input
        if hasattr(data, 'index'):
            return pd.Series(0.0, index=data.index)
        else:
            return pd.Series([0.0])


class NaNSafetyContext:
    """Context manager for NaN-safe operations."""
    
    def __init__(self, raise_on_nan: bool = False, default_value: float = 0.0):
        self.raise_on_nan = raise_on_nan
        self.default_value = default_value
        self.original_settings = {}
    
    def __enter__(self):
        # Store original numpy settings
        self.original_settings = {
            'invalid': np.geterr()['invalid'],
            'divide': np.geterr()['divide'],
            'over': np.geterr()['over']
        }
        
        # Set numpy to handle errors appropriately
        if self.raise_on_nan:
            np.seterr(invalid='raise', divide='raise', over='raise')
        else:
            np.seterr(invalid='ignore', divide='ignore', over='ignore')
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original numpy settings
        np.seterr(**self.original_settings)
        
        # Handle exceptions if needed
        if exc_type and not self.raise_on_nan:
            # Log the exception but don't re-raise
            return True  # Suppress exception
        
        return False


# Convenience functions for common operations
def safe_mean(values: Union[List, pd.Series, np.ndarray], default: float = 0.0) -> float:
    """Safely calculate mean, handling NaN/Inf values."""
    try:
        if isinstance(values, pd.Series):
            clean_values = clean_series(values)
        elif isinstance(values, np.ndarray):
            clean_values = pd.Series(values)
            clean_values = clean_series(clean_values)
        else:
            clean_values = pd.Series([safe_float(v, default) for v in values])
        
        if clean_values.empty:
            return default
        
        result = clean_values.mean()
        return safe_float(result, default)
    except Exception:
        return default


def safe_std(values: Union[List, pd.Series, np.ndarray], default: float = 0.0) -> float:
    """Safely calculate standard deviation, handling NaN/Inf values."""
    try:
        if isinstance(values, pd.Series):
            clean_values = clean_series(values)
        elif isinstance(values, np.ndarray):
            clean_values = pd.Series(values)
            clean_values = clean_series(clean_values)
        else:
            clean_values = pd.Series([safe_float(v, default) for v in values])
        
        if clean_values.empty or len(clean_values) < 2:
            return default
        
        result = clean_values.std()
        return safe_float(result, default)
    except Exception:
        return default


def safe_min_max(values: Union[List, pd.Series, np.ndarray], default_min: float = 0.0, default_max: float = 1.0) -> tuple:
    """Safely calculate min and max, handling NaN/Inf values."""
    try:
        if isinstance(values, pd.Series):
            clean_values = clean_series(values)
        elif isinstance(values, np.ndarray):
            clean_values = pd.Series(values)
            clean_values = clean_series(clean_values)
        else:
            clean_values = pd.Series([safe_float(v) for v in values])
        
        if clean_values.empty:
            return default_min, default_max
        
        min_val = safe_float(clean_values.min(), default_min)
        max_val = safe_float(clean_values.max(), default_max)
        
        return min_val, max_val
    except Exception:
        return default_min, default_max