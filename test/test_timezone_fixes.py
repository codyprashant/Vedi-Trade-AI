import pytest
import pandas as pd
import pytz
from datetime import datetime, timezone
import numpy as np

from app.utils_time import as_utc_index, last_closed, normalize_timestamp


class TestTimezoneFixes:
    """Test suite for timezone and data handling fixes."""
    
    def test_as_utc_index_with_rangeindex(self):
        """Test that as_utc_index handles RangeIndex properly."""
        # Create DataFrame with RangeIndex (integer index)
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'close': [100, 101, 102, 103, 104]
        })
        
        # Should not raise an error and return unchanged DataFrame
        result = as_utc_index(df)
        
        # Should be the same DataFrame (RangeIndex unchanged)
        assert isinstance(result.index, pd.RangeIndex)
        assert len(result) == 5
        pd.testing.assert_frame_equal(result, df)
    
    def test_as_utc_index_with_datetime_index(self):
        """Test that as_utc_index handles DatetimeIndex properly."""
        # Create DataFrame with naive DatetimeIndex
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        result = as_utc_index(df)
        
        # Should have UTC timezone
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'
    
    def test_as_utc_index_with_timezone_aware_index(self):
        """Test that as_utc_index handles already timezone-aware DatetimeIndex."""
        # Create DataFrame with timezone-aware DatetimeIndex
        dates = pd.date_range('2024-01-01', periods=5, freq='1h', tz='US/Eastern')
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        result = as_utc_index(df)
        
        # Should be converted to UTC
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'
    
    def test_last_closed_with_integer_input(self):
        """Test that last_closed handles integer input gracefully."""
        # Should not raise an error when given integer input
        result = last_closed(123, "15min")
        
        # Should return a valid UTC timestamp
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None
    
    def test_last_closed_with_timestamp_input(self):
        """Test that last_closed handles proper timestamp input."""
        # Test with naive timestamp
        ts = pd.Timestamp('2024-01-01 12:30:00')
        result = last_closed(ts, "15min")
        
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None
        # Should be floored to 15-minute boundary
        assert result.minute in [0, 15, 30, 45]
    
    def test_last_closed_with_timezone_aware_timestamp(self):
        """Test that last_closed handles timezone-aware timestamp."""
        # Test with timezone-aware timestamp
        ts = pd.Timestamp('2024-01-01 12:30:00', tz='US/Eastern')
        result = last_closed(ts, "1h")
        
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None
        assert str(result.tz) == 'UTC'
        # Should be floored to hour boundary
        assert result.minute == 0
    
    def test_last_closed_with_string_input(self):
        """Test that last_closed handles string input."""
        result = last_closed("2024-01-01 12:30:00", "1h")
        
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None
        assert result.minute == 0
    
    def test_normalize_timestamp_with_timezone_aware(self):
        """Test that normalize_timestamp handles timezone-aware timestamps properly."""
        # Test with UTC timestamp (should return as-is)
        ts_utc = pd.Timestamp('2024-01-01 12:00:00', tz='UTC')
        result = normalize_timestamp(ts_utc)
        
        assert result == ts_utc
        assert str(result.tz) == 'UTC'
        
        # Test with non-UTC timezone (should convert)
        ts_et = pd.Timestamp('2024-01-01 12:00:00', tz='US/Eastern')
        result = normalize_timestamp(ts_et)
        
        assert isinstance(result, pd.Timestamp)
        assert str(result.tz) == 'UTC'
        # Check that timezone was converted (same moment, different timezone)
        assert str(result.tz) != str(ts_et.tz)
    
    def test_normalize_timestamp_with_naive(self):
        """Test that normalize_timestamp handles naive timestamps."""
        ts_naive = pd.Timestamp('2024-01-01 12:00:00')
        result = normalize_timestamp(ts_naive)
        
        assert isinstance(result, pd.Timestamp)
        assert str(result.tz) == 'UTC'
    
    def test_normalize_timestamp_with_string(self):
        """Test that normalize_timestamp handles string input."""
        result = normalize_timestamp("2024-01-01 12:00:00")
        
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None
    
    def test_normalize_timestamp_with_none(self):
        """Test that normalize_timestamp handles None input."""
        result = normalize_timestamp(None)
        assert result is None
    
    def test_edge_cases_empty_dataframe(self):
        """Test edge cases with empty DataFrames."""
        empty_df = pd.DataFrame()
        result = as_utc_index(empty_df)
        assert result.empty
        pd.testing.assert_frame_equal(result, empty_df)
    
    def test_edge_cases_single_row_dataframe(self):
        """Test edge cases with single-row DataFrames."""
        df = pd.DataFrame({
            'time': [pd.Timestamp('2024-01-01')],
            'close': [100]
        })
        
        result = as_utc_index(df)
        assert len(result) == 1
        assert isinstance(result.index, pd.RangeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])