"""
Timezone Conversion Utilities
=============================

Centralized timezone handling for consistent datetime processing across the trading system.
All trading data is normalized to Eastern Time (America/New_York) for consistent market hours.
"""

import logging
import pandas as pd
from typing import Optional, List

# ARCHITECTURAL FIX: Remove constants.py dependency
TARGET_TIMEZONE = 'America/New_York'  # Target timezone for trading
DEFAULT_UTC_TIMEZONE = 'UTC'  # Default UTC timezone
DATETIME_KEYWORDS = ('time', 'date', 'timestamp')  # Keywords for datetime detection

logger = logging.getLogger(__name__)


def convert_to_target_timezone(dt_series: pd.Series, source_timezone: Optional[str] = None) -> pd.Series:
    """
    Convert datetime series to target timezone (America/New_York).
    
    Args:
        dt_series: Pandas datetime series
        source_timezone: Source timezone (defaults to UTC if naive)
        
    Returns:
        Datetime series converted to target timezone
    """
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        dt_series = pd.to_datetime(dt_series)
    
    if dt_series.dt.tz is None:
        # Naive timestamps - localize to source timezone first
        source_tz = source_timezone or DEFAULT_UTC_TIMEZONE
        dt_series = dt_series.dt.tz_localize(source_tz)
        logger.info(f"Localized naive timestamps to {source_tz}")
    
    if str(dt_series.dt.tz) != TARGET_TIMEZONE:
        # Convert to target timezone
        original_tz = str(dt_series.dt.tz)
        dt_series = dt_series.dt.tz_convert(TARGET_TIMEZONE)
        logger.info(f"Converted timestamps from {original_tz} to {TARGET_TIMEZONE}")
    
    return dt_series


def convert_datetime_index_to_column(df: pd.DataFrame, column_name: str = 'datetime') -> pd.DataFrame:
    """
    Convert DatetimeIndex to a datetime column with proper timezone handling.
    
    Args:
        df: DataFrame with DatetimeIndex
        column_name: Name for the datetime column
        
    Returns:
        DataFrame with datetime column in target timezone
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Convert timezone before resetting index
    if df.index.tz is None:
        # Naive DatetimeIndex - assume UTC and convert to target timezone
        df.index = df.index.tz_localize(DEFAULT_UTC_TIMEZONE).tz_convert(TARGET_TIMEZONE)
        logger.info(f"Converted naive DatetimeIndex from {DEFAULT_UTC_TIMEZONE} to {TARGET_TIMEZONE}")
    elif str(df.index.tz) != TARGET_TIMEZONE:
        # Timezone-aware but not target timezone - convert
        original_tz = str(df.index.tz)
        df.index = df.index.tz_convert(TARGET_TIMEZONE)
        logger.info(f"Converted DatetimeIndex from {original_tz} to {TARGET_TIMEZONE}")
    
    df = df.reset_index()
    # Handle both unnamed index ('index') and named index (e.g., 'timestamp')
    index_col_name = df.columns[0]  # First column is the reset index
    df.rename(columns={index_col_name: column_name}, inplace=True)
    logger.info(f"Converted DatetimeIndex ('{index_col_name}') to {column_name} column")
    
    return df


def find_datetime_candidates(df: pd.DataFrame) -> List[str]:
    """
    Find potential datetime columns based on column names.
    
    Args:
        df: DataFrame to search
        
    Returns:
        List of column names that might contain datetime data
    """
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in DATETIME_KEYWORDS):
            candidates.append(col)
    return candidates


def create_synthetic_datetime(df: pd.DataFrame, 
                            column_name: str = 'datetime',
                            start_date: str = '2023-01-01',
                            freq: str = '15min') -> pd.DataFrame:
    """
    Create synthetic datetime column when no datetime data is available.
    
    Args:
        df: DataFrame to add datetime column to
        column_name: Name for the datetime column
        start_date: Start date for synthetic timestamps
        freq: Frequency for synthetic timestamps
        
    Returns:
        DataFrame with synthetic datetime column in target timezone
    """
    logger.warning(f"No datetime column found, creating synthetic timestamps in {TARGET_TIMEZONE}")
    df[column_name] = pd.date_range(
        start=start_date,
        periods=len(df),
        freq=freq,
        tz=TARGET_TIMEZONE
    )
    return df


def ensure_datetime_column(df: pd.DataFrame, column_name: str = 'datetime') -> pd.DataFrame:
    """
    Comprehensive datetime column normalization with timezone conversion.
    
    This function handles all common datetime formats and ensures consistent
    timezone handling across the trading system.
    
    Args:
        df: DataFrame to normalize
        column_name: Target datetime column name
        
    Returns:
        DataFrame with normalized datetime column in target timezone
    """
    # Strategy 1: If already has datetime column, ensure proper typing and timezone
    if column_name in df.columns:
        df[column_name] = convert_to_target_timezone(df[column_name])
        return df
    
    # Strategy 2: If DatetimeIndex, convert to column with timezone handling
    if isinstance(df.index, pd.DatetimeIndex):
        return convert_datetime_index_to_column(df, column_name)
    
    # Strategy 3: Look for datetime-like columns
    datetime_candidates = find_datetime_candidates(df)
    if datetime_candidates:
        # Use the first candidate and convert to target timezone
        df[column_name] = convert_to_target_timezone(df[datetime_candidates[0]])
        logger.info(f"Created {column_name} column from {datetime_candidates[0]} and converted to {TARGET_TIMEZONE}")
        return df
    
    # Strategy 4: As last resort, create synthetic datetime
    return create_synthetic_datetime(df, column_name)


def validate_datetime_column(df: pd.DataFrame, column_name: str = 'datetime') -> bool:
    """
    Validate that the datetime column is properly formatted and in target timezone.
    
    Args:
        df: DataFrame to validate
        column_name: Name of datetime column
        
    Returns:
        True if datetime column is valid
        
    Raises:
        ValueError: If datetime column is invalid
    """
    if column_name not in df.columns:
        raise ValueError(f"Missing {column_name} column")
    
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        raise ValueError(f"{column_name} column is not datetime type")
    
    if df[column_name].dt.tz is None:
        raise ValueError(f"{column_name} column lacks timezone information")
    
    if str(df[column_name].dt.tz) != TARGET_TIMEZONE:
        logger.warning(f"{column_name} column timezone is {df[column_name].dt.tz}, expected {TARGET_TIMEZONE}")
    
    return True