"""
Pure Functional Data Splitter for 3-Way Splits
===============================================

Complete rewrite of the data splitter as a pure functional module.
Eliminates data leakage by providing clean 3-way splits (optimize/validate/test)
with temporal consistency and configurable gap days.

This module replaces the class-based DataSplitter with pure functions that:
- Create immutable DataSplit objects with temporal ordering validation
- Support both chronological and walk-forward splitting strategies
- Enforce gap days between splits to prevent data leakage
- Provide comprehensive metadata for optimization engines

Key Functions:
- chronological_split(): Single 3-way split with configurable ratios
- walk_forward_splitter(): Iterator of 3-way splits for robust validation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterator, Dict, Tuple
from pathlib import Path
import sys

# ARCHITECTURAL FIX: Remove constants.py dependency
# Use function parameter defaults instead of global constants

# Import the new DataSplit structure
sys.path.append(str(Path(__file__).parent))
from data_structures import DataSplit

logger = logging.getLogger(__name__)


def chronological_split(data: pd.DataFrame, 
                        ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),  # train, validation, test
                        gap_days: int = 1) -> DataSplit:
    """
    Create a single 3-way chronological split with temporal consistency.
    
    Splits data into optimize/validate/test periods with configurable ratios
    and gap days between splits to prevent data leakage.
    
    Args:
        data: DataFrame with datetime column and OHLCV data
        ratios: Tuple of (optimize_ratio, validate_ratio, test_ratio)
        gap_days: Number of days to skip between splits
        
    Returns:
        DataSplit object with optimize/validate/test data
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or data is insufficient
        AssertionError: If temporal ordering fails
    """
    if data.empty:
        raise ValueError("Cannot split empty DataFrame")
    
    if 'datetime' not in data.columns:
        raise ValueError("DataFrame must have 'datetime' column for chronological split")
    
    # Validate ratios
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    optimize_ratio, validate_ratio, test_ratio = ratios
    
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive")
    
    # Sort data by datetime
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # Check minimum data requirements for 3-way split
    min_bars_per_split = 50
    if len(data) < min_bars_per_split * 3:
        raise ValueError(f"Insufficient data for 3-way split: {len(data)} bars, need at least {min_bars_per_split * 3}")
    
    # Calculate base split points
    total_bars = len(data)
    
    if gap_days == 0:
        # Simple index-based splitting for testing (no gaps)
        optimize_end_idx = int(total_bars * optimize_ratio)
        validate_end_idx = optimize_end_idx + int(total_bars * validate_ratio)
        
        optimize_data = data.iloc[:optimize_end_idx].copy()
        validate_data = data.iloc[optimize_end_idx:validate_end_idx].copy()
        test_data = data.iloc[validate_end_idx:].copy()
    else:
        # Apply gap days by converting to datetime-based splits
        optimize_end_idx = int(total_bars * optimize_ratio)
        optimize_data = data.iloc[:optimize_end_idx].copy()
        
        # Calculate gap start time
        if not optimize_data.empty:
            optimize_end_time = optimize_data['datetime'].max()
            gap_start_time = optimize_end_time + pd.DateOffset(days=gap_days)
            
            # Find validate start index after gap
            validate_mask = data['datetime'] >= gap_start_time
            validate_candidates = data[validate_mask]
            
            if validate_candidates.empty:
                raise ValueError(f"No data available after {gap_days}-day gap from optimize period")
            
            validate_start_idx = validate_candidates.index[0]
            validate_end_idx = min(validate_start_idx + int(total_bars * validate_ratio), len(data))
            
            validate_data = data.iloc[validate_start_idx:validate_end_idx].copy()
            
            # Calculate test data with gap
            if not validate_data.empty:
                validate_end_time = validate_data['datetime'].max()
                test_gap_start_time = validate_end_time + pd.DateOffset(days=gap_days)
                
                # Find test start index after gap
                test_mask = data['datetime'] >= test_gap_start_time
                test_candidates = data[test_mask]
                
                if test_candidates.empty:
                    raise ValueError(f"No data available for test period after {gap_days}-day gap from validate period")
                
                test_start_idx = test_candidates.index[0]
                test_data = data.iloc[test_start_idx:].copy()
                
                if test_data.empty:
                    raise ValueError("No data available for test period")
            else:
                raise ValueError("Validate data is empty after gap adjustment")
        else:
            raise ValueError("Optimize data is empty")
    
    # Validate minimum sizes after gap adjustments
    if len(optimize_data) < min_bars_per_split:
        raise ValueError(f"Optimize data too small after gap adjustment: {len(optimize_data)} bars")
    if len(validate_data) < min_bars_per_split:
        raise ValueError(f"Validate data too small after gap adjustment: {len(validate_data)} bars")
    if len(test_data) < min_bars_per_split:
        raise ValueError(f"Test data too small after gap adjustment: {len(test_data)} bars")
    
    # Create metadata
    metadata = {
        'split_type': 'chronological',
        'requested_ratios': ratios,
        'actual_ratios': (
            len(optimize_data) / total_bars,
            len(validate_data) / total_bars,
            len(test_data) / total_bars
        ),
        'gap_days': gap_days,
        'total_bars': total_bars,
        'train_bars': len(optimize_data),
        'validation_bars': len(validate_data),
        'test_bars': len(test_data),
        'created_at': datetime.now().isoformat(),
        'optimize_period': {
            'start': optimize_data['datetime'].min(),
            'end': optimize_data['datetime'].max()
        },
        'validate_period': {
            'start': validate_data['datetime'].min(),
            'end': validate_data['datetime'].max()
        },
        'test_period': {
            'start': test_data['datetime'].min(),
            'end': test_data['datetime'].max()
        }
    }
    
    logger.info(f"Chronological 3-way split created: "
               f"optimize={len(optimize_data)} bars, "
               f"validate={len(validate_data)} bars, "
               f"test={len(test_data)} bars, "
               f"gap_days={gap_days}")
    
    # Create and return DataSplit object (will validate temporal ordering)
    return DataSplit(
        train=optimize_data,
        validation=validate_data,
        test=test_data,
        metadata=metadata
    )


def walk_forward_splitter(data: pd.DataFrame,
                         optimize_window_size: int,
                         validate_window_size: int,
                         test_window_size: int,
                         step_size: int,
                         gap_days: int = 1) -> Iterator[DataSplit]:
    """
    Generate walk-forward 3-way splits with temporal consistency.
    
    Creates an iterator of DataSplit objects where each split maintains
    temporal ordering with configurable gaps between optimize/validate/test periods.
    
    Args:
        data: DataFrame with datetime column and OHLCV data
        optimize_window_size: Number of bars for optimize period
        validate_window_size: Number of bars for validate period  
        test_window_size: Number of bars for test period
        step_size: Number of bars to step forward for next split
        gap_days: Number of days to skip between periods
        
    Yields:
        DataSplit objects with temporal ordering
        
    Raises:
        ValueError: If insufficient data or invalid parameters
    """
    if data.empty:
        raise ValueError("Cannot split empty DataFrame")
    
    if 'datetime' not in data.columns:
        raise ValueError("DataFrame must have 'datetime' column for walk-forward analysis")
    
    # Validate parameters
    if any(size <= 0 for size in [optimize_window_size, validate_window_size, test_window_size, step_size]):
        raise ValueError("All window sizes and step size must be positive")
    
    if gap_days < 0:
        raise ValueError("Gap days must be non-negative")
    
    # Sort data by datetime
    data = data.sort_values('datetime').reset_index(drop=True)
    
    total_window_size = optimize_window_size + validate_window_size + test_window_size
    
    # Check minimum data requirements
    if len(data) < total_window_size:
        raise ValueError(f"Insufficient data for walk-forward analysis: "
                        f"{len(data)} bars available, need at least {total_window_size}")
    
    split_count = 0
    current_start = 0
    
    while current_start + optimize_window_size <= len(data):
        try:
            # Extract optimize data
            optimize_end = current_start + optimize_window_size
            optimize_data = data.iloc[current_start:optimize_end].copy()
            
            if optimize_data.empty:
                break
            
            # Calculate validate start with gap
            optimize_end_time = optimize_data['datetime'].max()
            validate_gap_start_time = optimize_end_time + pd.DateOffset(days=gap_days)
            
            # Find validate start index after gap
            validate_mask = data['datetime'] >= validate_gap_start_time
            validate_candidates = data[validate_mask]
            
            if validate_candidates.empty:
                logger.debug(f"No data available for validate period after gap at split {split_count + 1}")
                break
            
            validate_start = validate_candidates.index[0]
            validate_end = validate_start + validate_window_size
            
            if validate_end > len(data):
                logger.debug(f"Insufficient data for validate period at split {split_count + 1}")
                break
            
            validate_data = data.iloc[validate_start:validate_end].copy()
            
            if validate_data.empty:
                break
            
            # Calculate test start with gap
            validate_end_time = validate_data['datetime'].max()
            test_gap_start_time = validate_end_time + pd.DateOffset(days=gap_days)
            
            # Find test start index after gap
            test_mask = data['datetime'] >= test_gap_start_time
            test_candidates = data[test_mask]
            
            if test_candidates.empty:
                logger.debug(f"No data available for test period after gap at split {split_count + 1}")
                break
            
            test_start = test_candidates.index[0]
            test_end = test_start + test_window_size
            
            if test_end > len(data):
                logger.debug(f"Insufficient data for test period at split {split_count + 1}")
                break
            
            test_data = data.iloc[test_start:test_end].copy()
            
            if test_data.empty:
                break
            
            # Create metadata for this split
            split_metadata = {
                'split_type': 'walk_forward',
                'split_number': split_count + 1,
                'gap_days': gap_days,
                'window_sizes': {
                    'optimize': optimize_window_size,
                    'validate': validate_window_size,
                    'test': test_window_size
                },
                'step_size': step_size,
                'optimize_period': {
                    'start': optimize_data['datetime'].min(),
                    'end': optimize_data['datetime'].max(),
                    'bars': len(optimize_data),
                    'start_idx': current_start,
                    'end_idx': optimize_end
                },
                'validate_period': {
                    'start': validate_data['datetime'].min(),
                    'end': validate_data['datetime'].max(),
                    'bars': len(validate_data),
                    'start_idx': validate_start,
                    'end_idx': validate_end
                },
                'test_period': {
                    'start': test_data['datetime'].min(),
                    'end': test_data['datetime'].max(),
                    'bars': len(test_data),
                    'start_idx': test_start,
                    'end_idx': test_end
                },
                'created_at': datetime.now().isoformat()
            }
            
            # Create DataSplit object (will validate temporal ordering)
            data_split = DataSplit(
                train=optimize_data,
                validation=validate_data,
                test=test_data,
                metadata=split_metadata
            )
            
            split_count += 1
            logger.debug(f"Walk-forward split {split_count} created: "
                        f"optimize={len(optimize_data)} bars, "
                        f"validate={len(validate_data)} bars, "
                        f"test={len(test_data)} bars")
            
            yield data_split
            
            # Move window forward by step size
            current_start += step_size
            
        except Exception as e:
            logger.warning(f"Failed to create walk-forward split {split_count + 1}: {e}")
            break
    
    if split_count == 0:
        raise ValueError("No valid walk-forward splits could be created with the given parameters")
    
    logger.info(f"Walk-forward analysis complete: {split_count} splits created")


def validate_split_ratios(ratios: Tuple[float, float, float]) -> bool:
    """
    Validate that split ratios are valid for 3-way splitting.
    
    Args:
        ratios: Tuple of (optimize_ratio, validate_ratio, test_ratio)
        
    Returns:
        True if ratios are valid
        
    Raises:
        ValueError: If ratios are invalid
    """
    if len(ratios) != 3:
        raise ValueError("Must provide exactly 3 ratios for optimize/validate/test")
    
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive")
    
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    # Check minimum reasonable ratios
    min_ratio = 0.05  # 5% minimum
    if any(r < min_ratio for r in ratios):
        raise ValueError(f"All ratios must be at least {min_ratio} ({min_ratio*100}%)")
    
    return True


def calculate_adaptive_walk_forward_parameters(data: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate optimal walk-forward parameters based on data characteristics.
    
    Analyzes data frequency and trading patterns to suggest optimal window sizes
    and step sizes for walk-forward analysis.
    
    Args:
        data: DataFrame with datetime column
        
    Returns:
        Dictionary with suggested parameters
    """
    if data.empty or 'datetime' not in data.columns:
        raise ValueError("Data must have datetime column for parameter calculation")
    
    total_bars = len(data)
    
    # Calculate data frequency
    time_deltas = data['datetime'].diff().dropna()
    if len(time_deltas) == 0:
        median_frequency_minutes = 15.0  # Default
    else:
        median_seconds = time_deltas.median().total_seconds()
        median_frequency_minutes = max(1.0, median_seconds / 60.0)
    
    # Calculate bars per day
    bars_per_day = min(1440.0, 1440.0 / median_frequency_minutes)
    
    # Calculate minimum training requirements (30 days or 5000 bars, whichever is larger)
    min_train_bars = max(5000, int(bars_per_day * 30))
    
    # Ensure we don't exceed available data
    if total_bars < min_train_bars * 3:
        # Reduce requirements for small datasets
        min_train_bars = max(1000, total_bars // 4)
    
    # Calculate optimal parameters
    optimize_window_size = max(min_train_bars, int(total_bars * 0.6))
    validate_window_size = max(int(bars_per_day * 10), int(total_bars * 0.2))  # 10 days minimum
    test_window_size = max(int(bars_per_day * 10), int(total_bars * 0.2))  # 10 days minimum
    step_size = max(int(bars_per_day * 5), int(optimize_window_size * 0.2))  # 5 days minimum
    
    # Final validation to ensure parameters fit within data (including gap overhead)
    # Account for gaps between optimize->validate and validate->test (2 gaps total)
    gap_overhead = int(2 * bars_per_day * 1)  # Assume 1 day gap by default
    total_required = optimize_window_size + validate_window_size + test_window_size + gap_overhead
    
    if total_required > total_bars:
        # Scale down proportionally, keeping some buffer
        scale_factor = total_bars * 0.8 / total_required  # More conservative scaling
        optimize_window_size = int(optimize_window_size * scale_factor)
        validate_window_size = int(validate_window_size * scale_factor)
        test_window_size = int(test_window_size * scale_factor)
        step_size = int(step_size * scale_factor)
        
        # Recalculate with smaller gap if still doesn't fit
        total_required_scaled = optimize_window_size + validate_window_size + test_window_size + gap_overhead
        if total_required_scaled > total_bars:
            gap_overhead = 0  # Remove gaps for very small datasets
    
    # Calculate adaptive gap_days based on dataset size
    adaptive_gap_days = 1 if gap_overhead > 0 else 0
    
    return {
        'optimize_window_size': optimize_window_size,
        'validate_window_size': validate_window_size,
        'test_window_size': test_window_size,
        'step_size': step_size,
        'gap_days': adaptive_gap_days,  # Adaptive gap based on data size
        'bars_per_day': bars_per_day,
        'frequency_minutes': median_frequency_minutes,
        'total_bars': total_bars,
        'estimated_splits': max(1, (total_bars - optimize_window_size) // step_size)
    }


# Legacy compatibility removed - use chronological_split() and walk_forward_splitter() for all splitting needs


if __name__ == "__main__":
    # Test the new functional data splitter
    import numpy as np
    
    print("Testing Functional Data Splitter...")
    
    # Create test data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='1h')
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    print(f"Test data: {len(test_data)} bars from {test_data['datetime'].min().date()} to {test_data['datetime'].max().date()}")
    
    # Test chronological split
    print("\n1. Testing chronological 3-way split...")
    try:
        chrono_split = chronological_split(test_data, ratios=(0.6, 0.2, 0.2), gap_days=1)
        print(f"   Success: {chrono_split}")
        print(f"   Ratios: {chrono_split.ratios}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test walk-forward split
    print("\n2. Testing walk-forward 3-way splits...")
    try:
        # Calculate adaptive parameters
        params = calculate_adaptive_walk_forward_parameters(test_data)
        print(f"   Adaptive parameters: {params}")
        
        # Create a few walk-forward splits (limit to 3 for testing)
        splits = list(walk_forward_splitter(
            test_data,
            optimize_window_size=params['optimize_window_size'],
            validate_window_size=params['validate_window_size'],
            test_window_size=params['test_window_size'],
            step_size=params['step_size'],
            gap_days=1
        ))
        
        print(f"   Created {len(splits)} walk-forward splits")
        for i, split in enumerate(splits[:3]):  # Show first 3
            print(f"   Split {i+1}: {split}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nFunctional Data Splitter test complete!")