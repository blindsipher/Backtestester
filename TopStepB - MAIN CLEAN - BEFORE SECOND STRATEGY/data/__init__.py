"""
3-Way Split Data Module for Trading System
==========================================

Clean, focused data infrastructure with 3-way split functionality that eliminates data leakage:
- File loading (CSV, Parquet) with GUI file picker
- Synthetic data generation
- 3-way splits (optimize/validate/test) with temporal ordering
- Basic data validation

This module has been completely rewritten to support the new DataSplit architecture
that provides proper data separation for trading strategy optimization.
"""

# Core functionality imports
from .data_loader import (
    # Main data loading classes
    DataLoader,
    InteractiveDataLoader,

    # Utility functions
    load_data_from_file,
    create_synthetic_data
)

# Import new 3-way split functionality
from .data_structures import DataSplit, validate_data_split
from .data_splitter import (
    # New 3-way split functions
    chronological_split,
    walk_forward_splitter,
    validate_split_ratios,
    calculate_adaptive_walk_forward_parameters,
    
)

from .data_validator import (
    # Simple validation
    BasicDataValidator,
    quick_data_check,
    fix_basic_issues
)

# Standard library imports
import logging
import pandas as pd
from typing import Dict, List, Any, Union, Tuple

# Set up module logger
logger = logging.getLogger(__name__)

# Import constants for consistency  
# ARCHITECTURAL FIX: Remove constants.py dependency - define locally
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train, validation, test ratios
DEFAULT_GAP_DAYS = 1  # Gap days between splits

# Module metadata
__version__ = "3.0.0"
__author__ = "Trading System - 3-Way Split Data Module"
__description__ = "Clean data infrastructure with 3-way splits for trading systems"

# Main convenience functions
def load_data_interactive() -> pd.DataFrame:
    """
    Interactive data loading with GUI file picker
    
    Returns:
        DataFrame with loaded data
    """
    loader = InteractiveDataLoader()
    return loader.load_data_interactive()

def create_test_data(bars: int = 5000, symbol: str = "TEST") -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing
    
    Args:
        bars: Number of bars to generate
        symbol: Symbol name for the data
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    return create_synthetic_data(bars=bars, symbol=symbol)

def create_data_splits(data: pd.DataFrame,
                      split_method: str = "chronological", 
                      ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
                      gap_days: int = DEFAULT_GAP_DAYS,
                      **kwargs) -> Union[DataSplit, List[DataSplit]]:
    """
    PUBLIC API: Create secure data splits for the pipeline.
    
    This is the main entry point for all data splitting operations.
    Handles both chronological and walk-forward splitting with proper validation.
    
    Args:
        data: Full dataset to split
        split_method: "chronological" or "walk_forward"
        ratios: (train, validation, test) ratios for chronological splits
        gap_days: Gap days between splits to prevent data leakage
        **kwargs: Additional parameters for walk-forward splitting
        
    Returns:
        DataSplit (chronological) or List[DataSplit] (walk-forward)
        
    Raises:
        ValueError: If splits cannot be created with given parameters
    """
    logger.info(f"Data module creating {split_method} splits with ratios {ratios}")
    
    if split_method == "chronological":
        return split_for_backtest(
            data=data,
            method="chronological", 
            ratios=ratios,
            gap_days=gap_days
        )
    elif split_method == "walk_forward":
        return split_for_backtest(
            data=data,
            method="walk_forward",
            gap_days=gap_days,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def split_for_backtest(data: pd.DataFrame, 
                      method: str = "chronological",
                      ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
                      gap_days: int = DEFAULT_GAP_DAYS,
                      **kwargs) -> Union[DataSplit, List[DataSplit]]:
    """
    Split data for backtesting with 3-way splits (optimize/validate/test).
    
    This function replaces the old 2-way split system with proper 3-way splits
    that eliminate data leakage by providing separate optimize, validate, and test datasets.
    
    Args:
        data: DataFrame to split
        method: "chronological" or "walk_forward"
        ratios: Tuple of (optimize_ratio, validate_ratio, test_ratio) for chronological splits
        gap_days: Number of days to skip between splits
        **kwargs: Additional parameters for walk-forward (window sizes, step_size)
        
    Returns:
        DataSplit object for chronological method
        List[DataSplit] for walk_forward method
        
    Raises:
        ValueError: If method is unknown or parameters are invalid
    """
    if method == "chronological":
        validate_split_ratios(ratios)
        return chronological_split(data, ratios=ratios, gap_days=gap_days)
    
    elif method == "walk_forward":
        # Extract walk-forward parameters
        optimize_window_size = kwargs.get('optimize_window_size')
        validate_window_size = kwargs.get('validate_window_size')
        test_window_size = kwargs.get('test_window_size')
        step_size = kwargs.get('step_size')
        
        # If parameters not provided, use adaptive calculation
        if any(param is None for param in [optimize_window_size, validate_window_size, test_window_size, step_size]):
            logger.info("Walk-forward parameters not fully specified, using adaptive calculation")
            adaptive_params = calculate_adaptive_walk_forward_parameters(data)
            
            optimize_window_size = optimize_window_size or adaptive_params['optimize_window_size']
            validate_window_size = validate_window_size or adaptive_params['validate_window_size']
            test_window_size = test_window_size or adaptive_params['test_window_size']
            step_size = step_size or adaptive_params['step_size']
            
            logger.info(f"Using adaptive parameters: optimize={optimize_window_size}, "
                       f"validate={validate_window_size}, test={test_window_size}, step={step_size}")
        
        # Convert iterator to list for compatibility with existing code
        splits = list(walk_forward_splitter(
            data,
            optimize_window_size=optimize_window_size,
            validate_window_size=validate_window_size,
            test_window_size=test_window_size,
            step_size=step_size,
            gap_days=gap_days
        ))
        
        return splits
    
    else:
        raise ValueError(f"Unknown split method: {method}")

def prepare_data_pipeline(file_path: str = None, 
                         interactive: bool = True,
                         split_method: str = "chronological",
                         split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
                         gap_days: int = 1) -> Dict[str, Any]:
    """
    Complete data pipeline: load, validate, and split with 3-way splits.
    
    This function has been updated to use the new DataSplit architecture
    that provides proper data separation for optimization.
    
    Args:
        file_path: Optional file path (if None, will prompt for file)
        interactive: Use interactive file picker
        split_method: How to split the data ("chronological" or "walk_forward")
        split_ratios: Ratios for chronological split (optimize, validate, test)
        gap_days: Days to skip between splits
        
    Returns:
        Dictionary with DataSplit objects and metadata
    """
    try:
        # Step 1: Load data
        if interactive and file_path is None:
            data = load_data_interactive()
        elif file_path:
            data = load_data_from_file(file_path)
        else:
            raise ValueError("Must provide file_path or set interactive=True")
        
        if data is None or data.empty:
            raise ValueError("No data loaded")
        
        # Step 2: Basic validation and fixes
        if not quick_data_check(data):
            logger.warning("Data failed basic validation, attempting fixes...")
            data = fix_basic_issues(data)
        
        # Step 3: Split data using new 3-way system
        split_result = split_for_backtest(
            data, 
            method=split_method, 
            ratios=split_ratios,
            gap_days=gap_days
        )
        
        # Step 4: Package results
        result = {
            'success': True,
            'full_data': data,
            'split_method': split_method,
            'total_bars': len(data),
            'date_range': {
                'start': data['datetime'].min(),
                'end': data['datetime'].max()
            }
        }
        
        # Add split-specific results
        if split_method == "chronological":
            # Single DataSplit object
            data_split = split_result
            result.update({
                'data_split': data_split,
                'train_data': data_split.train,
                'validation_data': data_split.validation,
                'test_data': data_split.test,
                'split_metadata': data_split.metadata,
                'split_summary': data_split.summary()
            })
        elif split_method == "walk_forward":
            # List of DataSplit objects
            splits_list = split_result
            result.update({
                'data_splits': splits_list,
                'total_splits': len(splits_list),
                'walk_forward_metadata': {
                    'splits_count': len(splits_list),
                    'first_split': splits_list[0].summary() if splits_list else None,
                    'last_split': splits_list[-1].summary() if splits_list else None
                }
            })
        
        logger.info(f"Data pipeline complete: {len(data)} bars, {split_method} split")
        return result
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'full_data': None
        }

# Legacy compatibility removed - use split_for_backtest() for all splitting needs

# Export all public functions
__all__ = [
    # Main classes
    'DataLoader',
    'InteractiveDataLoader', 
    'BasicDataValidator',
    
    # New 3-way split functionality
    'DataSplit',
    'chronological_split',
    'walk_forward_splitter',
    'validate_data_split',
    'validate_split_ratios',
    'calculate_adaptive_walk_forward_parameters',
    
    # Convenience functions
    'load_data_interactive',
    'load_data_from_file',
    'create_test_data',
    'create_synthetic_data',
    'split_for_backtest',
    'prepare_data_pipeline',
    
    # Utility functions
    'quick_data_check',
    'fix_basic_issues',
    
]

# Module initialization
logger.info(f"3-Way Split Data Module loaded (v{__version__})")

def show_examples():
    """Show usage examples for the new 3-way split system"""
    examples = """
3-WAY SPLIT DATA MODULE EXAMPLES
================================

1. INTERACTIVE DATA LOADING:
from data import load_data_interactive
df = load_data_interactive()

2. COMPLETE PIPELINE WITH 3-WAY SPLITS:
from data import prepare_data_pipeline
result = prepare_data_pipeline(
    interactive=True, 
    split_method="chronological",
    split_ratios=DEFAULT_SPLIT_RATIOS,  # train, validation, test
    gap_days=1
)
data_split = result['data_split']
train_data = data_split.train
validation_data = data_split.validation
test_data = data_split.test

3. SYNTHETIC DATA FOR TESTING:
from data import create_test_data
test_data = create_test_data(bars=5000, symbol="NQ")

4. MANUAL 3-WAY SPLITTING:
from data import split_for_backtest
data_split = split_for_backtest(
    df, 
    method="chronological", 
    ratios=DEFAULT_SPLIT_RATIOS,
    gap_days=1
)

5. WALK-FORWARD 3-WAY SPLITS:
from data import split_for_backtest
walk_forward_splits = split_for_backtest(
    df, 
    method="walk_forward",
    gap_days=1
    # Parameters will be calculated adaptively if not provided
)

6. QUICK VALIDATION:
from data import quick_data_check, fix_basic_issues
is_valid = quick_data_check(df)
if not is_valid:
    df = fix_basic_issues(df)

7. DATASPLIT OBJECT USAGE:
# DataSplit objects provide clean interfaces
print(f"Split summary: {data_split}")
print(f"Ratios: {data_split.ratios}")
print(f"Date ranges: {data_split.get_date_ranges()}")
print(f"Full summary: {data_split.summary()}")

3-WAY SPLIT BENEFITS:
====================
NEW (3-way): optimize_data, validate_data, test_data

- optimize_data: Used for parameter tuning during optimization
- validate_data: Used for evaluation during optimization (out-of-sample)
- test_data: Reserved for final validation (never seen during optimization)

This eliminates data leakage by ensuring optimization never sees test data.
"""
    print(examples)