"""
Data Structures for 3-Way Split System
=====================================

Immutable data structures for the new 3-way split system that eliminates data leakage
by providing separate train/validation/test datasets with temporal ordering validation.

This module defines the core DataSplit dataclass that serves as the foundation for all
data splitting operations in the trading optimization system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """
    Immutable dataclass representing a 3-way data split for trading strategy optimization.
    
    This structure eliminates data leakage by enforcing temporal ordering and providing
    separate datasets for training, validation, and testing.
    
    Fields:
        train: DataFrame used for parameter tuning and model fitting (institutional terminology)
        validation: DataFrame used for performance evaluation during optimization
        test: DataFrame reserved for final validation (unused during optimization)
        metadata: Dictionary containing split information and timestamps
    
    Temporal Ordering:
        train.index.max() < validation.index.min() < test.index.min()
        
    This ensures no data leakage between phases and maintains realistic trading conditions.
    """
    
    # ARCHITECTURAL FIX: Use institutional terminology (train/validation/test)
    train: pd.DataFrame         # Was 'optimize' - training data for model fitting
    validation: pd.DataFrame    # Was 'validate' - validation data for hyperparameter selection
    test: pd.DataFrame          # Held-out test data for final evaluation
    metadata: Dict[str, Any]
    
    
    def __post_init__(self):
        """
        Validate the data split for temporal ordering and completeness.
        
        Raises:
            ValueError: If temporal ordering is violated or data is missing
            AssertionError: If any of the DataFrames are empty
        """
        # Check that none of the splits are empty
        if self.train.empty:
            raise AssertionError("Train data cannot be empty")
        if self.validation.empty:
            raise AssertionError("Validation data cannot be empty")
        if self.test.empty:
            raise AssertionError("Test data cannot be empty")
        
        # Validate temporal ordering if datetime column exists
        if 'datetime' in self.train.columns and 'datetime' in self.validation.columns and 'datetime' in self.test.columns:
            train_end = self.train['datetime'].max()
            validation_start = self.validation['datetime'].min()
            validation_end = self.validation['datetime'].max()
            test_start = self.test['datetime'].min()
            
            # Check gap_days configuration for temporal ordering rules
            gap_days = self.metadata.get('gap_days', 1) if hasattr(self, 'metadata') and self.metadata else 1
            
            # Determine gap description based on gap_days setting
            gap_desc = "after or at" if gap_days == 0 else "before"
            
            # Validate train->validation ordering
            if (gap_days == 0 and train_end > validation_start) or (gap_days > 0 and train_end >= validation_start):
                raise ValueError(
                    f"Temporal ordering violation: train data ends at {train_end} "
                    f"but validation data starts at {validation_start}. "
                    f"With gap_days={gap_days}, validation must start {gap_desc} train end."
                )
            
            # Validate validation->test ordering  
            if (gap_days == 0 and validation_end > test_start) or (gap_days > 0 and validation_end >= test_start):
                raise ValueError(
                    f"Temporal ordering violation: validation data ends at {validation_end} "
                    f"but test data starts at {test_start}. "
                    f"With gap_days={gap_days}, test must start {gap_desc} validation end."
                )
            
            logger.debug(f"Temporal ordering validated: "
                        f"train ({train_end}) < validation ({validation_start} to {validation_end}) < test ({test_start})")
        else:
            logger.warning("No datetime column found - temporal ordering validation skipped")
    
    @property
    def train_bars(self) -> int:
        """Number of bars in train dataset"""
        return len(self.train)
    
    @property
    def validation_bars(self) -> int:
        """Number of bars in validation dataset"""
        return len(self.validation)
    
    @property
    def test_bars(self) -> int:
        """Number of bars in test dataset"""
        return len(self.test)
    
    @property
    def total_bars(self) -> int:
        """Total number of bars across all datasets"""
        return self.train_bars + self.validation_bars + self.test_bars
    
    @property
    def ratios(self) -> Dict[str, float]:
        """Calculate actual ratios of each split"""
        total = self.total_bars
        return {
            'train': self.train_bars / total if total > 0 else 0.0,
            'validation': self.validation_bars / total if total > 0 else 0.0,
            'test': self.test_bars / total if total > 0 else 0.0
        }
    
    def get_date_ranges(self) -> Dict[str, Dict[str, Optional[pd.Timestamp]]]:
        """
        Get date ranges for each split if datetime column exists.
        
        Returns:
            Dictionary with start/end dates for each split
        """
        ranges = {}
        
        for split_name, data in [('train', self.train), ('validation', self.validation), ('test', self.test)]:
            if 'datetime' in data.columns and not data.empty:
                ranges[split_name] = {
                    'start': data['datetime'].min(),
                    'end': data['datetime'].max()
                }
            else:
                ranges[split_name] = {
                    'start': None,
                    'end': None
                }
        
        return ranges
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the data split.
        
        Returns:
            Dictionary with split statistics and metadata
        """
        date_ranges = self.get_date_ranges()
        ratios = self.ratios
        
        summary = {
            'total_bars': self.total_bars,
            'splits': {
                'train': {
                    'bars': self.train_bars,
                    'ratio': ratios['train'],
                    'date_range': date_ranges['train']
                },
                'validation': {
                    'bars': self.validation_bars,
                    'ratio': ratios['validation'],
                    'date_range': date_ranges['validation']
                },
                'test': {
                    'bars': self.test_bars,
                    'ratio': ratios['test'],
                    'date_range': date_ranges['test']
                }
            },
            'metadata': self.metadata
        }
        
        return summary
    
    def __str__(self) -> str:
        """String representation of the DataSplit"""
        ratios = self.ratios
        return (f"DataSplit(train={self.train_bars} bars ({ratios['train']:.1%}), "
                f"validation={self.validation_bars} bars ({ratios['validation']:.1%}), "
                f"test={self.test_bars} bars ({ratios['test']:.1%}))")
    
    def __repr__(self) -> str:
        """Detailed representation of the DataSplit"""
        return (f"DataSplit(train={self.train.shape}, validation={self.validation.shape}, "
                f"test={self.test.shape}, metadata_keys={list(self.metadata.keys())})")


def validate_data_split(data_split: DataSplit) -> bool:
    """
    Validate a DataSplit object for completeness and correctness.
    
    Args:
        data_split: DataSplit object to validate
        
    Returns:
        True if split is valid, False otherwise
    """
    try:
        # Check basic structure
        if not isinstance(data_split, DataSplit):
            logger.error("Object is not a DataSplit instance")
            return False
        
        # Check for required columns in all splits
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        
        for split_name, data in [('train', data_split.train), 
                                ('validation', data_split.validation), 
                                ('test', data_split.test)]:
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column '{col}' in {split_name} data")
                    return False
        
        # Check minimum data requirements
        if data_split.train_bars < 100:
            logger.warning(f"Very small train dataset: {data_split.train_bars} bars")
        
        if data_split.validation_bars < 50:
            logger.warning(f"Very small validation dataset: {data_split.validation_bars} bars")
        
        if data_split.test_bars < 50:
            logger.warning(f"Very small test dataset: {data_split.test_bars} bars")
        
        # Post-init validation will check temporal ordering
        return True
        
    except Exception as e:
        logger.error(f"DataSplit validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the DataSplit structure
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Testing DataSplit structure...")
    
    # Create test data with temporal ordering
    base_date = datetime(2020, 1, 1)
    
    # Optimize data: Jan-Jun 2020
    optimize_dates = pd.date_range(base_date, base_date + timedelta(days=180), freq='1h')
    optimize_data = pd.DataFrame({
        'datetime': optimize_dates,
        'open': np.random.randn(len(optimize_dates)).cumsum() + 100,
        'high': np.random.randn(len(optimize_dates)).cumsum() + 102,
        'low': np.random.randn(len(optimize_dates)).cumsum() + 98,
        'close': np.random.randn(len(optimize_dates)).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, len(optimize_dates))
    })
    
    # Validate data: Jul-Sep 2020 (with 1-day gap)
    validate_start = base_date + timedelta(days=181)
    validate_dates = pd.date_range(validate_start, validate_start + timedelta(days=90), freq='1h')
    validate_data = pd.DataFrame({
        'datetime': validate_dates,
        'open': np.random.randn(len(validate_dates)).cumsum() + 100,
        'high': np.random.randn(len(validate_dates)).cumsum() + 102,
        'low': np.random.randn(len(validate_dates)).cumsum() + 98,
        'close': np.random.randn(len(validate_dates)).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, len(validate_dates))
    })
    
    # Test data: Oct-Dec 2020 (with 1-day gap)
    test_start = validate_start + timedelta(days=91)
    test_dates = pd.date_range(test_start, test_start + timedelta(days=90), freq='1h')
    test_data = pd.DataFrame({
        'datetime': test_dates,
        'open': np.random.randn(len(test_dates)).cumsum() + 100,
        'high': np.random.randn(len(test_dates)).cumsum() + 102,
        'low': np.random.randn(len(test_dates)).cumsum() + 98,
        'close': np.random.randn(len(test_dates)).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, len(test_dates))
    })
    
    # Create metadata
    metadata = {
        'split_type': 'chronological',
        'created_at': datetime.now().isoformat(),
        'gap_days': 1,
        'symbol': 'TEST'
    }
    
    # Test DataSplit creation
    try:
        data_split = DataSplit(
            train=optimize_data,
            validation=validate_data,
            test=test_data,
            metadata=metadata
        )
        
        print(f"[SUCCESS] DataSplit created successfully: {data_split}")
        print(f"[SUCCESS] Summary: {data_split.summary()}")
        print(f"[SUCCESS] Validation: {validate_data_split(data_split)}")
        
    except Exception as e:
        print(f"[ERROR] DataSplit creation failed: {e}")
    
    print("DataSplit test complete!")