"""
Simplified Data Validator
=========================

Basic data validation for trading data quality assurance.
Focuses on essential checks without the complexity of market schedule analysis.

Data validation is crucial - bad data leads to bad trading decisions!
This validator checks the essentials: OHLC relationships, data completeness, and basic integrity.
"""

import logging
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Simple validation result container"""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    # Basic metrics
    total_rows: int = 0
    valid_rows: int = 0
    quality_score: float = 1.0
    
    def add_error(self, message: str):
        """Add error and mark validation as failed"""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add info message"""
        self.info.append(message)
    
    def get_summary(self) -> str:
        """Get readable summary"""
        status = "PASSED" if self.passed else "FAILED"
        summary = f"""
DATA VALIDATION: {status}
Quality Score: {self.quality_score:.1%}
Total Rows: {self.total_rows:,}
Valid Rows: {self.valid_rows:,}

Issues:
- Errors: {len(self.errors)}
- Warnings: {len(self.warnings)}
"""
        
        if self.errors:
            summary += "\nCritical Errors:\n"
            for error in self.errors[:3]:
                summary += f"  • {error}\n"
                
        if self.warnings:
            summary += "\nWarnings:\n"
            for warning in self.warnings[:3]:
                summary += f"  • {warning}\n"
        
        return summary


class BasicDataValidator:
    """
    Simple, focused data validator for trading data
    
    Focuses on the most important checks:
    - OHLC price relationships (High >= Open/Close, etc.)
    - Data completeness (no missing values in critical columns)
    - Basic data types and ranges
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, apply stricter validation thresholds
        """
        self.strict_mode = strict_mode
        self.required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        # Validation thresholds
        self.min_rows = 100 if not strict_mode else 500
        self.max_price_change = 0.50 if not strict_mode else 0.20  # 50% or 20% max change
        self.min_quality_score = 0.80 if not strict_mode else 0.95
    
    def validate(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationResult:
        """
        Validate trading data
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for context
            
        Returns:
            ValidationResult with findings
        """
        result = ValidationResult()
        result.total_rows = len(data)
        
        logger.info(f"Validating {symbol}: {len(data)} rows")
        
        try:
            # Basic structure checks
            self._check_basic_structure(data, result)
            if not result.passed:
                return result
            
            # OHLC relationship checks
            self._check_ohlc_relationships(data, result)
            
            # Data completeness checks
            self._check_data_completeness(data, result)
            
            # Price continuity checks
            self._check_price_continuity(data, result)
            
            # Volume data checks
            self._check_volume_data(data, result)
            
            # DateTime checks
            self._check_datetime_data(data, result)
            
            # Calculate quality score
            self._calculate_quality_score(data, result)
            
            logger.info(f"Validation complete: {'PASSED' if result.passed else 'FAILED'} (Score: {result.quality_score:.1%})")
            
        except Exception as e:
            result.add_error(f"Validation failed: {str(e)}")
            logger.error(f"Validation error: {e}")
        
        return result
    
    def _check_basic_structure(self, data: pd.DataFrame, result: ValidationResult):
        """Check basic DataFrame structure"""
        # Check if empty
        if data.empty:
            result.add_error("DataFrame is empty")
            return
        
        # Check minimum rows
        if len(data) < self.min_rows:
            result.add_error(f"Insufficient data: {len(data)} rows < {self.min_rows} required")
            return
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
            return
        
        result.add_info("Basic structure validation passed")
    
    def _check_ohlc_relationships(self, data: pd.DataFrame, result: ValidationResult):
        """
        Check OHLC price relationships
        
        In valid OHLC data:
        - High must be >= Open, Close, and Low
        - Low must be <= Open, Close, and High
        If these relationships are violated, the data is corrupted.
        """
        ohlc_cols = ['open', 'high', 'low', 'close']
        
        # Check for missing OHLC columns
        missing_ohlc = [col for col in ohlc_cols if col not in data.columns]
        if missing_ohlc:
            result.add_error(f"Missing OHLC columns: {missing_ohlc}")
            return
        
        # Check High >= Open, Close, Low
        high_violations = (
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['high'] < data['low'])
        ).sum()
        
        if high_violations > 0:
            result.add_error(f"High price violations: {high_violations} bars where High < Open/Close/Low")
        
        # Check Low <= Open, Close, High
        low_violations = (
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['low'] > data['high'])
        ).sum()
        
        if low_violations > 0:
            result.add_error(f"Low price violations: {low_violations} bars where Low > Open/Close/High")
        
        # Check for non-positive prices
        for col in ohlc_cols:
            negative_prices = (data[col] <= 0).sum()
            if negative_prices > 0:
                result.add_error(f"Non-positive prices in {col}: {negative_prices} values")
        
        if high_violations == 0 and low_violations == 0:
            result.add_info("OHLC relationship validation passed")
    
    def _check_data_completeness(self, data: pd.DataFrame, result: ValidationResult):
        """Check for missing data"""
        for col in self.required_columns:
            if col not in data.columns:
                continue
            
            null_count = data[col].isnull().sum()
            if null_count > 0:
                null_percent = (null_count / len(data)) * 100
                if null_percent > 5:  # More than 5% missing
                    result.add_error(f"Too much missing data in {col}: {null_percent:.1f}%")
                else:
                    result.add_warning(f"Missing data in {col}: {null_count} values ({null_percent:.1f}%)")
        
        # Calculate valid rows
        required_data = data[self.required_columns]
        result.valid_rows = len(required_data.dropna())
        
        completeness = result.valid_rows / len(data) if len(data) > 0 else 0
        if completeness < 0.95:  # Less than 95% complete
            result.add_warning(f"Data completeness: {completeness:.1%}")
        else:
            result.add_info(f"Data completeness: {completeness:.1%}")
    
    def _check_price_continuity(self, data: pd.DataFrame, result: ValidationResult):
        """Check for unrealistic price movements"""
        if 'close' not in data.columns or len(data) < 2:
            return
        
        # Calculate price changes
        price_changes = data['close'].pct_change().dropna()
        
        # Check for extreme movements
        extreme_moves = (abs(price_changes) > self.max_price_change).sum()
        if extreme_moves > 0:
            max_change = abs(price_changes).max() * 100
            if self.strict_mode:
                result.add_error(f"Extreme price movements: {extreme_moves} bars with >{self.max_price_change*100}% change (max: {max_change:.1f}%)")
            else:
                result.add_warning(f"Large price movements: {extreme_moves} bars with >{self.max_price_change*100}% change (max: {max_change:.1f}%)")
        
        result.add_info("Price continuity check completed")
    
    def _check_volume_data(self, data: pd.DataFrame, result: ValidationResult):
        """Check volume data quality"""
        if 'volume' not in data.columns:
            result.add_warning("No volume column found")
            return
        
        # Check for negative volumes
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            result.add_error(f"Negative volume values: {negative_volume}")
        
        # Check for zero volumes
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume > 0:
            zero_percent = (zero_volume / len(data)) * 100
            if zero_percent > 10:  # More than 10% zero volume
                result.add_warning(f"High zero volume: {zero_percent:.1f}%")
            elif zero_volume < 10:
                result.add_info(f"Minimal zero volume: {zero_volume} bars")
        
        result.add_info("Volume validation completed")
    
    def _check_datetime_data(self, data: pd.DataFrame, result: ValidationResult):
        """Check datetime column integrity"""
        if 'datetime' not in data.columns:
            result.add_error("No datetime column found")
            return
        
        # Check for null datetimes
        null_datetime = data['datetime'].isnull().sum()
        if null_datetime > 0:
            result.add_error(f"Null datetime values: {null_datetime}")
        
        # Check if datetime is properly typed
        if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
            result.add_warning("Datetime column is not datetime type")
        
        # Check chronological order
        if not data['datetime'].is_monotonic_increasing:
            result.add_warning("Data is not in chronological order")
        
        # Check for duplicates
        duplicate_times = data['datetime'].duplicated().sum()
        if duplicate_times > 0:
            result.add_warning(f"Duplicate timestamps: {duplicate_times}")
        
        result.add_info("DateTime validation completed")
    
    def _calculate_quality_score(self, data: pd.DataFrame, result: ValidationResult):
        """Calculate overall quality score"""
        score = 1.0
        
        # Penalize errors heavily
        score -= len(result.errors) * 0.2
        
        # Penalize warnings moderately
        score -= len(result.warnings) * 0.05
        
        # Data completeness factor
        completeness = result.valid_rows / len(data) if len(data) > 0 else 0
        score *= completeness
        
        # Ensure score is between 0 and 1
        result.quality_score = max(0.0, min(1.0, score))
        
        # Final pass/fail decision
        if result.quality_score < self.min_quality_score:
            result.add_error(f"Quality score {result.quality_score:.1%} below minimum {self.min_quality_score:.1%}")


def quick_data_check(data: pd.DataFrame) -> bool:
    """
    Quick data integrity check
    
    Fast check to determine if data is basically valid.
    Good for quick validation before more expensive operations.
    
    Args:
        data: DataFrame to check
        
    Returns:
        True if basic checks pass
    """
    try:
        # Must have data
        if data.empty:
            return False
        
        # Must have basic columns
        required = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required):
            return False
        
        # Must have some valid data
        if data[required].isnull().all().any():
            return False
        
        # Basic OHLC check
        sample_size = min(1000, len(data))  # Check first 1000 rows for speed
        sample = data.head(sample_size)
        
        if (sample['high'] < sample['low']).any():
            return False
        
        if (sample[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False
        
        return True
        
    except Exception:
        return False


def fix_basic_issues(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix basic data issues automatically
    
    Attempts to fix common data problems automatically. 
    Conservative approach - only fixes obvious issues.
    
    Args:
        data: DataFrame to fix
        
    Returns:
        DataFrame with basic issues fixed
    """
    if data.empty:
        return data
    
    fixed_data = data.copy()
    fixes_applied = []
    
    try:
        # Fix 1: Ensure datetime column exists and is proper type
        if 'datetime' not in fixed_data.columns:
            # Try to create from index
            if isinstance(fixed_data.index, pd.DatetimeIndex):
                fixed_data['datetime'] = fixed_data.index
                fixed_data = fixed_data.reset_index(drop=True)
                fixes_applied.append("Created datetime column from index")
        
        # Fix 2: Convert datetime to proper type
        if 'datetime' in fixed_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(fixed_data['datetime']):
                fixed_data['datetime'] = pd.to_datetime(fixed_data['datetime'])
                fixes_applied.append("Converted datetime to proper type")
        
        # Fix 3: Sort by datetime
        if 'datetime' in fixed_data.columns:
            if not fixed_data['datetime'].is_monotonic_increasing:
                fixed_data = fixed_data.sort_values('datetime').reset_index(drop=True)
                fixes_applied.append("Sorted data chronologically")
        
        # Fix 4: Remove duplicate rows
        initial_rows = len(fixed_data)
        fixed_data = fixed_data.drop_duplicates().reset_index(drop=True)
        if len(fixed_data) < initial_rows:
            fixes_applied.append(f"Removed {initial_rows - len(fixed_data)} duplicate rows")
        
        # Fix 5: Fix obvious OHLC violations (swap high/low if inverted)
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in fixed_data.columns for col in ohlc_cols):
            # Fix high < low inversions
            invalid_mask = fixed_data['high'] < fixed_data['low']
            if invalid_mask.any():
                # Swap high and low
                temp = fixed_data.loc[invalid_mask, 'high'].copy()
                fixed_data.loc[invalid_mask, 'high'] = fixed_data.loc[invalid_mask, 'low']
                fixed_data.loc[invalid_mask, 'low'] = temp
                fixes_applied.append(f"Fixed {invalid_mask.sum()} high/low inversions")
        
        # Fix 6: Replace negative volumes with 0
        if 'volume' in fixed_data.columns:
            negative_volume = (fixed_data['volume'] < 0).sum()
            if negative_volume > 0:
                fixed_data.loc[fixed_data['volume'] < 0, 'volume'] = 0
                fixes_applied.append(f"Fixed {negative_volume} negative volumes")
        
        if fixes_applied:
            logger.info(f"Applied fixes: {'; '.join(fixes_applied)}")
        
        return fixed_data
        
    except Exception as e:
        logger.error(f"Error fixing data: {e}")
        return data  # Return original if fixes fail


if __name__ == "__main__":
    # Test the validator
    print("Testing Data Validator...")
    
    # Create test data with some issues
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 101,
        'volume': np.random.randint(100, 1000, len(dates))
    })
    
    # Introduce some issues
    test_data.iloc[100, test_data.columns.get_loc('high')] = test_data.iloc[100]['low'] - 1  # High < Low
    test_data.iloc[200:205, test_data.columns.get_loc('volume')] = -100  # Negative volume
    test_data.iloc[300] = test_data.iloc[299]  # Duplicate row
    
    print(f"Test data created: {len(test_data)} rows with intentional issues")
    
    # Test validation
    validator = BasicDataValidator()
    result = validator.validate(test_data, "TEST")
    
    print("\nValidation Result:")
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Quality Score: {result.quality_score:.1%}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\nErrors found:")
        for error in result.errors:
            print(f"  - {error}")
    
    # Test quick check
    print(f"\nQuick check: {'PASSED' if quick_data_check(test_data) else 'FAILED'}")
    
    # Test auto-fix
    print("\nTesting auto-fix...")
    fixed_data = fix_basic_issues(test_data)
    
    # Re-validate fixed data
    fixed_result = validator.validate(fixed_data, "TEST_FIXED")
    print(f"Fixed data quality: {fixed_result.quality_score:.1%}")
    print(f"Fixed data errors: {len(fixed_result.errors)}")
    
    print("\nValidator test complete!")