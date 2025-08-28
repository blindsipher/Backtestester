"""
Simplified Data Loader
======================

Clean file-based data loading with GUI file picker support.
Supports CSV and Parquet files with automatic format detection.
"""

import logging
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional

from utils.timezone_utils import ensure_datetime_column

# ARCHITECTURAL FIX: Remove constants.py dependency - define constants locally
DATA_EXTENSIONS = {'.csv', '.parquet'}  # Supported data file extensions
SEARCH_PATTERNS = ['*.csv', '*.parquet']  # File search patterns
MAX_DISPLAYED_FILES = 10  # Maximum files to display in selection
FILE_SIZE_MB_DIVISOR = 1024 * 1024  # Bytes to MB conversion
SYNTHETIC_START_DATE = '2023-01-01'  # Default start date for synthetic data

# Try to import tkinter for file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataLoader:
    """Simple data loader for CSV and Parquet files"""
    
    def __init__(self):
        self.supported_formats = list(DATA_EXTENSIONS)
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file with automatic format detection
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.parquet':
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df)} rows from Parquet: {file_path}")
                
            elif suffix == '.csv':
                # Try to detect datetime column and parse it
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows from CSV: {file_path}")
                
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            # Ensure datetime column exists and is properly formatted
            df = ensure_datetime_column(df, 'datetime')
            
            # Normalize column names to lowercase for strategy compatibility
            df.columns = df.columns.str.lower()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise


def load_data_file(file_path: str) -> pd.DataFrame:
    """Load data from a file path without any interactive prompts."""
    loader = DataLoader()
    return loader.load_file(file_path)


# Backwards compatibility alias
def load_data_from_file(file_path: str) -> pd.DataFrame:
    """Compatibility wrapper for load_data_file."""
    return load_data_file(file_path)


class InteractiveDataLoader:
    """Interactive data loader with GUI file picker"""

    def load_data_interactive(self) -> Optional[pd.DataFrame]:
        """
        Interactive data loading with GUI file picker
        
        Returns:
            DataFrame with loaded data or None if cancelled
        """
        print("\nINTERACTIVE DATA LOADING")
        print("=" * 40)
        
        # Show options
        print("Data source options:")
        if TKINTER_AVAILABLE:
            print("1. Browse for file (GUI)")
            print("2. Show nearby files")
            print("3. Enter file path manually")
            print("4. Create synthetic data")
            
            choice = input("Choose option (1-4) [1]: ").strip() or "1"
        else:
            print("1. Show nearby files") 
            print("2. Enter file path manually")
            print("3. Create synthetic data")
            
            choice = input("Choose option (1-3) [1]: ").strip() or "1"
            # Adjust choice for no-GUI environment
            if choice == "1":
                choice = "2"  # Show files
            elif choice == "2": 
                choice = "3"  # Manual entry
            elif choice == "3":
                choice = "4"  # Synthetic
        
        try:
            if choice == "1" and TKINTER_AVAILABLE:
                return self._browse_for_file()
            elif choice == "2":
                return self._show_and_select_files()
            elif choice == "3":
                return self._manual_file_input()
            elif choice == "4":
                return self._create_synthetic_interactive()
            else:
                print("Invalid choice")
                return None
                
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _browse_for_file(self) -> Optional[pd.DataFrame]:
        """Open GUI file browser"""
        if not TKINTER_AVAILABLE:
            print("GUI not available")
            return self._manual_file_input()
        
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)
            
            # File types for trading data
            filetypes = [
                ("Trading Data", "*.parquet;*.csv"),
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
            
            file_path = filedialog.askopenfilename(
                title="Select trading data file",
                filetypes=filetypes,
                initialdir=os.getcwd()
            )
            
            root.destroy()
            
            if file_path:
                print(f"Selected: {file_path}")
                return load_data_file(file_path)
            else:
                print("No file selected")
                return None
                
        except Exception as e:
            print(f"File browser error: {e}")
            return self._manual_file_input()
    
    def _show_and_select_files(self) -> Optional[pd.DataFrame]:
        """Show nearby data files and let user select"""
        print("\nLooking for data files...")
        
        # Search common locations
        search_paths = [
            Path.cwd(),
            Path.cwd() / "data", 
            Path.cwd() / "market_data",
            Path.home() / "Downloads"
        ]
        
        found_files = []
        for search_path in search_paths:
            if search_path.exists():
                # Look for data files using patterns from constants
                for pattern in SEARCH_PATTERNS:
                    found_files.extend(search_path.glob(pattern))
        
        if not found_files:
            print("No data files found nearby")
            return self._manual_file_input()
        
        # Show files with size info
        print("Found data files:")
        for i, file_path in enumerate(found_files[:MAX_DISPLAYED_FILES], 1):
            try:
                size_mb = file_path.stat().st_size / FILE_SIZE_MB_DIVISOR
                print(f"{i:2d}. {file_path.name} ({size_mb:.1f} MB)")
            except OSError:
                print(f"{i:2d}. {file_path.name}")
        
        print(f"{len(found_files) + 1:2d}. Enter custom path")
        
        # Get user choice
        try:
            choice = int(input(f"Select file (1-{len(found_files) + 1}): "))
            
            if 1 <= choice <= len(found_files):
                selected_file = found_files[choice - 1]
                print(f"Loading: {selected_file}")
                return load_data_file(str(selected_file))
            elif choice == len(found_files) + 1:
                return self._manual_file_input()
            else:
                print("Invalid choice")
                return None
                
        except ValueError:
            print("Please enter a number")
            return None
    
    def _manual_file_input(self) -> Optional[pd.DataFrame]:
        """Manual file path entry"""
        while True:
            file_path = input("Enter file path: ").strip()
            
            if not file_path:
                return None
            
            path = Path(file_path)
            
            if not path.exists():
                print(f"File not found: {file_path}")
                
                # Suggest similar files in the directory
                if path.parent.exists():
                    similar_files = list(path.parent.glob(f"*{path.stem}*"))
                    if similar_files:
                        print("Similar files found:")
                        for f in similar_files[:5]:
                            print(f"  {f}")
                continue
            
            try:
                return load_data_file(file_path)
            except Exception as e:
                print(f"Error loading file: {e}")
                continue
    
    def _create_synthetic_interactive(self) -> pd.DataFrame:
        """Create synthetic data with user input"""
        print("\nSYNTHETIC DATA GENERATION")
        print("-" * 30)
        
        try:
            bars = input("Number of bars [5000]: ").strip()
            bars = int(bars) if bars else 5000  # Default synthetic bars
            
            symbol = input("Symbol name [TEST]: ").strip() or "TEST"
            
            timeframe = input("Timeframe (1min, 5min, 15min, 1H, 1D) [15min]: ").strip() or "15min"
            
            print(f"Generating {bars} bars of {symbol} data...")
            return create_synthetic_data(bars=bars, symbol=symbol, timeframe=timeframe)
            
        except ValueError:
            print("Invalid input, using defaults")
            return create_synthetic_data()
def interactive_data_setup() -> pd.DataFrame:
    """
    Convenience function for interactive data loading
    
    Returns:
        DataFrame with loaded data
    """
    loader = InteractiveDataLoader()
    return loader.load_data_interactive()


def create_synthetic_data(bars: int = 5000,  # Default synthetic bars
                         symbol: str = "TEST",
                         timeframe: str = "15min",  # Default timeframe
                         base_price: Optional[float] = None) -> pd.DataFrame:
    """
    Create realistic synthetic OHLCV data with symbol-appropriate pricing.
    
    Uses PriceSeedResolver to determine appropriate price levels based on:
    1. User-supplied base_price (highest priority)
    2. Real market data (when available)  
    3. MarketSpec typical_price definitions
    4. System default fallback
    
    Args:
        bars: Number of bars to generate
        symbol: Futures contract symbol (e.g., "ES", "NQ", "CL")
        timeframe: Time frequency (pandas freq string)
        base_price: Optional user override for starting price level
        
    Returns:
        DataFrame with synthetic OHLCV data using appropriate price levels
    """
    from utils.price_seed_resolver import PriceSeedResolver
    
    # Resolve symbol-appropriate price seed
    resolver = PriceSeedResolver()
    price_seed = resolver.resolve(symbol, user_override=base_price)
    
    print(f"Generating {bars} bars of synthetic {symbol} data...")
    print(f"Using base price: ${price_seed.base_price:.2f}")
    
    # Create date range
    dates = pd.date_range(
        start=SYNTHETIC_START_DATE,
        periods=bars,
        freq=timeframe
    )
    
    # Generate realistic price movements
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    # Create returns with some trend and volatility clustering
    returns = np.zeros(bars)
    vol = 0.002  # Base volatility
    
    for i in range(1, bars):
        # Volatility clustering (GARCH-like)
        vol = 0.95 * vol + 0.05 * 0.002 + 0.01 * abs(returns[i-1])
        vol = np.clip(vol, 0.0005, 0.01)
        
        # Add some trend components
        trend = 0.0001 * np.sin(i / 200) + 0.00005 * np.sin(i / 50)
        returns[i] = np.random.normal(trend, vol)
    
    # Convert returns to prices using resolved seed
    actual_base_price = price_seed.base_price
    prices = [actual_base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, actual_base_price * 0.1))  # Floor price
    
    # Create OHLC data with proper relationships
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = prices[i-1] if i > 0 else close
        
        # Generate realistic intra-bar movement
        volatility = np.random.uniform(0.0005, 0.002)
        mid_price = (open_price + close) / 2
        range_size = mid_price * volatility
        
        # Create high/low with proper relationships
        potential_high = mid_price + range_size
        potential_low = mid_price - range_size
        
        high = max(open_price, close, potential_high)
        low = min(open_price, close, potential_low)
        
        # Add small random adjustments while maintaining OHLC rules
        high += np.random.uniform(0, mid_price * 0.001)
        low -= np.random.uniform(0, mid_price * 0.001)
        
        # Final validation of OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Round to realistic tick size
        tick_size = 0.25  # ES futures tick size
        open_price = round(open_price / tick_size) * tick_size
        high = round(high / tick_size) * tick_size
        low = round(low / tick_size) * tick_size
        close = round(close / tick_size) * tick_size
        
        # Generate volume
        volume = max(10, int(np.random.lognormal(5, 0.5)))  # Log-normal distribution
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'symbol': symbol
        })
    
    df = pd.DataFrame(data)
    
    # Validation check
    high_valid = (df['high'] >= df[['open', 'close']].max(axis=1)).all()
    low_valid = (df['low'] <= df[['open', 'close']].min(axis=1)).all()
    
    print(f"Generated {len(df)} bars")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"OHLC validation: {'PASS' if high_valid and low_valid else 'FAIL'}")
    
    return df


if __name__ == "__main__":
    # Test the loader
    print("Testing Data Loader...")
    
    # Test synthetic data
    test_data = create_synthetic_data(bars=1000, symbol="NQ")
    print(f"Created test data: {len(test_data)} bars")
    print(test_data.head())
    
    # Test file operations
    test_file = "test_data.parquet"
    test_data.to_parquet(test_file, index=False)
    print(f"Saved test data to {test_file}")
    
    # Load it back
    loader = DataLoader()
    loaded_data = loader.load_file(test_file)
    print(f"Loaded data: {len(loaded_data)} bars")
    
    # Cleanup
    Path(test_file).unlink()
    print("Test complete!")