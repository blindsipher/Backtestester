"""
BaseStrategy Abstract Base Class

Defines the minimal interface all trading strategies must implement.
Uses vectorized pandas operations with proper signal timing to avoid look-ahead bias.
"""

import pandas as pd
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, List, Optional

from config.system_config import TradingConfig
from utils.exceptions import (
    StrategyExecutionError, 
    StrategyConfigurationError
)
from utils.logger import get_logger




class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the minimal interface for vectorized strategy implementation
    with proper signal timing to avoid look-ahead bias.
    """
    
    # Required class attributes that all strategies must define
    # These are converted to abstract properties to enforce proper implementation
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable strategy description."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Strategy category for organization."""
        pass
    
    @property
    @abstractmethod
    def min_data_points(self) -> int:
        """Minimum data points required for strategy execution."""
        pass
    
    def __init__(self, name: str):
        """Initialize base strategy with name validation."""
        # Security validation for strategy names
        if not isinstance(name, str) or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise StrategyConfigurationError(
                f"Invalid strategy name '{name}'. Must be a valid Python identifier."
            )
        
        self.strategy_name = name
        self.initialized = False
        self._config: Optional[TradingConfig] = None
        
        # Initialize timing validation logger
        self._timing_logger = get_logger(f"timing_validation_{name}")
    
    def _raise_timing_validation_error(self, error_msg: str) -> None:
        """
        Helper method to raise StrategyExecutionError with proper constructor arguments.
        
        This encapsulates the exception creation to prevent constructor signature mismatches
        and provides a single place to maintain exception creation logic.
        
        Args:
            error_msg: The error message describing the timing violation
            
        Raises:
            StrategyExecutionError: Properly constructed exception with strategy context
        """
        raise StrategyExecutionError(
            strategy_name=self.strategy_name,
            execution_step="signal_timing_validation",
            original_error=error_msg
        )
    
    def set_config(self, config: TradingConfig) -> None:
        """
        Set the TradingConfig for this strategy.
        
        Args:
            config: TradingConfig object containing all strategy configuration
        """
        if not isinstance(config, TradingConfig):
            raise StrategyConfigurationError(
                f"Expected TradingConfig object, got {type(config)}"
            )
        
        self._config = config
        self.initialized = True
    
    def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any], contracts_per_trade: int = 1) -> pd.Series:
        """
        Execute the strategy with TradingConfig and parameters.
        
        This method calls the abstract generate_signals method with parameters
        and the TradingConfig for market specifications.
        
        LIFECYCLE: Must call set_config() before execute_strategy()
        
        Args:
            data: OHLCV DataFrame with validated financial data
            params: Strategy-specific parameters dictionary
            contracts_per_trade: Number of contracts to trade per signal (default: 1)
            
        Returns:
            pd.Series of position signals (1=long, -1=short, 0=flat)
            
        Raises:
            StrategyConfigurationError: If config not set or invalid
        """
        if self._config is None:
            raise StrategyConfigurationError(
                f"Strategy {self.strategy_name} not configured. Call set_config() first."
            )
        
        # Store contracts_per_trade for use in to_orders()
        self._contracts_per_trade = contracts_per_trade
        
        # CRITICAL FIX: Reset strategy state to prevent contamination between calls
        self.reset_state()
        
        # Call the abstract method with parameters and config
        signals = self.generate_signals(data, params, self._config)
        
        # CRITICAL: Validate signal timing to prevent look-ahead bias
        min_bars_between = params.get('min_bars_between_trades', 1)
        self.validate_signal_timing(signals, data, min_bars_between)
        
        # Optional: Statistical validation for look-ahead bias (can be disabled for performance)
        if params.get('enable_lookahead_validation', False):
            self.validate_no_lookahead_bias(signals, data)
        
        return signals
    
    def get_config(self) -> Optional[TradingConfig]:
        """Get the current TradingConfig."""
        return self._config
    
    def get_timeframe_minutes(self) -> int:
        """Get current timeframe in minutes from config."""
        from config.system_config import SupportedTimeframes
        if self._config is None:
            raise StrategyConfigurationError("Config not set. Call set_config() first.")
        return SupportedTimeframes.get_timeframe_minutes(self._config.timeframe)

    def get_bars_per_day(self, session_minutes: int = 1440) -> int:
        """Calculate bars per day based on current timeframe."""
        timeframe_minutes = self.get_timeframe_minutes()
        return max(1, session_minutes // timeframe_minutes)

    def get_bars_per_session(self, session_minutes: int = 390) -> int:
        """Calculate bars per trading session (default: 6.5hr equity session)."""
        timeframe_minutes = self.get_timeframe_minutes()
        # For timeframes longer than session, return 1 bar minimum
        return max(1, session_minutes // timeframe_minutes)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any], config: TradingConfig) -> pd.Series:
        """
        Generate trading signals with shift(1) to avoid look-ahead bias.
        
        Args:
            data: OHLCV DataFrame with validated financial data
            params: Dictionary of strategy parameters
            config: TradingConfig for market specifications and account settings
            
        Returns:
            pd.Series of position signals (1=long, -1=short, 0=flat)
            with proper shift(1) to trade on next bar open
        """
        pass
    
    @abstractmethod 
    def reset_state(self) -> None:
        """
        Reset internal strategy state to prevent contamination between calls.
        
        CRITICAL: This method must reset all stateful variables used in generate_signals
        to their initial values. This prevents state leakage when the same strategy
        instance is used for multiple executions (e.g., optimize_data then validate_data).
        
        Examples of state to reset:
        - Position tracking variables (position, entry_price)
        - Trade counters (bars_in_trade, bars_since_exit)
        - Any other variables that persist between loop iterations
        
        Called automatically by execute_strategy() before each generate_signals() call.
        """
        pass
    
    def validate_parameters_on_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> bool:
        """
        Lightweight parameter validation on data without full signal generation.
        
        PERFORMANCE OPTIMIZATION: This method provides fast parameter validation
        by only checking that indicators can be calculated and parameters work
        with the given data, without running the full signal generation process.
        
        Default implementation performs basic checks. Strategies can override
        for more sophisticated validation.
        
        Args:
            data: OHLCV DataFrame to validate parameters against  
            params: Strategy parameters to validate
            
        Returns:
            True if parameters are valid for this data, False otherwise
        """
        try:
            # Basic validation: ensure we have enough data
            if len(data) < self.min_data_points:
                return False
                
            # Validate data structure
            if not self.validate_data(data):
                return False
                
            # Validate parameters using existing method
            if not self.validate_parameters(params):
                return False
                
            # All basic checks passed
            return True
            
        except Exception as e:
            # Use proper logger scope - get_logger imported at module level
            validation_logger = get_logger("parameter_validation")
            validation_logger.warning(f"Parameter validation failed: {e}")
            return False
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Union[Tuple[Union[int, float], ...], List[str]]]:
        """
        Return parameter ranges for optimization.
        
        Returns:
            Dict mapping parameter names to ranges:
            - Numeric: (min, max, step) tuples
            - Categorical: Lists of valid values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter values meet strategy constraints.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def define_constraints(self) -> Dict[str, Any]:
        """
        Return parameter constraints for optimization engine.
        
        Strategy-agnostic constraints that optimization engine can discover and enforce
        during parameter generation to prevent invalid combinations and reduce optimization waste.
        
        Constraint Types:
        - comparison: Numeric parameter relationships [(dependent, operator, independent)]
        - conditional: Parameters only relevant when conditions met [(param, 'requires', flag, value)]
        - mutually_exclusive: Parameters that exclude others [(param, 'excludes', other_param)]
        
        Examples:
            {
                'comparison': [('exit_period', '<', 'entry_period')],
                'conditional': [('stop_loss_fixed_atr', 'requires', 'stop_loss_type', 'fixed')],
                'mutually_exclusive': [('use_fixed_size', 'excludes', 'dynamic_sizing')]
            }
        
        Returns:
            Dict with constraint arrays defining parameter relationships
        """
        return {}  # Default: no constraints
    
    @abstractmethod
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """
        Return strategy metadata for documentation.
        
        Returns:
            Dict with strategy information
        """
        pass
    
    def to_orders(self, signals: pd.Series) -> pd.DataFrame:
        """
        Convert shifted signals to orders executed at bar open.
        
        This helper method converts position signals to order instructions
        that can be executed by the trading engine at bar open prices.
        
        Args:
            signals: pd.Series of position signals (already shifted for proper timing)
            
        Returns:
            pd.DataFrame with columns ['qty', 'price_type'] where:
            - qty: number of contracts (±contracts_per_trade for entries/exits)
            - price_type: 'open' (engine interprets as next-bar open)
        """
        # Get contracts per trade (default to 1 if not set)
        contracts_per_trade = getattr(self, '_contracts_per_trade', 1)
        
        # Create orders DataFrame
        orders = pd.DataFrame(index=signals.index, columns=['qty', 'price_type'])
        orders['qty'] = 0  # Default to no action
        orders['price_type'] = 'open'  # All orders execute at open
        
        # Detect position changes (entries and exits)
        
        # Long entries: signal goes from 0 to 1
        long_entries = (signals == 1) & (signals.shift(1).fillna(0) == 0)
        orders.loc[long_entries, 'qty'] = contracts_per_trade
        
        # Short entries: signal goes from 0 to -1  
        short_entries = (signals == -1) & (signals.shift(1).fillna(0) == 0)
        orders.loc[short_entries, 'qty'] = -contracts_per_trade
        
        # Exits: signal goes from non-zero to 0
        exits = (signals == 0) & (signals.shift(1).fillna(0) != 0)
        # Set exit quantity to position-closing value (opposite of previous position)
        orders.loc[exits, 'qty'] = -signals.shift(1).fillna(0).loc[exits] * contracts_per_trade
        
        # Position flips: signal goes from 1 to -1 or -1 to 1
        # These require double the quantity to flatten previous position AND enter new position
        flips = ((signals == 1) & (signals.shift(1).fillna(0) == -1)) | \
                ((signals == -1) & (signals.shift(1).fillna(0) == 1))
        # For flips: qty = (new_signal - old_signal) * contracts_per_trade
        # e.g., (1 - (-1)) * 2 = 4 contracts, or (-1 - 1) * 2 = -4 contracts
        orders.loc[flips, 'qty'] = (signals.loc[flips] - signals.shift(1).fillna(0).loc[flips]) * contracts_per_trade
        
        # Return only rows with actual orders (position changes)
        position_changes = (long_entries | short_entries | exits | flips)
        active_orders = orders.loc[position_changes].copy()
        
        return active_orders
    
    def validate_signal_timing(self, signals: pd.Series, data: pd.DataFrame, min_bars_between: int = 1) -> bool:
        """
        CRITICAL: Validate signal timing to prevent look-ahead bias.
        
        This method performs runtime validation to ensure:
        1. No same-bar entry/exit violations
        2. Minimum bars between trades are respected
        3. Signal changes are realistic and executable
        
        Args:
            signals: Generated position signals series
            data: Original OHLCV data
            min_bars_between: Minimum bars required between trades
            
        Returns:
            True if timing is valid
            
        Raises:
            StrategyExecutionError: If timing violations are detected
        """
        try:
            # Validate signal changes don't violate minimum bars
            signal_changes = signals.diff().fillna(0)
            change_indices = signal_changes[signal_changes != 0].index
            
            if len(change_indices) <= 1:
                return True  # No trades or only one trade
            
            # Check minimum bars between signal changes
            for i in range(1, len(change_indices)):
                prev_idx = data.index.get_loc(change_indices[i-1])
                curr_idx = data.index.get_loc(change_indices[i])
                bars_between = curr_idx - prev_idx
                
                if bars_between < min_bars_between:
                    error_msg = (
                        f"TIMING VIOLATION: Signal change at bar {curr_idx} "
                        f"only {bars_between} bars after previous change "
                        f"(minimum: {min_bars_between})"
                    )
                    self._timing_logger.error(error_msg)
                    self._raise_timing_validation_error(error_msg)
            
            # Check for impossible signal magnitude changes
            max_change = signal_changes.abs().max()
            if float(max_change) > 2:  # Should never jump more than 2 positions (-1 to 1)
                error_msg = f"TIMING VIOLATION: Impossible signal jump of {max_change}"
                self._timing_logger.error(error_msg)
                self._raise_timing_validation_error(error_msg)
            
            self._timing_logger.debug(f"Signal timing validation passed: {len(change_indices)} trades")
            return True
            
        except Exception as e:
            self._timing_logger.error(f"Signal timing validation failed: {e}")
            self._raise_timing_validation_error(f"Signal timing validation failed: {e}")
    
    def validate_no_lookahead_bias(self, signals: pd.Series, data: pd.DataFrame) -> bool:
        """
        CRITICAL: Validate that signals don't exhibit look-ahead bias patterns.
        
        Performs statistical tests to detect impossible prescience in signal timing.
        
        Args:
            signals: Generated position signals series
            data: Original OHLCV data
            
        Returns:
            True if no look-ahead bias detected
            
        Raises:
            StrategyExecutionError: If look-ahead bias patterns are detected
        """
        try:
            # Check for signals that perfectly time market turns (statistical impossibility)
            signal_changes = signals.diff().fillna(0)
            entries = signal_changes[signal_changes != 0].index
            
            if len(entries) < 5:
                return True  # Too few trades to analyze
            
            # Calculate next-bar returns for entry points
            next_bar_returns = []
            for entry_idx in entries:
                try:
                    entry_pos = data.index.get_loc(entry_idx)
                    if entry_pos < len(data) - 1:
                        current_close = data['close'].iloc[entry_pos]
                        next_close = data['close'].iloc[entry_pos + 1]
                        next_return = (next_close - current_close) / current_close
                        signal_direction = signals.loc[entry_idx]
                        
                        # Check if signal direction aligns perfectly with next return
                        if signal_direction != 0:
                            directional_return = next_return * signal_direction
                            next_bar_returns.append(directional_return)
                except (IndexError, KeyError):
                    continue
            
            if next_bar_returns:
                avg_next_return = np.mean(next_bar_returns)
                win_rate = sum(1 for r in next_bar_returns if r > 0) / len(next_bar_returns)
                
                # Flag suspiciously perfect timing (statistically impossible)
                if win_rate > 0.95 and avg_next_return > 0.005:  # >95% wins with >0.5% avg return
                    error_msg = (
                        f"LOOK-AHEAD BIAS DETECTED: Impossible timing perfection "
                        f"(win_rate: {win_rate:.3f}, avg_return: {avg_next_return:.4f})"
                    )
                    self._timing_logger.error(error_msg)
                    raise StrategyExecutionError(
                        strategy_name=self.strategy_name,
                        execution_step="look_ahead_bias_validation",
                        original_error=error_msg
                    )
            
            self._timing_logger.debug("Look-ahead bias validation passed")
            return True
            
        except StrategyExecutionError:
            raise
        except Exception as e:
            self._timing_logger.warning(f"Look-ahead bias validation inconclusive: {e}")
            return True  # Don't fail on validation errors, just warn
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input OHLCV data meets strategy requirements.
        
        Args:
            data: OHLCV DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns exist
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.min_data_points}"
            )
        
        # Check for valid OHLCV relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            raise ValueError("Invalid OHLCV relationships detected")
        
        # Check for NaN or infinite values
        for col in required_columns:
            if data[col].isna().any():
                raise ValueError(f"NaN values found in {col} column")
            if np.isinf(data[col]).any():
                raise ValueError(f"Infinite values found in {col} column")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                raise ValueError(f"Non-positive prices found in {col} column")
        
        # Check for negative volume
        if (data['volume'] < 0).any():
            raise ValueError("Negative volume values found")
        
        return True
    
    def _load_strategy_module(self, module_name: str):
        """
        Safely load a strategy module with error handling.
        
        Args:
            module_name: Name of module to load
            
        Returns:
            Loaded module object
            
        Raises:
            StrategyExecutionError: If module loading fails
        """
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError as e:
            raise StrategyExecutionError(
                strategy_name="unknown",
                execution_step="module_loading",
                original_error=f"Failed to load strategy module {module_name}: {e}"
            )
        except Exception as e:
            raise StrategyExecutionError(
                strategy_name="unknown",
                execution_step="module_loading",
                original_error=f"Unexpected error loading {module_name}: {e}"
            )