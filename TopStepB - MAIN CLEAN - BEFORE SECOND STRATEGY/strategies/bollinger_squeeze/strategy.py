"""
Bollinger Band Squeeze Breakout Strategy Implementation

A volatility-based breakout strategy using the TTM Squeeze methodology.
Identifies low volatility periods (squeeze) and trades the subsequent breakouts.
"""
import pandas as pd
import numpy as np
from numba import njit
from typing import Dict, Any, Union, Tuple, List

from ..base import BaseStrategy
from config.system_config import TradingConfig
from .indicators import calculate_all_indicators
from .parameters import get_default_parameters, get_parameter_ranges, validate_parameters


@njit
def _stateful_loop_jit(open_prices, high_prices, low_prices, close_prices, entry,
                       atr, exit_upper, exit_lower, bb_upper, bb_lower,
                       exit_method, stop_loss_atr_multiplier, risk_reward_ratio):
    n = len(open_prices)
    final_signals = np.zeros(n, dtype=np.int8)
    entry_logs = np.zeros(n, dtype=np.int8)
    exit_logs = np.zeros(n, dtype=np.int8)
    entry_price_log = np.zeros(n, dtype=np.float64)
    stop_log = np.zeros(n, dtype=np.float64)
    target_log = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    target_price = 0.0
    bars_in_trade = 0
    bars_since_exit = 0

    for i in range(1, n):
        if position != 0:
            bars_in_trade += 1
            exit_triggered = False

            if position == 1:
                stop_hit = low_prices[i - 1] <= stop_loss
            else:
                stop_hit = high_prices[i - 1] >= stop_loss

            if stop_hit:
                exit_triggered = True
                exit_logs[i] = 1 if position == 1 else -1
            elif exit_method == 0:
                if position == 1 and close_prices[i - 1] >= target_price:
                    exit_triggered = True
                    exit_logs[i] = 2
                elif position == -1 and close_prices[i - 1] <= target_price:
                    exit_triggered = True
                    exit_logs[i] = -2
            elif exit_method == 1:
                if position == 1 and close_prices[i - 1] < exit_lower[i - 1]:
                    exit_triggered = True
                    exit_logs[i] = 3
                elif position == -1 and close_prices[i - 1] > exit_upper[i - 1]:
                    exit_triggered = True
                    exit_logs[i] = -3
            elif exit_method == 2:
                if position == 1 and close_prices[i - 1] >= bb_upper[i - 1]:
                    exit_triggered = True
                    exit_logs[i] = 4
                elif position == -1 and close_prices[i - 1] <= bb_lower[i - 1]:
                    exit_triggered = True
                    exit_logs[i] = -4

            if exit_triggered:
                position = 0
                stop_loss = 0.0
                target_price = 0.0
                bars_since_exit = 1
                bars_in_trade = 0
        else:
            if bars_since_exit > 0:
                bars_since_exit += 1

            entry_signal = entry[i - 1]

            if entry_signal != 0:
                position = int(entry_signal)
                entry_price = open_prices[i]
                if position == 1:
                    stop_loss = entry_price - (atr[i - 1] * stop_loss_atr_multiplier)
                    if exit_method == 0:
                        target_price = entry_price + (entry_price - stop_loss) * risk_reward_ratio
                    else:
                        target_price = 0.0
                else:
                    stop_loss = entry_price + (atr[i - 1] * stop_loss_atr_multiplier)
                    if exit_method == 0:
                        target_price = entry_price - (stop_loss - entry_price) * risk_reward_ratio
                    else:
                        target_price = 0.0

                bars_in_trade = 0
                bars_since_exit = 0
                entry_logs[i] = position
                entry_price_log[i] = entry_price
                stop_log[i] = stop_loss
                target_log[i] = target_price

        final_signals[i] = position

    return final_signals, entry_logs, exit_logs, entry_price_log, stop_log, target_log


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze breakout strategy for futures trading.
    
    This strategy implements the TTM Squeeze methodology:
    - Identifies periods when Bollinger Bands are inside Keltner Channels (squeeze)
    - Waits for momentum confirmation and breakout
    - Enters in the direction of the breakout with proper risk management
    - Uses multiple exit methods for optimal trade management
    """
    
    # Required class attributes
    name = "bollinger_squeeze"
    description = "TTM Squeeze breakout strategy using Bollinger Bands and Keltner Channels"
    category = "breakout"
    min_data_points = 250  # Need enough for trend filter and indicators
    
    def __init__(self):
        """Initialize strategy following BaseStrategy contract."""
        super().__init__(self.name)
        
        # Add debug logger for signal timing validation
        from utils.logger import get_logger
        self._debug_logger = get_logger(f"signal_timing_{self.name}")
    
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any], config: TradingConfig) -> pd.Series:
        """
        Generate trading signals based on Bollinger Band squeeze and breakout.
        
        Args:
            data: OHLCV DataFrame with datetime index
            params: Strategy parameters dictionary
            config: TradingConfig for market specifications and account settings
            
        Returns:
            pd.Series: Trading signals (-1, 0, 1) with proper position holding
        """
        
        # Validate inputs
        if not self.validate_parameters(params):
            raise ValueError("Invalid parameters")
        
        self.validate_data(data)
        
        # Calculate all indicators (GPU aware)
        use_gpu = getattr(self, 'use_gpu', None)
        indicators = calculate_all_indicators(data, params, use_gpu=use_gpu)
        
        # Phase 1: Generate entry conditions (vectorized for performance)
        entry_signals = self._generate_entry_signals_vectorized(data, indicators, params)
        
        # Phase 2: Apply stateful position management (mirrors deployment)
        final_signals = self._apply_stateful_position_management(data, entry_signals, indicators, params)
        
        return final_signals
    
    def _generate_entry_signals_vectorized(self, data: pd.DataFrame, indicators: Dict, params: Dict[str, Any]) -> pd.Series:
        """Generate entry signals using vectorized operations."""
        
        # Initialize entry signals
        entry_signals = pd.Series(0, index=data.index)
        
        # Extract key data
        close = data['close']
        
        # Extract indicators
        squeeze = indicators['squeeze']
        momentum = indicators['momentum']
        breakout_upper = indicators['breakout_upper']
        breakout_lower = indicators['breakout_lower']
        
        # Momentum conditions
        if params['use_momentum_filter']:
            momentum_bullish = momentum > params['momentum_threshold']
            momentum_bearish = momentum < -params['momentum_threshold']
        else:
            momentum_bullish = pd.Series(True, index=data.index)
            momentum_bearish = pd.Series(True, index=data.index)
        
        # Trend filter conditions
        if params['use_trend_filter']:
            trend_filter = indicators['trend_filter']
            long_trend = close > trend_filter
            short_trend = close < trend_filter
        else:
            long_trend = pd.Series(True, index=data.index)
            short_trend = pd.Series(True, index=data.index)
        
        # Volume filter conditions
        if params['volume_filter']:
            volume_ratio = indicators['volume_ratio']
            volume_ok = volume_ratio >= params['min_volume_ratio']
        else:
            volume_ok = pd.Series(True, index=data.index)
        
        # Breakout conditions
        long_breakout = close > breakout_upper.shift(1)
        short_breakout = close < breakout_lower.shift(1)
        
        # Check if squeeze was active on previous bar AND minimum duration met
        squeeze_setup = squeeze.shift(1)
        squeeze_duration = indicators['squeeze_duration']
        squeeze_ready = squeeze_duration.shift(1) >= params['min_squeeze_bars']  # CRITICAL FIX
        
        # Generate entry conditions
        long_entry = (
            squeeze_setup &                    # Squeeze was active
            squeeze_ready &                    # CRITICAL: Minimum squeeze duration met
            long_breakout &                    # Breakout to upside
            momentum_bullish &                 # Momentum confirmation
            long_trend &                       # Trend filter
            volume_ok                          # Volume confirmation
        )
        
        short_entry = (
            squeeze_setup &                    # Squeeze was active
            squeeze_ready &                    # CRITICAL: Minimum squeeze duration met
            short_breakout &                   # Breakout to downside
            momentum_bearish &                 # Momentum confirmation
            short_trend &                      # Trend filter
            volume_ok                          # Volume confirmation
        )
        
        # Set entry signals
        entry_signals[long_entry] = 1
        entry_signals[short_entry] = -1
        
        return entry_signals
    
    def _apply_stateful_position_management(self, data: pd.DataFrame, entry_signals: pd.Series,
                                          indicators: Dict, params: Dict[str, Any]) -> pd.Series:
        """Apply stateful position management using Numba for performance."""
        self._debug_logger.debug(
            "Applying stateful management via Numba JIT with vectorized stop-loss checks"
        )

        open_prices = data['open'].to_numpy(dtype=np.float64)
        high_prices = data['high'].to_numpy(dtype=np.float64)
        low_prices = data['low'].to_numpy(dtype=np.float64)
        close_prices = data['close'].to_numpy(dtype=np.float64)
        entry_tensor = entry_signals.to_numpy(dtype=np.int8)
        atr = indicators['atr'].to_numpy(dtype=np.float64)

        exit_method_map = {'fixed_rr': 0, 'trailing_donchian': 1, 'opposite_band': 2}
        exit_method = exit_method_map[params['exit_method']]

        zeros = np.zeros(len(data), dtype=np.float64)
        exit_upper = indicators.get('exit_upper', pd.Series(0, index=data.index)).to_numpy(dtype=np.float64) if exit_method == 1 else zeros
        exit_lower = indicators.get('exit_lower', pd.Series(0, index=data.index)).to_numpy(dtype=np.float64) if exit_method == 1 else zeros
        bb_upper = indicators.get('bb_upper', pd.Series(0, index=data.index)).to_numpy(dtype=np.float64) if exit_method == 2 else zeros
        bb_lower = indicators.get('bb_lower', pd.Series(0, index=data.index)).to_numpy(dtype=np.float64) if exit_method == 2 else zeros

        final_signals, entry_logs, exit_logs, entry_prices, stops, targets = _stateful_loop_jit(
            open_prices, high_prices, low_prices, close_prices, entry_tensor, atr,
            exit_upper, exit_lower, bb_upper, bb_lower,
            exit_method, params['stop_loss_atr_multiplier'], params['risk_reward_ratio']
        )

        reason_map = {
            1: 'stop_loss_long',
            -1: 'stop_loss_short',
            2: 'fixed_rr_long',
            -2: 'fixed_rr_short',
            3: 'trailing_donchian_long',
            -3: 'trailing_donchian_short',
            4: 'opposite_band_long',
            -4: 'opposite_band_short',
        }

        for i in range(len(data)):
            if entry_logs[i] != 0:
                etype = 'long' if entry_logs[i] == 1 else 'short'
                self._debug_logger.debug(
                    f"Bar {i}: {etype} entry at open price {entry_prices[i]}, stop: {stops[i]}, target: {targets[i]}"
                )
            if exit_logs[i] != 0:
                self._debug_logger.debug(f"Bar {i}: Exit triggered - {reason_map[exit_logs[i]]}")

        return pd.Series(final_signals, index=data.index)
    
    def reset_state(self) -> None:
        """
        Reset internal strategy state to prevent contamination between calls.
        
        CRITICAL: This method resets all stateful variables used in the stateful loop iteration
        to prevent contamination between optimization trials. Called automatically by 
        execute_strategy() before each generate_signals() call.
        
        Stateful variables that must be reset:
        - position: Current position (0=flat, 1=long, -1=short)  
        - entry_price: Price at which current position was entered
        - bars_in_trade: Number of bars since position entry
        - bars_since_exit: Number of bars since last position exit
        """
        # Note: This strategy uses stateful variables within the loop iteration
        # but they are re-initialized at the start of _apply_stateful_position_management()
        # so no persistent state variables need to be reset.
        pass
    
    def get_parameter_ranges(self) -> Dict[str, Union[Tuple[Union[int, float], ...], List[str]]]:
        """Return parameter ranges for optimization."""
        return get_parameter_ranges()
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter dictionary."""
        return validate_parameters(params)
    
    def define_constraints(self) -> Dict[str, Any]:
        """
        Return parameter constraints for optimization engine.
        
        Bollinger squeeze strategy constraints:
        - exit_donchian_period must be less than breakout_period for proper exit timing
        - breakout_period must be less than or equal to bb_period for valid Bollinger Band analysis
        """
        return {
            'comparison': [
                ('exit_donchian_period', '<', 'breakout_period'),
                ('breakout_period', '<=', 'bb_period')
            ]
        }
    
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Return strategy metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "min_data_points": self.min_data_points,
            "parameter_count": len(get_default_parameters()),
            "author": "TopStepX Team",
            "version": "1.0.0",
            "indicators_used": ["Bollinger Bands", "Keltner Channels", "ATR", "Donchian Channels", "Momentum"],
            "suitable_markets": ["Futures", "Stocks", "ETFs", "Forex"],
            "trading_style": "Volatility Breakout",
            "risk_profile": "Medium-High",
            "typical_holding_period": "Short to Medium Term",
            "methodology": "TTM Squeeze",
        }