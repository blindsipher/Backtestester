"""
Bollinger Band Squeeze Strategy Parameters

Defines parameter ranges and defaults for volatility breakout trading
using the TTM Squeeze methodology.
"""

import logging

def _get_logger():
    """Get logger instance - multiprocessing safe"""
    return logging.getLogger(__name__)

DEFAULT_PARAMETERS = {
    # Bollinger Bands Parameters
    "bb_period": 20,                    # Bollinger Bands moving average period
    "bb_std_dev": 2.0,                 # Standard deviations for bands
    
    # Keltner Channels Parameters
    "kc_period": 20,                   # Keltner Channel moving average period
    "kc_atr_multiplier": 1.5,          # ATR multiplier for channel width
    
    # Breakout Detection
    "breakout_period": 15,             # Period for Donchian breakout confirmation
    "min_squeeze_bars": 6,             # Minimum bars in squeeze before valid breakout
    
    # Momentum Filter
    "use_momentum_filter": True,       # Use momentum direction for entries
    "momentum_period": 12,             # Period for momentum calculation
    "momentum_threshold": 0.0,         # Minimum momentum for entry
    
    # Risk Management (Position sizing handled by CLI --contracts-per-trade)
    "stop_loss_atr_multiplier": 2.0,   # ATR multiplier for initial stop loss
    "atr_period": 20,                  # ATR calculation period
    
    # Exit Strategy
    "exit_method": "trailing_donchian", # Exit method: trailing_donchian, opposite_band, fixed_rr
    "exit_donchian_period": 7,         # Period for trailing Donchian exit (max value within constraint range)
    "risk_reward_ratio": 2.0,          # Risk/reward ratio for fixed target
    "trail_stop_atr_multiplier": 1.5,  # ATR multiplier for trailing stop
    
    # Filters
    "use_trend_filter": True,          # Use long-term trend filter
    "trend_filter_period": 200,        # Period for trend filter SMA
    "volume_filter": False,            # Require volume confirmation
    "min_volume_ratio": 1.2,           # Minimum volume vs average
}

PARAMETER_RANGES = {
    # INSTITUTIONAL FIX: TTM Squeeze parameter ranges tightened to eliminate instability
    # Previous ranges caused 60% zero-trade trials and extreme PNL swings due to
    # mathematically impossible parameter combinations (6.2B total combinations)
    
    # Bollinger Bands Parameters - INSTITUTIONAL TTM SQUEEZE STANDARDS
    "bb_period": (15, 30, 5),        # TTM standard: 20, allow 15-30 for optimization
    "bb_std_dev": (1.5, 2.5, 0.1),   # TTM standard: 2.0, allow 1.5-2.5 for fine-tuning
    
    # Keltner Channels Parameters - INSTITUTIONAL TTM SQUEEZE STANDARDS  
    "kc_period": (15, 30, 5),        # TTM standard: 20, match BB period range
    "kc_atr_multiplier": (1.0, 2.5, 0.1),  # TTM standard: 1.5, allow 1.0-2.5 for optimization
    
    # Breakout Detection - INSTITUTIONAL TRADING STANDARDS
    "breakout_period": (8, 25, 1),   # Practical range for ES futures breakout detection
    "min_squeeze_bars": (3, 12, 1),  # Minimum 3 bars for valid squeeze, max 12 for practicality
    
    # Momentum Filter - OPTIMIZED FOR ES FUTURES
    "use_momentum_filter": [True, False],
    "momentum_period": (8, 20, 2),   # Keep existing range (reasonable)
    "momentum_threshold": (-0.3, 0.3, 0.1),  # Tighter threshold for cleaner signals
    
    # Risk Management - ES FUTURES OPTIMIZED (Position sizing handled by CLI --contracts-per-trade)
    "stop_loss_atr_multiplier": (1.5, 2.5, 0.25),  # Practical stop distances for ES
    "atr_period": (14, 21, 7),       # Standard ATR periods: 14 or 21
    
    # Exit Strategy - INSTITUTIONAL TRADING STANDARDS
    "exit_method": ["trailing_donchian", "opposite_band", "fixed_rr"],
    "exit_donchian_period": (5, 10, 1),  # INSTITUTIONAL FIX: 5-10 bars prevents whipsaw exits
    "risk_reward_ratio": (1.5, 3.0, 0.25),  # REALISTIC: 1.5-3.0 achievable for ES futures
    "trail_stop_atr_multiplier": (1.25, 2.0, 0.25),  # Tighter trailing stops
    
    # Filters - OPTIMIZED RANGES
    "use_trend_filter": [True, False],
    "trend_filter_period": (150, 250, 50),  # Shorter range around 200-period standard
    "volume_filter": [True, False],
    "min_volume_ratio": (1.1, 1.8, 0.1),   # Tighter volume confirmation range
}

PARAMETER_DESCRIPTIONS = {
    "bb_period": "Period for Bollinger Bands moving average",
    "bb_std_dev": "Number of standard deviations for Bollinger Bands",
    "kc_period": "Period for Keltner Channel moving average",
    "kc_atr_multiplier": "ATR multiplier for Keltner Channel width",
    "breakout_period": "Period for Donchian breakout confirmation",
    "min_squeeze_bars": "Minimum consecutive bars in squeeze before valid breakout",
    "use_momentum_filter": "Whether to use momentum direction for entry filtering",
    "momentum_period": "Period for momentum oscillator calculation",
    "momentum_threshold": "Minimum momentum value required for entry",
    "stop_loss_atr_multiplier": "ATR multiplier for initial stop loss placement",
    "atr_period": "Period for Average True Range calculation",
    "exit_method": "Method for position exits: trailing Donchian, opposite band touch, or fixed R:R",
    "exit_donchian_period": "Period for trailing Donchian exit channel",
    "risk_reward_ratio": "Risk to reward ratio for fixed target exits",
    "trail_stop_atr_multiplier": "ATR multiplier for trailing stop distance",
    "use_trend_filter": "Whether to use long-term trend filter",
    "trend_filter_period": "Period for trend filter moving average",
    "volume_filter": "Whether to require volume confirmation for breakouts",
    "min_volume_ratio": "Minimum volume ratio vs recent average for valid breakout",
}

def get_default_parameters():
    """Return copy of default parameters."""
    return DEFAULT_PARAMETERS.copy()

def get_parameter_ranges():
    """Return copy of parameter ranges for optimization."""
    return PARAMETER_RANGES.copy()

def validate_parameters(params):
    """
    Validate parameter values are within acceptable ranges.
    More permissive validation with better constraint handling and logging.
    
    Args:
        params: Dictionary of parameters to validate
        
    Returns:
        bool: True if all parameters are valid
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Merge with defaults for complete validation
    full_params = {**DEFAULT_PARAMETERS, **params}
    
    _get_logger().debug(f"Validating Bollinger Squeeze parameters: {full_params}")
    
    try:
        # Check all required parameters are present
        required_params = set(DEFAULT_PARAMETERS.keys())
        provided_params = set(full_params.keys())
        
        missing = required_params - provided_params
        if missing:
            error_msg = f"Missing required parameters: {missing}"
            _get_logger().warning(error_msg)
            raise ValueError(error_msg)
        
        # Validate numeric ranges
        for param, value in full_params.items():
            if param not in PARAMETER_RANGES:
                continue
                
            param_range = PARAMETER_RANGES[param]
            
            # Handle categorical parameters (lists)
            if isinstance(param_range, list):
                if value not in param_range:
                    error_msg = f"{param} value {value} not in allowed values: {param_range}"
                    _get_logger().warning(error_msg)
                    raise ValueError(error_msg)
            # Handle numeric ranges (tuples)
            elif isinstance(param_range, tuple):
                min_val, max_val, _ = param_range
                if not min_val <= value <= max_val:
                    error_msg = f"{param} value {value} outside range [{min_val}, {max_val}]"
                    _get_logger().warning(error_msg)
                    raise ValueError(error_msg)
        
        # Validate logical constraints with better error messages
        breakout_period = full_params['breakout_period']
        exit_donchian_period = full_params['exit_donchian_period']
        bb_period = full_params['bb_period']
        min_squeeze_bars = full_params['min_squeeze_bars']
        
        # Allow BB and KC periods to be different (more flexible)
        if full_params['bb_period'] != full_params['kc_period']:
            _get_logger().debug(f"BB period ({bb_period}) != KC period ({full_params['kc_period']}) - allowing flexibility")
        
        # INSTITUTIONAL CONSTRAINT: Breakout period should be reasonable vs BB period
        if breakout_period > bb_period:
            error_msg = f"breakout_period ({breakout_period}) should not be larger than bb_period ({bb_period})"
            _get_logger().warning(error_msg)
            raise ValueError(error_msg)
        
        # INSTITUTIONAL CONSTRAINT: Exit period must be less than breakout period for proper trade management
        if exit_donchian_period >= breakout_period:
            error_msg = f"exit_donchian_period ({exit_donchian_period}) must be less than breakout_period ({breakout_period}) for proper exit timing"
            _get_logger().warning(error_msg)
            raise ValueError(error_msg)
        
        # Minimum squeeze bars validation
        if min_squeeze_bars < 2:
            error_msg = f"min_squeeze_bars ({min_squeeze_bars}) must be at least 2"
            _get_logger().warning(error_msg)
            raise ValueError(error_msg)
        
        # Position sizing validation removed - handled by CLI --contracts-per-trade parameter
        # This ensures single source of truth and institutional compliance
        
        _get_logger().debug("Bollinger Squeeze parameter validation passed")
        return True
        
    except Exception as e:
        _get_logger().error(f"Bollinger Squeeze parameter validation failed: {e}")
        _get_logger().error(f"Failed parameters: {full_params}")
        raise