"""
Bollinger Band Squeeze Indicator Calculations

Implements the TTM Squeeze methodology with Bollinger Bands, Keltner Channels,
momentum oscillator, and breakout detection for futures trading.
"""
import pandas as pd
import numpy as np


def calculate_bollinger_bands(data, period, std_dev):
    """
    Calculate Bollinger Bands using EMA for consistency with Keltner Channels.
    
    FIXED: Use EMA instead of SMA for methodological consistency across all indicators.
    This provides more responsive signals appropriate for volatility breakout strategies.
    
    Args:
        data: Price series (typically close)
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    # Use EMA for middle band (consistent with Keltner Channels)
    middle_band = data.ewm(span=period, adjust=False).mean()
    
    # Use rolling standard deviation (this remains rolling as EMA std would be complex)
    std = data.rolling(window=period).std(ddof=1)
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_keltner_channels(data, period, atr_multiplier):
    """
    Calculate Keltner Channels using ATR.
    
    Args:
        data: DataFrame with OHLC data
        period: Moving average period
        atr_multiplier: ATR multiplier for channel width
        
    Returns:
        tuple: (upper_channel, middle_channel, lower_channel)
    """
    # Calculate middle line (EMA of close)
    middle_channel = data['close'].ewm(span=period, adjust=False).mean()
    
    # Calculate ATR
    atr = calculate_atr(data, period)
    
    # Calculate channels
    upper_channel = middle_channel + (atr * atr_multiplier)
    lower_channel = middle_channel - (atr * atr_multiplier)
    
    return upper_channel, middle_channel, lower_channel


def calculate_atr(data, period):
    """
    Calculate Average True Range.
    
    Args:
        data: DataFrame with OHLC data
        period: ATR period
        
    Returns:
        pd.Series: ATR values
    """
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


def detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower):
    """
    Detect when Bollinger Bands are inside Keltner Channels (squeeze condition).
    
    Args:
        bb_upper: Bollinger Band upper
        bb_lower: Bollinger Band lower
        kc_upper: Keltner Channel upper
        kc_lower: Keltner Channel lower
        
    Returns:
        pd.Series: Boolean series indicating squeeze condition
    """
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    return squeeze


def calculate_momentum_oscillator(data, period):
    """
    Calculate momentum oscillator for TTM Squeeze.
    
    Uses Linear Regression of close price to determine momentum direction.
    
    Args:
        data: DataFrame with OHLC data
        period: Period for momentum calculation
        
    Returns:
        pd.Series: Momentum values
    """
    close = data['close']
    
    # Calculate linear regression slope over the period
    def linreg_slope(series):
        x = np.arange(len(series))
        if len(series) < 2:
            return 0
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    momentum = close.rolling(window=period).apply(linreg_slope, raw=False)
    
    return momentum


def calculate_donchian_channels(data, period):
    """
    Calculate Donchian Channels (highest high, lowest low).
    
    Args:
        data: DataFrame with OHLC data
        period: Lookback period
        
    Returns:
        tuple: (upper_channel, lower_channel)
    """
    upper_channel = data['high'].rolling(window=period).max()
    lower_channel = data['low'].rolling(window=period).min()
    
    return upper_channel, lower_channel


def calculate_squeeze_duration(squeeze_signal):
    """
    Calculate how many consecutive bars the squeeze has been active.
    
    Args:
        squeeze_signal: Boolean series of squeeze conditions
        
    Returns:
        pd.Series: Number of consecutive squeeze bars
    """
    # Group consecutive squeeze periods
    squeeze_groups = (squeeze_signal != squeeze_signal.shift(1)).cumsum()
    
    # Count bars in each group where squeeze is True
    squeeze_duration = squeeze_signal.groupby(squeeze_groups).cumsum()
    
    # Zero out duration when squeeze is False
    squeeze_duration = squeeze_duration.where(squeeze_signal, 0)
    
    return squeeze_duration


def calculate_volume_ratio(data, period):
    """
    Calculate volume ratio vs recent average.
    
    Args:
        data: DataFrame with volume data
        period: Period for average volume calculation
        
    Returns:
        pd.Series: Volume ratio
    """
    avg_volume = data['volume'].rolling(window=period).mean()
    volume_ratio = data['volume'] / avg_volume
    
    return volume_ratio.fillna(1.0)


def calculate_all_indicators(data, params):
    """
    Calculate all indicators required for the Bollinger Squeeze strategy.
    
    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters
        
    Returns:
        dict: Dictionary containing all calculated indicators
    """
    indicators = {}
    
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        data['close'], 
        params['bb_period'], 
        params['bb_std_dev']
    )
    indicators['bb_upper'] = bb_upper
    indicators['bb_middle'] = bb_middle
    indicators['bb_lower'] = bb_lower
    
    # Calculate Keltner Channels
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        data, 
        params['kc_period'], 
        params['kc_atr_multiplier']
    )
    indicators['kc_upper'] = kc_upper
    indicators['kc_middle'] = kc_middle
    indicators['kc_lower'] = kc_lower
    
    # Detect squeeze
    indicators['squeeze'] = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
    indicators['squeeze_duration'] = calculate_squeeze_duration(indicators['squeeze'])
    
    # Calculate momentum
    indicators['momentum'] = calculate_momentum_oscillator(data, params['momentum_period'])
    
    # Calculate breakout channels
    breakout_upper, breakout_lower = calculate_donchian_channels(
        data, 
        params['breakout_period']
    )
    indicators['breakout_upper'] = breakout_upper
    indicators['breakout_lower'] = breakout_lower
    
    # Calculate exit channels
    exit_upper, exit_lower = calculate_donchian_channels(
        data, 
        params['exit_donchian_period']
    )
    indicators['exit_upper'] = exit_upper
    indicators['exit_lower'] = exit_lower
    
    # Calculate ATR for risk management
    indicators['atr'] = calculate_atr(data, params['atr_period'])
    
    # Calculate trend filter using EMA for consistency
    if params['use_trend_filter']:
        # FIXED: Use EMA instead of SMA for methodological consistency
        indicators['trend_filter'] = data['close'].ewm(
            span=params['trend_filter_period'], adjust=False
        ).mean()
    
    # Calculate volume ratio
    if params['volume_filter']:
        indicators['volume_ratio'] = calculate_volume_ratio(
            data, 
            params['bb_period']  # Use same period as BB
        )
    
    return indicators