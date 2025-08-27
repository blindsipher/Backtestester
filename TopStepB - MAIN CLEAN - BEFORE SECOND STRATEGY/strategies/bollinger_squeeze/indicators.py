"""
Bollinger Band Squeeze Indicator Calculations

Implements the TTM Squeeze methodology with Bollinger Bands, Keltner Channels,
momentum oscillator, and breakout detection for futures trading.
"""
import pandas as pd
import numpy as np
import torch


def _to_tensor(series: pd.Series, device: torch.device) -> torch.Tensor:
    """Convert pandas Series to a torch tensor on the specified device."""
    return torch.as_tensor(series.to_numpy(), dtype=torch.float32, device=device)


def _ema_tensor(data: torch.Tensor, span: int) -> torch.Tensor:
    """Simple EMA implementation using PyTorch operations."""
    alpha = 2.0 / (span + 1)
    ema = torch.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, data.numel()):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rolling_std_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(data)
    rolled = data.unfold(0, window, 1)
    std = rolled.std(dim=1, unbiased=True)
    pad = torch.full((window - 1,), float("nan"), device=data.device)
    return torch.cat([pad, std])


def _rolling_max_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    rolled = data.unfold(0, window, 1)
    maxv = rolled.max(dim=1).values
    pad = torch.full((window - 1,), float("nan"), device=data.device)
    return torch.cat([pad, maxv])


def _rolling_min_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    rolled = data.unfold(0, window, 1)
    minv = rolled.min(dim=1).values
    pad = torch.full((window - 1,), float("nan"), device=data.device)
    return torch.cat([pad, minv])


def _rolling_mean_tensor(data: torch.Tensor, window: int) -> torch.Tensor:
    rolled = data.unfold(0, window, 1)
    mean = rolled.mean(dim=1)
    pad = torch.full((window - 1,), float("nan"), device=data.device)
    return torch.cat([pad, mean])


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


# ----- GPU/Torch implementations -----

def calculate_bollinger_bands_tensor(close: torch.Tensor, period: int, std_dev: float) -> tuple:
    middle_band = _ema_tensor(close, period)
    std = _rolling_std_tensor(close, period)
    upper_band = middle_band + std * std_dev
    lower_band = middle_band - std * std_dev
    return upper_band, middle_band, lower_band


def calculate_atr_tensor(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, period: int) -> torch.Tensor:
    high_low = high - low
    prev_close = torch.cat([close[:1], close[:-1]])
    high_close = torch.abs(high - prev_close)
    low_close = torch.abs(low - prev_close)
    true_range = torch.max(torch.stack([high_low, high_close, low_close]), dim=0).values
    return _ema_tensor(true_range, period)


def calculate_keltner_channels_tensor(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor,
                                     period: int, atr_multiplier: float) -> tuple:
    middle = _ema_tensor(close, period)
    atr = calculate_atr_tensor(high, low, close, period)
    upper = middle + atr * atr_multiplier
    lower = middle - atr * atr_multiplier
    return upper, middle, lower


def detect_squeeze_tensor(bb_upper: torch.Tensor, bb_lower: torch.Tensor,
                          kc_upper: torch.Tensor, kc_lower: torch.Tensor) -> torch.Tensor:
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


def calculate_momentum_oscillator_tensor(close: torch.Tensor, period: int) -> torch.Tensor:
    if period <= 1:
        return torch.zeros_like(close)
    windows = close.unfold(0, period, 1)
    x = torch.arange(period, device=close.device, dtype=torch.float32)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    slopes = ((windows - windows.mean(dim=1, keepdim=True)) * (x - x_mean)).sum(dim=1) / denom
    pad = torch.zeros(period - 1, device=close.device)
    return torch.cat([pad, slopes])


def calculate_donchian_channels_tensor(high: torch.Tensor, low: torch.Tensor, period: int) -> tuple:
    upper = _rolling_max_tensor(high, period)
    lower = _rolling_min_tensor(low, period)
    return upper, lower


def calculate_squeeze_duration_tensor(squeeze_signal: torch.Tensor) -> torch.Tensor:
    duration = torch.zeros_like(squeeze_signal, dtype=torch.int32)
    count = 0
    for i in range(squeeze_signal.numel()):
        if bool(squeeze_signal[i]):
            count += 1
            duration[i] = count
        else:
            count = 0
            duration[i] = 0
    return duration


def calculate_volume_ratio_tensor(volume: torch.Tensor, period: int) -> torch.Tensor:
    avg_volume = _rolling_mean_tensor(volume, period)
    ratio = volume / avg_volume
    ratio[torch.isnan(ratio)] = 1.0
    return ratio


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


def calculate_all_indicators(data, params, use_gpu: bool | None = None):
    """Calculate all indicators required for the Bollinger Squeeze strategy.

    This function can utilize GPU acceleration via PyTorch when available.

    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters
        use_gpu: Force GPU usage if True/False. When None the function will
            automatically use CUDA when :func:`torch.cuda.is_available`.

    Returns:
        dict: Dictionary containing all calculated indicators as pandas Series
    """
    device = torch.device("cuda" if (use_gpu if use_gpu is not None else torch.cuda.is_available()) else "cpu")

    if device.type == "cuda":
        close = _to_tensor(data['close'], device)
        high = _to_tensor(data['high'], device)
        low = _to_tensor(data['low'], device)
        volume = _to_tensor(data['volume'], device)

        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands_tensor(close, params['bb_period'], params['bb_std_dev'])
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels_tensor(high, low, close, params['kc_period'], params['kc_atr_multiplier'])
        squeeze = detect_squeeze_tensor(bb_upper, bb_lower, kc_upper, kc_lower)
        squeeze_duration = calculate_squeeze_duration_tensor(squeeze)
        momentum = calculate_momentum_oscillator_tensor(close, params['momentum_period'])
        breakout_upper, breakout_lower = calculate_donchian_channels_tensor(high, low, params['breakout_period'])
        exit_upper, exit_lower = calculate_donchian_channels_tensor(high, low, params['exit_donchian_period'])
        atr = calculate_atr_tensor(high, low, close, params['atr_period'])

        indicators = {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'kc_upper': kc_upper,
            'kc_middle': kc_middle,
            'kc_lower': kc_lower,
            'squeeze': squeeze,
            'squeeze_duration': squeeze_duration,
            'momentum': momentum,
            'breakout_upper': breakout_upper,
            'breakout_lower': breakout_lower,
            'exit_upper': exit_upper,
            'exit_lower': exit_lower,
            'atr': atr,
        }

        if params['use_trend_filter']:
            indicators['trend_filter'] = _ema_tensor(close, params['trend_filter_period'])

        if params['volume_filter']:
            indicators['volume_ratio'] = calculate_volume_ratio_tensor(volume, params['bb_period'])

        # Convert tensors back to pandas Series for compatibility with existing pipeline
        return {k: pd.Series(v.detach().cpu().numpy(), index=data.index) for k, v in indicators.items()}

    # --- CPU fallback ---
    indicators: dict[str, pd.Series] = {}

    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        data['close'], params['bb_period'], params['bb_std_dev'])
    indicators['bb_upper'] = bb_upper
    indicators['bb_middle'] = bb_middle
    indicators['bb_lower'] = bb_lower

    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        data, params['kc_period'], params['kc_atr_multiplier'])
    indicators['kc_upper'] = kc_upper
    indicators['kc_middle'] = kc_middle
    indicators['kc_lower'] = kc_lower

    indicators['squeeze'] = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
    indicators['squeeze_duration'] = calculate_squeeze_duration(indicators['squeeze'])
    indicators['momentum'] = calculate_momentum_oscillator(data, params['momentum_period'])

    breakout_upper, breakout_lower = calculate_donchian_channels(data, params['breakout_period'])
    indicators['breakout_upper'] = breakout_upper
    indicators['breakout_lower'] = breakout_lower

    exit_upper, exit_lower = calculate_donchian_channels(data, params['exit_donchian_period'])
    indicators['exit_upper'] = exit_upper
    indicators['exit_lower'] = exit_lower

    indicators['atr'] = calculate_atr(data, params['atr_period'])

    if params['use_trend_filter']:
        indicators['trend_filter'] = data['close'].ewm(span=params['trend_filter_period'], adjust=False).mean()

    if params['volume_filter']:
        indicators['volume_ratio'] = calculate_volume_ratio(data, params['bb_period'])

    return indicators