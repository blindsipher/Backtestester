"""
Bollinger Band Squeeze Breakout Strategy Module

A volatility-based breakout strategy that identifies periods of low volatility
(squeeze) followed by explosive breakouts in the direction of momentum.
"""
from .strategy import BollingerSqueezeStrategy
from .parameters import (
    get_default_parameters,
    get_parameter_ranges, 
    validate_parameters
)

__all__ = [
    'BollingerSqueezeStrategy',
    'get_default_parameters',
    'get_parameter_ranges',
    'validate_parameters'
]