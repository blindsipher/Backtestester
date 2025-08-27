"""
Price Seed Resolver for Symbol-Aware Synthetic Data Generation
============================================================

Resolves appropriate price levels for synthetic data generation based on:
1. User-supplied override (highest priority)
2. Real market data (when available)
3. MarketSpec typical_price definitions  
4. System default fallback

This ensures synthetic data generation is symbol-agnostic and uses
realistic price levels for each futures contract.
"""

import logging
from typing import Optional
from dataclasses import dataclass

# Import market specifications
from config.system_config import TopStepMarkets
# ARCHITECTURAL FIX: Remove constants.py dependency
DEFAULT_BASE_PRICE = 6000.0  # Default futures base price

logger = logging.getLogger(__name__)


@dataclass
class PriceSeed:
    """Price seed parameters for synthetic data generation."""
    base_price: float
    volatility: float = 0.002  # Default 0.2% daily volatility


class PriceSeedResolver:
    """
    Resolves symbol-appropriate price seeds for synthetic data generation.
    
    Follows priority order:
    1. User override (explicit base_price parameter)
    2. Real market data analysis (if available)  
    3. MarketSpec typical_price field
    4. System DEFAULT_BASE_PRICE with warning
    """
    
    def __init__(self):
        """Initialize the price seed resolver."""
        self.logger = logging.getLogger(__name__)
    
    def resolve(self, symbol: str, user_override: Optional[float] = None) -> PriceSeed:
        """
        Resolve appropriate price seed for the given symbol.
        
        Args:
            symbol: Futures contract symbol (e.g., "ES", "NQ", "CL")
            user_override: Optional user-supplied base price (highest priority)
            
        Returns:
            PriceSeed with appropriate base_price and volatility
        """
        symbol_upper = symbol.upper()
        
        # Priority 1: User override
        if user_override is not None and user_override > 0:
            self.logger.info(f"Using user override price for {symbol}: ${user_override:.2f}")
            return PriceSeed(base_price=user_override)
        
        # Priority 2: Real market data (uploaded data files handle this)
        # Real market data integration not needed - users upload their own data files
        
        # Priority 3: MarketSpec typical_price
        market_spec_price = self._get_market_spec_price(symbol_upper)
        if market_spec_price:
            self.logger.info(f"Using MarketSpec price for {symbol}: ${market_spec_price:.2f}")
            return PriceSeed(base_price=market_spec_price)
        
        # Priority 4: System default with warning
        self.logger.warning(f"Unknown symbol '{symbol}' - using default price ${DEFAULT_BASE_PRICE:.2f}. "
                          f"Consider adding typical_price to MarketSpec for {symbol_upper}")
        return PriceSeed(base_price=DEFAULT_BASE_PRICE)
    
    def _get_market_spec_price(self, symbol: str) -> Optional[float]:
        """
        Get typical price from MarketSpec definition.
        
        Args:
            symbol: Symbol to look up (should be uppercase)
            
        Returns:
            Typical price if defined, None otherwise
        """
        try:
            if hasattr(TopStepMarkets, symbol):
                market_spec = getattr(TopStepMarkets, symbol)
                if hasattr(market_spec, 'typical_price') and market_spec.typical_price:
                    return float(market_spec.typical_price)
        except Exception as e:
            self.logger.debug(f"Error accessing MarketSpec for {symbol}: {e}")
        
        return None
    
    def _get_recent_market_price(self, symbol: str) -> Optional[object]:
        """
        Get recent market price statistics from real data store.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Price statistics if available, None otherwise
            
        Note: This is a placeholder for future real data integration.
        """
        # Real market data lookup not implemented - users upload their own data files
        # This method remains as placeholder for potential future integration
        return None
    
    def get_supported_symbols(self) -> list[str]:
        """
        Get list of symbols with MarketSpec definitions.
        
        Returns:
            List of supported symbol names
        """
        supported = []
        for attr_name in dir(TopStepMarkets):
            if not attr_name.startswith('_'):
                try:
                    attr = getattr(TopStepMarkets, attr_name)
                    if hasattr(attr, 'symbol'):
                        supported.append(attr.symbol)
                except Exception:
                    continue
        
        return sorted(supported)