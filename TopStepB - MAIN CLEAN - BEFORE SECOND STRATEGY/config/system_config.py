"""
Complete System Configuration
TIER 1: Foundation - Centralized Configuration Hub

This module provides comprehensive system configuration for the optimization engine.
Designed primarily for TopStep trading with modular expansion capabilities.

Key Features:
- TopStep-specific account configurations with risk rules
- Complete futures market specifications with CME/CBOT/NYMEX/COMEX instruments
- Flexible commission and slippage models
- Cross-platform organized file system structure
- Validation criteria optimized for futures trading
- Modular design for easy expansion to other prop firms or personal accounts

Design Philosophy:
- TopStep-first but easily extensible
- Clean, readable configuration classes
- No complex exports - simple direct imports
- Real-world trading constraints and requirements
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from datetime import time
from pathlib import Path
import os


# =============================================================================
# RUNTIME CONFIGURATION WITH ENVIRONMENT VARIABLES
# =============================================================================

@dataclass
class RuntimeConfiguration:
    """Runtime configuration with environment variable support"""
    base_directory: Path = field(default_factory=lambda: Path(os.getenv('TOPSTEP_BASE_DIR', str(Path.home() / '.topstep_engine'))))
    initial_equity: float = field(default_factory=lambda: float(os.getenv('INITIAL_EQUITY', '10000.0')))
    default_timezone: str = field(default_factory=lambda: os.getenv('TRADING_TIMEZONE', 'America/New_York'))
    session_start: str = field(default_factory=lambda: os.getenv('SESSION_START', '18:00'))
    session_end: str = field(default_factory=lambda: os.getenv('SESSION_END', '16:10'))
    
    @classmethod
    def get_runtime_config(cls) -> 'RuntimeConfiguration':
        """Get runtime configuration instance"""
        return cls()


# =============================================================================
# DIRECTORY STRUCTURE CONFIGURATION
# =============================================================================

class SystemPaths:
    """
    Centralized cross-platform path configuration
    
    All paths are organized by strategy/market/timeframe/date for clear tracking.
    Uses user home directory by default for cross-platform compatibility.
    """
    
    # Base directory - now configurable via environment variables
    BASE_DIR = RuntimeConfiguration.get_runtime_config().base_directory
    
    # Main directories
    LOGS_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    DATA_CACHE_DIR = BASE_DIR / "data_cache"
    STRATEGY_PACKAGES_DIR = BASE_DIR / "strategy_packages"
    
    # Data cache subdirectories
    MARKET_DATA_CACHE = DATA_CACHE_DIR / "market_data"
    OPTIMIZATION_CHECKPOINTS = DATA_CACHE_DIR / "optimization_checkpoints"
    VALIDATION_CACHE = DATA_CACHE_DIR / "validation_results"
    
    # Strategy deployment
    DEPLOYED_STRATEGIES = STRATEGY_PACKAGES_DIR / "deployed_strategies"
    
    @classmethod
    def create_run_directory(cls, strategy_name: str, market: str, timeframe: str, timestamp: str) -> Dict[str, Path]:
        """
        Create organized directory structure for a specific run
        
        Returns dict with all relevant paths for the run
        """
        run_id = f"{strategy_name}_{market}_{timeframe}_{timestamp}"
        
        paths = {
            'run_logs': cls.LOGS_DIR / run_id,
            'run_results': cls.RESULTS_DIR / run_id,
            'optimization_log': cls.LOGS_DIR / run_id / "optimization_log.txt",
            'error_log': cls.LOGS_DIR / run_id / "error_log.txt", 
            'validation_log': cls.LOGS_DIR / run_id / "validation_log.txt",
            'performance_log': cls.LOGS_DIR / run_id / "performance_log.txt",
            'optimization_results': cls.RESULTS_DIR / run_id / "optimization_results.json",
            'top_strategies': cls.RESULTS_DIR / run_id / "top_25_strategies.csv",
            'validation_report': cls.RESULTS_DIR / run_id / "validation_report.html",
            'deployment_packages': cls.RESULTS_DIR / run_id / "deployment_packages"
        }
        
        # Create directories
        for path_key, path_value in paths.items():
            if path_key.endswith('_log') or path_key in ['optimization_results', 'top_strategies', 'validation_report']:
                path_value.parent.mkdir(parents=True, exist_ok=True)
            else:
                path_value.mkdir(parents=True, exist_ok=True)
        
        return paths


# =============================================================================
# MARKET SPECIFICATIONS
# =============================================================================

@dataclass
class MarketSpec:
    """
    Complete market specification for futures instruments
    
    Designed for CME Group products but extensible to other exchanges
    """
    symbol: str
    name: str
    exchange: str
    
    # Contract specifications
    tick_size: Decimal
    tick_value: Decimal
    contract_size: int
    currency: str = "USD"
    
    # Trading hours (Eastern Time - TopStep requirement)
    session_start: time = time(18, 0)  # 6:00 PM ET
    session_end: time = time(16, 10)   # 4:10 PM ET (TopStep cutoff)
    
    # Market-specific settings
    margin_requirement: Optional[Decimal] = None
    point_value: Optional[Decimal] = None
    minimum_fluctuation: Optional[str] = None
    
    # Platform availability
    available_on_t4: bool = True
    micro_contract: bool = False
    
    # Synthetic data generation support
    typical_price: Optional[float] = None  # Typical price level for synthetic data generation
    
    def __post_init__(self):
        """Calculate derived values"""
        if self.point_value is None:
            self.point_value = self.tick_value / self.tick_size
    
    # =============================================================================
    # TICK CALCULATION METHODS
    # =============================================================================
    
    def round_to_tick(self, price: Union[float, Decimal]) -> float:
        """
        Round price to nearest valid tick for this market
        
        Args:
            price: Price to round
            
        Returns:
            Price rounded to nearest valid tick
            
        Example:
            ES market (tick_size=0.25): 4387.23 → 4387.25
            NQ market (tick_size=0.25): 15234.37 → 15234.25
            YM market (tick_size=1.0): 34567.3 → 34567.0
        """
        price_decimal = Decimal(str(price))
        tick_size_decimal = self.tick_size
        
        # Round to nearest tick
        ticks = (price_decimal / tick_size_decimal).quantize(Decimal('1'), rounding='ROUND_HALF_UP')
        rounded_price = ticks * tick_size_decimal
        
        return float(rounded_price)
    
    def price_to_ticks(self, price: Union[float, Decimal], reference_price: Union[float, Decimal] = 0) -> int:
        """
        Convert price difference to tick count
        
        Args:
            price: Current price
            reference_price: Reference price (default 0 for absolute ticks)
            
        Returns:
            Number of ticks difference
            
        Example:
            ES: price_to_ticks(4387.25, 4386.75) → 2 ticks
            NQ: price_to_ticks(15234.25, 15233.75) → 2 ticks
        """
        price_diff = Decimal(str(price)) - Decimal(str(reference_price))
        ticks = price_diff / self.tick_size
        
        return int(ticks.quantize(Decimal('1'), rounding='ROUND_HALF_UP'))
    
    def ticks_to_price_change(self, ticks: int) -> float:
        """
        Convert tick count to price change
        
        Args:
            ticks: Number of ticks
            
        Returns:
            Price change in points
            
        Example:
            ES: ticks_to_price_change(4) → 1.0 (4 * 0.25)
            YM: ticks_to_price_change(3) → 3.0 (3 * 1.0)
        """
        price_change = Decimal(ticks) * self.tick_size
        return float(price_change)
    
    def calculate_tick_pnl(self, entry_price: Union[float, Decimal], exit_price: Union[float, Decimal], 
                          quantity: int = 1, side: str = 'long') -> float:
        """
        Calculate P&L based on realistic tick movement
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of contracts
            side: 'long' or 'short'
            
        Returns:
            P&L in dollars
            
        Example:
            ES long: entry=4387.25, exit=4388.00, qty=1 → $37.50 (3 ticks * $12.50)
            NQ short: entry=15234.25, exit=15233.75, qty=2 → $5.00 (2 ticks * $5.00 / 2 contracts)
        """
        # Round prices to valid ticks first
        entry_rounded = Decimal(str(self.round_to_tick(entry_price)))
        exit_rounded = Decimal(str(self.round_to_tick(exit_price)))
        
        # Calculate tick difference
        price_diff = exit_rounded - entry_rounded
        tick_count = price_diff / self.tick_size
        
        # Calculate P&L
        if side.lower() == 'long':
            pnl = tick_count * self.tick_value * Decimal(quantity)
        else:  # short
            pnl = -tick_count * self.tick_value * Decimal(quantity)
        
        return float(pnl)
    
    def validate_tick_price(self, price: Union[float, Decimal]) -> bool:
        """
        Check if price aligns with tick size
        
        Args:
            price: Price to validate
            
        Returns:
            True if price is valid for this market
            
        Example:
            ES: validate_tick_price(4387.25) → True
            ES: validate_tick_price(4387.23) → False
        """
        price_decimal = Decimal(str(price))
        remainder = price_decimal % self.tick_size
        
        # Allow small floating-point errors (within 0.1% of tick size)
        tolerance = self.tick_size * Decimal('0.001')
        
        return remainder <= tolerance or (self.tick_size - remainder) <= tolerance
    
    def get_tick_info(self) -> Dict[str, Any]:
        """
        Get comprehensive tick information for this market
        
        Returns:
            Dictionary with tick specifications and examples
        """
        return {
            'symbol': self.symbol,
            'tick_size': float(self.tick_size),
            'tick_value': float(self.tick_value),
            'point_value': float(self.point_value) if self.point_value else None,
            'contract_size': self.contract_size,
            'currency': self.currency,
            'examples': {
                'price_rounded_from_4387.23': self.round_to_tick(4387.23),
                'ticks_in_1_point': int(Decimal('1.0') / self.tick_size),
                'dollar_value_per_tick': float(self.tick_value),
                'valid_price_4387.25': self.validate_tick_price(4387.25),
                'invalid_price_4387.23': self.validate_tick_price(4387.23)
            }
        }
    
    def calculate_slippage_cost(self, price: Union[float, Decimal], slippage_ticks: int = 1, 
                               quantity: int = 1) -> float:
        """
        Calculate realistic slippage cost in dollars
        
        Args:
            price: Entry/exit price
            slippage_ticks: Number of ticks slippage (default 1)
            quantity: Number of contracts
            
        Returns:
            Slippage cost in dollars
            
        Example:
            ES: calculate_slippage_cost(4387.25, 1, 1) → $12.50
            NQ: calculate_slippage_cost(15234.25, 2, 1) → $10.00
        """
        slippage_cost = Decimal(slippage_ticks) * self.tick_value * Decimal(quantity)
        return float(slippage_cost)


# =============================================================================
# TOPSTEP FUTURES MARKETS
# =============================================================================

class TopStepMarkets:
    """
    Complete TopStep-supported futures markets with accurate specifications
    
    Based on CME Group contract specifications and TopStep documentation
    Updated with corrected tick values and complete market coverage
    """
    
    # CME Equity Index Futures
    ES = MarketSpec(
        symbol="ES", name="E-mini S&P 500", exchange="CME",
        tick_size=Decimal("0.25"), tick_value=Decimal("12.50"), contract_size=1,
        typical_price=6340.0  # December 2024 levels
    )
    
    MES = MarketSpec(
        symbol="MES", name="Micro E-mini S&P 500", exchange="CME", 
        tick_size=Decimal("0.25"), tick_value=Decimal("1.25"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    NQ = MarketSpec(
        symbol="NQ", name="E-mini NASDAQ 100", exchange="CME",
        tick_size=Decimal("0.25"), tick_value=Decimal("5.00"), contract_size=1,
        typical_price=21000.0  # Typical NASDAQ 100 levels
    )
    
    MNQ = MarketSpec(
        symbol="MNQ", name="Micro E-mini NASDAQ 100", exchange="CME",
        tick_size=Decimal("0.25"), tick_value=Decimal("0.50"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    RTY = MarketSpec(
        symbol="RTY", name="E-mini Russell 2000", exchange="CME",
        tick_size=Decimal("0.10"), tick_value=Decimal("5.00"), contract_size=1,
        typical_price=2300.0  # Typical Russell 2000 levels
    )
    
    M2K = MarketSpec(
        symbol="M2K", name="Micro E-mini Russell 2000", exchange="CME",
        tick_size=Decimal("0.10"), tick_value=Decimal("0.50"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    YM = MarketSpec(
        symbol="YM", name="Mini-DOW", exchange="CBOT",
        tick_size=Decimal("1.0"), tick_value=Decimal("5.00"), contract_size=1,
        typical_price=45000.0  # Typical DOW levels
    )
    
    MYM = MarketSpec(
        symbol="MYM", name="Micro Mini-DOW", exchange="CBOT",
        tick_size=Decimal("1.0"), tick_value=Decimal("0.50"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    NKD = MarketSpec(
        symbol="NKD", name="Nikkei 225", exchange="CME",
        tick_size=Decimal("5.0"), tick_value=Decimal("3.50"), contract_size=1
    )
    
    MBT = MarketSpec(
        symbol="MBT", name="Micro Bitcoin", exchange="CME",
        tick_size=Decimal("5.0"), tick_value=Decimal("0.50"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    MET = MarketSpec(
        symbol="MET", name="Micro Ether", exchange="CME",
        tick_size=Decimal("0.10"), tick_value=Decimal("0.10"), contract_size=1,
        available_on_t4=False, micro_contract=True
    )
    
    # CME Energy Futures
    CL = MarketSpec(
        symbol="CL", name="Crude Oil", exchange="NYMEX",
        tick_size=Decimal("0.01"), tick_value=Decimal("10.00"), contract_size=1000,
        typical_price=75.0  # Typical crude oil price range
    )
    
    QM = MarketSpec(
        symbol="QM", name="E-mini Crude Oil", exchange="NYMEX",
        tick_size=Decimal("0.025"), tick_value=Decimal("12.50"), contract_size=500
    )
    
    MCL = MarketSpec(
        symbol="MCL", name="Micro Crude Oil", exchange="NYMEX",
        tick_size=Decimal("0.01"), tick_value=Decimal("1.00"), contract_size=100,
        available_on_t4=False, micro_contract=True
    )
    
    NG = MarketSpec(
        symbol="NG", name="Natural Gas", exchange="NYMEX",
        tick_size=Decimal("0.001"), tick_value=Decimal("10.00"), contract_size=10000
    )
    
    QG = MarketSpec(
        symbol="QG", name="E-mini Natural Gas", exchange="NYMEX",
        tick_size=Decimal("0.0001"), tick_value=Decimal("1.00"), contract_size=10000
    )
    
    MNG = MarketSpec(
        symbol="MNG", name="Micro Natural Gas", exchange="NYMEX",
        tick_size=Decimal("0.001"), tick_value=Decimal("1.00"), contract_size=1000,
        available_on_t4=False, micro_contract=True
    )
    
    RB = MarketSpec(
        symbol="RB", name="RBOB Gasoline", exchange="NYMEX",
        tick_size=Decimal("0.0001"), tick_value=Decimal("4.20"), contract_size=42000
    )
    
    HO = MarketSpec(
        symbol="HO", name="Heating Oil", exchange="NYMEX",
        tick_size=Decimal("0.0001"), tick_value=Decimal("4.20"), contract_size=42000
    )
    
    PL = MarketSpec(
        symbol="PL", name="Platinum", exchange="NYMEX",
        tick_size=Decimal("0.10"), tick_value=Decimal("10.00"), contract_size=100
    )
    
    # CME Metals Futures
    GC = MarketSpec(
        symbol="GC", name="Gold", exchange="COMEX",
        tick_size=Decimal("0.10"), tick_value=Decimal("10.00"), contract_size=100
    )
    
    MGC = MarketSpec(
        symbol="MGC", name="Micro Gold", exchange="COMEX",
        tick_size=Decimal("0.10"), tick_value=Decimal("1.00"), contract_size=10,
        available_on_t4=False, micro_contract=True
    )
    
    SI = MarketSpec(
        symbol="SI", name="Silver", exchange="COMEX",
        tick_size=Decimal("0.005"), tick_value=Decimal("25.00"), contract_size=5000
    )
    
    SIL = MarketSpec(
        symbol="SIL", name="Micro Silver", exchange="COMEX",
        tick_size=Decimal("0.005"), tick_value=Decimal("2.50"), contract_size=500,
        available_on_t4=False, micro_contract=True
    )
    
    HG = MarketSpec(
        symbol="HG", name="Copper", exchange="COMEX",
        tick_size=Decimal("0.0005"), tick_value=Decimal("12.50"), contract_size=25000
    )
    
    MHG = MarketSpec(
        symbol="MHG", name="Micro Copper", exchange="COMEX",
        tick_size=Decimal("0.0005"), tick_value=Decimal("1.25"), contract_size=2500,
        available_on_t4=False, micro_contract=True
    )
    
    # CME FX Futures
    FX_6A = MarketSpec(
        symbol="6A", name="Australian Dollar", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("10.00"), contract_size=100000
    )
    
    FX_6B = MarketSpec(
        symbol="6B", name="British Pound", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("6.25"), contract_size=62500
    )
    
    FX_6C = MarketSpec(
        symbol="6C", name="Canadian Dollar", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("10.00"), contract_size=100000
    )
    
    FX_6E = MarketSpec(
        symbol="6E", name="Euro FX", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("12.50"), contract_size=125000
    )
    
    FX_6J = MarketSpec(
        symbol="6J", name="Japanese Yen", exchange="CME",
        tick_size=Decimal("0.000001"), tick_value=Decimal("12.50"), contract_size=12500000
    )
    
    FX_6S = MarketSpec(
        symbol="6S", name="Swiss Franc", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("12.50"), contract_size=125000
    )
    
    E7 = MarketSpec(
        symbol="E7", name="E-mini Euro FX", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("6.25"), contract_size=62500
    )
    
    M6E = MarketSpec(
        symbol="M6E", name="Micro Euro FX", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("1.25"), contract_size=12500,
        available_on_t4=False, micro_contract=True
    )
    
    M6A = MarketSpec(
        symbol="M6A", name="Micro AUD/USD", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("1.00"), contract_size=10000,
        available_on_t4=False, micro_contract=True
    )
    
    M6B = MarketSpec(
        symbol="M6B", name="Micro GBP/USD", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("0.625"), contract_size=6250,
        available_on_t4=False, micro_contract=True
    )
    
    FX_6M = MarketSpec(
        symbol="6M", name="Mexican Peso", exchange="CME",
        tick_size=Decimal("0.000025"), tick_value=Decimal("12.50"), contract_size=500000
    )
    
    FX_6N = MarketSpec(
        symbol="6N", name="New Zealand Dollar", exchange="CME",
        tick_size=Decimal("0.0001"), tick_value=Decimal("10.00"), contract_size=100000
    )
    
    # CME Agricultural Futures
    HE = MarketSpec(
        symbol="HE", name="Lean Hogs", exchange="CME",
        tick_size=Decimal("0.00025"), tick_value=Decimal("10.00"), contract_size=40000,
        session_start=time(9, 30), session_end=time(14, 5)  # CME Ag hours (ET)
    )
    
    LE = MarketSpec(
        symbol="LE", name="Live Cattle", exchange="CME",
        tick_size=Decimal("0.00025"), tick_value=Decimal("10.00"), contract_size=40000,
        session_start=time(9, 30), session_end=time(14, 5)  # CME Ag hours (ET)
    )
    
    # CBOT Agricultural Futures
    ZC = MarketSpec(
        symbol="ZC", name="Corn", exchange="CBOT",
        tick_size=Decimal("0.25"), tick_value=Decimal("12.50"), contract_size=5000,
        session_start=time(20, 0), session_end=time(14, 20)  # CBOT hours (ET)
    )
    
    ZW = MarketSpec(
        symbol="ZW", name="Wheat", exchange="CBOT",
        tick_size=Decimal("0.25"), tick_value=Decimal("12.50"), contract_size=5000,
        session_start=time(20, 0), session_end=time(14, 20)  # CBOT hours (ET)
    )
    
    ZS = MarketSpec(
        symbol="ZS", name="Soybeans", exchange="CBOT", 
        tick_size=Decimal("0.25"), tick_value=Decimal("12.50"), contract_size=5000,
        session_start=time(20, 0), session_end=time(14, 20)  # CBOT hours (ET)
    )
    
    ZM = MarketSpec(
        symbol="ZM", name="Soybean Meal", exchange="CBOT",
        tick_size=Decimal("0.10"), tick_value=Decimal("10.00"), contract_size=100,
        session_start=time(20, 0), session_end=time(14, 20)  # CBOT hours (ET)
    )
    
    ZL = MarketSpec(
        symbol="ZL", name="Soybean Oil", exchange="CBOT",
        tick_size=Decimal("0.0001"), tick_value=Decimal("11.20"), contract_size=112000,
        session_start=time(20, 0), session_end=time(14, 20)  # CBOT hours (ET)
    )
    
    # CBOT Interest Rate/Bond Futures
    ZT = MarketSpec(
        symbol="ZT", name="2-Year Note", exchange="CBOT",
        tick_size=Decimal("0.0015625"), tick_value=Decimal("15.625"), contract_size=200000  # 1/64 point
    )
    
    ZF = MarketSpec(
        symbol="ZF", name="5-Year Note", exchange="CBOT",
        tick_size=Decimal("0.0015625"), tick_value=Decimal("31.25"), contract_size=100000  # 1/64 point
    )
    
    ZN = MarketSpec(
        symbol="ZN", name="10-Year Note", exchange="CBOT",
        tick_size=Decimal("0.0003125"), tick_value=Decimal("31.25"), contract_size=100000  # 1/32 point
    )
    
    TN = MarketSpec(
        symbol="TN", name="10-Year Ultra Note", exchange="CBOT",
        tick_size=Decimal("0.0003125"), tick_value=Decimal("31.25"), contract_size=100000  # 1/32 point
    )
    
    ZB = MarketSpec(
        symbol="ZB", name="30-Year Bond", exchange="CBOT",
        tick_size=Decimal("0.0003125"), tick_value=Decimal("31.25"), contract_size=100000  # 1/32 point
    )
    
    UB = MarketSpec(
        symbol="UB", name="Ultra Bond", exchange="CBOT",
        tick_size=Decimal("0.0015625"), tick_value=Decimal("15.625"), contract_size=100000  # 1/64 point
    )
    
    @classmethod
    def get_all_markets(cls) -> Dict[str, MarketSpec]:
        """Get all defined markets as a dictionary"""
        markets = {}
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, MarketSpec):
                markets[attr_value.symbol] = attr_value
        return markets
    
    @classmethod
    def get_major_markets(cls) -> Dict[str, MarketSpec]:
        """Get the most commonly traded markets"""
        return {
            "ES": cls.ES,
            "NQ": cls.NQ, 
            "RTY": cls.RTY,
            "YM": cls.YM,
            "CL": cls.CL,
            "GC": cls.GC
        }


# =============================================================================
# ACCOUNT CONFIGURATIONS
# =============================================================================

@dataclass
class AccountConfig:
    """
    Account configuration with risk management rules
    
    Designed for prop firm accounts but extensible to personal accounts
    """
    name: str
    account_size: Decimal
    
    # TopStep-specific risk rules
    profit_target: Decimal
    daily_loss_limit: Decimal
    trailing_max_drawdown: Decimal
    max_position_size: int
    min_trading_days: int = 2
    
    # Trading constraints
    must_close_by: time = time(16, 10)  # 4:10 PM ET
    can_resume_at: time = time(18, 0)   # 6:00 PM ET
    weekend_resume: time = time(18, 0)  # Sunday 6:00 PM ET
    
    # Scaling rules (TopStep specific)
    scaling_plan: Dict[str, int] = field(default_factory=dict)
    
    # Account type
    is_prop_firm: bool = True
    firm_name: str = "TopStep"
    
    # ===================================================================
    # TOPSTEP COMPLIANCE FIX - Add missing starting_balance attribute
    # ===================================================================
    
    @property
    def starting_balance(self) -> Decimal:
        """Alias for account_size - required by validation modules"""
        return self.account_size
    
    @starting_balance.setter
    def starting_balance(self, value: Decimal):
        """Allow setting starting_balance (updates account_size)"""
        self.account_size = value


class TopStepAccounts:
    """
    TopStep Trading Combine account configurations
    
    Based on current TopStep specifications as of 2024
    """
    
    TRADING_COMBINE_50K = AccountConfig(
        name="TopStep $50K Trading Combine",
        account_size=Decimal("50000"),
        profit_target=Decimal("3000"),
        daily_loss_limit=Decimal("-1000"),
        trailing_max_drawdown=Decimal("-2000"),
        max_position_size=5,
        scaling_plan={
            "base": 5,
            "after_1000_profit": 7,
            "after_2000_profit": 10
        }
    )
    
    TRADING_COMBINE_100K = AccountConfig(
        name="TopStep $100K Trading Combine", 
        account_size=Decimal("100000"),
        profit_target=Decimal("6000"),
        daily_loss_limit=Decimal("-2000"),
        trailing_max_drawdown=Decimal("-3000"),
        max_position_size=10,
        scaling_plan={
            "base": 10,
            "after_2000_profit": 15,
            "after_4000_profit": 20
        }
    )
    
    TRADING_COMBINE_150K = AccountConfig(
        name="TopStep $150K Trading Combine",
        account_size=Decimal("150000"), 
        profit_target=Decimal("9000"),
        daily_loss_limit=Decimal("-3000"),
        trailing_max_drawdown=Decimal("-4500"),
        max_position_size=15,
        scaling_plan={
            "base": 15,
            "after_3000_profit": 20,
            "after_6000_profit": 25
        }
    )
    
    # Express Funded Accounts (same specs but faster qualification)
    EXPRESS_FUNDED_50K = AccountConfig(
        name="TopStep $50K Express Funded",
        account_size=Decimal("50000"),
        profit_target=Decimal("3000"),
        daily_loss_limit=Decimal("-1000"),
        trailing_max_drawdown=Decimal("-2000"),
        max_position_size=5,
        min_trading_days=2,
        scaling_plan={
            "base": 5,
            "after_1000_profit": 7,
            "after_2000_profit": 10
        }
    )
    
    @classmethod
    def get_default_account(cls) -> AccountConfig:
        """Get default account for testing"""
        return cls.TRADING_COMBINE_50K


# =============================================================================
# COMMISSION AND SLIPPAGE MODELS
# =============================================================================

@dataclass
class CommissionModel:
    """
    Commission structure for different brokers/platforms
    """
    name: str
    commission_per_side: Decimal
    commission_per_round_trip: Decimal
    
    # Additional fees
    exchange_fees: Decimal = Decimal("0")
    regulatory_fees: Decimal = Decimal("0")
    platform_fees: Decimal = Decimal("0")
    
    def total_round_trip_cost(self) -> Decimal:
        """Calculate total cost per round trip"""
        return (self.commission_per_round_trip + 
                (self.exchange_fees * 2) + 
                (self.regulatory_fees * 2) + 
                self.platform_fees)


@dataclass  
class SlippageModel:
    """
    Market impact and slippage modeling
    """
    name: str
    
    # Slippage in ticks
    market_order_slippage_ticks: Decimal = Decimal("0.5")
    limit_order_slippage_ticks: Decimal = Decimal("0.1")
    
    # Latency simulation
    order_latency_ms: int = 50
    fill_latency_ms: int = 25
    
    # Market impact (for larger sizes)
    impact_per_contract: Decimal = Decimal("0.1")  # ticks per contract
    max_impact_ticks: Decimal = Decimal("2.0")


class ExecutionModels:
    """
    Commission and slippage models for different scenarios
    """
    
    # Commission models
    TOPSTEP_COMMISSION = CommissionModel(
        name="TopStep Standard",
        commission_per_side=Decimal("2.50"),
        commission_per_round_trip=Decimal("5.00"),
        exchange_fees=Decimal("0.85"),
        regulatory_fees=Decimal("0.02")
    )
    
    GENERIC_FUTURES = CommissionModel(
        name="Generic Futures Broker",
        commission_per_side=Decimal("2.50"), 
        commission_per_round_trip=Decimal("5.00"),
        exchange_fees=Decimal("1.00"),
        regulatory_fees=Decimal("0.05")
    )
    
    LOW_COST_BROKER = CommissionModel(
        name="Low Cost Futures",
        commission_per_side=Decimal("1.00"),
        commission_per_round_trip=Decimal("2.00"),
        exchange_fees=Decimal("0.85"),
        regulatory_fees=Decimal("0.02")
    )
    
    # Slippage models
    CONSERVATIVE_SLIPPAGE = SlippageModel(
        name="Conservative Slippage",
        market_order_slippage_ticks=Decimal("1.0"),
        limit_order_slippage_ticks=Decimal("0.2"),
        order_latency_ms=100,
        fill_latency_ms=50
    )
    
    REALISTIC_SLIPPAGE = SlippageModel(
        name="Realistic Slippage",
        market_order_slippage_ticks=Decimal("0.5"),
        limit_order_slippage_ticks=Decimal("0.1"),
        order_latency_ms=50,
        fill_latency_ms=25
    )
    
    OPTIMISTIC_SLIPPAGE = SlippageModel(
        name="Optimistic Slippage", 
        market_order_slippage_ticks=Decimal("0.25"),
        limit_order_slippage_ticks=Decimal("0.05"),
        order_latency_ms=25,
        fill_latency_ms=10
    )


# =============================================================================
# OPTIMIZATION DEFAULTS
# =============================================================================

class OptimizationDefaults:
    """
    Default settings for optimization runs
    
    Balanced for thoroughness and reasonable computation time
    """
    
    # Core optimization settings
    BRUTE_FORCE_EVALUATIONS = 2000
    BAYESIAN_CALLS = 400
    TOP_STRATEGIES_FOR_INJECTION = 25
    
    # System preferences
    DEFAULT_ACCOUNT = TopStepAccounts.TRADING_COMBINE_50K
    DEFAULT_COMMISSION_MODEL = ExecutionModels.TOPSTEP_COMMISSION
    DEFAULT_SLIPPAGE_MODEL = ExecutionModels.REALISTIC_SLIPPAGE
    
    # Performance constraints
    MAX_OPTIMIZATION_TIME_HOURS = 8
    MEMORY_CONSERVATIVE_MODE = False
    USE_PARALLEL_PROCESSING = True
    
    # Validation settings
    ENABLE_OUT_OF_SAMPLE_VALIDATION = True
    ENABLE_MONTE_CARLO_VALIDATION = True
    ENABLE_WALK_FORWARD_VALIDATION = True


# =============================================================================
# TIMEFRAME SPECIFICATIONS
# =============================================================================

class SupportedTimeframes:
    """
    Supported timeframes for analysis
    
    No default timeframe - user must specify in main_runner
    """
    
    SUPPORTED = [
        "1m", "5m", "10m", "15m", "20m", "30m", "45m", "1h", "4h", "1d"
    ]
    
    INTRADAY = [
        "1m", "5m", "10m", "15m", "20m", "30m", "45m", "1h"
    ]
    
    COMMON_FOR_SCALPING = ["1m", "5m"]
    COMMON_FOR_SWING = ["15m", "30m", "1h"]
    COMMON_FOR_POSITION = ["4h", "1d"]
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """Validate if timeframe is supported"""
        return timeframe in cls.SUPPORTED
    
    @classmethod
    def get_timeframe_minutes(cls, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            "1m": 1, "5m": 5, "10m": 10, "15m": 15, "20m": 20,
            "30m": 30, "45m": 45, "1h": 60, "4h": 240, "1d": 1440
        }
        return mapping.get(timeframe, 0)


# =============================================================================
# VALIDATION CRITERIA
# =============================================================================

@dataclass
class ValidationCriteria:
    """
    Performance validation criteria for strategy acceptance
    
    Optimized for futures trading with realistic expectations
    """
    
    # Risk-adjusted performance
    minimum_sharpe_ratio: float = 1.2
    minimum_sortino_ratio: float = 1.5
    minimum_gain_to_pain_ratio: float = 1.0
    
    # Win/loss metrics
    minimum_win_rate: float = 0.42  # 42%
    minimum_profit_factor: float = 1.4
    
    # Statistical significance
    minimum_number_of_trades: int = 100
    maximum_monte_carlo_failure_rate: float = 0.05  # 5%
    
    # Overfitting protection
    maximum_out_of_sample_degradation: float = 0.30  # 30%
    
    # Robustness requirements
    require_regime_consistency: bool = True
    require_walk_forward_validity: bool = True
    
    # Walk-forward settings
    walk_forward_window_months: int = 6
    walk_forward_step_months: int = 1
    
    # ===================================================================
    # TOPSTEP COMPLIANCE FIX - Add missing minimum_trades attribute
    # ===================================================================
    
    @property
    def minimum_trades(self) -> int:
        """Alias for minimum_number_of_trades - required by some modules"""
        return self.minimum_number_of_trades
    
    @minimum_trades.setter
    def minimum_trades(self, value: int):
        """Allow setting minimum_trades (updates minimum_number_of_trades)"""
        self.minimum_number_of_trades = value
    
    def meets_criteria(self, metrics: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        Check if strategy metrics meet validation criteria
        
        Returns (passed, list_of_failures)
        """
        failures = []
        
        if metrics.get('sharpe_ratio', 0) < self.minimum_sharpe_ratio:
            failures.append(f"Sharpe ratio {metrics.get('sharpe_ratio', 0):.2f} < {self.minimum_sharpe_ratio}")
        
        if metrics.get('sortino_ratio', 0) < self.minimum_sortino_ratio:
            failures.append(f"Sortino ratio {metrics.get('sortino_ratio', 0):.2f} < {self.minimum_sortino_ratio}")
        
        if metrics.get('win_rate', 0) < self.minimum_win_rate:
            failures.append(f"Win rate {metrics.get('win_rate', 0):.1%} < {self.minimum_win_rate:.1%}")
        
        if metrics.get('profit_factor', 0) < self.minimum_profit_factor:
            failures.append(f"Profit factor {metrics.get('profit_factor', 0):.2f} < {self.minimum_profit_factor}")
        
        if metrics.get('number_of_trades', 0) < self.minimum_number_of_trades:
            failures.append(f"Number of trades {metrics.get('number_of_trades', 0)} < {self.minimum_number_of_trades}")
        
        return len(failures) == 0, failures


# =============================================================================
# PLATFORM INTEGRATION SETTINGS
# =============================================================================

class PlatformSettings:
    """
    Platform-specific settings and configurations
    """
    
    # NinjaTrader integration
    NINJATRADER_ENABLED = True
    NINJATRADER_TCP_PORT = 8080
    NINJATRADER_TCP_HOST = "localhost"
    
    # Data feed settings
    PREFERRED_DATA_FEED = "Rithmic"
    BACKUP_DATA_FEED = "Kinetick"
    
    # Real-time data requirements
    REAL_TIME_DATA_REQUIRED = False  # For optimization, historical is sufficient
    
    # Export formats
    STRATEGY_EXPORT_FORMATS = ["py", "cs", "json"]  # Python, C# (NinjaScript), JSON
    
    # Deployment preferences
    AUTO_DEPLOY_TO_PLATFORM = False
    REQUIRE_MANUAL_DEPLOYMENT_APPROVAL = True





# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_system_config() -> tuple[bool, List[str]]:
    """
    Validate system configuration for consistency and completeness
    
    Returns (is_valid, list_of_issues)
    """
    issues = []
    
    # Validate paths exist
    try:
        SystemPaths.BASE_DIR.mkdir(parents=True, exist_ok=True)
        SystemPaths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SystemPaths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create base directories: {e}")
    
    # Validate market specs
    markets = TopStepMarkets.get_all_markets()
    if len(markets) == 0:
        issues.append("No markets defined")
    
    # Validate account configs - ADD NULL CHECKS
    try:
        default_account = TopStepAccounts.get_default_account()
        
        # Check if account exists
        if default_account is None:
            issues.append("Default account is None")
        # Check if account_size exists and is valid
        elif default_account.account_size is None:
            issues.append("Default account size is None")
        elif default_account.account_size <= 0:
            issues.append("Invalid default account size")
            
    except Exception as e:
        issues.append(f"Default account configuration error: {e}")
    
    # Validate timeframes
    if len(SupportedTimeframes.SUPPORTED) == 0:
        issues.append("No timeframes defined")
    
    return len(issues) == 0, issues


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def get_market_spec(symbol: str) -> Optional[MarketSpec]:
    """Get market specification by symbol"""
    markets = TopStepMarkets.get_all_markets()
    return markets.get(symbol.upper())


def get_default_optimization_config() -> Dict[str, Any]:
    """Get default optimization configuration"""
    return {
        'brute_force_evaluations': OptimizationDefaults.BRUTE_FORCE_EVALUATIONS,
        'bayesian_calls': OptimizationDefaults.BAYESIAN_CALLS,
        'top_strategies': OptimizationDefaults.TOP_STRATEGIES_FOR_INJECTION,
        'account': OptimizationDefaults.DEFAULT_ACCOUNT,
        'commission_model': OptimizationDefaults.DEFAULT_COMMISSION_MODEL,
        'slippage_model': OptimizationDefaults.DEFAULT_SLIPPAGE_MODEL
    }


def create_run_paths(strategy_name: str, market: str, timeframe: str, timestamp: str) -> Dict[str, Path]:
    """Create organized directory structure for a strategy run"""
    return SystemPaths.create_run_directory(strategy_name, market, timeframe, timestamp)


def get_validation_criteria() -> ValidationCriteria:
    """Get default validation criteria"""
    return ValidationCriteria()

# =============================================================================
# VALIDATION BRIDGE FUNCTIONS (ADD TO THE END OF YOUR EXISTING system_config.py)
# =============================================================================

def configure_for_account(account_type: str, verbose: bool = True) -> AccountConfig:
    """
    Configure account settings by type name (needed by validation modules)
    
    Args:
        account_type: "topstep_50k", "topstep_100k", "topstep_150k", etc.
        verbose: Whether to print configuration info
        
    Returns:
        AccountConfig instance
    """
    account_map = {
        "topstep_50k": TopStepAccounts.TRADING_COMBINE_50K,
        "topstep_100k": TopStepAccounts.TRADING_COMBINE_100K, 
        "topstep_150k": TopStepAccounts.TRADING_COMBINE_150K,
        "express_50k": TopStepAccounts.EXPRESS_FUNDED_50K
    }
    
    account = account_map.get(account_type.lower(), TopStepAccounts.TRADING_COMBINE_50K)
    
    if verbose:
        print(f"Configured account: {account.name}")
    
    return account


def configure_for_market(symbol: str, verbose: bool = True) -> MarketSpec:
    """
    Configure market specifications by symbol (needed by validation modules)
    
    Args:
        symbol: Market symbol like "ES", "NQ", "MNQ", etc.
        verbose: Whether to print configuration info
        
    Returns:
        MarketSpec instance with dictionary-like access
    """
    market = get_market_spec(symbol)
    
    if market is None:
        # Default to MNQ if symbol not found
        market = TopStepMarkets.MNQ
        if verbose:
            print(f"Market '{symbol}' not found, defaulting to {market.symbol}")
    elif verbose:
        print(f"Configured market: {market.symbol} - {market.name}")
    
    # Add dictionary-like access for validation modules
    if not hasattr(market, '__getitem__'):
        def __getitem__(self, key):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(f"'{key}' not found in market spec")
        
        def get(self, key, default=None):
            return getattr(self, key, default)
        
        market.__class__.__getitem__ = __getitem__
        market.__class__.get = get
    
    return market


# =============================================================================
# SIMPLE BACKTEST ENGINE (needed by validation modules)
# =============================================================================

@dataclass
class BacktestResult:
    """Simple backtest result container"""
    metrics: Dict[str, float]
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    strategy_name: str = "Unknown"
    
    def __post_init__(self):
        """Convert trades to DataFrame if needed"""
        if self.trades and isinstance(self.trades, list):
            import pandas as pd
            self.trades = pd.DataFrame(self.trades)




# =============================================================================
# TRADING CONFIG - UNIFIED CONFIGURATION INTERFACE
# =============================================================================

@dataclass
class TradingConfig:
    """
    Consolidated configuration for a trading run.
    
    This class unifies market specifications, account configuration,
    execution models, and strategy settings into a single interface.
    """
    market_spec: MarketSpec
    account_config: AccountConfig
    commission_model: CommissionModel
    slippage_model: SlippageModel
    symbol: str
    timeframe: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.timeframe:
            raise ValueError("Timeframe cannot be empty")


def create_trading_config(
    symbol: str,
    timeframe: str,
    account_type: str = "topstep_50k",
    commission_model: Optional[CommissionModel] = None,
    slippage_model: Optional[SlippageModel] = None
) -> TradingConfig:
    """
    Factory function to create a TradingConfig instance.
    
    Args:
        symbol: Trading symbol (e.g., "ES", "NQ", "YM")
        timeframe: Chart timeframe (e.g., "1h", "5m", "1d")
        account_type: Account type from TopStepAccounts
        commission_model: Optional commission model (defaults to TopStep)
        slippage_model: Optional slippage model (defaults to realistic)
        
    Returns:
        Configured TradingConfig instance
        
    Raises:
        ValueError: If symbol or account_type not found
    """
    # Get market specification
    market_spec = TopStepMarkets.get_all_markets().get(symbol.upper())
    if not market_spec:
        raise ValueError(f"Market spec for '{symbol}' not found. Available: {list(TopStepMarkets.get_all_markets().keys())}")
    
    # Get account configuration
    account_config = None
    if hasattr(TopStepAccounts, account_type.upper()):
        account_config = getattr(TopStepAccounts, account_type.upper())
    else:
        # Try with underscores removed for names like "TRADING_COMBINE_50K"
        clean_name = account_type.replace("_", "").upper()
        account_config = configure_for_account(clean_name, verbose=False)
    
    if not account_config:
        raise ValueError(f"Account config for '{account_type}' not found")
    
    # Use defaults if not provided
    if commission_model is None:
        commission_model = ExecutionModels.TOPSTEP_COMMISSION
    if slippage_model is None:
        slippage_model = ExecutionModels.REALISTIC_SLIPPAGE
    
    return TradingConfig(
        market_spec=market_spec,
        account_config=account_config,
        commission_model=commission_model,
        slippage_model=slippage_model,
        symbol=symbol.upper(),
        timeframe=timeframe
    )


# =============================================================================
# MODULE TEST FUNCTION
# =============================================================================

def test_system_config():
    """Test system configuration"""
    print("Testing System Configuration...")
    
    # Test directory creation
    try:
        SystemPaths.create_run_directory("test_strategy", "ES", "5m", "2024-01-01_12-00")
        print("Directory creation successful")
    except Exception as e:
        print(f"Directory creation failed: {e}")
    
    # Test market specs
    try:
        es_spec = TopStepMarkets.ES
        print(f"Market spec: {es_spec.symbol} - {es_spec.name}")
    except Exception as e:
        print(f"Market spec failed: {e}")
    
    # Test account config
    try:
        account = TopStepAccounts.get_default_account()
        print(f"Account config: {account.name} - ${account.account_size}")
    except Exception as e:
        print(f"Account config failed: {e}")
    
    # Test validation
    is_valid, issues = validate_system_config()
    if is_valid:
        print("System configuration validation passed")
    else:
        print(f"System configuration issues: {issues}")
    
    print("System configuration test complete")


if __name__ == "__main__":
    test_system_config()