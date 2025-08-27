"""
Configuration Module
TIER 1: Foundation - Centralized Configuration System

This module provides clean, organized access to all system configurations.
Designed for TopStep trading with modular expansion capabilities.

Key Features:
- Simple, direct imports for all configuration classes
- TopStep-specific account and market configurations
- Cross-platform organized file system structure
- Commission and slippage models for realistic simulation
- Validation criteria optimized for futures trading
- Unified backtesting system for validation support
- Easy expansion for other prop firms or personal accounts

Module Status: COMPLETE - Configuration Layer Ready

Quick Start Examples:
===================

# Import main configuration classes
from config import TopStepMarkets, TopStepAccounts, OptimizationDefaults

# Get market specifications
es_market = TopStepMarkets.ES
nq_market = TopStepMarkets.NQ

# Get account configurations
account_50k = TopStepAccounts.TRADING_COMBINE_50K
account_100k = TopStepAccounts.TRADING_COMBINE_100K

# Use strategy backtesting for validation
from config import configure_for_account, configure_for_market
account = configure_for_account("topstep_50k")
market = configure_for_market("MNQ")
result = strategy.backtest(data)

# Get default optimization settings
defaults = OptimizationDefaults
print(f"Default brute force: {defaults.BRUTE_FORCE_EVALUATIONS}")
print(f"Default Bayesian: {defaults.BAYESIAN_CALLS}")

# Create organized run directories
from config import create_run_paths
paths = create_run_paths("momentum", "ES", "5m", "2024-01-15_14-30")

# Get execution models
from config import ExecutionModels
commission = ExecutionModels.TOPSTEP_COMMISSION
slippage = ExecutionModels.REALISTIC_SLIPPAGE

# Validate system configuration
from config import validate_system_config
is_valid, issues = validate_system_config()
"""

from typing import Dict, Any

# Import all configuration classes and functions from system_config
from .system_config import (
    # Core configuration classes
    MarketSpec,
    AccountConfig,
    CommissionModel,
    SlippageModel,
    ValidationCriteria,
    
    # Market specifications
    TopStepMarkets,
    
    # Account configurations  
    TopStepAccounts,
    
    # Execution models
    ExecutionModels,
    
    # System defaults
    OptimizationDefaults,
    SupportedTimeframes,
    SystemPaths,
    PlatformSettings,
    
    # Utility functions
    get_market_spec,
    get_default_optimization_config,
    create_run_paths,
    get_validation_criteria,
    validate_system_config,
    test_system_config,
    
    # NEW: Validation bridge functions
    configure_for_account,
    configure_for_market,
    BacktestResult
)

# Direct access to individual markets (commonly used)
ES = TopStepMarkets.ES
MES = TopStepMarkets.MES
NQ = TopStepMarkets.NQ
MNQ = TopStepMarkets.MNQ
RTY = TopStepMarkets.RTY
M2K = TopStepMarkets.M2K
YM = TopStepMarkets.YM
MYM = TopStepMarkets.MYM
NKD = TopStepMarkets.NKD
MBT = TopStepMarkets.MBT
MET = TopStepMarkets.MET

# Energy markets
CL = TopStepMarkets.CL
QM = TopStepMarkets.QM
MCL = TopStepMarkets.MCL
NG = TopStepMarkets.NG
QG = TopStepMarkets.QG
MNG = TopStepMarkets.MNG
RB = TopStepMarkets.RB
HO = TopStepMarkets.HO
PL = TopStepMarkets.PL

# Metals markets
GC = TopStepMarkets.GC
MGC = TopStepMarkets.MGC
SI = TopStepMarkets.SI
SIL = TopStepMarkets.SIL
HG = TopStepMarkets.HG
MHG = TopStepMarkets.MHG

# Agricultural markets
ZC = TopStepMarkets.ZC
ZW = TopStepMarkets.ZW
ZS = TopStepMarkets.ZS
ZM = TopStepMarkets.ZM
ZL = TopStepMarkets.ZL
HE = TopStepMarkets.HE
LE = TopStepMarkets.LE

# Interest rate/bond markets
ZT = TopStepMarkets.ZT
ZF = TopStepMarkets.ZF
ZN = TopStepMarkets.ZN
TN = TopStepMarkets.TN
ZB = TopStepMarkets.ZB
UB = TopStepMarkets.UB

# Direct access to account configurations
TRADING_COMBINE_50K = TopStepAccounts.TRADING_COMBINE_50K
TRADING_COMBINE_100K = TopStepAccounts.TRADING_COMBINE_100K
TRADING_COMBINE_150K = TopStepAccounts.TRADING_COMBINE_150K
EXPRESS_FUNDED_50K = TopStepAccounts.EXPRESS_FUNDED_50K

# Direct access to execution models
TOPSTEP_COMMISSION = ExecutionModels.TOPSTEP_COMMISSION
GENERIC_FUTURES = ExecutionModels.GENERIC_FUTURES
LOW_COST_BROKER = ExecutionModels.LOW_COST_BROKER
CONSERVATIVE_SLIPPAGE = ExecutionModels.CONSERVATIVE_SLIPPAGE
REALISTIC_SLIPPAGE = ExecutionModels.REALISTIC_SLIPPAGE
OPTIMISTIC_SLIPPAGE = ExecutionModels.OPTIMISTIC_SLIPPAGE

# Module metadata
__version__ = "1.0.0"
__author__ = "TopStep Optimization Engine"

# Configuration status tracking
SYSTEM_CONFIG_AVAILABLE = True
MARKETS_LOADED = len(TopStepMarkets.get_all_markets())
ACCOUNTS_CONFIGURED = 5  # Number of TopStep account types
EXECUTION_MODELS_AVAILABLE = 6  # Commission + slippage models
VALIDATION_BRIDGE_AVAILABLE = True  # NEW: Validation bridge functions available

def get_config_status() -> Dict[str, Any]:
    """
    Get comprehensive configuration module status
    
    Returns:
        Dictionary with configuration status and capabilities
    """
    try:
        # Test system configuration
        is_valid, issues = validate_system_config()
        
        # Get market and account counts
        all_markets = TopStepMarkets.get_all_markets()
        major_markets = TopStepMarkets.get_major_markets()
        
        status = {
            'module_name': 'config',
            'version': __version__,
            'system_config_valid': is_valid,
            'configuration_issues': issues,
            
            # Market coverage
            'total_markets': len(all_markets),
            'major_markets': len(major_markets),
            'markets_available': list(all_markets.keys()),
            'major_markets_list': list(major_markets.keys()),
            
            # Account configurations
            'account_types_configured': ACCOUNTS_CONFIGURED,
            'default_account': TopStepAccounts.get_default_account().name,
            
            # Execution models
            'commission_models': 3,
            'slippage_models': 3,
            'default_commission': ExecutionModels.TOPSTEP_COMMISSION.name,
            'default_slippage': ExecutionModels.REALISTIC_SLIPPAGE.name,
            
            # Optimization defaults
            'default_brute_force': OptimizationDefaults.BRUTE_FORCE_EVALUATIONS,
            'default_bayesian': OptimizationDefaults.BAYESIAN_CALLS,
            'default_top_strategies': OptimizationDefaults.TOP_STRATEGIES_FOR_INJECTION,
            
            # System capabilities
            'supported_timeframes': len(SupportedTimeframes.SUPPORTED),
            'timeframes_list': SupportedTimeframes.SUPPORTED,
            'base_directory': str(SystemPaths.BASE_DIR),
            'ninjatrader_integration': PlatformSettings.NINJATRADER_ENABLED,
            
            # Validation criteria
            'validation_configured': True,
            'min_sharpe_ratio': get_validation_criteria().minimum_sharpe_ratio,
            'min_win_rate': get_validation_criteria().minimum_win_rate,
            
            # NEW: Validation support
            'validation_bridge_available': VALIDATION_BRIDGE_AVAILABLE,
            'validation_bridge_functions': True,
            
            'ready_for_production': is_valid and len(all_markets) > 0
        }
        
        return status
        
    except Exception as e:
        return {
            'module_name': 'config',
            'version': __version__,
            'system_config_valid': False,
            'error': str(e),
            'ready_for_production': False
        }


def print_config_status():
    """Print formatted configuration status"""
    status = get_config_status()
    
    print("\n" + "="*60)
    print("CONFIGURATION MODULE STATUS")
    print("="*60)
    
    print(f"\nModule: {status['module_name']} v{status['version']}")
    print(f"System Config Valid: {'VALID' if status['system_config_valid'] else 'INVALID'}")
    print(f"Production Ready: {'YES' if status['ready_for_production'] else 'NO'}")
    
    if status.get('configuration_issues'):
        print("\nWARNING - ISSUES:")
        for issue in status['configuration_issues'][:3]:
            print(f"   • {issue}")
    
    # Market coverage
    print("\nMARKET COVERAGE:")
    print(f"   Total Markets: {status.get('total_markets', 0)}")
    print(f"   Major Markets: {', '.join(status.get('major_markets_list', [])[:6])}")
    if status.get('total_markets', 0) > 6:
        print(f"   ... and {status.get('total_markets', 0) - 6} more")
    
    # Account configurations
    print("\nACCOUNT CONFIGURATIONS:")
    print(f"   Account Types: {status.get('account_types_configured', 0)}")
    print(f"   Default Account: {status.get('default_account', 'Unknown')}")
    
    # Optimization defaults
    print("\nOPTIMIZATION DEFAULTS:")
    print(f"   Brute Force: {status.get('default_brute_force', 0):,} evaluations")
    print(f"   Bayesian: {status.get('default_bayesian', 0)} calls")
    print(f"   Top Strategies: {status.get('default_top_strategies', 0)}")
    
    # Execution models
    print("\nEXECUTION MODELS:")
    print(f"   Commission Models: {status.get('commission_models', 0)}")
    print(f"   Default: {status.get('default_commission', 'Unknown')}")
    print(f"   Slippage Models: {status.get('slippage_models', 0)}")
    print(f"   Default: {status.get('default_slippage', 'Unknown')}")
    
    # Validation support
    print("\nVALIDATION SUPPORT:")
    print(f"   Validation Bridge: {'YES' if status.get('validation_bridge_available') else 'NO'}")
    print(f"   Bridge Functions: {'YES' if status.get('validation_bridge_functions') else 'NO'}")
    print(f"   Min Sharpe Ratio: {status.get('min_sharpe_ratio', 0)}")
    print(f"   Min Win Rate: {status.get('min_win_rate', 0):.1%}")
    
    # System settings
    print("\nSYSTEM SETTINGS:")
    print(f"   Base Directory: {status.get('base_directory', 'Unknown')}")
    print(f"   Supported Timeframes: {status.get('supported_timeframes', 0)}")
    print(f"   NinjaTrader Integration: {'YES' if status.get('ninjatrader_integration') else 'NO'}")
    
    print("\n" + "="*60)


def get_quick_setup() -> Dict[str, Any]:
    """
    Get a quick setup configuration for immediate use
    
    Returns:
        Dictionary with commonly used configuration objects
    """
    return {
        # Most commonly used markets
        'markets': {
            'ES': TopStepMarkets.ES,
            'NQ': TopStepMarkets.NQ,
            'RTY': TopStepMarkets.RTY,
            'YM': TopStepMarkets.YM,
            'CL': TopStepMarkets.CL,
            'GC': TopStepMarkets.GC
        },
        
        # Default account
        'account': TopStepAccounts.get_default_account(),
        
        # Default execution models
        'commission': ExecutionModels.TOPSTEP_COMMISSION,
        'slippage': ExecutionModels.REALISTIC_SLIPPAGE,
        
        # Optimization defaults
        'optimization': get_default_optimization_config(),
        
        # Validation criteria
        'validation': get_validation_criteria(),
        
        # Common timeframes
        'timeframes': SupportedTimeframes.COMMON_FOR_SCALPING + SupportedTimeframes.COMMON_FOR_SWING
    }


def show_examples():
    """Show common usage examples"""
    examples = """

CONFIGURATION MODULE EXAMPLES
=================================

1. BASIC MARKET ACCESS:
----------------------
from config import TopStepMarkets

# Get market specifications
es = TopStepMarkets.ES
nq = TopStepMarkets.NQ
print(f"ES tick value: ${es.tick_value}")

# Get all markets
all_markets = TopStepMarkets.get_all_markets()
print(f"Available markets: {list(all_markets.keys())}")

2. ACCOUNT CONFIGURATION:
------------------------
from config import TopStepAccounts

# Get account specifications
account_50k = TopStepAccounts.TRADING_COMBINE_50K
account_100k = TopStepAccounts.TRADING_COMBINE_100K

print(f"50K Account - Daily Loss Limit: ${account_50k.daily_loss_limit}")
print(f"100K Account - Profit Target: ${account_100k.profit_target}")

3. VALIDATION SETUP:
-------------------
from config import configure_for_account, configure_for_market

# Configure for validation
account = configure_for_account("topstep_50k")
market = configure_for_market("MNQ")

# Run strategy backtesting directly
result = strategy.backtest(data)

4. OPTIMIZATION SETUP:
---------------------
from config import OptimizationDefaults, get_default_optimization_config

# Get default settings
defaults = get_default_optimization_config()
print(f"Brute force evaluations: {defaults['brute_force_evaluations']}")
print(f"Bayesian calls: {defaults['bayesian_calls']}")

5. DIRECTORY CREATION:
---------------------
from config import create_run_paths

# Create organized run directories
paths = create_run_paths("momentum", "ES", "5m", "2024-01-15_14-30")
print(f"Logs: {paths['run_logs']}")
print(f"Results: {paths['run_results']}")

6. EXECUTION MODELS:
-------------------
from config import ExecutionModels

# Get commission and slippage models
commission = ExecutionModels.TOPSTEP_COMMISSION
slippage = ExecutionModels.REALISTIC_SLIPPAGE

total_cost = commission.total_round_trip_cost()
print(f"Total round-trip cost: ${total_cost}")

7. VALIDATION CRITERIA:
----------------------
from config import get_validation_criteria

criteria = get_validation_criteria()
print(f"Min Sharpe ratio: {criteria.minimum_sharpe_ratio}")
print(f"Min win rate: {criteria.minimum_win_rate:.1%}")

8. TIMEFRAME VALIDATION:
-----------------------
from config import SupportedTimeframes

# Check if timeframe is supported
is_valid = SupportedTimeframes.validate_timeframe("5m")
minutes = SupportedTimeframes.get_timeframe_minutes("1h")
print(f"5m is valid: {is_valid}")
print(f"1h = {minutes} minutes")

9. QUICK SETUP:
--------------
from config import get_quick_setup

setup = get_quick_setup()
es_market = setup['markets']['ES']
default_account = setup['account']
optimization_config = setup['optimization']

10. SYSTEM VALIDATION:
--------------------
from config import validate_system_config

is_valid, issues = validate_system_config()
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")

11. COMPLETE STATUS CHECK:
-------------------------
from config import print_config_status

print_config_status()  # Shows complete module status

"""
    print(examples)


def test_config_module():
    """Test the configuration module"""
    print("TESTING Configuration Module...")
    
    success_count = 0
    total_tests = 9  # Updated test count
    
    # Test 1: System configuration validation
    try:
        is_valid, issues = validate_system_config()
        print(f"PASS System validation: {'valid' if is_valid else 'has issues'}")
        success_count += 1
    except Exception as e:
        print(f"FAIL System validation failed: {e}")
    
    # Test 2: Market specifications
    try:
        markets = TopStepMarkets.get_all_markets()
        es_market = TopStepMarkets.ES
        print(f"PASS Markets loaded: {len(markets)} total, ES tick value: ${es_market.tick_value}")
        success_count += 1
    except Exception as e:
        print(f"FAIL Market specifications failed: {e}")
    
    # Test 3: Account configurations
    try:
        account = TopStepAccounts.get_default_account()
        print(f"PASS Account config: {account.name}")
        success_count += 1
    except Exception as e:
        print(f"FAIL Account configurations failed: {e}")
    
    # Test 4: Optimization defaults
    try:
        defaults = get_default_optimization_config()
        print(f"PASS Optimization defaults: {defaults['brute_force_evaluations']} BF, {defaults['bayesian_calls']} Bayesian")
        success_count += 1
    except Exception as e:
        print(f"FAIL Optimization defaults failed: {e}")
    
    # Test 5: Directory creation
    try:
        paths = create_run_paths("test", "ES", "5m", "2024-01-01_12-00")
        print(f"PASS Directory creation: {len(paths)} paths created")
        success_count += 1
    except Exception as e:
        print(f"FAIL Directory creation failed: {e}")
    
    # Test 6: Execution models
    try:
        commission = ExecutionModels.TOPSTEP_COMMISSION
        print(f"PASS Execution models: ${commission.commission_per_round_trip} commission")
        success_count += 1
    except Exception as e:
        print(f"FAIL Execution models failed: {e}")
    
    # Test 7: Validation criteria
    try:
        criteria = get_validation_criteria()
        print(f"PASS Validation criteria: Sharpe > {criteria.minimum_sharpe_ratio}")
        success_count += 1
    except Exception as e:
        print(f"FAIL Validation criteria failed: {e}")
    
    # Test 8: Validation bridge functions
    try:
        account = configure_for_account("topstep_50k", verbose=False)
        market = configure_for_market("MNQ", verbose=False)
        print(f"PASS Validation bridge functions: {account.name}, {market.symbol}")
        success_count += 1
    except Exception as e:
        print(f"FAIL Validation bridge functions failed: {e}")
    
    # Test 9: Quick setup
    try:
        setup = get_quick_setup()
        print(f"PASS Quick setup: {len(setup['markets'])} markets, {len(setup['timeframes'])} timeframes")
        success_count += 1
    except Exception as e:
        print(f"FAIL Quick setup failed: {e}")
    
    print(f"\nCONFIGURATION TEST RESULTS: {success_count}/{total_tests} tests passed")
    return success_count == total_tests


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core configuration classes
    'MarketSpec',
    'AccountConfig', 
    'CommissionModel',
    'SlippageModel',
    'ValidationCriteria',
    'SystemHealthLimits',
    
    # Main configuration providers
    'TopStepMarkets',
    'TopStepAccounts',
    'ExecutionModels',
    'OptimizationDefaults',
    'SupportedTimeframes',
    'SystemPaths',
    'PlatformSettings',
    
    # Individual market access (commonly used)
    'ES', 'MES', 'NQ', 'MNQ', 'RTY', 'M2K', 'YM', 'MYM',
    'NKD', 'MBT', 'MET', 'CL', 'QM', 'MCL', 'NG', 'QG', 'MNG',
    'GC', 'MGC', 'SI', 'SIL', 'HG', 'MHG', 'RB', 'HO', 'PL',
    'ZC', 'ZW', 'ZS', 'ZM', 'ZL', 'HE', 'LE',
    'ZT', 'ZF', 'ZN', 'TN', 'ZB', 'UB',
    
    # Individual account configurations
    'TRADING_COMBINE_50K', 'TRADING_COMBINE_100K', 'TRADING_COMBINE_150K',
    'EXPRESS_FUNDED_50K',
    
    # Individual execution models
    'TOPSTEP_COMMISSION', 'GENERIC_FUTURES', 'LOW_COST_BROKER',
    'CONSERVATIVE_SLIPPAGE', 'REALISTIC_SLIPPAGE', 'OPTIMISTIC_SLIPPAGE',
    
    # Utility functions
    'get_market_spec',
    'get_default_optimization_config',
    'create_run_paths',
    'get_validation_criteria',
    'validate_system_config',
    'test_system_config',
    
    # NEW: Validation bridge functions
    'configure_for_account',
    'configure_for_market',
    'BacktestResult',
    
    # Module utilities
    'get_config_status',
    'print_config_status',
    'get_quick_setup',
    'show_examples',
    'test_config_module',
    
    # Module metadata
    '__version__',
    'SYSTEM_CONFIG_AVAILABLE',
    'MARKETS_LOADED',
    'ACCOUNTS_CONFIGURED',
    'EXECUTION_MODELS_AVAILABLE',
    'VALIDATION_BRIDGE_AVAILABLE'
]

# Module initialization
if SYSTEM_CONFIG_AVAILABLE:
    # Validate configuration on import
    _is_valid, _issues = validate_system_config()
    if not _is_valid and len(_issues) > 0:
        print(f"WARNING: Configuration issues detected: {_issues[0]}")
        if len(_issues) > 1:
            print(f"   ... and {len(_issues) - 1} more issues")

# Final module status
if __name__ == "__main__":
    print_config_status()
    print()
    test_config_module()