"""
Strategy package for TopStep trading system.

Modular strategy architecture with automatic discovery and vectorized processing.
Each strategy subfolder is self-contained with its own indicators, parameters, and deployment template.
"""

import importlib
import pkgutil
import logging
from pathlib import Path
from typing import Dict, Type, Optional

from .base import BaseStrategy
from utils.exceptions import StrategyConfigurationError

logger = logging.getLogger(__name__)

def discover_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Automatically discover all strategy classes in strategy subfolders.
    
    Returns:
        Dict mapping strategy names to strategy classes
    """
    strategies = {}
    
    try:
        # Walk through all packages in strategies module
        for module_info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
            # Look for strategy.py modules (e.g., rsi_ema.strategy)
            if module_info.name.endswith(".strategy"):
                try:
                    # Import the strategy module
                    mod = importlib.import_module(module_info.name)
                    
                    # Look for strategy classes (try common names)
                    strategy_class = None
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseStrategy) and 
                            attr != BaseStrategy):
                            strategy_class = attr
                            break
                    
                    if strategy_class and hasattr(strategy_class, 'name'):
                        strategies[strategy_class.name] = strategy_class
                        logger.debug(f"Discovered strategy: {strategy_class.name}")
                    else:
                        logger.warning(f"No valid strategy class found in {module_info.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to import strategy module {module_info.name}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Strategy discovery failed: {e}")
        
    logger.info(f"Discovered {len(strategies)} strategies: {list(strategies.keys())}")
    return strategies

def load_strategy(name: str) -> Optional[Type[BaseStrategy]]:
    """
    Load a specific strategy by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy class or None if not found
    """
    strategies = discover_strategies()
    return strategies.get(name)

def get_all_strategies() -> Dict[str, Type[BaseStrategy]]:
    """
    Get all available strategies.
    
    Returns:
        Dict mapping strategy names to strategy classes
    """
    return discover_strategies()

def validate_strategy_folder(folder_path: Path) -> bool:
    """
    Validate that a strategy folder contains all required files.
    
    Args:
        folder_path: Path to strategy folder
        
    Returns:
        True if folder is valid
    """
    required_files = ['__init__.py', 'indicators.py', 'parameters.py', 'strategy.py', 'deployment_template.py']
    
    for filename in required_files:
        if not (folder_path / filename).exists():
            logger.warning(f"Strategy folder {folder_path} missing required file: {filename}")
            return False
            
    return True

__all__ = [
    'BaseStrategy',
    'discover_strategies', 
    'load_strategy',
    'get_all_strategies',
    'validate_strategy_folder'
]