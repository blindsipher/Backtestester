"""
Core Module - State and Configuration Management
===============================================

Core components for pipeline state management and configuration collection.
"""

from .state import PipelineState, ExecutionConfig, SplitConfig
from .config_collector import collect_cli_config, collect_interactive_config, is_cli_mode

__all__ = [
    'PipelineState',
    'ExecutionConfig', 
    'SplitConfig',
    'collect_cli_config',
    'collect_interactive_config',
    'is_cli_mode'
]