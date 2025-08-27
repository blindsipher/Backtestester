"""
App Module - Pipeline Orchestration
===================================

Clean orchestration layer for the trading strategy pipeline.
Uses existing modules without modification.
"""

from .core.config_collector import collect_cli_config, collect_interactive_config, is_cli_mode
from .core.state import PipelineState
from .pipeline import orchestrate_pipeline

__all__ = [
    "collect_cli_config",
    "collect_interactive_config",
    "is_cli_mode",
    "orchestrate_pipeline",
    "PipelineState",
]