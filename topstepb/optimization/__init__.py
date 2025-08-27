"""
Optuna Optimization Engine
==========================

Sophisticated parameter optimization module for trading strategies using Optuna.

This module provides institutional-grade optimization capabilities with:
- TPE sampling for efficient parameter space exploration
- MedianPruner for memory safety and performance
- 7-metric composite scoring system for robust strategy evaluation
- Multiprocessing support with resource management
- In-memory parameter sets for deployment modules

Key Components:
- OptunaEngine: Main orchestrator for optimization workflows
- CompositeScore: 7-metric scoring system for strategy evaluation
- ObjectiveFactory: Creates optimization objectives from strategies
- ParallelOptimizer: Safe multiprocessing wrapper

Integration:
The optimization engine integrates seamlessly with the existing pipeline
architecture via Phase 5 restoration in app.pipeline.py. It consumes
strategy interfaces (get_parameter_ranges, validate_parameters) and
outputs parameter sets ready for deployment module consumption.

Usage:
    from optimization import OptunaEngine
    
    engine = OptunaEngine()
    results = engine.run(pipeline_state)
    
    # Access optimized parameters for deployment
    best_params = results['best_parameters']  # List[Dict] - 1-500 sets
    
Author: Claude Code with Zen MCP Tools
Version: 1.0.0
"""

from .engine import OptunaEngine
from .scorers import CompositeScore
from .config.optuna_config import OptimizationConfig

__version__ = "1.0.0"
__author__ = "Claude Code with Zen MCP Tools"

__all__ = [
    'OptunaEngine',
    'CompositeScore', 
    'OptimizationConfig'
]

# Module metadata for pipeline integration
MODULE_INFO = {
    'name': 'optuna_optimization',
    'version': __version__,
    'description': 'Institutional-grade trading strategy optimization engine',
    'dependencies': ['optuna', 'pandas', 'numpy', 'sklearn'],
    'integration_point': 'app.pipeline.orchestrate_pipeline',
    'output_format': 'deployment_ready_parameters'
}