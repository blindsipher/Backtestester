"""
Deployment Module for Institutional Trading System
=================================================

Deploys optimized parameters from optimization phase into production-ready strategy files.
Implements the DEPLOYMENT phase of the institutional architecture pipeline.

Key Features:
- Parameter injection into deployment templates  
- Batch deployment of top-N parameter sets
- Production file generation with proper naming
- Template validation and error handling
- Market configuration integration

Following INIT_INSTITUTIONAL_ARCHITECTURE design principles.
"""

from .parameter_injector import ParameterInjector
from .deployment_engine import DeploymentEngine
from .template_validator import TemplateValidator
from .deployment_config import DeploymentConfig

# Public API
__all__ = [
    'DeploymentEngine',
    'ParameterInjector', 
    'TemplateValidator',
    'DeploymentConfig',
    'deploy_optimized_parameters'
]

def deploy_optimized_parameters(state, output_directory=None, max_deployments=10):
    """
    Convenience function to deploy optimized parameters.
    
    Args:
        state: PipelineState with best_parameters from optimization
        output_directory: Directory to save deployed files (optional)
        max_deployments: Maximum number of parameter sets to deploy
        
    Returns:
        Deployment result dictionary with file paths and status
    """
    from pathlib import Path
    
    if output_directory is None:
        output_directory = Path("deployed_strategies")
    
    engine = DeploymentEngine(output_directory=output_directory)
    return engine.deploy(state, max_deployments=max_deployments)