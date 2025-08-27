"""
Deployment Engine for Institutional Trading System  
=================================================

Main orchestrator for the deployment phase that injects optimized parameters
into production strategy templates.

Implements Phase 5: DEPLOYMENT from the institutional architecture design.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import datetime

from app.core.state import PipelineState
from .deployment_config import DeploymentConfig, ParameterSet  
from .parameter_injector import ParameterInjector
from .template_validator import TemplateValidator

logger = logging.getLogger(__name__)


class DeploymentEngine:
    """
    Main deployment engine that orchestrates parameter injection into strategy templates.
    
    Converts optimization results into production-ready strategy files with
    optimized parameters injected.
    """
    
    def __init__(self, output_directory: Optional[Path] = None, config: Optional[DeploymentConfig] = None):
        """
        Initialize deployment engine.
        
        Args:
            output_directory: Directory for deployed files (overrides config)
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()
        
        if output_directory:
            self.config.output_directory = Path(output_directory)
        
        self.injector = ParameterInjector(self.config)
        self.validator = TemplateValidator(self.config) 
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DeploymentEngine initialized: output_dir={self.config.output_directory}")
    
    def deploy(self, state: PipelineState, max_deployments: Optional[int] = None) -> Dict[str, Any]:
        """
        Deploy optimized parameters from pipeline state.
        
        Args:
            state: PipelineState with optimization results
            max_deployments: Override max deployments from config
            
        Returns:
            Deployment results with file paths and status
        """
        try:
            self.logger.info(f"Starting deployment for strategy: {state.strategy_name}")
            
            # Validation
            if not state.best_parameters:
                return {
                    'success': False,
                    'error': 'No optimized parameters available for deployment',
                    'deployed_files': [],
                    'deployment_count': 0
                }
            
            if not state.strategy_instance:
                return {
                    'success': False,
                    'error': 'No strategy instance available - cannot locate deployment template',
                    'deployed_files': [],
                    'deployment_count': 0
                }
            
            # Find deployment template
            template_result = self._find_deployment_template(state.strategy_name)
            if not template_result['found']:
                return {
                    'success': False,
                    'error': template_result['error'],
                    'deployed_files': [],
                    'deployment_count': 0
                }
            
            template_path = template_result['template_path']
            self.logger.info(f"Using deployment template: {template_path}")
            
            # Validate template if configured
            if self.config.validate_templates:
                validation_result = self.validator.validate_template(template_path)
                if not validation_result['valid']:
                    self.logger.warning(f"Template validation issues: {validation_result['errors']}")
                    # Continue anyway unless critical errors
            
            # Prepare parameter sets
            parameter_sets = self._prepare_parameter_sets(state.best_parameters, max_deployments)
            self.logger.info(f"Deploying {len(parameter_sets)} parameter sets")
            
            # Prepare simulation parameters from pipeline state
            simulation_params = {
                'slippage_ticks': getattr(state, 'slippage_ticks', 0.0),
                'commission_per_trade': getattr(state, 'commission_per_trade', 0.0),
                'contracts_per_trade': getattr(state, 'contracts_per_trade', 1)
            }
            
            # Deploy each parameter set
            deployed_files = []
            deployment_errors = []
            
            for param_set in parameter_sets:
                deploy_result = self._deploy_parameter_set(
                    template_path, param_set, state.trading_config, simulation_params, state.strategy_name
                )
                
                if deploy_result['success']:
                    deployed_files.append(deploy_result)
                    self.logger.info(f"Deployed rank #{param_set.rank}: {deploy_result['file_path']}")
                else:
                    deployment_errors.append(deploy_result)
                    self.logger.error(f"Failed to deploy rank #{param_set.rank}: {deploy_result['error']}")
            
            # Create deployment summary
            summary = self._create_deployment_summary(deployed_files, deployment_errors, state)
            
            return {
                'success': len(deployed_files) > 0,
                'deployed_files': deployed_files,
                'deployment_count': len(deployed_files),
                'failed_deployments': len(deployment_errors),
                'deployment_errors': deployment_errors,
                'output_directory': str(self.config.output_directory),
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Deployment engine failure: {e}',
                'deployed_files': [],
                'deployment_count': 0
            }
    
    def _find_deployment_template(self, strategy_name: str) -> Dict[str, Any]:
        """Find the deployment template for the given strategy."""
        
        # Standard template path pattern
        template_paths = [
            Path(f"strategies/{strategy_name}/deployment_template.py"),
            Path(f"strategies/{strategy_name.lower()}/deployment_template.py"),
            Path(f"strategies/{strategy_name}/template.py"),
        ]
        
        for template_path in template_paths:
            if template_path.exists():
                return {
                    'found': True,
                    'template_path': template_path,
                    'strategy_name': strategy_name
                }
        
        return {
            'found': False,
            'error': f'Deployment template not found for strategy: {strategy_name}. '
                    f'Searched: {[str(p) for p in template_paths]}',
            'template_path': None
        }
    
    def _prepare_parameter_sets(self, best_parameters: List[Dict[str, Any]], 
                               max_deployments: Optional[int]) -> List[ParameterSet]:
        """Prepare parameter sets from optimization results."""
        
        limit = max_deployments or self.config.max_deployments
        limited_params = best_parameters[:limit]
        
        parameter_sets = []
        for rank, param_data in enumerate(limited_params, 1):
            # Filter by score threshold if configured
            score = param_data.get('composite_score', 0.0)
            if score < self.config.min_score_threshold:
                self.logger.info(f"Skipping rank #{rank} - score {score:.4f} below threshold {self.config.min_score_threshold}")
                continue
            
            param_set = ParameterSet(
                parameters=param_data.get('parameters', {}),
                composite_score=score,
                trial_number=param_data.get('trial_number', rank),
                rank=rank,
                metrics=param_data.get('metrics', {})
            )
            parameter_sets.append(param_set)
        
        return parameter_sets
    
    
    def _deploy_parameter_set(self, template_path: Path, parameter_set: ParameterSet,
                             trading_config, simulation_params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """Deploy a single parameter set to a production file."""
        try:
            # Read template
            template_content = template_path.read_text(encoding=self.config.template_encoding)
            
            # Inject parameters using enhanced TradingConfig system
            deployed_content = self.injector.inject_parameters(
                template_content, parameter_set, trading_config, simulation_params
            )
            
            # Generate output filename
            filename = parameter_set.generate_filename(self.config, strategy_name)
            output_path = self.config.output_directory / filename
            
            # Handle existing files
            if output_path.exists() and not self.config.overwrite_existing:
                return {
                    'success': False,
                    'error': f'Output file already exists and overwrite disabled: {output_path}',
                    'file_path': str(output_path)
                }
            
            # Write deployed file
            output_path.write_text(deployed_content, encoding=self.config.template_encoding)
            
            return {
                'success': True,
                'file_path': str(output_path),
                'filename': filename,
                'rank': parameter_set.rank,
                'composite_score': parameter_set.composite_score,
                'trial_number': parameter_set.trial_number,
                'parameter_count': len(parameter_set.parameters),
                'file_size': len(deployed_content)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to deploy parameter set rank #{parameter_set.rank}: {e}',
                'rank': parameter_set.rank
            }
    
    def _create_deployment_summary(self, deployed_files: List[Dict[str, Any]], 
                                  deployment_errors: List[Dict[str, Any]],
                                  state: PipelineState) -> Dict[str, Any]:
        """Create deployment summary for reporting."""
        
        successful_deployments = len(deployed_files)
        failed_deployments = len(deployment_errors)
        
        if successful_deployments > 0:
            best_score = max(f['composite_score'] for f in deployed_files)
            worst_score = min(f['composite_score'] for f in deployed_files)
            avg_score = sum(f['composite_score'] for f in deployed_files) / successful_deployments
        else:
            best_score = worst_score = avg_score = 0.0
        
        return {
            'strategy_name': state.strategy_name,
            'deployment_timestamp': datetime.datetime.now().isoformat(),
            'total_attempts': successful_deployments + failed_deployments,
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'success_rate': successful_deployments / (successful_deployments + failed_deployments) if (successful_deployments + failed_deployments) > 0 else 0.0,
            'score_statistics': {
                'best_score': best_score,
                'worst_score': worst_score, 
                'average_score': avg_score
            },
            'output_directory': str(self.config.output_directory),
            'config_used': {
                'max_deployments': self.config.max_deployments,
                'min_score_threshold': self.config.min_score_threshold,
                'file_prefix': self.config.file_prefix,
                'overwrite_existing': self.config.overwrite_existing
            }
        }
    
    def list_deployed_files(self) -> List[Dict[str, Any]]:
        """List all deployed files in the output directory."""
        
        deployed_files = []
        pattern = f"{self.config.file_prefix}*{self.config.file_suffix}"
        
        for file_path in self.config.output_directory.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                deployed_files.append({
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'size': stat.st_size,
                    'created': datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by creation time (most recent first)
        deployed_files.sort(key=lambda x: x['created'], reverse=True)
        
        return deployed_files
    
    def cleanup_deployed_files(self, keep_latest: int = 10) -> Dict[str, Any]:
        """Clean up old deployed files, keeping only the most recent."""
        
        try:
            deployed_files = self.list_deployed_files()
            
            if len(deployed_files) <= keep_latest:
                return {
                    'success': True,
                    'files_removed': 0,
                    'files_kept': len(deployed_files),
                    'message': f'No cleanup needed - only {len(deployed_files)} files exist'
                }
            
            # Remove older files
            files_to_remove = deployed_files[keep_latest:]
            removed_count = 0
            
            for file_info in files_to_remove:
                try:
                    Path(file_info['file_path']).unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not remove {file_info['file_path']}: {e}")
            
            return {
                'success': True,
                'files_removed': removed_count,
                'files_kept': keep_latest,
                'total_files_before': len(deployed_files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cleanup failed: {e}',
                'files_removed': 0
            }