"""
Pipeline Orchestrator
====================

Clean orchestration of existing modules without business logic.
Coordinates data, config, strategy, and utils modules directly.
"""

import logging
from typing import Dict, Any

# Import utilities
from utils.logger import setup_logging
from utils.error_handling import ErrorResultFactory

# Import domain modules directly
from data import create_test_data, load_data_from_file, create_data_splits
from config.system_config import create_trading_config
from strategies import discover_strategies

from .core.state import PipelineState
from .core.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

def orchestrate_pipeline(state: PipelineState) -> Dict[str, Any]:
    """
    Orchestrate the complete pipeline by delegating to existing modules directly.
    
    Args:
        state: PipelineState with user configuration
        
    Returns:
        Dictionary with pipeline results and complete configuration
    """
    
    try:
        setup_logging()
        logger.info(f"Starting pipeline orchestration for {state.strategy_name}")
        
        # Phase 1: Data Loading - Delegate to data module
        state.update_phase("data_loading")
        if state.data_file_path:
            logger.info(f"Loading data from file: {state.data_file_path}")
            full_data = load_data_from_file(state.data_file_path)
        else:
            logger.info(f"Creating synthetic test data with {state.synthetic_bars} bars")
            full_data = create_test_data(bars=state.synthetic_bars, symbol=state.symbol)
        
        if full_data is None or full_data.empty:
            return _create_error_result(state, 'Data loading resulted in empty dataset')
        
        state.full_data = full_data
        logger.info(f"Data loading phase completed: {len(full_data)} bars loaded")
        
        # Phase 2: Strategy Discovery - Delegate to strategies module
        state.update_phase("strategy_discovery")
        available_strategies = discover_strategies()
        
        if state.strategy_name not in available_strategies:
            available_list = list(available_strategies.keys())
            error = f"Strategy '{state.strategy_name}' not found. Available: {available_list}"
            return _create_error_result(state, error)
        
        strategy_class = available_strategies[state.strategy_name]
        strategy_instance = strategy_class()
        state.strategy_instance = strategy_instance
        logger.info(f"Strategy discovery phase completed: {state.strategy_name}")
        
        strategy_result = {
            'success': True,
            'name': state.strategy_name,
            'class': strategy_class,
            'instance': strategy_instance,
            'parameter_ranges': strategy_instance.get_parameter_ranges(),
            'available_strategies': list(available_strategies.keys())
        }
        
        # Phase 3: Trading Configuration - Delegate to config module
        state.update_phase("trading_config")
        trading_config = create_trading_config(
            symbol=state.symbol,
            timeframe=state.timeframe,
            account_type=state.account_type
        )
        
        if trading_config is None:
            return _create_error_result(state, 'Failed to create trading configuration')
        
        state.trading_config = trading_config
        logger.info(f"Trading config phase completed: {state.symbol} {state.timeframe}")
        
        # Phase 4: Data Splitting - Delegate to data module
        state.update_phase("data_splitting")
        logger.info(f"Delegating {state.split_type} data splitting to data module")
        
        data_splits = create_data_splits(
            data=state.full_data,
            split_method=state.split_type,
            ratios=state.split_ratios,
            gap_days=state.gap_days
        )
        
        # Create secure orchestrator for coordination only
        secure_orchestrator = PipelineOrchestrator(state)
        
        # Load pre-created splits into orchestrator for secure access
        if state.split_type == "chronological":
            splits_loaded = secure_orchestrator.load_data_splits(data_splits)
        elif state.split_type == "walk_forward":
            splits_loaded = secure_orchestrator.load_walk_forward_splits(data_splits)
        else:
            return _create_error_result(state, f"Unknown split type: {state.split_type}")
        
        if not splits_loaded:
            return _create_error_result(state, "Failed to load data splits into orchestrator")
            
        # Store orchestrator in state for optimization phase
        state.secure_orchestrator = secure_orchestrator
        logger.info(f"Data splitting phase completed: {state.split_type}")
        
        split_result = {
            'success': True,
            'result': None,  # No DataSplit objects exposed
            'secure_access': True,
            'orchestrator_available': True
        }
        
        # Phase 5: Parameter Optimization - Delegate to optimization module
        state.update_phase("optimization")
        from optimization import OptunaEngine
        
        from optimization.config.optuna_config import OptimizationConfig
        
        opt_config = OptimizationConfig()
        opt_config.limits.max_trials = state.max_trials
        opt_config.limits.max_workers = state.max_workers  # <-- RECONNECT THE MULTICORE WIRE
        opt_config.limits.timeout_per_trial = state.timeout_per_trial  # CLI: --timeout-per-trial
        opt_config.limits.memory_limit_mb = state.memory_per_worker_mb  # CLI: --memory-per-worker-mb
        opt_config.limits.results_top_n = state.results_top_n  # CLI: --results-top-n
        
        engine = OptunaEngine(config=opt_config)
        
        try:
            optimization_result = engine.run(pipeline_state=state)
            
            if optimization_result and not optimization_result.get('skipped', False):
                state.optimization_result = optimization_result
                state.best_parameters = optimization_result.get('best_parameters', [])
                logger.info(f"Parameter optimization completed with {len(state.best_parameters)} parameter sets")
            else:
                state.optimization_result = optimization_result or {'skipped': True}
                logger.info("Parameter optimization skipped or failed")
                
        except Exception as e:
            error_msg = f"Parameter optimization failed: {e}"
            logger.warning(f"Optimization failed, continuing without optimization: {error_msg}")
            state.optimization_result = {'skipped': True, 'error': error_msg}
        
        # Phase 6: Deployment (if optimization results are available)
        if state.best_parameters and len(state.best_parameters) > 0:
            state.update_phase("deployment")
            from deployment import DeploymentEngine
            from deployment.deployment_config import DeploymentConfig
            
            # Create deployment configuration
            deployment_config = DeploymentConfig()
            # Use CLI parameters if available
            if hasattr(state, 'max_deployments'):
                deployment_config.max_deployments = state.max_deployments
            
            # Initialize deployment engine
            deployment_engine = DeploymentEngine(config=deployment_config)
            
            # Note: Simulation parameters are already stored in state and passed to deployment_engine.deploy()
            
            try:
                # Deploy parameter sets - use results_top_n to ensure exact count matching
                deployment_result = deployment_engine.deploy(state, max_deployments=state.results_top_n)
                
                if deployment_result['success']:
                    state.deployment_result = deployment_result
                    logger.info(f"Deployment phase completed: {deployment_result['deployment_count']} files deployed")
                else:
                    logger.warning(f"Deployment phase failed: {deployment_result.get('error', 'Unknown error')}")
                    state.deployment_result = deployment_result
                    
            except Exception as e:
                error_msg = f"Deployment phase failed: {e}"
                logger.warning(f"Deployment failed, continuing without deployment: {error_msg}")
                state.deployment_result = {'success': False, 'error': error_msg, 'deployed_files': []}
        else:
            logger.info("Skipping deployment phase - no optimization results available")
            state.deployment_result = {'skipped': True, 'reason': 'No optimization results'}
        
        # Phase 7: Final Assembly
        state.update_phase("complete")
        result = _create_success_result(state, strategy_result, split_result)
        
        logger.info("Pipeline orchestration completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Pipeline orchestration failed: {str(e)}"
        state.add_error(error_msg)
        logger.error(error_msg, exc_info=True)
        return _create_error_result(state, error_msg)





def _create_success_result(state: PipelineState, strategy_config: Dict, split_result: Dict) -> Dict[str, Any]:
    """Create successful pipeline result with secure orchestrator support"""
    
    # ARCHITECTURAL FIX: Use secure orchestrator instead of exposing DataSplit objects
    if hasattr(state, 'secure_orchestrator') and state.secure_orchestrator:
        # Get safe metadata from orchestrator (no data exposure)
        orchestrator = state.secure_orchestrator
        
        # Run validation to ensure no data leakage
        validation_result = orchestrator.validate_no_data_leakage()
        
        split_info = {
            'type': state.split_type,
            'secure_access': True,
            'data_leakage_validation': validation_result,
            'orchestrator_available': True,
            'total_bars': len(state.full_data),
            'split_method': state.split_type
        }
        
        # Add safe metadata if available
        if validation_result.get('valid', False):
            split_info.update({
                'date_ranges': validation_result.get('date_ranges', {}),
                'temporal_order_valid': validation_result.get('temporal_order_valid', False),
                'data_overlap_detected': validation_result.get('data_overlap_detected', True)
            })
        
    else:
        # Fallback for legacy compatibility (should not happen in production)
        logger.warning("No secure orchestrator available - falling back to legacy split result")
        split_info = {
            'type': state.split_type,
            'secure_access': False,
            'legacy_fallback': True,
            'total_bars': len(state.full_data) if state.full_data is not None else 0
        }
    
    result = {
        'success': True,
        'pipeline_state': state,
        'strategy_config': strategy_config,
        'trading_config': state.trading_config,
        'execution_config': {
            'slippage_ticks': state.slippage_ticks,
            'commission_per_trade': state.commission_per_trade,
            'contracts_per_trade': state.contracts_per_trade
        },
        'data_config': {
            'total_bars': len(state.full_data),
            'split_method': state.split_type,
            'has_file': state.data_file_path is not None
        },
        'split_info': split_info,    # Secure split information (no raw data exposed)
        'secure_orchestrator': hasattr(state, 'secure_orchestrator') and state.secure_orchestrator is not None,
        'ready_for_deployment': True,
        'summary': state.get_summary()
    }
    
    # Include optimization results if available
    if state.optimization_result:
        result['optimization_result'] = state.optimization_result
        result['best_parameters'] = state.best_parameters  # Ready for deployment module consumption
        result['optimization_summary'] = state.optimization_result.get('study_summary', {})
        result['export_paths'] = state.optimization_result.get('export_paths', {})
        
        # Update deployment readiness
        if state.best_parameters:
            result['ready_for_optimization_deployment'] = True
            result['parameter_sets_available'] = len(state.best_parameters)
        else:
            result['ready_for_optimization_deployment'] = False
            result['parameter_sets_available'] = 0
    else:
        result['ready_for_optimization_deployment'] = False
        result['parameter_sets_available'] = 0
    
    return result


def _create_error_result(state: PipelineState, error_message: str) -> Dict[str, Any]:
    """Create standardized error result with pipeline context"""
    additional_data = {
        'pipeline_state': state,
        'errors': state.errors,
        'warnings': state.warnings,
        'phase_failed': state.pipeline_phase
    }
    
    return ErrorResultFactory.create_error_result(
        error_message=error_message,
        error_context="pipeline_orchestration",
        additional_data=additional_data
    )