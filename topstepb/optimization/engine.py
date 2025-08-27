"""
Optuna Optimization Engine
===========================

Main orchestrator for trading strategy parameter optimization using Optuna.

This module provides the central OptunaEngine class that coordinates all
optimization activities. It integrates with the existing pipeline architecture
and handles study creation, optimization execution, and result processing.

Key Features:
- TPE sampling with MedianPruner for efficient optimization
- PostgreSQL storage with unlimited worker scalability
- High-concurrency support with connection pooling
- Comprehensive error handling and recovery
- Structured result output for deployment modules
- Integration with existing pipeline state management

The engine outputs 1-500 parameter sets in deployment-ready format without
handling deployment injection (separate deployment module responsibility).
"""

import logging
import time
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import optuna
from optuna import Trial, Study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
import numpy as np

# PostgreSQL and SQLAlchemy imports

# Import existing system components
from app.core.state import PipelineState

# Import optimization components
from .config.optuna_config import OptimizationConfig, create_study_paths
from .objective import ObjectiveFactory
from .parallel import ParallelOptimizer

logger = logging.getLogger(__name__)


class OptunaEngine:
    """
    Main optimization engine using Optuna for trading strategy parameter optimization.
    
    Coordinates the complete optimization workflow from study creation through
    result export. Integrates seamlessly with the existing pipeline architecture
    while providing institutional-grade optimization capabilities.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the Optuna optimization engine.
        
        Args:
            config: Optimization configuration (uses default if None)
        """
        self.config = config or OptimizationConfig()
        self.config.validate()
        
        self.study: Optional[Study] = None
        self.objective_factory = ObjectiveFactory(self.config)
        self.parallel_optimizer = ParallelOptimizer(self.config)
        
        # Runtime state
        self.start_time: Optional[float] = None
        self.optimization_interrupted = False
        self.last_checkpoint_time = 0
        
        # Suppress verbose Optuna logging - show only warnings and errors
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info(f"OptunaEngine initialized with {self.config.limits.max_trials} max trials")
    
    def run(self, pipeline_state: PipelineState) -> Dict[str, Any]:
        """
        Run complete optimization workflow.
        
        This is the main entry point called by the pipeline orchestrator.
        It performs the full optimization workflow and returns structured
        results ready for deployment module consumption.
        
        Args:
            pipeline_state: Complete pipeline state with strategy, data, and config
            
        Returns:
            Comprehensive optimization results including:
            - best_parameters: List of top-N parameter sets (1-500)
            - study_summary: Optimization performance summary
            - export_paths: File paths to detailed results
            - optimization_metadata: Execution statistics and configuration
        """
        self.start_time = time.time()
        
        try:
            logger.info(f"Starting optimization for {pipeline_state.strategy_name}")
            
            # 1. Validate pipeline state
            validation_result = self._validate_pipeline_state(pipeline_state)
            if not validation_result['valid']:
                return self._create_error_result(f"Pipeline validation failed: {validation_result['error']}")
            
            # 2. Prepare optimization data
            optimization_data = self._prepare_optimization_data(pipeline_state)
            if not optimization_data['success']:
                return self._create_error_result(f"Data preparation failed: {optimization_data['error']}")
            
            # 3. Create and configure Optuna study
            study_setup = self._create_optuna_study(pipeline_state)
            if not study_setup['success']:
                return self._create_error_result(f"Study creation failed: {study_setup['error']}")
            
            self.study = study_setup['study']
            
            # 4. Set up signal handlers for graceful interruption
            self._setup_signal_handlers()
            
            # 5. Create objective function with secure data access
            objective_function = self.objective_factory.create_objective(
                strategy_instance=pipeline_state.strategy_instance,
                authorized_accesses=optimization_data['authorized_accesses'],
                trading_config=pipeline_state.trading_config,
                execution_config={
                    'slippage_ticks': pipeline_state.slippage_ticks,
                    'commission_per_trade': pipeline_state.commission_per_trade
                }
            )
            
            # 6. Run optimization with multiprocessing support
            optimization_result = self._run_optimization(objective_function, pipeline_state)
            
            if not optimization_result['success']:
                return self._create_error_result(f"Optimization failed: {optimization_result['error']}")
            
            # 7. Process and export results
            results = self._process_optimization_results(pipeline_state, study_setup['paths'])
            
            logger.info(f"Optimization completed successfully in {time.time() - self.start_time:.1f}s")
            return results
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            self.optimization_interrupted = True
            return self._create_interrupted_result(pipeline_state)
            
        except Exception as e:
            logger.error(f"Optimization engine error: {e}", exc_info=True)
            return self._create_error_result(f"Engine error: {str(e)}")
    
    def _validate_pipeline_state(self, state: PipelineState) -> Dict[str, Any]:
        """Validate pipeline state has all required components for optimization"""
        if not state.strategy_instance:
            return {'valid': False, 'error': 'No strategy instance'}
        
        if not state.trading_config:
            return {'valid': False, 'error': 'No trading configuration'}
        
        if state.full_data is None or state.full_data.empty:
            return {'valid': False, 'error': 'No optimization data'}
        
        # Check strategy has required optimization interfaces
        if not hasattr(state.strategy_instance, 'get_parameter_ranges'):
            return {'valid': False, 'error': 'Strategy missing get_parameter_ranges method'}
        
        if not hasattr(state.strategy_instance, 'validate_parameters'):
            return {'valid': False, 'error': 'Strategy missing validate_parameters method'}
        
        parameter_ranges = state.strategy_instance.get_parameter_ranges()
        if not parameter_ranges:
            return {'valid': False, 'error': 'Strategy provided empty parameter ranges'}
        
        return {'valid': True}
    
    def _prepare_optimization_data(self, state: PipelineState) -> Dict[str, Any]:
        """Prepare secure data access using PipelineOrchestrator (no DataSplit objects exposed)"""
        try:
            # ARCHITECTURAL FIX: Use secure orchestrator instead of direct DataSplit access
            if not hasattr(state, 'secure_orchestrator') or not state.secure_orchestrator:
                return {'success': False, 'error': 'No secure orchestrator available - pipeline integration issue'}
            
            orchestrator = state.secure_orchestrator
            
            # Request authorized data for optimization phase
            # SECURITY: Test data is never provided to optimization modules
            try:
                if state.split_type == "chronological":
                    # Get single authorized data access
                    access = orchestrator.get_authorized_data("optimization", "optimization")
                    if access.train_data is None or access.train_data.empty or access.validation_data is None or access.validation_data.empty:
                        return {'success': False, 'error': 'Failed to get authorized optimization data'}
                    
                    # Package as list for consistent processing
                    authorized_accesses = [access]
                    logger.info(f"Authorized single chronological data access: "
                               f"train={len(access.train_data)}, validation={len(access.validation_data)}, test=WITHHELD")
                    
                elif state.split_type == "walk_forward":
                    # Get walk-forward authorized data accesses
                    authorized_accesses = orchestrator.get_walk_forward_splits("optimization")
                    if not authorized_accesses:
                        return {'success': False, 'error': 'No walk-forward splits authorized for optimization'}
                    
                    logger.info(f"Authorized {len(authorized_accesses)} walk-forward data accesses")
                    
                    # Log first access for verification
                    first_access = authorized_accesses[0]
                    logger.debug(f"First authorized access: train={len(first_access.train_data)}, "
                                f"validation={len(first_access.validation_data)}, test=WITHHELD")
                    
                else:
                    return {'success': False, 'error': f'Unknown split type: {state.split_type}'}
                
            except Exception as e:
                return {'success': False, 'error': f'Failed to get authorized data access: {e}'}
            
            # Validate all authorized accesses have required data
            for i, access in enumerate(authorized_accesses):
                if access.train_data is None or access.train_data.empty:
                    return {'success': False, 'error': f'Authorized access {i+1} has no train data'}
                if access.validation_data is None or access.validation_data.empty:
                    return {'success': False, 'error': f'Authorized access {i+1} has no validation data'}
                
                # Test data should be None/withheld during optimization
                if access.test_data is not None:
                    logger.warning(f"Authorized access {i+1} unexpectedly has test data - security violation!")
                
                logger.debug(f"Authorized access {i+1} validated: "
                            f"train={len(access.train_data)}, validation={len(access.validation_data)}")
            
            logger.info(f"Successfully prepared {len(authorized_accesses)} secure data accesses for optimization")
            return {'success': True, 'authorized_accesses': authorized_accesses, 'orchestrator': orchestrator}
            
        except Exception as e:
            logger.error(f"Secure data access preparation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_optuna_study(self, state: PipelineState) -> Dict[str, Any]:
        """Create and configure Optuna study with PostgreSQL storage"""
        try:
            # Create study directory structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_paths = create_study_paths(
                state.strategy_name, 
                state.symbol, 
                state.timeframe, 
                timestamp
            )
            
            # Ensure directories exist
            for path in study_paths.values():
                if isinstance(path, Path):
                    path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure PostgreSQL storage
            database_url = self.config.storage.get_database_url()
            study_name = self.config.storage.get_study_name(
                state.strategy_name, state.symbol, state.timeframe, timestamp
            )
            
            # Configure sampler and pruner - OPTIMIZED for Phase 3
            sampler = TPESampler(
                n_startup_trials=self.config.tpe_sampler.n_startup_trials,    # 50 for faster activation
                multivariate=self.config.tpe_sampler.multivariate,            # True for correlations
                group=self.config.tpe_sampler.group,                          # True for mixed types
                prior_weight=self.config.tpe_sampler.prior_weight,
                consider_prior=self.config.tpe_sampler.consider_prior,
                consider_endpoints=self.config.tpe_sampler.consider_endpoints,
                warn_independent_sampling=self.config.tpe_sampler.warn_independent_sampling,  # False to suppress warnings
                seed=getattr(self.config.tpe_sampler, 'seed', 42)             # ADDED: Reproducible results
            )
            
            pruner = MedianPruner(
                n_startup_trials=self.config.median_pruner.n_startup_trials,
                n_warmup_steps=self.config.median_pruner.n_warmup_steps,
                interval_steps=self.config.median_pruner.interval_steps,
                n_min_trials=self.config.median_pruner.n_min_trials
            )
            
            # Try PostgreSQL first, fallback to SQLite if unavailable
            logger.info(f"Creating Optuna RDBStorage with PostgreSQL: {database_url}")
            storage = None
            storage_type = "postgresql"
            
            try:
                # OPTIMIZED: Dynamic connection pool sizing based on expected worker count
                expected_workers = self.config.limits.max_workers or mp.cpu_count()
                optimized_pool_size = max(self.config.storage.pool_size, expected_workers + 10)
                optimized_max_overflow = max(self.config.storage.max_overflow, optimized_pool_size)
                
                engine_kwargs = {
                    'pool_size': optimized_pool_size,
                    'max_overflow': optimized_max_overflow,
                    'pool_timeout': self.config.storage.pool_timeout,
                    'pool_recycle': self.config.storage.pool_recycle,
                }

                # Use the same simple approach that works in our direct test
                storage = RDBStorage(url=database_url, engine_kwargs=engine_kwargs) # <-- Pass the args here
                logger.info("PostgreSQL RDBStorage created successfully with custom connection pooling.")
                
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
                logger.info("Falling back to SQLite storage for local optimization")
                
                # Fallback to SQLite storage
                try:
                    # Create SQLite database in results directory
                    sqlite_path = study_paths['run_results'] / f"{study_name}.db"
                    sqlite_url = f"sqlite:///{sqlite_path}"
                    storage = RDBStorage(url=sqlite_url)
                    storage_type = "sqlite"
                    logger.info(f"SQLite storage created successfully: {sqlite_path}")
                except Exception as sqlite_error:
                    logger.error(f"Failed to create SQLite storage: {sqlite_error}")
                    return {'success': False, 'error': f'Both PostgreSQL and SQLite storage failed: {e}, {sqlite_error}'}
            
            if storage is None:
                return {'success': False, 'error': 'No storage backend available'}
            
            # Create study with proper storage
            try:
                logger.info(f"Creating Optuna study: {study_name}")
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction='maximize',  # Maximize composite score
                    load_if_exists=True
                )
                logger.info("Optuna study created successfully")
            except Exception as e:
                logger.error(f"Failed to create Optuna study: {e}")
                return {'success': False, 'error': f'Study creation failed: {e}'}
            
            logger.info(f"Created Optuna study '{study_name}' with {storage_type} storage")
            
            return {
                'success': True,
                'study': study,
                'paths': study_paths,
                'database_url': database_url,
                'study_name': study_name
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, gracefully stopping optimization...")
            self.optimization_interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _run_optimization(self, objective_function, state: PipelineState) -> Dict[str, Any]:
        """Run the actual optimization with PostgreSQL-backed resource monitoring"""
        try:
            # PostgreSQL can handle unlimited concurrent workers
            max_workers = self.config.limits.max_workers or 1
            
            # Validate worker count for system resources (no PostgreSQL limits)
            max_workers = self._validate_postgresql_concurrency(max_workers)
            
            logger.info(f"Running optimization with {max_workers} workers, "
                       f"max {self.config.limits.max_trials} trials")
            
            # Run optimization
            if max_workers > 1:
                # Parallel optimization with unlimited PostgreSQL scalability
                self.parallel_optimizer.run_parallel_optimization(
                    study=self.study,
                    objective=objective_function,
                    n_trials=self.config.limits.max_trials,
                    n_jobs=max_workers,
                    timeout=self.config.limits.max_optimization_time,
                    callbacks=[self._checkpoint_callback]
                )
            else:
                # Single-threaded optimization
                self.study.optimize(
                    objective_function,
                    n_trials=self.config.limits.max_trials,
                    timeout=self.config.limits.max_optimization_time,
                    callbacks=[self._checkpoint_callback]
                )
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _checkpoint_callback(self, study: Study, trial: Trial):
        """Callback for periodic checkpointing during optimization"""
        current_time = time.time()
        
        # Checkpoint every N trials or every few minutes
        if (trial.number % self.config.limits.checkpoint_interval == 0 or
            current_time - self.last_checkpoint_time > 300):  # 5 minutes
            
            try:
                logger.debug(f"Checkpoint at trial {trial.number}: "
                           f"best_value={study.best_value:.4f}, "
                           f"elapsed={current_time - self.start_time:.1f}s")
                
                # Save study state (handled by SQLite storage automatically)
                self.last_checkpoint_time = current_time
                
                # Check for interruption
                if self.optimization_interrupted:
                    logger.info("Optimization interrupted, stopping...")
                    study.stop()
                
                # Check memory usage (simplified monitoring)
                try:
                    import psutil
                    process = psutil.Process()
                    memory_usage = int(process.memory_info().rss / (1024 * 1024))
                    if memory_usage > self.config.limits.memory_limit_mb * 2:  # Emergency threshold
                        logger.warning(f"High memory usage ({memory_usage}MB), consider stopping")
                except ImportError:
                    # Skip memory monitoring if psutil not available
                    pass
                
            except Exception as e:
                logger.warning(f"Checkpoint callback error: {e}")
    
    def _process_optimization_results(self, state: PipelineState, study_paths: Dict[str, Path]) -> Dict[str, Any]:
        """Process optimization results and export for deployment modules"""
        try:
            if not self.study.trials:
                return self._create_error_result("No trials completed")
            
            # Get top trials based on composite score
            n_results = min(self.config.limits.results_top_n, len(self.study.trials))
            
            # Filter valid trials (not pruned or failed)
            valid_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not valid_trials:
                return self._create_error_result("No valid trials completed")
            
            # Sort by objective value (composite score) and take top N
            sorted_trials = sorted(valid_trials, key=lambda t: t.value or 0, reverse=True)
            top_trials = sorted_trials[:n_results]
            
            logger.info(f"Processing {len(top_trials)} top trials from {len(valid_trials)} valid trials")
            
            # No file exports - deployment uses in-memory parameter sets only
            logger.info("Using in-memory parameter sets for deployment (no file exports)")
            
            # Extract parameter sets for deployment modules
            best_parameters = []
            for trial in top_trials:
                param_set = {
                    'parameters': trial.params.copy(),
                    'composite_score': trial.value,
                    'trial_number': trial.number,
                    'metrics': self._extract_trial_metrics(trial)
                }
                best_parameters.append(param_set)
            
            # Create optimization summary
            summary = self._create_optimization_summary(valid_trials, top_trials)
            
            # Create complete result structure
            result = {
                'success': True,
                'best_parameters': best_parameters,  # Ready for deployment modules
                'study_summary': summary,
                'export_paths': {},
                'optimization_metadata': {
                    'strategy_name': state.strategy_name,
                    'symbol': state.symbol,
                    'timeframe': state.timeframe,
                    'total_trials': len(self.study.trials),
                    'valid_trials': len(valid_trials),
                    'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
                    'best_score': self.study.best_value,
                    'optimization_time': time.time() - self.start_time,
                    'configuration': self.config.to_dict()
                }
            }
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Result processing failed: {str(e)}")
    
    def _extract_trial_metrics(self, trial: Trial) -> Dict[str, float]:
        """Extract individual metrics from trial user attributes"""
        metrics = {}
        
        for key, value in trial.user_attrs.items():
            if key.startswith('metric_') or key.startswith('norm_') or key.startswith('contrib_'):
                metrics[key] = value
        
        return metrics
    
    def _create_optimization_summary(self, valid_trials: List[Trial], top_trials: List[Trial]) -> Dict[str, Any]:
        """Create optimization performance summary"""
        if not valid_trials:
            return {'error': 'No valid trials to summarize'}
        
        scores = [t.value for t in valid_trials if t.value is not None]
        
        return {
            'best_score': max(scores) if scores else 0.0,
            'median_score': float(np.median(scores)) if scores else 0.0,
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'score_std': float(np.std(scores)) if scores else 0.0,
            'score_range': [float(min(scores)), float(max(scores))] if scores else [0.0, 0.0],
            'convergence_achieved': len(top_trials) >= min(10, len(valid_trials) // 10),
            'parameter_importance': self._calculate_parameter_importance(),
            'optimization_efficiency': len(valid_trials) / len(self.study.trials) if self.study.trials else 0.0
        }
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using Optuna's built-in methods"""
        try:
            if len(self.study.trials) < 10:  # Need minimum trials for importance
                return {}
            
            importance = optuna.importance.get_param_importances(
                self.study,
                evaluator=optuna.importance.FanovaImportanceEvaluator()
            )
            
            return {param: float(imp) for param, imp in importance.items()}
            
        except Exception as e:
            logger.warning(f"Failed to calculate parameter importance: {e}")
            return {}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'success': False,
            'error': error_message,
            'best_parameters': [],
            'study_summary': {'error': error_message},
            'export_paths': {},
            'optimization_metadata': {
                'total_trials': len(self.study.trials) if self.study else 0,
                'optimization_time': time.time() - self.start_time if self.start_time else 0,
                'configuration': self.config.to_dict()
            }
        }
    
    def _create_interrupted_result(self, state: PipelineState) -> Dict[str, Any]:
        """Create result for interrupted optimization"""
        if self.study and self.study.trials:
            # Try to salvage partial results
            try:
                return self._process_optimization_results(state, {})
            except:
                pass
        
        return {
            'success': False,
            'error': 'Optimization interrupted by user',
            'best_parameters': [],
            'study_summary': {'interrupted': True},
            'export_paths': {},
            'optimization_metadata': {
                'interrupted': True,
                'total_trials': len(self.study.trials) if self.study else 0,
                'optimization_time': time.time() - self.start_time if self.start_time else 0
            }
        }
    
    
    def _validate_postgresql_concurrency(self, requested_workers: int) -> int:
        """
        Validate worker count for PostgreSQL - delegate to ParallelOptimizer for consistency.
        
        PostgreSQL connection pooling settings (pool_size=50, max_overflow=100) 
        support up to 150 concurrent connections for high-throughput optimization.
        """
        if requested_workers <= 1:
            return requested_workers
        
        # Log PostgreSQL capacity for high concurrency workloads
        if requested_workers > 16:
            logger.info(
                f"Using high concurrency ({requested_workers} workers) with PostgreSQL. "
                f"Connection pooling supports up to {self.config.storage.pool_size + self.config.storage.max_overflow} "
                f"concurrent connections for optimal throughput."
            )
        
        # Delegate detailed validation to ParallelOptimizer to avoid duplication
        # ParallelOptimizer handles memory, CPU, and system resource validation
        return requested_workers