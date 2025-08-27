"""
Parallel Optimization Wrapper
==============================

Safe multiprocessing wrapper for Optuna optimization with resource monitoring.

This module provides the ParallelOptimizer class that handles multiprocessing
safety, memory monitoring, and resource management for Optuna studies. It
integrates with the existing resource management infrastructure and provides
fallback mechanisms for robust operation.

Key Features:
- Safe multiprocessing with memory limits
- CPU and memory monitoring via resource_manager
- Graceful worker failure handling
- Fallback to single-process execution
- Progress monitoring and reporting
- Proper process cleanup

The parallel optimizer ensures optimization runs efficiently while respecting
system resource constraints and maintaining stability.
"""

import logging
import time
import multiprocessing as mp
import psutil
from typing import Dict, Any, List, Optional, Callable
# ProcessPoolExecutor removed - using Optuna's native parallelism
import threading
import queue

# Optional GPU support
try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

from optuna.study import Study
from optuna.trial import Trial

# Import optimization components
from .config.optuna_config import OptimizationConfig

logger = logging.getLogger(__name__)


class ParallelOptimizer:
    """
    Safe multiprocessing wrapper for Optuna optimization.
    
    Handles resource monitoring, worker management, and provides fallback
    mechanisms to ensure robust optimization execution across different
    system configurations.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize parallel optimizer.
        
        Args:
            config: Optimization configuration with resource limits
        """
        self.config = config
        # Optimized worker configuration with dynamic memory calculation
        total_system_memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        optimized_total_memory_mb = int(total_system_memory_mb * 0.90)  # Use 90% of system RAM
        
        self.worker_config = {
            'max_workers': mp.cpu_count(),
            'safe_max_workers': mp.cpu_count(),  # Use ALL CPU cores for maximum parallelization
            'memory_per_worker_mb': 1500,  # Will be empirically tuned in Phase 2
            'total_memory_limit_mb': optimized_total_memory_mb  # FIXED: Dynamic calculation
        }

        # Runtime state
        self.active_workers = 0
        self.total_memory_usage = 0
        self.optimization_start_time = 0
        self.progress_queue = queue.Queue()
        self.stop_event = threading.Event()

        # GPU management
        self.available_gpus: List[int] = []
        if torch and torch.cuda.is_available():
            try:
                detected = list(range(torch.cuda.device_count()))
                if config.limits.gpu_device_ids:
                    self.available_gpus = [g for g in config.limits.gpu_device_ids if g in detected]
                else:
                    self.available_gpus = detected
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"GPU detection failed: {e}")
                self.available_gpus = []
        self.gpu_memory_limit_mb = config.limits.gpu_memory_limit_mb
        self.worker_gpu_map: Dict[int, int] = {}

        if self.available_gpus:
            logger.info(f"Detected GPUs: {self.available_gpus}")
        else:
            logger.info("No GPUs detected or torch unavailable")
        
        logger.info(f"ParallelOptimizer initialized: "
                   f"max_workers={self.worker_config['safe_max_workers']}, "
                   f"memory_limit={config.limits.memory_limit_mb}MB")
    
    def run_parallel_optimization(self,
                                 study: Study,
                                 objective: Callable[[Trial], float],
                                 n_trials: int,
                                 n_jobs: int,
                                 timeout: Optional[int] = None,
                                 callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Run optimization with multiprocessing support.
        
        Args:
            study: Optuna study to optimize
            objective: Objective function to minimize/maximize
            n_trials: Total number of trials to run
            n_jobs: Number of parallel workers
            timeout: Maximum optimization time in seconds
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary with optimization results and statistics
        """
        self.optimization_start_time = time.time()
        
        try:
            # Validate worker configuration
            safe_n_jobs = self._validate_worker_count(n_jobs)
            
            if safe_n_jobs == 1:
                logger.info("Using single-threaded optimization")
                return self._run_single_threaded(study, objective, n_trials, timeout, callbacks)
            
            logger.info(f"Starting parallel optimization with {safe_n_jobs} workers")
            
            # Set up worker limits
            worker_limits = self._setup_worker_limits(safe_n_jobs)
            
            # Start progress monitoring
            monitor_thread = threading.Thread(
                target=self._monitor_progress,
                args=(study, n_trials),
                daemon=True
            )
            monitor_thread.start()
            
            # Run parallel optimization
            result = self._run_multiprocessing_optimization(
                study=study,
                objective=objective,
                n_trials=n_trials,
                n_jobs=safe_n_jobs,
                timeout=timeout,
                callbacks=callbacks,
                worker_limits=worker_limits
            )
            
            # Stop monitoring
            self.stop_event.set()
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel optimization failed: {e}")
            # Fallback to single-threaded
            logger.info("Falling back to single-threaded optimization")
            return self._run_single_threaded(study, objective, n_trials, timeout, callbacks)
    
    def _validate_worker_count(self, requested_workers: int) -> int:
        """
        Validate and adjust worker count based on system capabilities.
        
        Args:
            requested_workers: Desired number of workers
            
        Returns:
            Safe number of workers to use
        """
        # Get system constraints
        max_safe_workers = self.worker_config['safe_max_workers']
        cpu_count = mp.cpu_count()
        available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
        
        # Calculate memory-based limit with empirical safety factor
        memory_per_worker = self.config.limits.memory_limit_mb
        memory_limited_workers = max(1, available_memory_mb // (memory_per_worker * 1.25))  # Optimized safety factor: 1.5x → 1.25x
        
        # Use most restrictive limit (removed CPU core reservation)
        safe_workers = min(
            requested_workers,
            max_safe_workers,
            cpu_count,  # OPTIMIZED: Use ALL CPU cores for maximum performance
            memory_limited_workers
        )
        
        if safe_workers != requested_workers:
            logger.info(f"Adjusted worker count from {requested_workers} to {safe_workers} "
                       f"(CPU: {cpu_count}, Memory: {memory_limited_workers}, Safe: {max_safe_workers})")
        
        return max(1, safe_workers)
    
    def _setup_worker_limits(self, n_workers: int) -> Dict[str, Any]:
        """
        Set up resource limits for worker processes.
        
        Args:
            n_workers: Number of worker processes
            
        Returns:
            Dictionary with worker limit configuration
        """
        # Calculate per-worker memory limit
        total_available_mb = psutil.virtual_memory().available // (1024 * 1024)
        per_worker_memory_mb = min(
            self.config.limits.memory_limit_mb,
            total_available_mb // (n_workers + 1)  # +1 for main process
        )
        
        limits = {
            'memory_limit_mb': per_worker_memory_mb,
            # REMOVED: Per-worker CPU limits - let OS scheduler handle efficiently
            'timeout_per_trial': self.config.limits.timeout_per_trial,
            'max_trials_per_worker': max(10, 1000 // n_workers)  # Distribute trials
        }
        
        logger.debug(f"Worker limits: {limits}")
        return limits
    
    def _run_multiprocessing_optimization(self,
                                        study: Study,
                                        objective: Callable[[Trial], float],
                                        n_trials: int,
                                        n_jobs: int,
                                        timeout: Optional[int],
                                        callbacks: Optional[List[Callable]],
                                        worker_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization using multiprocessing"""
        try:
            # Instead of using ProcessPoolExecutor which has pickling issues,
            # we run Optuna's optimize directly in the main process with n_jobs
            # This avoids pickling the objective function closure

            logger.info(f"Running Optuna optimization with n_jobs={n_jobs}")
            # Prepare GPU assignments
            if self.available_gpus:
                self.worker_gpu_map = {
                    i: self.available_gpus[i % len(self.available_gpus)]
                    for i in range(n_jobs)
                }
                logger.info(f"GPU assignments: {self.worker_gpu_map}")

                def gpu_objective(trial: Trial) -> float:
                    proc = mp.current_process()
                    idx = (proc._identity[0] - 1) if proc._identity else 0
                    gpu_id = self.worker_gpu_map.get(idx, self.available_gpus[idx % len(self.available_gpus)])
                    if torch:
                        try:
                            torch.cuda.set_device(gpu_id)
                            if self.gpu_memory_limit_mb > 0:
                                try:
                                    total = torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024)
                                    frac = min(1.0, self.gpu_memory_limit_mb / total)
                                    torch.cuda.set_per_process_memory_fraction(frac, device=gpu_id)
                                except Exception as e:  # pragma: no cover - depends on torch version
                                    logger.debug(f"GPU memory limit not applied: {e}")
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Failed to set GPU {gpu_id} for worker {idx}: {e}")
                    return objective(trial)

            else:
                gpu_objective = objective

            # Run optimization directly with Optuna's native parallelism
            study.optimize(
                gpu_objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                callbacks=callbacks or [],
                gc_after_trial=True,
                show_progress_bar=False
            )
            
            return {
                'success': True,
                'trials_completed': len(study.trials),
                'optimization_time': time.time() - self.optimization_start_time,
                'workers_used': n_jobs,
                'worker_limits': worker_limits
            }
                
        except Exception as e:
            logger.error(f"Multiprocessing optimization failed: {e}")
            raise
    
    # Removed unused legacy ProcessPoolExecutor methods:
    # - _run_optuna_parallel() - duplicated _run_multiprocessing_optimization logic
    # - _worker_initializer() - manual worker management replaced by Optuna's native parallelism
    
    def _run_single_threaded(self,
                           study: Study,
                           objective: Callable[[Trial], float],
                           n_trials: int,
                           timeout: Optional[int],
                           callbacks: Optional[List[Callable]]) -> Dict[str, Any]:
        """Fallback single-threaded optimization"""
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks or [],
                gc_after_trial=True
            )
            
            return {
                'success': True,
                'trials_completed': len(study.trials),
                'optimization_time': time.time() - self.optimization_start_time,
                'workers_used': 1,
                'fallback_used': True
            }
            
        except Exception as e:
            logger.error(f"Single-threaded optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'trials_completed': len(study.trials),
                'optimization_time': time.time() - self.optimization_start_time,
                'workers_used': 1,
                'fallback_used': True
            }
    
    def _monitor_progress(self, study: Study, target_trials: int) -> None:
        """
        Monitor optimization progress and resource usage.
        
        Args:
            study: Optuna study to monitor
            target_trials: Target number of trials
        """
        last_trial_count = 0
        last_log_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_trials = len(study.trials)
                current_time = time.time()
                
                # OPTIMIZED: Log progress every 60 seconds or when significant progress is made (reduced monitoring overhead)
                if (current_time - last_log_time > 60 or 
                    current_trials - last_trial_count >= 20):
                    
                    # Monitor system resources (simplified)
                    try:
                        process = psutil.Process()
                        memory_usage = int(process.memory_info().rss / (1024 * 1024))
                    except:
                        memory_usage = 512  # Fallback
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # Calculate progress metrics
                    progress_percent = (current_trials / target_trials * 100) if target_trials > 0 else 0
                    elapsed_time = current_time - self.optimization_start_time
                    trials_per_minute = (current_trials / elapsed_time * 60) if elapsed_time > 0 else 0
                    
                    # Estimate completion time
                    if trials_per_minute > 0:
                        remaining_trials = max(0, target_trials - current_trials)
                        eta_minutes = remaining_trials / trials_per_minute
                    else:
                        eta_minutes = 0

                    gpu_log = ""
                    if self.available_gpus and torch:
                        gpu_stats = []
                        for gid in self.available_gpus:
                            try:
                                util = torch.cuda.utilization(gid) if hasattr(torch.cuda, "utilization") else 0
                                mem_alloc = torch.cuda.memory_allocated(gid) // (1024 * 1024)
                                mem_total = torch.cuda.get_device_properties(gid).total_memory // (1024 * 1024)
                                gpu_stats.append(f"GPU{gid}: {util}% {mem_alloc}/{mem_total}MB")
                            except Exception:
                                continue
                        gpu_log = ", GPU: " + "; ".join(gpu_stats) if gpu_stats else ""

                    logger.info(
                        f"Optimization progress: {current_trials}/{target_trials} trials "
                        f"({progress_percent:.1f}%), {trials_per_minute:.1f} trials/min, "
                        f"ETA: {eta_minutes:.1f}min, Memory: {memory_usage}MB, CPU: {cpu_percent:.1f}%"
                        f"{gpu_log}"
                    )
                    
                    # Check for resource issues
                    if memory_usage > self.config.limits.memory_limit_mb * 1.5:
                        logger.warning(f"High memory usage detected: {memory_usage}MB")

                    if cpu_percent > 95:
                        logger.warning(f"High CPU usage detected: {cpu_percent}%")

                    if self.available_gpus and torch and self.gpu_memory_limit_mb > 0:
                        for gid in self.available_gpus:
                            try:
                                mem_alloc = torch.cuda.memory_allocated(gid) // (1024 * 1024)
                                if mem_alloc > self.gpu_memory_limit_mb * 1.5:
                                    logger.warning(
                                        f"High GPU{gid} memory usage detected: {mem_alloc}MB"
                                    )
                            except Exception:
                                continue
                    
                    last_trial_count = current_trials
                    last_log_time = current_time
                
                # Check completion
                if current_trials >= target_trials:
                    logger.info("Target trials reached, monitoring complete")
                    break
                
                # OPTIMIZED: Reduced monitoring frequency for better performance
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def get_resource_usage_report(self) -> Dict[str, Any]:
        """
        Get current resource usage report.

        Returns:
            Dictionary with resource usage statistics
        """
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            gpu_info = []
            if self.available_gpus and torch:
                for gid in self.available_gpus:
                    try:
                        util = torch.cuda.utilization(gid) if hasattr(torch.cuda, "utilization") else 0
                        mem_alloc = torch.cuda.memory_allocated(gid) // (1024 * 1024)
                        mem_total = torch.cuda.get_device_properties(gid).total_memory // (1024 * 1024)
                        gpu_info.append({
                            'id': gid,
                            'util_percent': util,
                            'memory_used_mb': mem_alloc,
                            'memory_total_mb': mem_total
                        })
                    except Exception:
                        continue

            return {
                'memory_total_mb': memory.total // (1024 * 1024),
                'memory_available_mb': memory.available // (1024 * 1024),
                'memory_used_mb': memory.used // (1024 * 1024),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'cpu_count': mp.cpu_count(),
                'gpu': gpu_info,
                'active_workers': self.active_workers,
                'optimization_running': not self.stop_event.is_set()
            }

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Clean up resources and stop monitoring"""
        self.stop_event.set()
        logger.info("ParallelOptimizer cleanup completed")
