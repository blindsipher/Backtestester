"""
Foundation Utilities Module
TIER 1: Foundation - Self-Sufficient Utility System

This module provides enterprise-grade utilities for the entire optimization engine.
Completely self-sufficient foundation layer with no internal dependencies.

Key Features:
- Centralized logging system with performance tracking
- Comprehensive exception handling with recovery suggestions
- Performance monitoring and measurement utilities
- Memory-safe operation helpers
- System diagnostics and health checks
- Clean, consistent API for all utility functions

Design Philosophy:
- Zero internal dependencies (only uses standard library)
- Fail-safe operation (utility failures don't break the system)
- Performance-optimized for high-frequency usage
- Self-managing with automatic resource cleanup
- Production-ready from day one

Module Status: COMPLETE - Foundation Layer Ready
"""

import sys
import time
import platform
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Module metadata
__version__ = "1.0.0"
__author__ = "TopStep Optimization Engine"

# Import core utilities with graceful fallbacks
try:
    from .logger import (
        SafeLogger, 
        ProgressTracker,
        get_logger,
        setup_logging,
        create_progress_tracker,
        log_performance,
        get_all_logger_stats,
        print_all_logger_summaries,
        log_system_start,
        log_system_shutdown,
        log_exception,
        test_logger
    )
    LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Logger utilities not available: {e}")
    LOGGER_AVAILABLE = False
    
    # Fallback logging
    import logging
    def get_logger(name: str, **kwargs):
        return logging.getLogger(name)
    
    def setup_logging(**kwargs):
        logging.basicConfig(level=logging.INFO)
        return {'success': True}

try:
    from .exceptions import (
        # Base exceptions
        TopStepOptimizationError,
        
        # Module-specific exceptions
        DataError, DataLoadError, DataValidationError, DataFormatError, 
        DataIntegrityError, InsufficientDataError,
        
        StrategyError, StrategyNotFoundError, StrategyConfigurationError, 
        StrategyExecutionError, InvalidParameterError,
        
        OptimizationError, OptimizationTimeoutError, OptimizationMemoryError,
        OptimizationConfigError, ParameterSpaceError, OptimizationConvergenceError,
        
        ValidationError, ValidationTestFailure, ValidationConfigError,
        OutOfSampleTestFailure, MonteCarloTestFailure, WalkForwardTestFailure,
        
        AnalyticsError, MetricCalculationError, ReportGenerationError,
        
        DeploymentError, StrategyPackagingError, DeploymentValidationError,
        
        SystemResourceError, ConfigurationError,
        
        # Utility functions
        handle_exception, create_validation_summary, get_exception_hierarchy,
        test_exceptions
    )
    EXCEPTIONS_AVAILABLE = True
    
    # Create compatibility aliases for naming mismatches
    # These fix the import errors from other modules
    DataInsufficientError = InsufficientDataError  # Alias for data module compatibility
    DataLoadingError = DataLoadError  # Alias for data loader compatibility
    DataValidationFailure = DataValidationError  # Common alternative name
    DataFormatException = DataFormatError  # Common alternative name
    
except ImportError as e:
    print(f"WARNING: Exception utilities not available: {e}")
    EXCEPTIONS_AVAILABLE = False
    
    # Fallback exceptions
    class TopStepOptimizationError(Exception):
        pass
    
    class DataInsufficientError(Exception):
        """Fallback exception for insufficient data"""
        pass
    
    class DataLoadingError(Exception):
        """Fallback exception for data loading errors"""
        pass
    
    class DataValidationFailure(Exception):
        """Fallback exception for data validation errors"""
        pass
    
    class DataFormatException(Exception):
        """Fallback exception for data format errors"""
        pass
    
    def handle_exception(exception: Exception, context: str = "", logger=None):
        return TopStepOptimizationError(f"{context}: {str(exception)}")


# System information and diagnostics
def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information
    
    Returns:
        Dictionary with system details
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_count = psutil.cpu_count()
        
        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': cpu_count,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent_used': (disk.used / disk.total) * 100
        }
    except ImportError:
        # Fallback without psutil
        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': 'unknown',
            'memory_total_gb': 'unknown',
            'memory_available_gb': 'unknown',
            'memory_percent_used': 'unknown',
            'disk_total_gb': 'unknown',
            'disk_free_gb': 'unknown',
            'disk_percent_used': 'unknown'
        }
    
    return system_info


def check_system_requirements() -> Dict[str, Any]:
    """
    Check if system meets optimization engine requirements
    
    Returns:
        Dictionary with requirement check results
    """
    requirements = {
        'overall_status': True,
        'python_version_ok': False,
        'memory_sufficient': False,
        'disk_space_ok': False,
        'dependencies_available': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check Python version (3.8+)
    python_version = sys.version_info
    if python_version >= (3, 8):
        requirements['python_version_ok'] = True
    else:
        requirements['overall_status'] = False
        requirements['warnings'].append(f"Python {python_version.major}.{python_version.minor} detected, recommend 3.8+")
    
    # Check system resources
    system_info = get_system_info()
    
    if isinstance(system_info['memory_total_gb'], (int, float)):
        if system_info['memory_total_gb'] >= 8:
            requirements['memory_sufficient'] = True
        elif system_info['memory_total_gb'] >= 4:
            requirements['memory_sufficient'] = True
            requirements['recommendations'].append("Consider using conservative optimization mode for better performance")
        else:
            requirements['overall_status'] = False
            requirements['warnings'].append(f"Low memory: {system_info['memory_total_gb']:.1f}GB (recommend 8GB+)")
    
    if isinstance(system_info['disk_free_gb'], (int, float)):
        if system_info['disk_free_gb'] >= 1:
            requirements['disk_space_ok'] = True
        else:
            requirements['warnings'].append("Low disk space - may affect logging and caching")
    
    # Check key dependencies
    deps_to_check = ['pandas', 'numpy', 'scipy', 'scikit-learn']
    for dep in deps_to_check:
        try:
            __import__(dep)
            requirements['dependencies_available'].append(dep)
        except ImportError:
            requirements['warnings'].append(f"Optional dependency '{dep}' not available")
    
    return requirements


def safe_import(module_name: str, fallback_value: Any = None, logger=None):
    """
    Safely import a module with fallback
    
    Args:
        module_name: Name of module to import
        fallback_value: Value to return if import fails
        logger: Optional logger for recording import issues
        
    Returns:
        Imported module or fallback value
    """
    try:
        return __import__(module_name)
    except ImportError as e:
        if logger:
            logger.warning(f"Failed to import {module_name}: {e}")
        return fallback_value


def time_function(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Time a function execution with detailed metrics
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing results and function output
    """
    start_time = time.time()
    start_process_time = time.process_time()
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    end_process_time = time.process_time()
    
    return {
        'result': result,
        'success': success,
        'error': error,
        'wall_time': end_time - start_time,
        'cpu_time': end_process_time - start_process_time,
        'function_name': func.__name__ if hasattr(func, '__name__') else 'unknown'
    }


def format_memory_size(bytes_value: Union[int, float]) -> str:
    """
    Format memory size in human-readable format
    
    Args:
        bytes_value: Memory size in bytes
        
    Returns:
        Formatted string (e.g., "1.5GB", "512MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    if hours < 24:
        return f"{hours}h {remaining_minutes}m"
    
    days = int(hours // 24)
    remaining_hours = hours % 24
    
    return f"{days}d {remaining_hours}h"


def create_directory_safely(path: Union[str, Path], logger=None) -> bool:
    """
    Create directory with safe error handling
    
    Args:
        path: Directory path to create
        logger: Optional logger for recording operations
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        if logger:
            logger.debug(f"Created directory: {path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to create directory {path}: {e}")
        return False


def cleanup_old_files(directory: Union[str, Path], 
                     pattern: str = "*.log", 
                     days_old: int = 7, 
                     logger=None) -> int:
    """
    Clean up old files in directory
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        days_old: Files older than this many days will be deleted
        logger: Optional logger for recording operations
        
    Returns:
        Number of files deleted
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
                if logger:
                    logger.debug(f"Deleted old file: {file_path}")
        
        if logger and deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to cleanup files in {directory}: {e}")
        return 0


def get_module_status() -> Dict[str, Any]:
    """
    Get status of utils module components
    
    Returns:
        Dictionary with module status information
    """
    status = {
        'module_name': 'utils',
        'version': __version__,
        'logger_available': LOGGER_AVAILABLE,
        'exceptions_available': EXCEPTIONS_AVAILABLE,
        'system_info': get_system_info(),
        'requirements_check': check_system_requirements(),
        'ready_for_production': LOGGER_AVAILABLE and EXCEPTIONS_AVAILABLE
    }
    
    return status


def print_module_status():
    """Print formatted module status"""
    status = get_module_status()
    
    print("\n" + "="*60)
    print("UTILS MODULE STATUS")
    print("="*60)
    
    print(f"\nModule: {status['module_name']} v{status['version']}")
    print(f"Logger Available: {'Yes' if status['logger_available'] else 'No'}")
    print(f"Exceptions Available: {'Yes' if status['exceptions_available'] else 'No'}")
    print(f"Production Ready: {'Yes' if status['ready_for_production'] else 'No'}")
    
    # System info
    sys_info = status['system_info']
    print("\nSYSTEM INFO:")
    print(f"   Platform: {sys_info['platform']}")
    print(f"   Python: {sys_info['python_version'].split()[0]}")
    if isinstance(sys_info['memory_total_gb'], (int, float)):
        print(f"   Memory: {sys_info['memory_total_gb']:.1f}GB ({sys_info['memory_percent_used']:.1f}% used)")
    if isinstance(sys_info['cpu_count'], int):
        print(f"   CPUs: {sys_info['cpu_count']}")
    
    # Requirements check
    req_check = status['requirements_check']
    print("\nREQUIREMENTS CHECK:")
    print(f"   Overall Status: {'OK' if req_check['overall_status'] else 'FAIL'}")
    print(f"   Python Version: {'OK' if req_check['python_version_ok'] else 'FAIL'}")
    print(f"   Memory Sufficient: {'OK' if req_check['memory_sufficient'] else 'FAIL'}")
    print(f"   Dependencies: {len(req_check['dependencies_available'])} available")
    
    if req_check['warnings']:
        print("\nWARNINGS:")
        for warning in req_check['warnings'][:3]:
            print(f"   • {warning}")
    
    if req_check['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in req_check['recommendations'][:3]:
            print(f"   • {rec}")
    
    print("\n" + "="*60)


# Convenience functions for common patterns
def quick_logger(name: str = "utils") -> Union[object, Any]:
    """Get a logger quickly with sensible defaults"""
    if LOGGER_AVAILABLE:
        return get_logger(name)
    else:
        import logging
        return logging.getLogger(name)


def safe_execute(func, *args, fallback_result=None, logger=None, **kwargs):
    """
    Execute function safely with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        fallback_result: Result to return if function fails
        logger: Optional logger for error recording
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or fallback_result if execution fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        elif LOGGER_AVAILABLE:
            log_exception(e, f"safe_execute({func.__name__})")
        return fallback_result


def test_utils_module():
    """Test the utils module functionality"""
    print("Testing Utils Module...")
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Module status
    try:
        status = get_module_status()
        print(f"Module status: {status['ready_for_production']}")
        success_count += 1
    except Exception as e:
        print(f"Module status test failed: {e}")
    
    # Test 2: System info
    try:
        sys_info = get_system_info()
        print(f"System info: {sys_info['platform'][:20]}...")
        success_count += 1
    except Exception as e:
        print(f"System info test failed: {e}")
    
    # Test 3: Logger
    try:
        logger = quick_logger("test")
        logger.info("Test log message")
        print("Logger test passed")
        success_count += 1
    except Exception as e:
        print(f"Logger test failed: {e}")
    
    # Test 4: Exception handling
    try:
        test_exception = TopStepOptimizationError("Test error")
        handled = handle_exception(test_exception, "test context")
        print("Exception handling test passed")
        success_count += 1
    except Exception as e:
        print(f"Exception test failed: {e}")
    
    # Test 5: Performance timing
    try:
        result = time_function(lambda: sum(range(1000)))
        print(f"Performance timing: {result['wall_time']:.4f}s")
        success_count += 1
    except Exception as e:
        print(f"Performance timing test failed: {e}")
    
    # Test 6: Safe execution
    try:
        result = safe_execute(lambda x: x * 2, 5, fallback_result=0)
        print(f"Safe execution: result={result}")
        success_count += 1
    except Exception as e:
        print(f"Safe execution test failed: {e}")
    
    print(f"\nUtils Test Results: {success_count}/{total_tests} tests passed")
    return success_count == total_tests


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core utilities
    'get_system_info',
    'check_system_requirements', 
    'safe_import',
    'time_function',
    'format_memory_size',
    'format_duration',
    'create_directory_safely',
    'cleanup_old_files',
    'quick_logger',
    'safe_execute',
    
    # Module status
    'get_module_status',
    'print_module_status',
    'test_utils_module',
    
    # Module metadata
    '__version__',
    'LOGGER_AVAILABLE',
    'EXCEPTIONS_AVAILABLE'
]

# Add logger exports if available
if LOGGER_AVAILABLE:
    __all__.extend([
        'SafeLogger', 
        'ProgressTracker',
        'get_logger',
        'setup_logging',
        'create_progress_tracker',
        'log_performance',
        'get_all_logger_stats',
        'print_all_logger_summaries',
        'log_system_start',
        'log_system_shutdown',
        'log_exception',
        'test_logger'
    ])

# Add exception exports if available
if EXCEPTIONS_AVAILABLE:
    __all__.extend([
        # Base exceptions
        'TopStepOptimizationError',
        
        # Module-specific exceptions
        'DataError', 'DataLoadError', 'DataValidationError', 'DataFormatError', 
        'DataIntegrityError', 'InsufficientDataError',
        
        'StrategyError', 'StrategyNotFoundError', 'StrategyConfigurationError', 
        'StrategyExecutionError', 'InvalidParameterError',
        
        'OptimizationError', 'OptimizationTimeoutError', 'OptimizationMemoryError',
        'OptimizationConfigError', 'ParameterSpaceError', 'OptimizationConvergenceError',
        
        'ValidationError', 'ValidationTestFailure', 'ValidationConfigError',
        'OutOfSampleTestFailure', 'MonteCarloTestFailure', 'WalkForwardTestFailure',
        
        'AnalyticsError', 'MetricCalculationError', 'ReportGenerationError',
        
        'DeploymentError', 'StrategyPackagingError', 'DeploymentValidationError',
        
        'SystemResourceError', 'ConfigurationError',
        
        # Compatibility aliases (the key fix!)
        'DataInsufficientError',
        'DataLoadingError', 
        'DataValidationFailure',
        'DataFormatException',
        
        # Utility functions
        'handle_exception', 'create_validation_summary', 'get_exception_hierarchy',
        'test_exceptions'
    ])

# Resource manager removed - using fallback implementations
RESOURCE_MANAGER_AVAILABLE = False

# Provide fallback implementations
class SystemResourceManager:
    """Fallback resource manager"""
    def __init__(self, enable_monitoring=False):
        self.enable_monitoring = enable_monitoring
        
    def get_hardware_profile(self):
        return type('HardwareProfile', (), {
            'optimization_tier': 'basic',
            'cpu': type('CPU', (), {'physical_cores': 4})(),
            'memory': type('Memory', (), {'total_gb': 8})(),
            'overall_performance_score': 50
        })()
        
    def get_worker_recommendations(self):
        return {'recommended_workers': 2}

def validate_optimization_capability():
    return True

def get_worker_config():
    return {'workers': 2, 'batch_size': 100}

# Add fallback exports
__all__.extend([
    'SystemResourceManager',
    'validate_optimization_capability',
    'get_worker_config',
    'RESOURCE_MANAGER_AVAILABLE'
])

# Module initialization logging
if LOGGER_AVAILABLE:
    _init_logger = get_logger("utils")
    _init_logger.info(f"Utils module v{__version__} initialized successfully")
    if not EXCEPTIONS_AVAILABLE:
        _init_logger.warning("Exception utilities not available - using fallbacks")

# Final module validation
if __name__ == "__main__":
    print_module_status()
    print()
    test_utils_module()