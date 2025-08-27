"""
Custom Exception Classes
TIER 1: Foundation - Self-Sufficient Exception Handling

This module provides a comprehensive hierarchy of custom exceptions for the
entire optimization engine. Completely self-sufficient with no internal dependencies.

Key Features:
- Hierarchical exception structure for precise error handling
- Rich error context with actionable error messages
- Performance-safe exception handling
- Automatic error classification and recovery suggestions
- Integration with logging system
- Exception chaining and context preservation
- Validation-specific exceptions with detailed diagnostics

Design Philosophy:
- Clear, actionable error messages
- Hierarchical structure for easy catching
- Rich context for debugging
- Recovery suggestions when possible
- Performance-safe (minimal overhead)
- Self-documenting exception types
"""

import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime


class TopStepOptimizationError(Exception):
    """
    Base exception for all optimization engine errors
    
    Features:
    - Rich error context
    - Automatic timestamp recording
    - Recovery suggestions
    - Error classification
    """
    
    def __init__(self, message: str, 
                 context: Optional[Dict[str, Any]] = None,
                 recovery_suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        """
        Initialize base exception
        
        Args:
            message: Human-readable error description
            context: Additional context data
            recovery_suggestion: Suggested fix for the error
            error_code: Machine-readable error code
        """
        super().__init__(message)
        
        self.message = message
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.error_code = error_code
        self.timestamp = datetime.now()
        self.traceback_info = self._capture_traceback()
    
    def _capture_traceback(self) -> str:
        """Capture current traceback for debugging"""
        try:
            return ''.join(traceback.format_stack()[:-1])
        except Exception:
            return "Traceback unavailable"
    
    def get_detailed_message(self) -> str:
        """Get detailed error message with context"""
        details = [f"Error: {self.message}"]
        
        if self.error_code:
            details.append(f"Code: {self.error_code}")
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            details.append(f"Context: {context_str}")
        
        if self.recovery_suggestion:
            details.append(f"Suggestion: {self.recovery_suggestion}")
        
        details.append(f"Time: {self.timestamp.isoformat()}")
        
        return " | ".join(details)
    
    def __str__(self) -> str:
        return self.get_detailed_message()


# =============================================================================
# DATA MODULE EXCEPTIONS
# =============================================================================

class DataError(TopStepOptimizationError):
    """Base exception for all data-related errors"""
    pass


class DataLoadError(DataError):
    """Error loading data from file or source"""
    
    def __init__(self, filepath: str, reason: str, **kwargs):
        message = f"Failed to load data from '{filepath}': {reason}"
        context = {"filepath": filepath, "reason": reason}
        recovery = "Check file exists, format is correct, and you have read permissions"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class DataValidationError(DataError):
    """Error in data validation"""
    
    def __init__(self, validation_failures: List[str], 
                 data_shape: Optional[tuple] = None, **kwargs):
        message = f"Data validation failed: {', '.join(validation_failures[:3])}"
        if len(validation_failures) > 3:
            message += f" (and {len(validation_failures) - 3} more issues)"
        
        context = {
            "validation_failures": validation_failures,
            "total_failures": len(validation_failures)
        }
        
        if data_shape:
            context["data_shape"] = data_shape
        
        recovery = "Fix data issues or use auto-fix functionality"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class DataFormatError(DataError):
    """Error in data format or structure"""
    
    def __init__(self, expected_format: str, actual_format: str, **kwargs):
        message = f"Data format mismatch: expected {expected_format}, got {actual_format}"
        context = {"expected": expected_format, "actual": actual_format}
        recovery = "Convert data to expected format or update format specification"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class DataIntegrityError(DataError):
    """Error in data integrity (OHLCV relationships, etc.)"""
    
    def __init__(self, integrity_issues: List[str], **kwargs):
        message = f"Data integrity issues: {', '.join(integrity_issues[:2])}"
        context = {"integrity_issues": integrity_issues}
        recovery = "Use data cleaning utilities or manual data correction"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class InsufficientDataError(DataError):
    """Not enough data for operation"""
    
    def __init__(self, required: int, available: int, operation: str = "analysis", **kwargs):
        message = f"Insufficient data for {operation}: need {required}, have {available}"
        context = {"required": required, "available": available, "operation": operation}
        recovery = "Provide more data or reduce requirements"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# STRATEGY MODULE EXCEPTIONS
# =============================================================================

class StrategyError(TopStepOptimizationError):
    """Base exception for all strategy-related errors"""
    pass


class StrategyNotFoundError(StrategyError):
    """Strategy not found in registry"""
    
    def __init__(self, strategy_name: str, available_strategies: Optional[List[str]] = None, **kwargs):
        message = f"Strategy '{strategy_name}' not found"
        context = {"strategy_name": strategy_name}
        
        if available_strategies:
            context["available_strategies"] = available_strategies
            recovery = f"Use one of: {', '.join(available_strategies)}"
        else:
            recovery = "Check strategy name spelling and ensure strategy is registered"
        
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class StrategyConfigurationError(StrategyError):
    """Error in strategy configuration"""
    
    def __init__(self, config_issues: List[str], strategy_name: str = "Unknown", **kwargs):
        message = f"Strategy '{strategy_name}' configuration error: {', '.join(config_issues[:2])}"
        context = {"strategy_name": strategy_name, "config_issues": config_issues}
        recovery = "Fix configuration parameters or use default configuration"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class StrategyExecutionError(StrategyError):
    """Error during strategy execution"""
    
    def __init__(self, strategy_name: str, execution_step: str, original_error: str, **kwargs):
        message = f"Strategy '{strategy_name}' failed at {execution_step}: {original_error}"
        context = {
            "strategy_name": strategy_name, 
            "execution_step": execution_step,
            "original_error": original_error
        }
        recovery = "Check strategy parameters and input data quality"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class InvalidParameterError(StrategyError):
    """Invalid parameter value for strategy"""
    
    def __init__(self, param_name: str, param_value: Any, 
                 valid_range: Optional[str] = None, **kwargs):
        message = f"Invalid parameter '{param_name}': {param_value}"
        context = {"param_name": param_name, "param_value": param_value}
        
        if valid_range:
            message += f" (valid range: {valid_range})"
            context["valid_range"] = valid_range
        
        recovery = "Use valid parameter values within specified ranges"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# OPTIMIZATION MODULE EXCEPTIONS
# =============================================================================

class OptimizationError(TopStepOptimizationError):
    """Base exception for all optimization-related errors"""
    pass


class OptimizationTimeoutError(OptimizationError):
    """Optimization timed out"""
    
    def __init__(self, timeout_minutes: float, completed_evaluations: int, **kwargs):
        message = f"Optimization timed out after {timeout_minutes:.1f} minutes ({completed_evaluations} evaluations)"
        context = {"timeout_minutes": timeout_minutes, "completed_evaluations": completed_evaluations}
        recovery = "Increase timeout or reduce optimization scope"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class OptimizationMemoryError(OptimizationError):
    """Insufficient memory for optimization"""
    
    def __init__(self, required_mb: float, available_mb: float, **kwargs):
        message = f"Insufficient memory: need {required_mb:.1f}MB, have {available_mb:.1f}MB"
        context = {"required_mb": required_mb, "available_mb": available_mb}
        recovery = "Reduce batch size, enable memory-conservative mode, or use smaller dataset"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class OptimizationConfigError(OptimizationError):
    """Error in optimization configuration"""
    
    def __init__(self, config_issue: str, **kwargs):
        message = f"Optimization configuration error: {config_issue}"
        recovery = "Check optimization parameters and constraints"
        super().__init__(message, recovery_suggestion=recovery, **kwargs)


class ParameterSpaceError(OptimizationError):
    """Error in parameter space definition"""
    
    def __init__(self, parameter: str, issue: str, **kwargs):
        message = f"Parameter space error for '{parameter}': {issue}"
        context = {"parameter": parameter, "issue": issue}
        recovery = "Fix parameter bounds and constraints"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class OptimizationConvergenceError(OptimizationError):
    """Optimization failed to converge"""
    
    def __init__(self, method: str, iterations: int, best_score: float, **kwargs):
        message = f"{method} failed to converge after {iterations} iterations (best: {best_score:.4f})"
        context = {"method": method, "iterations": iterations, "best_score": best_score}
        recovery = "Increase iterations, adjust parameters, or try different optimization method"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# VALIDATION MODULE EXCEPTIONS
# =============================================================================

class ValidationError(TopStepOptimizationError):
    """Base exception for all validation-related errors"""
    pass


class ValidationTestFailure(ValidationError):
    """A validation test failed"""
    
    def __init__(self, test_name: str, failure_reason: str, 
                 test_details: Optional[Dict[str, Any]] = None, **kwargs):
        message = f"Validation test '{test_name}' failed: {failure_reason}"
        context = {"test_name": test_name, "failure_reason": failure_reason}
        
        if test_details:
            context.update(test_details)
        
        recovery = "Review strategy parameters or adjust validation criteria"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class ValidationConfigError(ValidationError):
    """Error in validation configuration"""
    
    def __init__(self, config_issue: str, **kwargs):
        message = f"Validation configuration error: {config_issue}"
        recovery = "Check validation test parameters and settings"
        super().__init__(message, recovery_suggestion=recovery, **kwargs)


class OutOfSampleTestFailure(ValidationTestFailure):
    """Out-of-sample validation test failed"""
    
    def __init__(self, oos_score: float, min_required: float, **kwargs):
        failure_reason = f"Score {oos_score:.4f} below required {min_required:.4f}"
        context = {"oos_score": oos_score, "min_required": min_required}
        super().__init__("out_of_sample", failure_reason, context, **kwargs)


class MonteCarloTestFailure(ValidationTestFailure):
    """Monte Carlo validation test failed"""
    
    def __init__(self, p_value: float, significance_level: float, **kwargs):
        failure_reason = f"p-value {p_value:.4f} above significance level {significance_level:.4f}"
        context = {"p_value": p_value, "significance_level": significance_level}
        super().__init__("monte_carlo", failure_reason, context, **kwargs)


class WalkForwardTestFailure(ValidationTestFailure):
    """Walk-forward validation test failed"""
    
    def __init__(self, failing_periods: int, total_periods: int, **kwargs):
        failure_reason = f"{failing_periods}/{total_periods} periods failed"
        context = {"failing_periods": failing_periods, "total_periods": total_periods}
        super().__init__("walk_forward", failure_reason, context, **kwargs)


# =============================================================================
# ANALYTICS MODULE EXCEPTIONS
# =============================================================================

class AnalyticsError(TopStepOptimizationError):
    """Base exception for analytics-related errors"""
    pass


class MetricCalculationError(AnalyticsError):
    """Error calculating performance metrics"""
    
    def __init__(self, metric_name: str, calculation_error: str, **kwargs):
        message = f"Failed to calculate {metric_name}: {calculation_error}"
        context = {"metric_name": metric_name, "calculation_error": calculation_error}
        recovery = "Check input data quality and metric parameters"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class ReportGenerationError(AnalyticsError):
    """Error generating reports"""
    
    def __init__(self, report_type: str, generation_error: str, **kwargs):
        message = f"Failed to generate {report_type} report: {generation_error}"
        context = {"report_type": report_type, "generation_error": generation_error}
        recovery = "Check data availability and report template"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# DEPLOYMENT MODULE EXCEPTIONS
# =============================================================================

class DeploymentError(TopStepOptimizationError):
    """Base exception for deployment-related errors"""
    pass


class StrategyPackagingError(DeploymentError):
    """Error packaging strategy for deployment"""
    
    def __init__(self, strategy_name: str, packaging_error: str, **kwargs):
        message = f"Failed to package strategy '{strategy_name}': {packaging_error}"
        context = {"strategy_name": strategy_name, "packaging_error": packaging_error}
        recovery = "Check strategy configuration and output directory permissions"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class DeploymentValidationError(DeploymentError):
    """Error validating deployment package"""
    
    def __init__(self, validation_failures: List[str], **kwargs):
        message = f"Deployment validation failed: {', '.join(validation_failures[:2])}"
        context = {"validation_failures": validation_failures}
        recovery = "Fix deployment package issues before deployment"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# SYSTEM EXCEPTIONS
# =============================================================================

class SystemResourceError(TopStepOptimizationError):
    """System resource unavailable"""
    
    def __init__(self, resource_type: str, resource_details: str, **kwargs):
        message = f"System resource unavailable: {resource_type} - {resource_details}"
        context = {"resource_type": resource_type, "resource_details": resource_details}
        recovery = "Free up system resources or reduce operation scope"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


class ConfigurationError(TopStepOptimizationError):
    """System configuration error"""
    
    def __init__(self, config_component: str, config_issue: str, **kwargs):
        message = f"Configuration error in {config_component}: {config_issue}"
        context = {"config_component": config_component, "config_issue": config_issue}
        recovery = "Check system configuration files and settings"
        super().__init__(message, context=context, recovery_suggestion=recovery, **kwargs)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def handle_exception(exception: Exception, context: str = "Unknown operation", 
                    logger=None) -> TopStepOptimizationError:
    """
    Convert any exception to a TopStep exception with proper context
    
    Args:
        exception: Original exception
        context: Context where exception occurred
        logger: Optional logger to record exception
        
    Returns:
        TopStepOptimizationError subclass
    """
    # Map common exceptions to our hierarchy
    if isinstance(exception, FileNotFoundError):
        error = DataLoadError(str(exception), "File not found")
    elif isinstance(exception, MemoryError):
        error = OptimizationMemoryError(0, 0)  # Will be filled by caller
    elif isinstance(exception, TimeoutError):
        error = OptimizationTimeoutError(0, 0)  # Will be filled by caller
    elif isinstance(exception, ValueError):
        error = InvalidParameterError("unknown", str(exception))
    elif isinstance(exception, TopStepOptimizationError):
        # Already our exception type
        error = exception
    else:
        # Generic wrapper
        error = TopStepOptimizationError(
            f"{context}: {str(exception)}", 
            context={"original_exception": type(exception).__name__}
        )
    
    # Log if logger provided
    if logger:
        logger.error(f"Exception in {context}", exception=error)
    
    return error


def create_validation_summary(exceptions: List[Exception]) -> str:
    """
    Create a summary of validation exceptions
    
    Args:
        exceptions: List of exceptions from validation
        
    Returns:
        Human-readable summary
    """
    if not exceptions:
        return "All validation tests passed"
    
    summary_lines = [f"{len(exceptions)} validation issues:"]
    
    for i, exc in enumerate(exceptions[:5], 1):  # Show first 5
        if isinstance(exc, ValidationTestFailure):
            summary_lines.append(f"  {i}. {exc.context.get('test_name', 'Unknown test')}: {exc.message}")
        else:
            summary_lines.append(f"  {i}. {type(exc).__name__}: {str(exc)}")
    
    if len(exceptions) > 5:
        summary_lines.append(f"  ... and {len(exceptions) - 5} more issues")
    
    return "\n".join(summary_lines)


def get_exception_hierarchy() -> Dict[str, List[str]]:
    """Get the exception hierarchy for documentation"""
    hierarchy = {
        "TopStepOptimizationError": [
            "DataError",
            "StrategyError", 
            "OptimizationError",
            "ValidationError",
            "AnalyticsError",
            "DeploymentError",
            "SystemResourceError",
            "ConfigurationError"
        ],
        "DataError": [
            "DataLoadError",
            "DataValidationError", 
            "DataFormatError",
            "DataIntegrityError",
            "InsufficientDataError"
        ],
        "StrategyError": [
            "StrategyNotFoundError",
            "StrategyConfigurationError",
            "StrategyExecutionError", 
            "InvalidParameterError"
        ],
        "OptimizationError": [
            "OptimizationTimeoutError",
            "OptimizationMemoryError",
            "OptimizationConfigError",
            "ParameterSpaceError",
            "OptimizationConvergenceError"
        ],
        "ValidationError": [
            "ValidationTestFailure",
            "ValidationConfigError",
            "OutOfSampleTestFailure",
            "MonteCarloTestFailure", 
            "WalkForwardTestFailure"
        ],
        "AnalyticsError": [
            "MetricCalculationError",
            "ReportGenerationError"
        ],
        "DeploymentError": [
            "StrategyPackagingError",
            "DeploymentValidationError"
        ]
    }
    return hierarchy


def test_exceptions():
    """Test the exception system"""
    print("Testing exception system...")
    
    try:
        # Test base exception
        raise TopStepOptimizationError("Test error", 
                                      context={"test": True},
                                      recovery_suggestion="This is just a test")
    except TopStepOptimizationError as e:
        print(f"Base exception: {e.get_detailed_message()}")
    
    try:
        # Test data exception
        raise DataValidationError(["Missing columns", "Invalid dates"], data_shape=(1000, 5))
    except DataValidationError as e:
        print(f"Data exception: {e}")
    
    try:
        # Test validation exception
        raise OutOfSampleTestFailure(0.85, 1.0)
    except ValidationTestFailure as e:
        print(f"Validation exception: {e}")
    
    # Test exception hierarchy
    hierarchy = get_exception_hierarchy()
    print(f"Exception hierarchy: {len(hierarchy)} categories")
    
    print("Exception system test complete")
    return True


if __name__ == "__main__":
    test_exceptions()