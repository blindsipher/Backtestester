"""
Standardized Error Handling Utilities
=====================================

Centralized error handling patterns to ensure consistency across modules.
Provides factory functions for creating standardized error results.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# ARCHITECTURAL FIX: Remove constants.py dependency
ERROR_RESULT_KEYS = ('success', 'error', 'best_parameters', 'optimization_metadata')  # Error result keys

logger = logging.getLogger(__name__)


class ErrorResultFactory:
    """Factory for creating standardized error result dictionaries."""
    
    @staticmethod
    def create_error_result(error_message: str, 
                          error_context: Optional[str] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized error result dictionary.
        
        Args:
            error_message: The error message
            error_context: Optional context about where the error occurred
            additional_data: Optional additional data to include
            
        Returns:
            Standardized error result dictionary
        """
        result = {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'best_parameters': [],
            'optimization_metadata': {}
        }
        
        if error_context:
            result['error_context'] = error_context
        
        if additional_data:
            result.update(additional_data)
        
        return result
    
    @staticmethod
    def create_success_result(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized success result dictionary.
        
        Args:
            data: Result data to include
            
        Returns:
            Standardized success result dictionary
        """
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
        }
        result.update(data)
        return result
    
    @staticmethod
    def create_partial_success_result(data: Dict[str, Any], 
                                    warnings: List[str]) -> Dict[str, Any]:
        """
        Create a result for operations that succeeded with warnings.
        
        Args:
            data: Result data to include
            warnings: List of warning messages
            
        Returns:
            Partial success result dictionary
        """
        result = {
            'success': True,
            'partial': True,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat(),
        }
        result.update(data)
        return result


class PipelineErrorHandler:
    """Specialized error handler for pipeline operations."""
    
    @staticmethod
    def handle_phase_error(phase_name: str, 
                         error: Exception, 
                         state: Optional[Any] = None) -> Dict[str, Any]:
        """
        Handle errors that occur during pipeline phases.
        
        Args:
            phase_name: Name of the phase where error occurred
            error: The exception that occurred
            state: Optional pipeline state for additional context
            
        Returns:
            Standardized error result
        """
        error_msg = f"{phase_name} failed: {str(error)}"
        
        additional_data = {
            'phase': phase_name,
            'error_type': type(error).__name__
        }
        
        if state:
            additional_data['pipeline_state'] = getattr(state, 'pipeline_phase', 'unknown')
        
        logger.error(error_msg, exc_info=True)
        
        return ErrorResultFactory.create_error_result(
            error_message=error_msg,
            error_context=phase_name,
            additional_data=additional_data
        )
    
    @staticmethod
    def handle_validation_error(validation_type: str, 
                              validation_error: str,
                              data_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle validation errors with structured context.
        
        Args:
            validation_type: Type of validation that failed
            validation_error: Description of the validation failure
            data_context: Optional context about the data being validated
            
        Returns:
            Standardized validation error result
        """
        error_msg = f"{validation_type} validation failed: {validation_error}"
        
        additional_data = {
            'validation_type': validation_type,
            'validation_context': data_context or {}
        }
        
        logger.error(error_msg)
        
        return ErrorResultFactory.create_error_result(
            error_message=error_msg,
            error_context=f"{validation_type}_validation",
            additional_data=additional_data
        )


def log_and_return_error(error_message: str, 
                        context: Optional[str] = None,
                        exc_info: bool = False) -> Dict[str, Any]:
    """
    Convenience function to log error and return standardized error result.
    
    Args:
        error_message: The error message
        context: Optional context information
        exc_info: Whether to include exception information in log
        
    Returns:
        Standardized error result dictionary
    """
    full_message = f"{context}: {error_message}" if context else error_message
    logger.error(full_message, exc_info=exc_info)
    
    return ErrorResultFactory.create_error_result(
        error_message=error_message,
        error_context=context
    )


def safe_execute(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a function with standardized error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Either the function result or a standardized error result
    """
    try:
        result = func(*args, **kwargs)
        
        # If the function already returns a standardized result, return it
        if isinstance(result, dict) and 'success' in result:
            return result
        
        # Otherwise, wrap it in a success result
        return ErrorResultFactory.create_success_result({'result': result})
        
    except Exception as e:
        return PipelineErrorHandler.handle_phase_error(
            phase_name=func.__name__,
            error=e
        )


def validate_required_keys(data: Dict[str, Any], 
                         required_keys: List[str],
                         validation_context: str) -> Optional[str]:
    """
    Validate that required keys are present in a dictionary.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        validation_context: Context for error messages
        
    Returns:
        None if valid, error message if invalid
    """
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        return f"{validation_context} missing required keys: {missing_keys}"
    
    return None


def ensure_error_result_format(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a result dictionary has the standard error result format.
    
    Args:
        result: Result dictionary to standardize
        
    Returns:
        Standardized result dictionary
    """
    if not isinstance(result, dict):
        return ErrorResultFactory.create_error_result("Invalid result format")
    
    # Ensure required keys are present
    if 'success' not in result:
        result['success'] = False
    
    if not result['success'] and 'error' not in result:
        result['error'] = "Unknown error"
    
    # Add missing standard keys
    for key in ERROR_RESULT_KEYS:
        if key not in result:
            if key == 'best_parameters':
                result[key] = []
            elif key == 'optimization_metadata':
                result[key] = {}
    
    return result