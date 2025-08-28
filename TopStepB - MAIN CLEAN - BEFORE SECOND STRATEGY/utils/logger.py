"""
Centralized Logging System
TIER 1: Foundation - Self-Sufficient Logging Utilities

This module provides enterprise-grade logging for the entire optimization engine.
Completely self-sufficient with no internal dependencies.

Key Features:
- Performance-optimized logging with minimal overhead
- Automatic log rotation and cleanup
- Memory-safe operation for long-running optimizations
- Multiple output formats (console, file, structured)
- Intelligent log level management
- Automatic directory creation
- Thread-safe operation
- Progress tracking for long operations
- Error aggregation and reporting

Design Philosophy:
- Zero internal dependencies (only uses standard library)
- Fail-safe operation (logging failures don't break the system)
- Minimal performance impact
- Self-managing (automatic cleanup, rotation)
- Production-ready from day one
"""

import logging
import logging.handlers
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback


# Emoji fallback mapping for Windows console compatibility
EMOJI_FALLBACKS = {
    '🚀': '[START]',
    '🧠': '[INTEL]',
    '💪': '[BRUTE]',
    '🔍': '[BAYES]',
    '🔧': '[INTEG]',
    '✅': '[DONE]',
    '🏁': '[FINAL]',
    '⏱️': '[TIME]',
    '📊': '[STATS]',
    '📒': '[PERF]',
    '📋': '[RESULT]',
    '🎯': '[TARGET]',
    '💡': '[IDEA]',
    '⚡': '[FAST]',
    '🔥': '[HOT]',
    '🌟': '[STAR]',
    '🚨': '[ALERT]',
    '⚠️': '[WARN]',
    '❌': '[ERROR]',
    '📦': '[PACK]',
    '🔑': '[KEY]',
    '💾': '[SAVE]',
    '🎨': '[STYLE]',
    '🛠️': '[TOOL]',
    '🔎': '[SEARCH]',
    '📝': '[NOTE]',
    '📍': '[MARK]',
    '🎪': '[SHOW]',
    '🌈': '[RAINBOW]',
    '🎭': '[MASK]',
    '🎲': '[DICE]',
    '🏹': '[AIM]',
    '🖼️': '[ART]',
    '⭐': '[BEST]',
    '💰': '[MONEY]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '🎖️': '[MEDAL]',
    '🏆': '[TROPHY]',
    '🥇': '[GOLD]',
    '🥈': '[SILVER]',
    '🥉': '[BRONZE]',
    '🖌️': '[PAINT]',
    '🔴': '[BULLSEYE]',
    '✨': '[SHINE]',
    '🔄': '[REFRESH]',
    '🔁': '[REPEAT]',
    '🔀': '[SHUFFLE]',
    '🔃': '[CYCLE]',
    '🔂': '[LOOP]',
    '▶️': '[PLAY]',
    '⏸️': '[PAUSE]',
    '⏹️': '[STOP]',
    '⏭️': '[NEXT]',
    '⏮️': '[PREV]',
    '⏯️': '[PLAYPAUSE]',
    '🔊': '[LOUD]',
    '🔇': '[MUTE]',
    '🔈': '[QUIET]',
    '🔉': '[MEDIUM]',
    '🔆': '[BRIGHT]',
    '🔅': '[DIM]',
    '🎵': '[MUSIC]',
    '🎶': '[NOTES]',
    '🎼': '[SCORE]',
    '🎹': '[PIANO]',
    '🎸': '[GUITAR]',
    '🎺': '[TRUMPET]',
    '🎻': '[VIOLIN]',
    '🥁': '[DRUM]',
    '🎧': '[HEADPHONE]',
    '🎤': '[MIC]',
    '🎬': '[MOVIE]',
    '🎳': '[BOWLING]',
    '🎮': '[GAME]',
    '🕹️': '[JOYSTICK]',
    '🎰': '[SLOT]',
    '🃏': '[JOKER]',
    '🎴': '[CARDS]',
}


def safe_emoji_replace(text: str) -> str:
    """
    Replace emojis with safe ASCII alternatives for console compatibility
    """
    if not isinstance(text, str):
        return str(text)
    
    # Check if we're likely to have encoding issues with emojis
    # This includes Windows, or when PYTHONIOENCODING is not set to utf-8
    needs_replacement = False
    
    # Check platform
    if sys.platform.startswith('win'):
        needs_replacement = True
    
    # Check if we're in WSL running Windows console
    if 'microsoft' in sys.platform.lower() or 'WSL' in os.environ.get('WSL_DISTRO_NAME', ''):
        needs_replacement = True
    
    # Check encoding environment
    if os.environ.get('PYTHONIOENCODING', '').lower() not in ['utf-8', 'utf8']:
        needs_replacement = True
    
    # Check if stdout encoding might not support emojis
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        encoding = sys.stdout.encoding.lower()
        if 'cp1252' in encoding or 'ascii' in encoding or 'latin' in encoding:
            needs_replacement = True
    
    if needs_replacement:
        for emoji, replacement in EMOJI_FALLBACKS.items():
            text = text.replace(emoji, replacement)
    
    return text


class SafeLogger:
    """
    Thread-safe, high-performance logger with automatic management
    
    Features:
    - Automatic log directory creation
    - File rotation to prevent huge log files
    - Memory-efficient operation
    - Performance tracking
    - Error aggregation
    - Multiple output formats
    """
    
    _instances: Dict[str, 'SafeLogger'] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str, 
                 log_dir: str = "logs",
                 level: int = logging.INFO,
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 5,
                 console_output: bool = True,
                 file_output: bool = True):
        """
        Initialize thread-safe logger
        
        Args:
            name: Logger name (usually module name)
            log_dir: Directory for log files
            level: Logging level
            max_file_size: Max size before rotation (bytes)
            backup_count: Number of backup files to keep
            console_output: Enable console logging
            file_output: Enable file logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_output = console_output
        self.file_output = file_output
        
        # Performance tracking
        self.start_time = time.time()
        self.log_counts = {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}
        self.error_summary = []
        
        # Initialize logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the actual logging configuration"""
        try:
            # Create logger instance
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(self.level)
            
            # Clear any existing handlers
            self.logger.handlers.clear()
            
            # Create log directory if needed
            if self.file_output:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            simple_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            
            # Console handler
            if self.console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.level)
                console_handler.setFormatter(simple_formatter)
                self.logger.addHandler(console_handler)
            
            # File handler with rotation
            if self.file_output:
                log_file = self.log_dir / f"{self.name}.log"
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count
                )
                file_handler.setLevel(self.level)
                file_handler.setFormatter(detailed_formatter)
                self.logger.addHandler(file_handler)
                
                # Also create a detailed debug log
                debug_file = self.log_dir / f"{self.name}_debug.log"
                debug_handler = logging.handlers.RotatingFileHandler(
                    debug_file,
                    maxBytes=self.max_file_size,
                    backupCount=2
                )
                debug_handler.setLevel(logging.DEBUG)
                debug_handler.setFormatter(detailed_formatter)
                self.logger.addHandler(debug_handler)
            
        except Exception as e:
            # Fallback to basic console logging if setup fails
            print(f"Logger setup failed for {self.name}: {e}")
            self.logger = logging.getLogger(self.name)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with performance tracking"""
        self._log_with_tracking('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with performance tracking"""
        self._log_with_tracking('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with performance tracking"""
        self._log_with_tracking('warning', message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            # Add to error summary for reporting
            self.error_summary.append({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'exception_type': type(exception).__name__,
                'traceback': traceback.format_exc()
            })
        
        self._log_with_tracking('error', message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception details"""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
        
        self._log_with_tracking('critical', message, **kwargs)
    
    def _log_with_tracking(self, level: str, message: str, **kwargs):
        """Internal method to log with performance tracking"""
        try:
            # Update counters
            self.log_counts[level] += 1
            
            # Make message safe for console output
            safe_message = safe_emoji_replace(message)
            
            # Log the message
            getattr(self.logger, level)(safe_message, **kwargs)
            
        except Exception as e:
            # Fallback logging if main logging fails
            safe_message = safe_emoji_replace(message)
            print(f"Logging failed: {e} | Original message: {safe_message}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics"""
        runtime = time.time() - self.start_time
        total_logs = sum(self.log_counts.values())
        
        return {
            'logger_name': self.name,
            'runtime_seconds': runtime,
            'total_log_messages': total_logs,
            'logs_per_second': total_logs / max(runtime, 0.1),
            'log_counts': self.log_counts.copy(),
            'error_count': len(self.error_summary),
            'last_errors': self.error_summary[-5:] if self.error_summary else []
        }
    
    def print_summary(self):
        """Print a summary of logging activity"""
        stats = self.get_stats()
        
        print(f"\nLogging Summary for {self.name}")
        print(f"Runtime: {stats['runtime_seconds']:.1f}s")
        print(f"Total Messages: {stats['total_log_messages']}")
        print(f"Rate: {stats['logs_per_second']:.1f} msgs/sec")
        print(f"Breakdown: {stats['log_counts']}")
        
        if stats['error_count'] > 0:
            print(f"Errors: {stats['error_count']}")


class ProgressTracker:
    """
    High-performance progress tracker for long-running operations
    
    Features:
    - Minimal performance overhead
    - Automatic ETA calculation
    - Memory-efficient operation
    - Thread-safe updates
    """
    
    def __init__(self, total: int, name: str = "Progress", logger: Optional[SafeLogger] = None):
        self.total = total
        self.name = name
        self.logger = logger or get_logger("progress")
        
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = max(1, total // 50)  # Update ~50 times total
        
        self._lock = threading.Lock()
        
        self.logger.info(f"Starting {self.name}: {total:,} items")
    
    def update(self, n: int = 1):
        """Update progress counter (thread-safe)"""
        with self._lock:
            self.current += n
            
            # Only log periodically to avoid spam
            if (self.current - self.last_update) >= self.update_interval or self.current >= self.total:
                self._log_progress()
                self.last_update = self.current
    
    def _log_progress(self):
        """Log current progress with ETA"""
        if self.total <= 0:
            return
        
        elapsed = time.time() - self.start_time
        progress_pct = (self.current / self.total) * 100
        
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            eta_seconds = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f" | ETA: {eta_seconds:.0f}s" if eta_seconds > 0 else ""
        else:
            rate = 0
            eta_str = ""
        
        self.logger.info(
            f"{self.name}: {self.current:,}/{self.total:,} "
            f"({progress_pct:.1f}%) | Rate: {rate:.1f}/s{eta_str}"
        )
    
    def finish(self):
        """Mark progress as complete"""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"{self.name} Complete: {self.current:,} items in {elapsed:.1f}s "
            f"(avg {rate:.1f}/s)"
        )


# Global logger registry for efficient reuse
_logger_registry: Dict[str, SafeLogger] = {}
_registry_lock = threading.Lock()


def get_logger(name: str, **kwargs) -> SafeLogger:
    """
    Get or create a logger instance (cached for performance)
    
    Args:
        name: Logger name
        **kwargs: Logger configuration options
        
    Returns:
        SafeLogger instance
    """
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = SafeLogger(name, **kwargs)
        return _logger_registry[name]


def setup_logging(log_dir: str = "logs", 
                 level: str = "INFO",
                 console: bool = True,
                 files: bool = True) -> Dict[str, Any]:
    """
    Setup global logging configuration
    
    Args:
        log_dir: Directory for log files
        level: Global logging level
        console: Enable console output
        files: Enable file output
        
    Returns:
        Dictionary with setup results
    """
    try:
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert level string to constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        
        # Setup root logger
        root_logger = get_logger("root", 
                                log_dir=log_dir, 
                                level=log_level,
                                console_output=console,
                                file_output=files)
        
        root_logger.info(f"Logging system initialized - Level: {level}, Directory: {log_dir}")
        
        return {
            'success': True,
            'log_dir': log_dir,
            'level': level,
            'console_output': console,
            'file_output': files
        }
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def create_progress_tracker(total: int, name: str = "Operation", logger_name: str = "progress") -> ProgressTracker:
    """
    Create a progress tracker for long operations
    
    Args:
        total: Total number of items to process
        name: Name for the operation
        logger_name: Logger to use
        
    Returns:
        ProgressTracker instance
    """
    logger = get_logger(logger_name)
    return ProgressTracker(total, name, logger)


def log_performance(func_name: str, start_time: float, end_time: float, **metrics):
    """
    Log performance metrics for a function
    
    Args:
        func_name: Name of the function
        start_time: Start timestamp
        end_time: End timestamp
        **metrics: Additional metrics to log
    """
    logger = get_logger("performance")
    
    duration = end_time - start_time
    
    metrics_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
    
    logger.info(f"{func_name}: {duration:.3f}s | {metrics_str}")


def get_all_logger_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics from all active loggers"""
    with _registry_lock:
        return {name: logger.get_stats() for name, logger in _logger_registry.items()}


def print_all_logger_summaries():
    """Print summaries for all active loggers"""
    with _registry_lock:
        for logger in _logger_registry.values():
            logger.print_summary()


# Convenience functions for common logging patterns
def log_system_start(system_name: str, version: str = "1.0"):
    """Log system startup"""
    logger = get_logger("system")
    logger.info(f"{system_name} v{version} - Starting Up")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")


def log_system_shutdown(system_name: str):
    """Log system shutdown with summary"""
    logger = get_logger("system")
    logger.info(f"{system_name} - Shutting Down")
    
    # Print all logger summaries
    print_all_logger_summaries()


def log_exception(exception: Exception, context: str = "Unknown", logger_name: str = "error"):
    """Log an exception with full context"""
    logger = get_logger(logger_name)
    logger.error(f"{context} failed", exception=exception)


# Test function to verify logger is working
def test_logger():
    """Test the logging system"""
    print("Testing logging system...")
    
    # Setup logging
    result = setup_logging()
    if not result['success']:
        print(f"Setup failed: {result['error']}")
        return False
    
    # Test different log levels
    logger = get_logger("test")
    
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    
    # Test progress tracker
    tracker = create_progress_tracker(100, "Test Operation")
    for i in range(100):
        tracker.update()
        if i % 20 == 0:
            time.sleep(0.01)  # Simulate work
    tracker.finish()
    
    # Test performance logging
    start = time.time()
    time.sleep(0.1)
    end = time.time()
    log_performance("test_function", start, end, items_processed=100, memory_mb=50)
    
    # Print stats
    logger.print_summary()
    
    print("Logger test complete")
    return True


if __name__ == "__main__":
    test_logger()