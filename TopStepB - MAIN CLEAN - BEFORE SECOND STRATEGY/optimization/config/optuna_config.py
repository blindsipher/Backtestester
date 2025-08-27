"""
Optuna Configuration
====================

Comprehensive configuration for Optuna optimization engine following
institutional best practices for trading strategy optimization.

This configuration module defines:
- TPE sampler settings optimized for trading parameters
- MedianPruner configuration for memory safety
- 7-metric composite scoring weights from quantitative research
- Optimization limits and resource constraints
- Database storage and checkpoint settings

The configuration is designed for institutional-grade trading systems
with emphasis on robust parameter discovery and efficient resource usage.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path

# Reference existing system configuration
from config.system_config import SystemPaths


@dataclass
class TPESamplerConfig:
    """
    Tree-structured Parzen Estimator configuration optimized for trading parameters.
    
    Settings based on research for mixed parameter types (numeric/categorical/boolean)
    common in trading strategies.
    """
    # Number of random trials before TPE starts - OPTIMIZED for faster activation
    n_startup_trials: int = 50          # REDUCED: Faster TPE activation for exploration
    
    # Use multivariate TPE for parameter correlations
    multivariate: bool = True
    
    # Group parameters for mixed types (numeric + categorical)
    group: bool = True
    
    # Prior weight for regularization (higher = more conservative)
    prior_weight: float = 1.0
    
    # Consider only recent trials for adaptation (0 = all trials)
    consider_prior: int = 25
    
    # Consider only endpoints for categorical parameters
    consider_endpoints: bool = False
    
    # OPTIMIZED: Disable warnings and add reproducibility
    warn_independent_sampling: bool = False  # DISABLED: Suppress TPE warnings for cleaner output
    seed: int = 42                          # ADDED: Reproducible results for debugging


@dataclass
class MedianPrunerConfig:
    """
    MedianPruner configuration for early termination of unpromising trials.
    
    Settings balanced for trading strategy optimization where early metrics
    may not be representative of final performance.
    """
    # Number of trials before pruning starts (prevent premature pruning)
    n_startup_trials: int = 50
    
    # Number of warmup steps before pruning (let strategies stabilize)
    n_warmup_steps: int = 10
    
    # Pruning interval (every N steps)
    interval_steps: int = 5
    
    # Only prune if below median of recent trials (0 = all trials)
    n_min_trials: int = 5


@dataclass
class CompositeScoreWeights:
    """
    Weights for 7-metric composite scoring system.
    
    Based on institutional quantitative research for trading strategy evaluation.
    Weights sum to 1.0 and emphasize risk-adjusted returns over raw performance.
    """
    # DISCOVERY PHASE: Profitability-focused weights for unprofitable strategy discovery
    # Profitability metrics (55% total weight) - PRIMARY FOCUS
    profit_factor: float = 0.30     # PRIMARY: Must exceed 1.0 for profitability
    pnl: float = 0.25               # SECONDARY: Actual dollar returns
    
    # Confidence metrics (15% total weight)
    win_rate: float = 0.10          # CONFIDENCE: Consistency indicator
    trade_frequency: float = 0.05   # Activity level (normalized)
    
    # Risk-adjusted metrics (25% total weight) - REDUCED until profitable
    prop_firm_viability: float = 0.15   # NEW: TopStep rule adherence (replaces Sharpe)
    sortino_ratio: float = 0.10         # REDUCED: Until profitable
    
    # Risk constraint (minimal weight)
    max_drawdown: float = 0.05      # MINIMAL: Focus on finding profits first
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.prop_firm_viability + self.sortino_ratio + self.pnl + 
                self.max_drawdown + self.profit_factor + self.win_rate + 
                self.trade_frequency)
        
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Composite score weights must sum to 1.0, got {total}")


@dataclass
class MetricNormalizationBounds:
    """
    Normalization bounds for converting raw metrics to [0,1] scale.
    
    Based on analysis of institutional trading strategy performance ranges.
    Values outside bounds are clipped to prevent outlier distortion.
    """
    # Risk-adjusted returns - EXPANDED for discovery phase
    prop_firm_viability_min: float = 0.70   # High risk threshold (poor viability)
    prop_firm_viability_max: float = 0.95   # Excellent viability (safe operations)
    
    sortino_min: float = -2.0       # Worst acceptable Sortino  
    sortino_max: float = 6.0        # EXPANDED: Allow exceptional Sortino ratios for discovery
    
    # Returns (as percentages)
    pnl_min: float = -50.0          # Maximum acceptable loss
    pnl_max: float = 100.0          # Exceptional profit percentage
    
    # Risk (as percentages)
    max_drawdown_min: float = 0.0   # No drawdown (ideal)
    max_drawdown_max: float = 50.0  # Maximum tolerable drawdown
    
    # Strategy metrics
    profit_factor_min: float = 0.5  # Break-even threshold
    profit_factor_max: float = 3.0  # Excellent profit factor
    
    win_rate_min: float = 0.0       # No wins (worst case)
    win_rate_max: float = 100.0     # Perfect win rate
    
    # Trade frequency (trades per 1000 bars)
    trade_freq_min: float = 0.1     # Very low frequency
    trade_freq_max: float = 50.0    # High frequency threshold


@dataclass  
class OptimizationLimits:
    """
    Resource and execution limits for optimization runs.
    
    Designed to prevent runaway optimizations while allowing thorough
    parameter space exploration for institutional-grade strategies.
    """
    # Maximum number of trials per optimization - INSTITUTIONAL SCALE for comprehensive exploration
    max_trials: int = 50000             # INSTITUTIONAL: Support for large-scale optimization campaigns
    
    # Maximum time per individual trial (seconds)
    timeout_per_trial: int = 300
    
    # Maximum total optimization time (seconds) - 8 hours
    max_optimization_time: int = 28800
    
    # Number of top parameter sets to return (1-500) - Keep manageable for analysis
    results_top_n: int = 500
    
    # Memory limit per trial (MB) - Conservative 1500MB per worker
    memory_limit_mb: int = 1500

    # CPU cores to use (0 = auto-detect)
    max_workers: int = 0

    # GPU device selection (empty = auto-detect all)
    gpu_device_ids: List[int] = field(default_factory=list)

    # Per-worker GPU memory limit in MB (0 = no limit)
    gpu_memory_limit_mb: int = 0
    
    # Checkpoint interval (save every N trials)
    checkpoint_interval: int = 50
    
    # Minimum trade count threshold for penalty system
    minimum_trades_threshold: int = 30
    


@dataclass
class StorageConfig:
    """
    Database and file storage configuration for Optuna studies.
    
    Uses PostgreSQL for high-concurrency distributed optimization
    with unlimited worker scalability.
    """
    # Storage type - PostgreSQL only for high concurrency
    storage_type: str = "postgresql"
    
    # PostgreSQL connection parameters (Windows installation)
    host: str = "localhost"  # Local connection when running on Windows
    port: int = 1127  # Custom PostgreSQL port
    database: str = "optuna_optimization"
    username: str = "postgres"
    password: str = "1"
    
    # Study name template
    study_name_template: str = "{strategy}_{symbol}_{timeframe}_{timestamp}"
    
    # Enable study persistence
    enable_persistence: bool = True
    
    # Auto-create database
    create_database: bool = True
    
    # OPTIMIZED: Dynamic connection pool settings based on CPU cores for maximum parallelization
    pool_size: int = 60               # Base connections: CPU cores + buffer (dynamic in engine)
    max_overflow: int = 120           # Additional connections: 2x pool_size for burst capacity
    pool_timeout: int = 15            # Connection timeout: 15s - more patience for high concurrency
    pool_recycle: int = 3600          # Connection recycle: 1hr - reduce overhead
    
    def get_database_url(self, run_dir: Path = None) -> str:
        """Generate PostgreSQL database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_study_name(self, strategy: str, symbol: str, timeframe: str, timestamp: str) -> str:
        """Generate study name for specific optimization run"""
        return self.study_name_template.format(
            strategy=strategy, 
            symbol=symbol, 
            timeframe=timeframe, 
            timestamp=timestamp
        )


@dataclass
class OptimizationConfig:
    """
    Master configuration class combining all optimization settings.
    
    Provides single entry point for all optimization configuration
    with validation and environment variable overrides.
    """
    # Component configurations
    tpe_sampler: TPESamplerConfig = field(default_factory=TPESamplerConfig)
    median_pruner: MedianPrunerConfig = field(default_factory=MedianPrunerConfig)
    score_weights: CompositeScoreWeights = field(default_factory=CompositeScoreWeights)
    metric_bounds: MetricNormalizationBounds = field(default_factory=MetricNormalizationBounds)
    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Runtime overrides from environment
    verbose_logging: bool = field(default_factory=lambda: os.getenv('OPTUNA_VERBOSE', 'false').lower() == 'true')
    debug_mode: bool = field(default_factory=lambda: os.getenv('OPTUNA_DEBUG', 'false').lower() == 'true')
    
    def __post_init__(self):
        """Apply environment variable overrides"""
        # Override max_trials if set
        if 'OPTUNA_MAX_TRIALS' in os.environ:
            self.limits.max_trials = int(os.environ['OPTUNA_MAX_TRIALS'])
            
        # Override max_workers if set  
        if 'OPTUNA_MAX_WORKERS' in os.environ:
            self.limits.max_workers = int(os.environ['OPTUNA_MAX_WORKERS'])
            
        # Override results count if set
        if 'OPTUNA_RESULTS_TOP_N' in os.environ:
            self.limits.results_top_n = min(500, int(os.environ['OPTUNA_RESULTS_TOP_N']))
            
        # Override minimum trades threshold if set
        if 'OPTUNA_MIN_TRADES' in os.environ:
            self.limits.minimum_trades_threshold = max(1, int(os.environ['OPTUNA_MIN_TRADES']))

        # GPU configuration overrides
        if 'OPTUNA_GPU_DEVICES' in os.environ:
            self.limits.gpu_device_ids = [
                int(x) for x in os.environ['OPTUNA_GPU_DEVICES'].split(',') if x.strip().isdigit()
            ]
        if 'OPTUNA_GPU_MEMORY_MB' in os.environ:
            self.limits.gpu_memory_limit_mb = int(os.environ['OPTUNA_GPU_MEMORY_MB'])
            
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        # Ensure results_top_n doesn't exceed max_trials
        if self.limits.results_top_n > self.limits.max_trials:
            self.limits.results_top_n = self.limits.max_trials
            
        # Ensure startup trials are reasonable
        if self.tpe_sampler.n_startup_trials > self.limits.max_trials // 2:
            self.tpe_sampler.n_startup_trials = max(10, self.limits.max_trials // 10)
            
        # Ensure pruner settings are compatible
        if self.median_pruner.n_startup_trials > self.limits.max_trials // 3:
            self.median_pruner.n_startup_trials = max(5, self.limits.max_trials // 5)
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'tpe_sampler': {
                'n_startup_trials': self.tpe_sampler.n_startup_trials,
                'multivariate': self.tpe_sampler.multivariate,
                'group': self.tpe_sampler.group
            },
            'median_pruner': {
                'n_startup_trials': self.median_pruner.n_startup_trials,
                'n_warmup_steps': self.median_pruner.n_warmup_steps,
                'interval_steps': self.median_pruner.interval_steps
            },
            'score_weights': {
                'prop_firm_viability': self.score_weights.prop_firm_viability,
                'sortino_ratio': self.score_weights.sortino_ratio,
                'pnl': self.score_weights.pnl,
                'max_drawdown': self.score_weights.max_drawdown,
                'profit_factor': self.score_weights.profit_factor,
                'win_rate': self.score_weights.win_rate,
                'trade_frequency': self.score_weights.trade_frequency
            },
            'limits': {
                'max_trials': self.limits.max_trials,
                'timeout_per_trial': self.limits.timeout_per_trial,
                'results_top_n': self.limits.results_top_n,
                'max_workers': self.limits.max_workers,
                'minimum_trades_threshold': self.limits.minimum_trades_threshold,
                'gpu_device_ids': self.limits.gpu_device_ids,
                'gpu_memory_limit_mb': self.limits.gpu_memory_limit_mb,
            }
        }


# Default configuration instance
DEFAULT_CONFIG = OptimizationConfig()


def get_optimization_config() -> OptimizationConfig:
    """
    Get optimization configuration with validation.
    
    Returns:
        Validated OptimizationConfig instance
    """
    config = OptimizationConfig()
    config.validate()
    return config


def create_study_paths(strategy_name: str, symbol: str, timeframe: str, timestamp: str) -> Dict[str, Path]:
    """
    Create optimization study directory structure.
    
    Args:
        strategy_name: Name of strategy being optimized
        symbol: Trading symbol 
        timeframe: Data timeframe
        timestamp: Optimization timestamp
        
    Returns:
        Dictionary of paths for optimization run
    """
    return SystemPaths.create_run_directory(strategy_name, symbol, timeframe, timestamp)
