"""
Deployment Configuration for Parameter Injection System
======================================================

Configuration classes for the deployment module that handles parameter injection
into production strategy templates.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

@dataclass
class DeploymentConfig:
    """Configuration for deployment operations."""
    
    # Output settings
    output_directory: Path = Path("deployed_strategies")
    file_prefix: str = "optimized"
    file_suffix: str = ".py"
    overwrite_existing: bool = True
    
    # Deployment limits
    max_deployments: int = 50  # Maximum parameter sets to deploy
    min_score_threshold: float = 0.0  # Minimum composite score to deploy
    
    # Template settings
    template_encoding: str = "utf-8"
    validate_templates: bool = True
    backup_originals: bool = False
    
    # Market configuration
    include_market_config: bool = True
    symbol_override: Optional[str] = None
    timeframe_override: Optional[str] = None
    
    # Naming convention
    ranking_in_filename: bool = True  # Include rank (1, 2, 3...) in filename
    score_in_filename: bool = True    # Include score in filename
    timestamp_in_filename: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_deployments <= 0:
            raise ValueError("max_deployments must be positive")
        
        if not isinstance(self.output_directory, Path):
            self.output_directory = Path(self.output_directory)
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class MarketConfig:
    """Market configuration for strategy deployment."""
    
    symbol: str
    timeframe: str
    tick_size: float
    tick_value: float = 12.5  # Default for ES futures
    margin_requirement: float = 13000.0  # Default for ES
    
    # Market hours (optional)
    market_open: Optional[str] = None
    market_close: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template injection."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe, 
            'tick_size': self.tick_size,
            'tick_value': self.tick_value,
            'margin_requirement': self.margin_requirement,
            'market_open': self.market_open or 'N/A',
            'market_close': self.market_close or 'N/A'
        }


@dataclass  
class ParameterSet:
    """Container for a single parameter set with metadata."""
    
    parameters: Dict[str, Any]
    composite_score: float
    trial_number: int
    rank: int
    metrics: Dict[str, Any]
    
    def get_filename_components(self, config: DeploymentConfig) -> List[str]:
        """Generate filename components based on configuration."""
        components = [config.file_prefix]
        
        if config.ranking_in_filename:
            components.append(f"rank_{self.rank:03d}")
            
        if config.score_in_filename:
            components.append(f"score_{self.composite_score:.4f}")
            
        if config.timestamp_in_filename:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
            components.append(timestamp)
        
        return components
    
    def generate_filename(self, config: DeploymentConfig, strategy_name: str) -> str:
        """Generate complete filename for this parameter set."""
        components = self.get_filename_components(config)
        components.append(strategy_name.lower())
        
        filename = "_".join(components) + config.file_suffix
        return filename