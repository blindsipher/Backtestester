"""
Parameter Injection System for Strategy Deployment
==================================================

Injects optimized parameters from optimization results into deployment templates
to create production-ready strategy files.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

from .deployment_config import DeploymentConfig, ParameterSet
# Import TradingConfig to access complete MarketSpec
from config.system_config import TradingConfig

logger = logging.getLogger(__name__)


class ParameterInjector:
    """
    Handles parameter injection into deployment templates.
    
    Replaces {parameter_name} placeholders in template files with optimized values.
    Supports various data types and format validation.
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize parameter injector.
        
        Args:
            config: Deployment configuration (uses defaults if None)
        """
        self.config = config or DeploymentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Parameter type formatting rules
        self.format_rules = {
            bool: self._format_boolean,
            int: self._format_integer,
            float: self._format_float,
            str: self._format_string,
            list: self._format_list,
            dict: self._format_dict
        }
    
    def inject_parameters(self, 
                         template_content: str,
                         parameter_set: ParameterSet,
                         trading_config: TradingConfig,
                         simulation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject parameters into template content.
        
        Args:
            template_content: Raw template file content
            parameter_set: Optimized parameters to inject
            trading_config: Complete trading configuration with MarketSpec
            simulation_params: Optional CLI simulation parameters (slippage_ticks, commission_per_trade)
            
        Returns:
            Template content with parameters injected
        """
        try:
            # Start with the original content
            result = template_content
            
            # Inject strategy parameters
            result = self._inject_strategy_parameters(result, parameter_set)
            
            # Inject complete market configuration from TradingConfig
            if self.config.include_market_config:
                result = self._inject_trading_config(result, trading_config)
            
            # Inject simulation parameters if provided
            if simulation_params:
                result = self._inject_simulation_parameters(result, simulation_params)
            
            # Inject metadata comments
            result = self._inject_metadata_comments(result, parameter_set)
            
            # Validate that all placeholders were replaced
            remaining_placeholders = self._find_unreplaced_placeholders(result)
            if remaining_placeholders:
                self.logger.warning(f"Unreplaced placeholders found: {remaining_placeholders}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Parameter injection failed: {e}")
            raise
    
    def _inject_strategy_parameters(self, content: str, parameter_set: ParameterSet) -> str:
        """Inject strategy-specific parameters."""
        result = content
        
        for param_name, param_value in parameter_set.parameters.items():
            placeholder = f"{{{param_name}}}"
            
            # Format the parameter value appropriately
            formatted_value = self._format_parameter_value(param_value)
            
            # Replace all occurrences
            result = result.replace(placeholder, formatted_value)
            
            self.logger.debug(f"Injected {param_name}: {param_value} -> {formatted_value}")
        
        return result
    
    def _inject_trading_config(self, content: str, trading_config: TradingConfig) -> str:
        """Inject complete market specification from TradingConfig."""
        result = content
        market_spec = trading_config.market_spec
        
        # Extract comprehensive market parameters from MarketSpec
        market_params = {
            # Core market identification
            'symbol': market_spec.symbol,
            'timeframe': trading_config.timeframe,
            'exchange': market_spec.exchange,
            'currency': market_spec.currency,
            
            # Critical financial specifications (FIXES THE CALCULATION ERRORS)
            'tick_size': float(market_spec.tick_size),           # Price increment (e.g., 0.25 for ES)
            'tick_value': float(market_spec.tick_value),         # Dollar value per tick (e.g., 12.50 for ES)
            'contract_size': market_spec.contract_size,          # Contract multiplier
            
            # Additional market specifications
            'point_value': float(market_spec.point_value) if market_spec.point_value else float(market_spec.tick_value / market_spec.tick_size),
            'margin_requirement': float(market_spec.margin_requirement) if market_spec.margin_requirement else 0.0,
            
            # Market session information
            'session_start': market_spec.session_start.strftime('%H:%M') if market_spec.session_start else '18:00',
            'session_end': market_spec.session_end.strftime('%H:%M') if market_spec.session_end else '16:10',
            
            # Platform availability
            'available_on_t4': market_spec.available_on_t4,
            'micro_contract': market_spec.micro_contract,
        }
        
        # Inject all market parameters
        for param_name, param_value in market_params.items():
            placeholder = f"{{{param_name}}}"
            formatted_value = self._format_parameter_value(param_value)
            result = result.replace(placeholder, formatted_value)
            
            self.logger.debug(f"Injected market param {param_name}: {param_value} -> {formatted_value}")
        
        return result
    
    def _inject_simulation_parameters(self, content: str, simulation_params: Dict[str, Any]) -> str:
        """Inject simulation-specific parameters (slippage, commission, etc.)."""
        result = content
        
        # Inject CLI simulation parameters
        for param_name, param_value in simulation_params.items():
            placeholder = f"{{{param_name}}}"
            formatted_value = self._format_parameter_value(param_value)
            result = result.replace(placeholder, formatted_value)
            
            self.logger.debug(f"Injected simulation param {param_name}: {param_value} -> {formatted_value}")
        
        return result
    
    def _inject_metadata_comments(self, content: str, parameter_set: ParameterSet) -> str:
        """Inject minimal metadata header without optimization metrics."""
        # User requested NO optimization metrics in deployed files
        # Only include essential deployment information
        metadata_comment = f'''"""
OPTIMIZED STRATEGY DEPLOYMENT
=============================
Deployment Rank: #{parameter_set.rank}
Generated by Institutional Trading System Deployment Module
"""

'''
        
        # Insert after the first docstring (if any)
        docstring_pattern = r'^(""".*?""")\s*'
        match = re.match(docstring_pattern, content, re.DOTALL)
        
        if match:
            # Insert after existing docstring
            existing_docstring = match.group(1)
            rest_of_content = content[match.end():]
            result = existing_docstring + "\n\n" + metadata_comment + rest_of_content
        else:
            # Insert at the beginning
            result = metadata_comment + content
        
        return result
    
    def _format_parameter_value(self, value: Any) -> str:
        """Format a parameter value for code injection."""
        value_type = type(value)
        
        if value_type in self.format_rules:
            return self.format_rules[value_type](value)
        else:
            # Default: convert to string representation
            return repr(value)
    
    def _format_boolean(self, value: bool) -> str:
        """Format boolean values."""
        return "True" if value else "False"
    
    def _format_integer(self, value: int) -> str:
        """Format integer values."""
        return str(value)
    
    def _format_float(self, value: float) -> str:
        """Format float values with appropriate precision."""
        # Use reasonable precision to avoid floating point artifacts
        if abs(value) < 0.0001:
            return f"{value:.8f}"
        elif abs(value) < 1:
            return f"{value:.6f}"
        elif abs(value) < 1000:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"
    
    def _format_string(self, value: str) -> str:
        """Format string values with proper quoting."""
        return f'"{value}"'
    
    def _format_list(self, value: List) -> str:
        """Format list values."""
        formatted_items = [self._format_parameter_value(item) for item in value]
        return f"[{', '.join(formatted_items)}]"
    
    def _format_dict(self, value: Dict) -> str:
        """Format dictionary values."""
        formatted_items = []
        for k, v in value.items():
            key_str = self._format_parameter_value(k)
            val_str = self._format_parameter_value(v)
            formatted_items.append(f"{key_str}: {val_str}")
        return f"{{{', '.join(formatted_items)}}}"
    
    def _format_metrics_comment(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for inclusion in comments."""
        lines = []
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                if metric_name.lower().endswith(('pnl', 'profit', 'loss')):
                    lines.append(f"  {metric_name}: ${metric_value:.2f}")
                elif metric_name.lower().endswith(('rate', 'ratio', 'factor')):
                    lines.append(f"  {metric_name}: {metric_value:.4f}")
                else:
                    lines.append(f"  {metric_name}: {metric_value}")
            else:
                lines.append(f"  {metric_name}: {metric_value}")
        
        return "\n".join(lines) if lines else "  No metrics available"
    
    def _find_unreplaced_placeholders(self, content: str) -> List[str]:
        """Find any remaining {placeholder} patterns in the content (single-line only)."""
        # Only match placeholders that don't contain newlines (to avoid Python dict literals)
        # Exclude f-string patterns (containing brackets, quotes, or method calls)
        pattern = r'\{([^}\n\r\[\]\'\"\.]+)\}'
        matches = re.findall(pattern, content)
        
        # Filter out common f-string variable patterns
        filtered_matches = []
        for match in matches:
            # Skip if it looks like an f-string variable (no parameter-like naming)
            if not any(keyword in match.lower() for keyword in ['param', 'config', 'symbol', 'period', 'multiplier', 'threshold', 'method', 'ratio', 'filter']):
                continue
            filtered_matches.append(match)
        
        return list(set(filtered_matches))  # Remove duplicates
    
    def validate_template(self, template_path: Path) -> Dict[str, Any]:
        """
        Validate a deployment template.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Validation result with placeholders found and issues
        """
        try:
            if not template_path.exists():
                return {
                    'valid': False,
                    'error': f'Template file not found: {template_path}',
                    'placeholders': [],
                    'issues': ['File does not exist']
                }
            
            # Read template content
            content = template_path.read_text(encoding=self.config.template_encoding)
            
            # Find all placeholders
            placeholders = self._find_unreplaced_placeholders(content)
            
            # Check for common issues
            issues = []
            
            # Check for Python syntax issues in the template structure
            if not content.strip():
                issues.append("Template is empty")
            
            # Check for required deployment template structure
            if 'def generate_signal' not in content:
                issues.append("Missing generate_signal method (required for deployment)")
            
            if 'class' not in content:
                issues.append("No class definition found (may not be a valid strategy template)")
            
            return {
                'valid': len(issues) == 0,
                'placeholders': placeholders,
                'issues': issues,
                'template_path': str(template_path)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Template validation failed: {e}',
                'placeholders': [],
                'issues': [f'Validation error: {e}']
            }