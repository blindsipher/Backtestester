"""
Composite Scoring System
========================

Institutional-grade 7-metric composite scoring system for trading strategy evaluation.

This module implements sophisticated metric normalization and combination following
quantitative research best practices. The composite score balances risk-adjusted
returns, risk metrics, and strategy robustness to identify optimal parameter sets.

Key Features:
- 7-metric weighted composite scoring
- Industry-standard normalization bounds  
- Robust handling of edge cases (zero trades, invalid metrics)
- Individual metric storage for analysis
- Configurable weighting system

Metrics:
1. Prop Firm Viability (15%) - TopStep rule adherence and safety margin
2. Profit Factor (30%) - Gross profit/loss ratio (PRIMARY)
3. PNL (25%) - Profit and Loss percentage (SECONDARY)
4. Sortino Ratio (10%) - Downside risk focus  
5. Win Rate (10%) - Percentage winning trades
6. Max Drawdown (5%) - Peak-to-trough risk (inverted)
7. Trade Frequency (5%) - Normalized trading activity
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass

from .config.optuna_config import CompositeScoreWeights, MetricNormalizationBounds
from config.system_config import AccountConfig

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for individual metric calculation results"""
    raw_value: float
    normalized_value: float
    is_valid: bool
    weight: float
    contribution: float


class CompositeScore:
    """
    7-metric composite scoring system for trading strategy evaluation.
    
    Implements institutional best practices for combining multiple performance
    metrics into a single optimization objective. Handles edge cases gracefully
    and provides detailed metric breakdown for analysis.
    """
    
    def __init__(self, 
                 weights: Optional[CompositeScoreWeights] = None,
                 bounds: Optional[MetricNormalizationBounds] = None,
                 account_config: Optional[AccountConfig] = None):
        """
        Initialize composite scoring system.
        
        Args:
            weights: Metric weights (defaults to institutional configuration)
            bounds: Normalization bounds (defaults to research-based bounds)
            account_config: TopStep account configuration for viability scoring
        """
        self.weights = weights or CompositeScoreWeights()
        self.bounds = bounds or MetricNormalizationBounds()
        self.account_config = account_config
        
        # Validate configuration
        self.weights.__post_init__()
        
        logger.info(f"CompositeScore initialized with weights: "
                   f"PropFirmViability({self.weights.prop_firm_viability:.2f}) "
                   f"Sortino({self.weights.sortino_ratio:.2f}) "
                   f"PNL({self.weights.pnl:.2f}) "
                   f"MaxDD({self.weights.max_drawdown:.2f}) "
                   f"PF({self.weights.profit_factor:.2f}) "
                   f"WinRate({self.weights.win_rate:.2f}) "
                   f"TradeFq({self.weights.trade_frequency:.2f})")
    
    def calculate_composite_score(self, metrics: Dict[str, Any], minimum_trades_threshold: int = 30) -> Tuple[float, Dict[str, MetricResult]]:
        """
        Calculate weighted composite score from backtest metrics.
        
        Args:
            metrics: Dictionary of raw backtest metrics
            minimum_trades_threshold: Minimum trade count for penalty system
            
        Returns:
            Tuple of (composite_score, individual_metric_results)
            
        Raises:
            ValueError: If required metrics are missing or invalid
        """
        try:
            # Extract and validate required metrics
            required_metrics = [
                'daily_pnl_series', 'equity_curve', 'sortino_ratio', 'pnl', 'max_drawdown',
                'profit_factor', 'win_rate', 'total_trades'
            ]
            
            for metric in required_metrics:
                if metric not in metrics:
                    raise ValueError(f"Missing required metric: {metric}")
            
            # Calculate individual metric results
            metric_results = {}
            total_score = 0.0
            valid_weight_sum = 0.0
            
            # 1. Prop Firm Viability Score
            viability_result = self._calculate_prop_firm_viability_score(
                metrics['daily_pnl_series'], 
                metrics['equity_curve'],
                self.account_config
            )
            metric_results['prop_firm_viability'] = viability_result
            if viability_result.is_valid:
                total_score += viability_result.contribution
                valid_weight_sum += viability_result.weight
            
            # 2. Sortino Ratio  
            sortino_result = self._calculate_sortino_score(metrics['sortino_ratio'])
            metric_results['sortino_ratio'] = sortino_result
            if sortino_result.is_valid:
                total_score += sortino_result.contribution
                valid_weight_sum += sortino_result.weight
            
            # 3. PNL - INSTITUTIONAL FIX: Use dollar-based PNL for optimization
            # This eliminates account equity contamination in optimization results
            pnl_for_optimization = metrics.get('dollar_pnl_for_optimization', metrics.get('total_dollar_pnl', metrics['pnl']))
            pnl_result = self._calculate_dollar_pnl_score(pnl_for_optimization)
            metric_results['pnl'] = pnl_result
            if pnl_result.is_valid:
                total_score += pnl_result.contribution
                valid_weight_sum += pnl_result.weight
            
            # 4. Max Drawdown - INSTITUTIONAL FIX: Use dollar-based drawdown for optimization
            drawdown_for_optimization = metrics.get('max_drawdown_dollars', metrics['max_drawdown'])
            dd_result = self._calculate_dollar_drawdown_score(drawdown_for_optimization)
            metric_results['max_drawdown'] = dd_result
            if dd_result.is_valid:
                total_score += dd_result.contribution
                valid_weight_sum += dd_result.weight
            
            # 5. Profit Factor
            pf_result = self._calculate_profit_factor_score(metrics['profit_factor'])
            metric_results['profit_factor'] = pf_result
            if pf_result.is_valid:
                total_score += pf_result.contribution
                valid_weight_sum += pf_result.weight
            
            # 6. Win Rate
            wr_result = self._calculate_win_rate_score(metrics['win_rate'])
            metric_results['win_rate'] = wr_result
            if wr_result.is_valid:
                total_score += wr_result.contribution
                valid_weight_sum += wr_result.weight
            
            # 7. Trade Frequency
            tf_result = self._calculate_trade_frequency_score(
                metrics['total_trades'], 
                metrics.get('total_bars', 1000)  # Default assumption
            )
            metric_results['trade_frequency'] = tf_result
            if tf_result.is_valid:
                total_score += tf_result.contribution
                valid_weight_sum += tf_result.weight
            
            # Handle case where some metrics are invalid
            if valid_weight_sum == 0:
                logger.warning("All metrics invalid, returning score of 0")
                return 0.0, metric_results
            
            # Normalize by valid weights (rescale to 0-1 range)
            composite_score = total_score / valid_weight_sum if valid_weight_sum > 0 else 0.0
            
            # Apply trade count penalty with smooth function
            if 'total_trades' in metrics:
                total_trades = metrics['total_trades']
                # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
                if hasattr(total_trades, 'item'):
                    total_trades = total_trades.item()
                elif not np.isscalar(total_trades):
                    total_trades = float(total_trades)
                if total_trades < minimum_trades_threshold:
                    # Smooth penalty function - avoids optimization cliffs
                    penalty_factor = total_trades / minimum_trades_threshold
                    composite_score = composite_score * penalty_factor if composite_score > 0 else composite_score * (2 - penalty_factor)
                    logger.debug(f"Applied trade count penalty: {total_trades} trades < {minimum_trades_threshold} threshold, penalty_factor={penalty_factor:.3f}")
            
            # Log detailed breakdown if debug enabled
            if logger.isEnabledFor(logging.DEBUG):
                self._log_score_breakdown(composite_score, metric_results)
            
            return composite_score, metric_results
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            # Return minimal valid result for robustness
            return 0.0, self._create_empty_metric_results()
    
    def _calculate_prop_firm_viability_score(self, 
                                           daily_pnl_series: List[float], 
                                           equity_curve: List[float],
                                           account_config: Optional[AccountConfig]) -> MetricResult:
        """
        Calculate Prop Firm Viability Score based on TopStep rule adherence.
        
        Measures strategy's safety margin from daily loss limits and trailing drawdown.
        Uses average of worst 10% of daily viability scores for optimization stability.
        
        Args:
            daily_pnl_series: List of daily P&L values
            equity_curve: List of daily equity values
            account_config: TopStep account configuration with limits
            
        Returns:
            MetricResult with viability score and contribution
        """
        try:
            # Validate inputs
            if not account_config:
                logger.warning("No account config provided for prop firm viability score")
                return MetricResult(0.0, 0.0, False, self.weights.prop_firm_viability, 0.0)
            
            if not daily_pnl_series or not equity_curve:
                logger.warning("Missing daily PnL or equity data for viability score")
                return MetricResult(0.0, 0.0, False, self.weights.prop_firm_viability, 0.0)
            
            # Extract account limits (convert Decimal to float, handle negatives)
            daily_loss_limit = abs(float(account_config.daily_loss_limit))
            trailing_max_dd = abs(float(account_config.trailing_max_drawdown))
            
            # Calculate daily viability scores
            daily_viability_scores = []
            peak_equity = equity_curve[0]
            
            for i, (daily_pnl, current_equity) in enumerate(zip(daily_pnl_series, equity_curve)):
                # Ensure scalar extraction
                if hasattr(daily_pnl, 'item'):
                    daily_pnl = daily_pnl.item()
                elif not np.isscalar(daily_pnl):
                    daily_pnl = float(daily_pnl)
                    
                if hasattr(current_equity, 'item'):
                    current_equity = current_equity.item()
                elif not np.isscalar(current_equity):
                    current_equity = float(current_equity)
                
                # Update peak equity
                peak_equity = max(peak_equity, current_equity)
                current_drawdown = peak_equity - current_equity
                
                # Daily loss health (1.0 = perfect, 0.0 = at limit, <0 = violation)
                daily_loss_health = 1.0 - (max(0, -daily_pnl) / daily_loss_limit)
                
                # Trailing drawdown health
                trailing_dd_health = 1.0 - (current_drawdown / trailing_max_dd)
                
                # Most restrictive constraint determines daily viability
                daily_viability = min(daily_loss_health, trailing_dd_health)
                daily_viability_scores.append(daily_viability)
            
            # Calculate tail risk: average of worst 10% of days
            if not daily_viability_scores:
                return MetricResult(0.0, 0.0, False, self.weights.prop_firm_viability, 0.0)
                
            sorted_scores = sorted(daily_viability_scores)
            worst_10_percent_count = max(1, len(sorted_scores) // 10)
            tail_risk_score = float(np.mean(sorted_scores[:worst_10_percent_count]))
            
            # Validate result
            is_valid = not (np.isnan(tail_risk_score) or np.isinf(tail_risk_score))
            
            if not is_valid:
                tail_risk_score = self.bounds.prop_firm_viability_min
            
            # Normalize using bounds (0.95 = excellent, 0.70 = poor)
            normalized = self._normalize_metric(
                tail_risk_score, 
                self.bounds.prop_firm_viability_min, 
                self.bounds.prop_firm_viability_max
            )
            
            contribution = normalized * self.weights.prop_firm_viability
            
            return MetricResult(
                raw_value=tail_risk_score,
                normalized_value=normalized,
                is_valid=is_valid,
                weight=self.weights.prop_firm_viability,
                contribution=contribution
            )
            
        except Exception as e:
            logger.error(f"Error calculating prop firm viability score: {e}")
            return MetricResult(0.0, 0.0, False, self.weights.prop_firm_viability, 0.0)
    
    
    def _calculate_sortino_score(self, sortino_ratio: float) -> MetricResult:
        """Calculate normalized Sortino ratio score"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(sortino_ratio, 'item'):
            sortino_ratio = sortino_ratio.item()
        elif not np.isscalar(sortino_ratio):
            sortino_ratio = float(sortino_ratio)
        is_valid = not (np.isnan(sortino_ratio) or np.isinf(sortino_ratio))
        
        if not is_valid:
            sortino_ratio = self.bounds.sortino_min
        
        normalized = self._normalize_metric(
            sortino_ratio,
            self.bounds.sortino_min,
            self.bounds.sortino_max
        )
        
        contribution = normalized * self.weights.sortino_ratio
        
        return MetricResult(
            raw_value=sortino_ratio,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.sortino_ratio,
            contribution=contribution
        )
    
    def _calculate_pnl_score(self, pnl: float) -> MetricResult:
        """Calculate normalized PNL score"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(pnl, 'item'):
            pnl = pnl.item()
        elif not np.isscalar(pnl):
            pnl = float(pnl)
        
        # MAJOR BUG FIX: Remove inappropriate percentage conversion
        # PNL is already in percentage format from the backtest calculation.
        # The previous logic: if abs(pnl) < 5: pnl = pnl * 100
        # was causing massive inflation by multiplying percentages by 100 again.
        # For example: 2.4% -> 240% -> further scaled to millions in downstream calculations
        
        # PNL should remain as-is (already in percentage format)
        
        is_valid = not (np.isnan(pnl) or np.isinf(pnl))
        
        if not is_valid:
            pnl = self.bounds.pnl_min
        
        normalized = self._normalize_metric(
            pnl,
            self.bounds.pnl_min,
            self.bounds.pnl_max
        )
        
        contribution = normalized * self.weights.pnl
        
        return MetricResult(
            raw_value=pnl,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.pnl,
            contribution=contribution
        )
    
    def _calculate_dollar_pnl_score(self, dollar_pnl: float) -> MetricResult:
        """
        INSTITUTIONAL FIX: Calculate normalized dollar PNL score (account equity independent).
        
        This method uses pure dollar PNL amounts for optimization, eliminating
        account equity contamination in the optimization objective.
        """
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(dollar_pnl, 'item'):
            dollar_pnl = dollar_pnl.item()
        elif not np.isscalar(dollar_pnl):
            dollar_pnl = float(dollar_pnl)
        
        is_valid = not (np.isnan(dollar_pnl) or np.isinf(dollar_pnl))
        
        if not is_valid:
            # Use a reasonable default for invalid dollar PNL
            dollar_pnl = -1000.0  # Small loss in dollars
        
        # INSTITUTIONAL FIX: Use dollar-based bounds instead of percentage bounds
        # For ES futures: $-2500 (poor) to $7500 (excellent) based on institutional research
        dollar_pnl_min = -2500.0  # Poor performance (5% loss on $50K account)
        dollar_pnl_max = 7500.0   # Excellent performance (15% gain on $50K account)
        
        normalized = self._normalize_metric(
            dollar_pnl,
            dollar_pnl_min,
            dollar_pnl_max
        )
        
        contribution = normalized * self.weights.pnl
        
        return MetricResult(
            raw_value=dollar_pnl,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.pnl,
            contribution=contribution
        )
    
    def _calculate_drawdown_score(self, max_drawdown: float) -> MetricResult:
        """Calculate normalized max drawdown score (inverted - lower is better)"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(max_drawdown, 'item'):
            max_drawdown = max_drawdown.item()
        elif not np.isscalar(max_drawdown):
            max_drawdown = float(max_drawdown)
        # Ensure positive percentage format
        max_drawdown = abs(max_drawdown)
        if max_drawdown < 1:  # Assume decimal format
            max_drawdown = max_drawdown * 100
        
        is_valid = not (np.isnan(max_drawdown) or np.isinf(max_drawdown))
        
        if not is_valid:
            max_drawdown = self.bounds.max_drawdown_max  # Use maximum for invalid
        
        # Invert for scoring (lower drawdown = higher score)
        inverted_dd = self.bounds.max_drawdown_max - max_drawdown
        
        normalized = self._normalize_metric(
            inverted_dd,
            self.bounds.max_drawdown_min,
            self.bounds.max_drawdown_max
        )
        
        contribution = normalized * self.weights.max_drawdown
        
        return MetricResult(
            raw_value=max_drawdown,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.max_drawdown,
            contribution=contribution
        )
    
    def _calculate_dollar_drawdown_score(self, max_drawdown_dollars: float) -> MetricResult:
        """
        INSTITUTIONAL FIX: Calculate normalized dollar drawdown score (account equity independent).
        
        Uses pure dollar drawdown amounts for optimization, eliminating account equity
        contamination in drawdown-based optimization decisions.
        """
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(max_drawdown_dollars, 'item'):
            max_drawdown_dollars = max_drawdown_dollars.item()
        elif not np.isscalar(max_drawdown_dollars):
            max_drawdown_dollars = float(max_drawdown_dollars)
        
        # Ensure positive dollar amount
        max_drawdown_dollars = abs(max_drawdown_dollars)
        
        is_valid = not (np.isnan(max_drawdown_dollars) or np.isinf(max_drawdown_dollars))
        
        if not is_valid:
            # Use reasonable default for invalid dollar drawdown
            max_drawdown_dollars = 2500.0  # Moderate dollar drawdown
        
        # INSTITUTIONAL FIX: Use dollar-based bounds instead of percentage bounds
        # For ES futures: $0 (excellent) to $5000 (poor) based on institutional research
        dollar_dd_min = 0.0      # Excellent (no drawdown)
        dollar_dd_max = 5000.0   # Poor (10% drawdown on $50K account)
        
        # Invert for scoring (lower drawdown = higher score)
        inverted_dd = dollar_dd_max - max_drawdown_dollars
        
        normalized = self._normalize_metric(
            inverted_dd,
            0.0,  # min after inversion (worst case)
            dollar_dd_max  # max after inversion (best case)
        )
        
        contribution = normalized * self.weights.max_drawdown
        
        return MetricResult(
            raw_value=max_drawdown_dollars,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.max_drawdown,
            contribution=contribution
        )
    
    def _calculate_profit_factor_score(self, profit_factor: float) -> MetricResult:
        """Calculate normalized profit factor score"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(profit_factor, 'item'):
            profit_factor = profit_factor.item()
        elif not np.isscalar(profit_factor):
            profit_factor = float(profit_factor)
        is_valid = not (np.isnan(profit_factor) or np.isinf(profit_factor)) and profit_factor > 0
        
        if not is_valid:
            profit_factor = self.bounds.profit_factor_min
        
        normalized = self._normalize_metric(
            profit_factor,
            self.bounds.profit_factor_min,
            self.bounds.profit_factor_max
        )
        
        contribution = normalized * self.weights.profit_factor
        
        return MetricResult(
            raw_value=profit_factor,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.profit_factor,
            contribution=contribution
        )
    
    def _calculate_win_rate_score(self, win_rate: float) -> MetricResult:
        """Calculate normalized win rate score"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(win_rate, 'item'):
            win_rate = win_rate.item()
        elif not np.isscalar(win_rate):
            win_rate = float(win_rate)
        # Convert to percentage if needed
        if win_rate <= 1.0:  # Assume decimal format
            win_rate = win_rate * 100
        
        is_valid = not (np.isnan(win_rate) or np.isinf(win_rate)) and 0 <= win_rate <= 100
        
        if not is_valid:
            win_rate = self.bounds.win_rate_min
        
        normalized = self._normalize_metric(
            win_rate,
            self.bounds.win_rate_min,
            self.bounds.win_rate_max
        )
        
        contribution = normalized * self.weights.win_rate
        
        return MetricResult(
            raw_value=win_rate,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.win_rate,
            contribution=contribution
        )
    
    def _calculate_trade_frequency_score(self, total_trades: int, total_bars: int) -> MetricResult:
        """Calculate normalized trade frequency score"""
        # CRITICAL FIX: Ensure scalar extraction to prevent array boolean error
        if hasattr(total_trades, 'item'):
            total_trades = total_trades.item()
        elif not np.isscalar(total_trades):
            total_trades = int(total_trades)
        if hasattr(total_bars, 'item'):
            total_bars = total_bars.item()
        elif not np.isscalar(total_bars):
            total_bars = int(total_bars)
        # Calculate trades per 1000 bars for normalization
        if total_bars <= 0:
            trade_frequency = 0.0
            is_valid = False
        else:
            trade_frequency = (total_trades / total_bars) * 1000
            is_valid = not (np.isnan(trade_frequency) or np.isinf(trade_frequency))
        
        if not is_valid:
            trade_frequency = self.bounds.trade_freq_min
        
        normalized = self._normalize_metric(
            trade_frequency,
            self.bounds.trade_freq_min,
            self.bounds.trade_freq_max
        )
        
        contribution = normalized * self.weights.trade_frequency
        
        return MetricResult(
            raw_value=trade_frequency,
            normalized_value=normalized,
            is_valid=is_valid,
            weight=self.weights.trade_frequency,
            contribution=contribution
        )
    
    def _normalize_metric(self, value: float, min_bound: float, max_bound: float) -> float:
        """
        Normalize metric to [0,1] scale with soft scaling for discovery.
        
        FIXED: Replaces hard clipping with logistic soft scaling to allow
        exceptional performance discovery while maintaining bounded output.
        
        Args:
            value: Raw metric value
            min_bound: Minimum bound (maps to ~0)
            max_bound: Maximum bound (maps to ~1)
            
        Returns:
            Normalized value in [0,1] range with soft bounds
        """
        if max_bound == min_bound:
            return 0.5  # Avoid division by zero
        
        # Linear normalization first
        linear_norm = (value - min_bound) / (max_bound - min_bound)
        
        # Apply soft scaling with logistic function for values outside [0,1]
        if linear_norm < 0:
            # CRITICAL FIX: Prevent exponential overflow for extreme negative values
            # Clip linear_norm to safe range before exponential calculation
            safe_linear_norm = max(linear_norm, -10.0)  # Prevent exp overflow
            normalized = 1.0 / (1.0 + np.exp(-2.0 * safe_linear_norm)) * 0.5
        elif linear_norm > 1:
            # Exceptional values: soft approach to 1 but can exceed slightly
            excess = linear_norm - 1.0
            # CRITICAL FIX: Prevent exponential overflow for extreme positive values
            safe_excess = min(excess, 10.0)  # Prevent exp overflow
            normalized = 0.5 + 0.5 / (1.0 + np.exp(-3.0 * safe_excess))
            normalized = min(normalized, 1.2)  # Cap at 1.2 for stability
        else:
            # Normal range [0,1]: keep linear scaling
            normalized = linear_norm
        
        return float(normalized)
    
    def _log_score_breakdown(self, composite_score: float, metric_results: Dict[str, MetricResult]):
        """Log detailed score breakdown for debugging"""
        logger.debug(f"Composite Score Breakdown (Total: {composite_score:.4f}):")
        
        for metric_name, result in metric_results.items():
            if result.is_valid:
                logger.debug(f"  {metric_name}: raw={result.raw_value:.4f}, "
                           f"norm={result.normalized_value:.4f}, "
                           f"weight={result.weight:.3f}, "
                           f"contrib={result.contribution:.4f}")
            else:
                logger.debug(f"  {metric_name}: INVALID (raw={result.raw_value})")
    
    def _create_empty_metric_results(self) -> Dict[str, MetricResult]:
        """Create empty metric results for error cases"""
        empty_result = MetricResult(
            raw_value=0.0,
            normalized_value=0.0,
            is_valid=False,
            weight=0.0,
            contribution=0.0
        )
        
        return {
            'prop_firm_viability': empty_result,
            'sortino_ratio': empty_result,
            'pnl': empty_result,
            'max_drawdown': empty_result,
            'profit_factor': empty_result,
            'win_rate': empty_result,
            'trade_frequency': empty_result
        }
    
    def get_metric_importance(self) -> Dict[str, float]:
        """
        Get metric importance weights for analysis.
        
        Returns:
            Dictionary mapping metric names to importance weights
        """
        return {
            'prop_firm_viability': self.weights.prop_firm_viability,
            'sortino_ratio': self.weights.sortino_ratio,
            'pnl': self.weights.pnl,
            'max_drawdown': self.weights.max_drawdown,
            'profit_factor': self.weights.profit_factor,
            'win_rate': self.weights.win_rate,
            'trade_frequency': self.weights.trade_frequency
        }
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all required metrics are present and reasonable.
        
        Args:
            metrics: Dictionary of backtest metrics to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for missing metrics
        required_base_metrics = [
            'sortino_ratio', 'pnl', 'max_drawdown',
            'profit_factor', 'win_rate', 'total_trades'  
        ]
        for metric in required_base_metrics:
            if metric not in metrics:
                issues.append(f"Missing required metric: {metric}")
                
        # Special validation for prop firm viability inputs
        if 'daily_pnl_series' not in metrics or 'equity_curve' not in metrics:
            issues.append("Missing daily_pnl_series or equity_curve for prop firm viability scoring")
        
        # Check for obviously invalid values
        if 'total_trades' in metrics and metrics['total_trades'] < 0:
            issues.append("total_trades cannot be negative")
        
        if 'win_rate' in metrics:
            wr = metrics['win_rate']
            if wr < 0 or (wr > 1 and wr > 100):  # Allow both decimal and percentage
                issues.append(f"win_rate out of valid range: {wr}")
        
        if 'profit_factor' in metrics and metrics['profit_factor'] < 0:
            issues.append("profit_factor cannot be negative")
        
        return len(issues) == 0, issues
    
    def calculate_split_consistency(self, split_scores: List[float]) -> Dict[str, float]:
        """
        Calculate consistency metrics across walk-forward splits.
        
        Args:
            split_scores: List of composite scores from each split
            
        Returns:
            Dictionary with consistency metrics including coefficient of variation
        """
        if not split_scores or len(split_scores) < 2:
            return {
                'coefficient_of_variation': 0.0,
                'consistency_score': 0.0,
                'score_range': 0.0,
                'num_splits': len(split_scores) if split_scores else 0
            }
        
        try:
            split_array = np.array(split_scores)
            mean_score = np.mean(split_array)
            std_score = np.std(split_array)
            
            # Coefficient of Variation (lower is better)
            cv = std_score / abs(mean_score) if abs(mean_score) > 1e-6 else float('inf')
            
            # Consistency score (0-1, higher is better)
            # Based on inverse CV with saturation
            consistency_score = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
            
            # Score range (max - min)
            score_range = float(np.max(split_array) - np.min(split_array))
            
            return {
                'coefficient_of_variation': float(cv),
                'consistency_score': float(consistency_score),
                'score_range': score_range,
                'num_splits': len(split_scores),
                'mean_score': float(mean_score),
                'std_score': float(std_score)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating split consistency: {e}")
            return {
                'coefficient_of_variation': float('inf'),
                'consistency_score': 0.0,
                'score_range': 0.0,
                'num_splits': len(split_scores) if split_scores else 0
            }