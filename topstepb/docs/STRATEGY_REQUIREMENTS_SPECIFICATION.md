# 📋 **TopStepJ Strategy Requirements Specification**
## **Mandatory Technical Specification for Institutional Trading Strategy Development**

**Document Version**: 1.0  
**Target System**: TopStepJ v2.7+  
**Compliance Level**: Institutional Grade  
**Last Updated**: August 9, 2025

---

## 📖 **Table of Contents**

1. [**Executive Summary**](#executive-summary)
2. [**Architectural Compliance Requirements**](#architectural-compliance-requirements)
3. [**File System Convention Specification**](#file-system-convention-specification)
4. [**BaseStrategy Contract Requirements**](#basestrategy-contract-requirements)
5. [**Parameter System Architecture**](#parameter-system-architecture)
6. [**Dual-Implementation Synchronization**](#dual-implementation-synchronization)
7. [**Optimization Pipeline Integration**](#optimization-pipeline-integration)
8. [**Deployment Template Specification**](#deployment-template-specification)
9. [**Security & Data Access Requirements**](#security--data-access-requirements)
10. [**Testing & Validation Framework**](#testing--validation-framework)
11. [**Performance & Resource Constraints**](#performance--resource-constraints)
12. [**Compliance & Audit Requirements**](#compliance--audit-requirements)

---

# **Executive Summary**

This specification document defines the **mandatory technical requirements** that ALL trading strategies must satisfy to integrate with the TopStepJ institutional trading system. These requirements ensure:

- **Production Safety**: Strategies that pass validation will perform identically in live trading
- **Institutional Security**: Data leakage prevention and audit trail compliance
- **Optimization Compatibility**: Parallel processing without contamination
- **Deployment Readiness**: Automated parameter injection and packaging
- **Regulatory Compliance**: Complete audit trails and risk management integration

**CRITICAL**: This is not guidance—these are **hard requirements**. Strategies that do not satisfy these specifications will be rejected by the system during discovery, optimization, or deployment phases.

---

# **Architectural Compliance Requirements**

## 🏗️ **1. System Integration Mandate**

Every strategy MUST integrate with exactly **4 pipeline phases**:

| Phase | Integration Point | Validation Method |
|-------|------------------|-------------------|
| **Discovery** | Plugin system via `pkgutil.walk_packages` | Convention validation + BaseStrategy inheritance |
| **Optimization** | StatefulObjective pattern | Parameter mapping + constraint validation |
| **Deployment** | Template injection system | Placeholder validation + fence integrity |
| **Validation** | Bar-by-bar processing | Parity testing + signal timing validation |

**Failure at any phase = System rejection**

## 🔒 **2. Security Architecture Compliance**

All strategies MUST operate within the secure orchestration framework:

- **Data Access**: Only through `AuthorizedDataAccess` objects (no direct DataFrame access)
- **State Isolation**: Mandatory `reset_state()` implementation preventing optimization contamination
- **Audit Trail**: All strategy decisions must be logged with institutional metadata
- **Parameter Security**: Template injection via whitelisted placeholders only

---

# **File System Convention Specification**

## 📁 **1. Mandatory Directory Structure**

```
strategies/{strategy_name}/
├── __init__.py                    # REQUIRED: Strategy export for discovery
├── strategy.py                    # REQUIRED: Vectorized BaseStrategy implementation
├── parameters.py                  # REQUIRED: Parameter definitions and validation
├── indicators.py                  # REQUIRED: Vectorized calculations
└── deployment_template.py         # REQUIRED: Self-contained live trading logic
```

**Convention Validation Rules**:
- Strategy name MUST be valid Python identifier: `^[a-zA-Z_][a-zA-Z0-9_]*$`
- All 5 files MUST exist or discovery fails
- Directory name MUST match strategy class `name` property
- No additional files allowed in strategy directory (maintains clean namespace)

## 📋 **2. File-Specific Requirements**

### **`__init__.py` Requirements**
```python
# MANDATORY EXPORT FORMAT
from .strategy import StrategyClassName

__all__ = ['StrategyClassName']
```

### **`strategy.py` Requirements**
- MUST contain exactly one class inheriting from `BaseStrategy`
- Class name MUST follow PascalCase convention
- MUST implement all 8 abstract methods (no defaults allowed)
- MUST use `.shift(1)` for all look-ahead bias prevention

### **`parameters.py` Requirements**
- MUST define `DEFAULT_PARAMETERS` dictionary (complete parameter set)
- MUST define `PARAMETER_RANGES` dictionary (optimization bounds)
- MUST implement `validate_parameters(params)` function
- MUST implement `get_default_parameters()` and `get_parameter_ranges()` functions

### **`indicators.py` Requirements**
- MUST contain only vectorized pandas operations
- MUST NOT contain any strategy logic or signal generation
- All functions MUST be pure (no side effects or state)
- MUST handle edge cases (insufficient data, NaN values)

### **`deployment_template.py` Requirements**
- MUST be completely self-contained (no imports from strategy module)
- MUST contain exactly one class for live trading
- MUST implement `process_new_bar()` method with exact signature
- MUST use parameter injection placeholders: `{parameter_name}`

---

# **BaseStrategy Contract Requirements**

## ⚖️ **1. Abstract Method Implementation Matrix**

| Method | Return Type | Validation | Purpose |
|--------|-------------|------------|---------|
| `name` | `str` | Python identifier regex | Strategy discovery key |
| `description` | `str` | Non-empty string | Human documentation |
| `category` | `str` | Predefined categories | Strategy classification |
| `min_data_points` | `int` | Positive integer | Data sufficiency check |
| `generate_signals()` | `pd.Series` | Signal validation | Core trading logic |
| `reset_state()` | `None` | State isolation check | Contamination prevention |
| `get_parameter_ranges()` | `Dict` | Range format validation | Optimization integration |
| `validate_parameters()` | `bool` | Constraint satisfaction | Parameter safety |
| `get_strategy_metadata()` | `Dict` | Metadata schema | Documentation compliance |

## 🎯 **2. Signal Generation Requirements**

### **Signal Format Specification**
```python
def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any], 
                    config: TradingConfig) -> pd.Series:
    """
    MANDATORY RETURN FORMAT:
    - pd.Series with same index as data
    - Values: 1 (long), -1 (short), 0 (flat)
    - MUST use .shift(1) to prevent look-ahead bias
    - MUST handle edge cases (insufficient data)
    """
```

### **Timing Protocol Enforcement**
- **Entry Signals**: Use previous bar's condition to signal current bar
- **Exit Signals**: Use previous bar's close for exit decisions
- **Position Changes**: Maximum change magnitude = 2 (from -1 to +1)
- **Validation**: Automatic timing validation via `validate_signal_timing()`

## 🔄 **3. State Management Requirements**

### **Mandatory State Reset Implementation**
```python
def reset_state(self) -> None:
    """
    CRITICAL: Reset ALL instance variables used in generate_signals()
    
    MUST RESET:
    - Position tracking variables
    - Entry prices and stop levels
    - Bar counters and timers
    - Technical indicator state
    - Any cached calculations
    """
```

**State Contamination Detection**: System automatically detects state leakage between optimization trials and fails the strategy if contamination is found.

---

# **Parameter System Architecture**

## 📊 **1. Parameter Definition Requirements**

### **DEFAULT_PARAMETERS Specification**
```python
# MANDATORY FORMAT
DEFAULT_PARAMETERS = {
    "parameter_name": default_value,        # Must cover ALL strategy parameters
    "numeric_param": 20,                    # Numbers as int/float
    "categorical_param": "option_a",        # Strings for categories
    "boolean_param": True,                  # Explicit boolean values
}
```

### **PARAMETER_RANGES Specification**
```python
# MANDATORY OPTIMIZATION RANGES
PARAMETER_RANGES = {
    # Numeric parameters: (min, max, step)
    "period": (10, 50, 5),                  # Creates: 10, 15, 20, ..., 50
    "threshold": (1.0, 3.0, 0.1),          # Creates: 1.0, 1.1, 1.2, ..., 3.0
    
    # Categorical parameters: [options]
    "method": ["ema", "sma", "wma"],        # Discrete choice options
    "exit_type": ["fixed", "trailing"],     # String categoricals
}
```

## ✅ **2. Parameter Validation Requirements**

### **Validation Function Specification**
```python
def validate_parameters(params):
    """
    MANDATORY VALIDATION CHECKS:
    1. All required parameters present
    2. Values within acceptable ranges
    3. Logical constraints satisfied
    4. Data type validation
    5. Cross-parameter dependencies
    
    MUST raise ValueError with descriptive message for any violation
    """
```

### **Institutional Constraint Examples**
- **Range Validation**: `stop_loss_pct` must be between 0.5% and 5.0%
- **Logical Constraints**: `fast_period < slow_period` for moving averages
- **Data Sufficiency**: `min_data_points >= max(all_periods) * 2`
- **Risk Limits**: `max_position_size <= account_risk_limit`

## 🎯 **3. Optimization Constraint Definition**

### **Optional Constraint Enhancement**
```python
def define_constraints(self) -> Dict[str, Any]:
    """
    OPTIMIZATION EFFICIENCY: Define parameter relationships
    to reduce invalid combination testing
    """
    return {
        'comparison': [
            ('fast_ma', '<', 'slow_ma'),         # Numeric relationships
            ('stop_loss', '<', 'take_profit'),   # Risk management logic
        ],
        'conditional': [
            ('atr_multiplier', 'requires', 'use_atr', True),  # Feature dependencies
        ],
        'mutually_exclusive': [
            ('fixed_stop', 'excludes', 'trailing_stop'),      # Either/or logic
        ]
    }
```

---

# **Dual-Implementation Synchronization**

## ⚖️ **1. Implementation Parity Requirements**

### **CRITICAL**: Both implementations MUST produce identical signals

| Implementation | Purpose | Performance | Validation Method |
|----------------|---------|-------------|-------------------|
| **Vectorized** (`strategy.py`) | Optimization speed | 1000+ trials/min | Bulk signal generation |
| **Stateful** (`deployment_template.py`) | Live trading precision | Real-time processing | Bar-by-bar execution |

### **Synchronization Protocol**
1. **Signal Logic**: Identical decision trees in both implementations
2. **Timing Protocol**: Vectorized uses `.shift(1)`, stateful uses pending signals
3. **State Management**: Stateful mirrors vectorized position tracking exactly
4. **Exit Conditions**: Same stop/target calculations in both versions

## 🔍 **2. Parity Testing Framework**

### **Automated Parity Validation** (Expert Recommendation)
```python
def test_strategy_parity():
    """
    MANDATORY PARITY TEST:
    - Run identical OHLCV data through both implementations
    - Compare signal series element by element
    - HARD FAIL on any divergence with diff context
    - Must be part of CI pipeline, not advisory
    """
```

**Parity Test Requirements**:
- **Test Data**: Multiple market conditions (trending, sideways, volatile)
- **Parameter Sets**: Multiple parameter combinations from ranges
- **Edge Cases**: Insufficient data, extreme values, missing bars
- **Performance**: Test must complete in <30 seconds for CI compatibility

## ⏰ **3. Sacred Timing Protocol Implementation**

### **Vectorized Implementation Pattern**
```python
# CORRECT: Signal on close, execute on next open
entry_condition = indicator.shift(1) > threshold.shift(1)  # Previous bar's condition
signals[entry_condition] = 1  # Signal current bar for next bar execution
```

### **Stateful Implementation Pattern**
```python
# CORRECT: Pending signal execution
if current_indicator > threshold:
    self.pending_entry_signal = 1  # Signal detected this bar
    
# Process pending signals from previous bar
if self.pending_entry_signal != 0:
    entry_price = current_open  # Execute at this bar's open price
```

---

# **Optimization Pipeline Integration**

## 🔄 **1. StatefulObjective Pattern Requirements**

### **Strategy Class Integration**
- Strategy MUST be passed as **class** (not instance) to prevent serialization issues
- Each trial creates fresh strategy instance for complete state isolation
- Parameter mapping from ranges to Optuna search spaces automatic
- Constraint integration for optimization efficiency

### **Performance Requirements**
- **Parallel Processing**: Must support unlimited concurrent workers
- **Trial Isolation**: No shared state between parallel executions
- **Memory Efficiency**: Maximum 50MB memory per trial
- **Throughput**: Target 60+ trials/minute on standard hardware

## 🛡️ **2. Data Access Security**

### **AuthorizedDataAccess Integration**
```python
# STRATEGY RECEIVES SECURE DATA ACCESS OBJECTS
def run_optimization(authorized_accesses: List[AuthorizedDataAccess]):
    """
    Data access ONLY through authorized objects:
    - training_data = authorized_access.get_data('train')
    - validation_data = authorized_access.get_data('validation')
    - NO direct DataFrame access allowed
    """
```

### **Data Segregation Requirements**
- **Training Data**: Available during optimization for parameter search
- **Validation Data**: Available during optimization for overfitting prevention
- **Test Data**: NEVER available during optimization (secured by orchestrator)
- **Audit Trail**: All data access requests logged with strategy context

## 📊 **3. Composite Scoring Integration**

### **Institutional Metrics Requirements**
Strategy performance evaluated on **7 institutional metrics**:
- **PropFirmViability**: Custom metric for prop firm requirements
- **Sortino Ratio**: Downside deviation focus
- **Net PnL**: Dollar-based returns (contract-based math)
- **Max Drawdown**: Risk management metric
- **Profit Factor**: Win/loss ratio
- **Win Rate**: Trade success percentage
- **Trade Frequency**: Activity level assessment

---

# **Deployment Template Specification**

## 🔧 **1. Parameter Injection System**

### **Template Placeholder Requirements**
```python
# STRATEGY PARAMETERS - INJECTED BY DEPLOYMENT ENGINE
bb_period = {bb_period}              # Strategy-specific parameters
bb_std_dev = {bb_std_dev}           # Must match parameters.py keys exactly
stop_loss_pct = {stop_loss_pct}     # Type-safe injection (int/float/str/bool)

# MARKET CONFIGURATION - INJECTED BY TRADING CONFIG
symbol = {symbol}                    # Trading symbol
timeframe = {timeframe}             # Bar timeframe
tick_size = {tick_size}             # Market specification
tick_value = {tick_value}           # Dollar per tick
contracts_per_trade = {contracts_per_trade}  # Position sizing
```

### **Fencing Mechanism Requirements**
```python
# FENCE:START:SIMULATION
# These parameters are REMOVED during packaging for production
slippage_ticks = {slippage_ticks}
commission_per_trade = {commission_per_trade}
# FENCE:END:SIMULATION
```

**Fencing Rules**:
- FENCE blocks MUST be on separate lines
- Content between fences is removed during packaging
- Fence integrity validated before deployment
- No nested fences allowed

## 📱 **2. Live Trading Class Requirements**

### **Class Structure Specification**
```python
class Live{StrategyName}Strategy:
    """
    MANDATORY INTERFACE:
    - __init__(self): Initialize all state variables
    - process_new_bar(open, high, low, close, volume, timestamp) -> dict
    - Return dict with position info and strategy state
    """
```

### **State Management Requirements**
- **Efficient Storage**: Use `deque(maxlen=N)` for rolling data
- **Memory Optimization**: Calculate exact lookback requirements
- **State Variables**: Position, entry price, stops, targets, indicator state
- **Previous Bar Data**: Store previous close/high/low for exit decisions

### **Bar Processing Protocol**
```python
def process_new_bar(self, open_price: float, high: float, low: float,
                   close: float, volume: float, timestamp: str) -> dict:
    """
    MANDATORY PROCESSING ORDER:
    1. Store new bar data in deques
    2. Update all technical indicators
    3. Process pending entry signals (execute at open)
    4. Check exit conditions (use previous bar close)
    5. Generate new entry signals (pending for next bar)
    6. Update state tracking variables
    7. Return strategy state dictionary
    """
```

## 🔐 **3. Self-Containment Requirements**

### **Import Restrictions**
- **PROHIBITED**: Imports from strategy module (`from .indicators import`)
- **PROHIBITED**: Imports from parameters module (`from .parameters import`)
- **ALLOWED**: Standard library imports (`math`, `collections`, `datetime`)
- **ALLOWED**: Core libraries (`numpy`, `pandas` if absolutely necessary)

### **Code Duplication Requirements**
- All indicator calculations MUST be inlined in template
- Parameter validation MUST be inlined in template
- No shared code between vectorized and stateful implementations
- Template must be completely standalone for production deployment

---

# **Security & Data Access Requirements**

## 🛡️ **1. Data Access Control**

### **PipelineOrchestrator Integration**
- ALL data access MUST go through `AuthorizedDataAccess` objects
- NO direct access to raw DataFrames during optimization
- Audit trail logging for every data access request
- Temporal segregation enforced (train/validation/test boundaries)

### **Anti-Leakage Validation**
```python
# AUTOMATIC VALIDATION
orchestrator.validate_no_data_leakage()
# Returns validation result with:
# - temporal_order_valid: bool
# - data_overlap_detected: bool  
# - date_ranges: dict with split boundaries
```

## 🔍 **2. State Isolation Requirements**

### **Optimization Contamination Prevention**
- **Mandatory**: `reset_state()` called before every trial
- **Detection**: System monitors for state persistence between trials
- **Enforcement**: Trial fails if contamination detected
- **Audit**: State reset events logged with strategy context

### **Parallel Processing Safety**
- **Thread Safety**: Each worker gets isolated strategy instance
- **Memory Isolation**: No shared objects between workers
- **Resource Limits**: Per-worker memory and timeout constraints
- **Error Isolation**: Worker failures don't affect other trials

## 📋 **3. Audit Trail Requirements**

### **Institutional Logging Standards**
```python
# MANDATORY LOG STRUCTURE
{
    "timestamp": "2025-08-09T10:15:30.123Z",  # ISO-8601 format
    "strategy_name": "bollinger_squeeze",      # Strategy identifier
    "phase": "optimization",                   # Pipeline phase
    "action": "parameter_validation",          # Specific action
    "parameters": {...},                       # Parameter hash/values
    "result": "success",                       # Outcome
    "metadata": {...}                          # Additional context
}
```

---

# **Testing & Validation Framework**

## 🧪 **1. Mandatory Test Categories**

| Test Category | Purpose | Coverage Requirement |
|---------------|---------|---------------------|
| **Parity Tests** | Vectorized/stateful synchronization | 100% signal agreement |
| **Parameter Tests** | Validation logic coverage | All constraint branches |
| **Edge Case Tests** | Robustness under extreme conditions | Boundary value analysis |
| **Performance Tests** | Resource usage and timing | Memory/CPU limits |
| **Integration Tests** | Pipeline compatibility | End-to-end workflow |

## ✅ **2. Test Implementation Requirements**

### **Parity Test Framework**
```python
@pytest.mark.parametrize("params", generate_parameter_combinations())
def test_strategy_parity(params):
    """
    MANDATORY PARITY VALIDATION:
    - Generate test OHLCV data (multiple market conditions)
    - Run vectorized implementation
    - Run stateful implementation bar-by-bar  
    - Assert exact signal series equality
    - Log first divergence point with context
    """
```

### **Edge Case Test Requirements**
- **Insufficient Data**: Less than `min_data_points`
- **Extreme Parameters**: Boundary values from ranges
- **Market Conditions**: Flat, trending, volatile, gapped
- **Data Quality**: Missing bars, zero volume, price limits

## 📊 **3. Coverage Requirements**

### **Code Coverage Standards**
- **Strategy Logic**: 95% line coverage minimum
- **Parameter Validation**: 100% branch coverage required
- **Error Handling**: All exception paths tested
- **State Reset**: All state variables validated

---

# **Performance & Resource Constraints**

## ⚡ **1. Optimization Performance Requirements**

| Metric | Requirement | Validation Method |
|--------|-------------|-------------------|
| **Trial Throughput** | 60+ trials/minute | Automated benchmarking |
| **Memory per Trial** | <50MB peak usage | Resource monitoring |
| **Trial Duration** | <30 seconds maximum | Timeout enforcement |
| **Parallel Workers** | Unlimited scalability | Load testing |

## 🎯 **2. Live Trading Performance Requirements**

| Metric | Requirement | Purpose |
|--------|-------------|---------|
| **Bar Processing** | <100ms per bar | Real-time compatibility |
| **Memory Footprint** | <10MB per strategy | Resource efficiency |
| **State Update** | <10ms per update | Low-latency execution |
| **Indicator Calculation** | <50ms for complex indicators | Performance optimization |

## 📈 **3. Scalability Requirements**

### **Strategy Scaling**
- **Discovery**: Support 100+ strategies without performance degradation
- **Optimization**: Parallel processing across unlimited workers
- **Deployment**: Batch deployment of multiple parameter sets
- **Monitoring**: Resource usage tracking per strategy

### **Data Scaling**
- **Historical Data**: Support 10+ years of minute-level data
- **Memory Management**: Efficient sliding windows for indicators
- **Processing Speed**: Vectorized operations for bulk calculations
- **Storage**: Optimized data structures for real-time updates

---

# **Compliance & Audit Requirements**

## 📋 **1. Regulatory Compliance**

### **Audit Trail Standards**
- **Complete Lineage**: Every strategy decision must be traceable
- **Parameter History**: All optimization trials stored with metadata
- **Performance Records**: Trade-by-trade execution logs
- **Risk Management**: Position limits and drawdown tracking

### **Documentation Requirements**
- **Strategy Description**: Business logic and market rationale
- **Parameter Documentation**: Each parameter's purpose and constraints
- **Risk Assessment**: Maximum drawdown and position size analysis
- **Backtesting Results**: Historical performance with confidence intervals

## 🔐 **2. Security Compliance**

### **Data Security Requirements**
- **Encryption**: All strategy parameters encrypted at rest
- **Access Control**: Role-based access to strategy configurations
- **Audit Logging**: All access and modifications logged
- **Version Control**: Complete change history for all strategies

### **Production Deployment Security**
- **Code Signing**: All deployed strategies digitally signed
- **Integrity Validation**: Template validation before deployment
- **Runtime Monitoring**: Real-time performance and risk monitoring
- **Emergency Stops**: Automatic strategy disable on risk limit breach

## 📊 **3. Performance Reporting**

### **Institutional Metrics Reporting**
```python
# MANDATORY PERFORMANCE REPORT STRUCTURE
{
    "strategy_name": "bollinger_squeeze",
    "optimization_results": {
        "best_parameters": {...},
        "composite_score": 0.847,
        "individual_metrics": {
            "prop_firm_viability": 0.92,
            "sortino_ratio": 1.45,
            "net_pnl": 4750.00,
            "max_drawdown": -1250.00,
            "profit_factor": 1.73,
            "win_rate": 0.68,
            "trade_frequency": 2.3
        }
    },
    "risk_assessment": {...},
    "audit_trail": {...}
}
```

---

# **🎯 Summary: Critical Success Factors**

## ✅ **Mandatory Compliance Checklist**

Before deploying ANY strategy to TopStepJ, verify:

### **📁 Convention Compliance**
- [ ] 4-file directory structure complete
- [ ] All file naming conventions followed
- [ ] Plugin discovery validation passes
- [ ] Strategy class inheritance correct

### **⚖️ Contract Compliance**  
- [ ] All 8 abstract methods implemented
- [ ] Signal timing protocol enforced (.shift(1))
- [ ] State reset implementation complete
- [ ] Parameter validation comprehensive

### **🔄 Implementation Parity**
- [ ] Vectorized and stateful implementations synchronized
- [ ] Parity tests pass with 100% signal agreement
- [ ] Timing protocol identical in both versions
- [ ] State management mirrors exactly

### **🔧 Deployment Readiness**
- [ ] Template parameter placeholders complete
- [ ] Fencing mechanism properly implemented
- [ ] Self-contained template (no external imports)
- [ ] Bar processing protocol implemented correctly

### **🛡️ Security & Compliance**
- [ ] Data access through AuthorizedDataAccess only
- [ ] Audit trail logging implemented
- [ ] State isolation enforced
- [ ] Performance requirements met

## 🚨 **Failure Points - System Rejection Scenarios**

Your strategy will be **automatically rejected** if:

1. **Discovery Phase**: Missing files, wrong naming, inheritance failure
2. **Optimization Phase**: Parameter mapping failure, constraint violations
3. **Deployment Phase**: Template validation failure, placeholder errors
4. **Validation Phase**: Parity test failure, timing violations

## 🎯 **Success Definition**

A strategy is **production ready** when:
- All requirements in this specification are satisfied
- Automated test suite passes completely
- Performance benchmarks are met
- Expert review validation completed

---

**Remember**: These are not suggestions—they are **mandatory technical requirements** for institutional-grade trading strategy development in TopStepJ. Compliance ensures your strategies will perform reliably from optimization through live trading deployment.

**For implementation guidance and examples, refer to the companion `STRATEGY_DEVELOPMENT_GUIDE.md`.**