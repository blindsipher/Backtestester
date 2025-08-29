from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ValidationTestConfig:
    """Configuration for a single validation test.

    Attributes
    ----------
    enabled:
        Whether this test should run.
    params:
        Optional parameters understood by the individual test implementation.
        Thresholds for pass/fail decisions are also supplied here under the
        ``threshold`` key when applicable.
    """

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Container for all validation tests and their configurations."""

    in_sample: ValidationTestConfig = field(default_factory=ValidationTestConfig)
    out_of_sample: ValidationTestConfig = field(default_factory=ValidationTestConfig)
    in_sample_permutation: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False, params={"permutations": 1000, "threshold": 0.05})
    )
    out_of_sample_permutation: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False, params={"permutations": 1000, "threshold": 0.05})
    )
    noise_injection: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False, params={"simulations": 100, "sigma": 0.01})
    )
    monte_carlo: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False, params={"simulations": 100})
    )
    regime_testing: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )
