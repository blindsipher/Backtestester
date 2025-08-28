from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationTestConfig:
    """Configuration for a single validation test."""

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Container for all validation tests and their configurations."""

    in_sample: ValidationTestConfig = field(default_factory=ValidationTestConfig)
    out_of_sample: ValidationTestConfig = field(default_factory=ValidationTestConfig)
    in_sample_permutation: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )
    out_of_sample_permutation: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )
    noise_injection: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )
    monte_carlo: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )
    regime_testing: ValidationTestConfig = field(
        default_factory=lambda: ValidationTestConfig(enabled=False)
    )


class ValidationEngine:
    """Run configured validation tests over parameter sets.

    The current implementation is lightweight and deterministic. Each enabled
    test simply contributes to a list of executed tests, while the score is a
    sum of numeric parameter values. The structure allows future replacement
    with realistic validation logic operating on walk-forward data splits.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def run(self, parameter_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate parameter sets and return scored results.

        Args:
            parameter_sets: List of parameter dictionaries.

        Returns:
            List of validation result dictionaries containing the original
            parameters, a deterministic score, and the list of executed tests.
        """

        results: List[Dict[str, Any]] = []
        for params in parameter_sets:
            score = sum(
                value for value in params.values() if isinstance(value, (int, float))
            )
            executed = [
                name
                for name, cfg in self.config.__dict__.items()
                if isinstance(cfg, ValidationTestConfig) and cfg.enabled
            ]
            results.append({"params": params, "score": score, "tests": executed})
        return results
