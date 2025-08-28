"""Validation engine providing multiple statistical tests.

This module implements a configurable validation engine capable of running
several robustness tests over a collection of strategy parameter sets.  Each
test returns a metric and a pass/fail flag and contributes to an aggregated
score used by downstream analytics.

The implementation is intentionally lightweight but mirrors the structure of
an institutional pipeline.  Tests operate on in-sample and out-of-sample
return series which are supplied in each parameter dictionary under the keys
``in_sample_returns`` and ``out_of_sample_returns``.  If these keys are absent
the numeric values in the parameter dictionary are used as a stand‑in.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclasses


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


# ---------------------------------------------------------------------------
# Validation engine implementation


class ValidationEngine:
    """Run configured validation tests over parameter sets.

    Parameters
    ----------
    config:
        :class:`ValidationConfig` specifying which tests to run and their
        parameters.
    max_workers:
        Maximum number of worker threads to use when evaluating strategies.
    """

    def __init__(self, config: ValidationConfig | None = None, max_workers: int | None = None) -> None:
        self.config = config or ValidationConfig()
        self.max_workers = max_workers or min(32, os.cpu_count() or 1)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _extract_returns(params: Dict[str, Any], key: str) -> np.ndarray:
        data = params.get(key)
        if data is None:
            data = [v for v in params.values() if isinstance(v, (int, float))]
        return np.asarray(list(data), dtype=float)

    @staticmethod
    def _extract_in_sample_returns(params: Dict[str, Any]) -> np.ndarray:
        """Return combined train and validation returns for in-sample tests.

        The primary interface expects ``in_sample_returns`` but callers may
        provide explicit ``train_returns`` and ``validation_returns`` arrays.
        When provided, those arrays are concatenated to form the in-sample
        series.  This allows the validation engine to respect the
        train/validation split while maintaining backward compatibility with
        existing callers.
        """

        if "train_returns" in params or "validation_returns" in params:
            train = np.asarray(params.get("train_returns", []), dtype=float)
            val = np.asarray(params.get("validation_returns", []), dtype=float)
            return np.concatenate([train, val])
        return ValidationEngine._extract_returns(params, "in_sample_returns")

    @staticmethod
    def _permutation_metric(data: np.ndarray, permutations: int, rng: np.random.Generator) -> Tuple[float, float]:
        baseline = float(np.mean(data)) if data.size else 0.0
        count = 0
        for _ in range(permutations):
            if np.mean(rng.permutation(data)) >= baseline:
                count += 1
        p_value = (count + 1) / (permutations + 1)
        metric = 1.0 - p_value
        return metric, p_value

    # -- individual tests ------------------------------------------------

    def _test_in_sample(self, params: Dict[str, Any], *, threshold: float = 0.0, rng: np.random.Generator) -> Tuple[float, bool]:
        returns = self._extract_in_sample_returns(params)
        metric = float(np.mean(returns)) if returns.size else 0.0
        return metric, metric >= threshold

    def _test_out_of_sample(self, params: Dict[str, Any], *, threshold: float = 0.0, rng: np.random.Generator) -> Tuple[float, bool]:
        returns = self._extract_returns(params, "out_of_sample_returns")
        metric = float(np.mean(returns)) if returns.size else 0.0
        return metric, metric >= threshold

    def _test_in_sample_permutation(self, params: Dict[str, Any], *, permutations: int = 1000, threshold: float = 0.05, rng: np.random.Generator) -> Tuple[float, bool]:
        data = self._extract_in_sample_returns(params)
        metric, p_value = self._permutation_metric(data, permutations, rng)
        return metric, p_value <= threshold

    def _test_out_of_sample_permutation(self, params: Dict[str, Any], *, permutations: int = 1000, threshold: float = 0.05, rng: np.random.Generator) -> Tuple[float, bool]:
        data = self._extract_returns(params, "out_of_sample_returns")
        metric, p_value = self._permutation_metric(data, permutations, rng)
        return metric, p_value <= threshold

    def _test_noise_injection(self, params: Dict[str, Any], *, simulations: int = 100, sigma: float = 0.01, threshold: float = 0.0, rng: np.random.Generator) -> Tuple[float, bool]:
        data = self._extract_in_sample_returns(params)
        metrics = []
        for _ in range(simulations):
            noisy = data + rng.normal(0, sigma, size=data.size)
            metrics.append(float(np.mean(noisy)) if noisy.size else 0.0)
        metric = float(np.percentile(metrics, 5)) if metrics else 0.0
        return metric, metric >= threshold

    def _test_monte_carlo(self, params: Dict[str, Any], *, simulations: int = 100, threshold: float = 0.0, rng: np.random.Generator) -> Tuple[float, bool]:
        data = self._extract_in_sample_returns(params)
        metrics = []
        for _ in range(simulations):
            sample = rng.choice(data, size=data.size, replace=True) if data.size else np.array([])
            metrics.append(float(np.mean(sample)) if sample.size else 0.0)
        metric = float(np.percentile(metrics, 5)) if metrics else 0.0
        return metric, metric >= threshold

    def _test_regime_testing(self, params: Dict[str, Any], *, threshold: float = 0.0, rng: np.random.Generator) -> Tuple[float, bool]:
        data = self._extract_in_sample_returns(params)
        if data.size < 2:
            metric = float(np.mean(data)) if data.size else 0.0
            return metric, metric >= threshold
        mid = data.size // 2
        regime1 = float(np.mean(data[:mid])) if mid else 0.0
        regime2 = float(np.mean(data[mid:])) if data.size - mid else 0.0
        metric = min(regime1, regime2)
        return metric, metric >= threshold

    # -- execution -------------------------------------------------------

    def _run_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Derive a deterministic seed from the parameter set for reproducibility
        seed = int(abs(hash(str(sorted(params.items())))) % (2**32))
        rng = np.random.default_rng(seed)

        results: Dict[str, Dict[str, Any]] = {}
        executed: List[str] = []
        total_score = 0.0
        passed_all = True

        for name, cfg in self.config.__dict__.items():
            if not isinstance(cfg, ValidationTestConfig) or not cfg.enabled:
                continue

            test_fn = getattr(self, f"_test_{name}")
            metric, passed = test_fn(params, rng=rng, **cfg.params)
            results[name] = {"metric": metric, "passed": passed}
            executed.append(name)
            total_score += metric
            passed_all = passed_all and passed

        return {
            "params": params,
            "score": total_score,
            "tests": executed,
            "results": results,
            "passed": passed_all,
        }

    def run(self, parameter_sets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate parameter sets and return scored results."""

        params_list = list(parameter_sets)
        workers = min(self.max_workers, len(params_list)) or 1
        with ThreadPoolExecutor(max_workers=workers) as exe:
            return list(exe.map(self._run_single, params_list))


__all__ = ["ValidationConfig", "ValidationEngine", "ValidationTestConfig"]

