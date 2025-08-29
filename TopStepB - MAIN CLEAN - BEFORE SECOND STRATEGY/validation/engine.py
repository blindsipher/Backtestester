from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, Iterable, List

import numpy as np

from .config import ValidationConfig, ValidationTestConfig
from .tests import TEST_FUNCTIONS


class ValidationEngine:
    """Run configured validation tests over parameter sets."""

    def __init__(self, config: ValidationConfig | None = None, max_workers: int | None = None) -> None:
        self.config = config or ValidationConfig()
        self.max_workers = max_workers or min(32, os.cpu_count() or 1)

    def _run_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Derive a deterministic seed from the parameter set for reproducibility
        seed = int(abs(hash(str(sorted(params.items())))) % (2**32))
        rng = np.random.default_rng(seed)

        results: Dict[str, Dict[str, Any]] = {}
        executed: List[str] = []
        total_score = 0.0
        passed_all = True

        informational = {"noise_injection", "regime_testing"}

        for name, cfg in self.config.__dict__.items():
            if not isinstance(cfg, ValidationTestConfig) or not cfg.enabled:
                continue

            test_fn = TEST_FUNCTIONS[name]
            metric, passed = test_fn(params, rng=rng, **cfg.params)
            results[name] = {"metric": metric, "passed": passed}
            executed.append(name)

            if name not in informational:
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
        params_list = list(parameter_sets)
        workers = min(self.max_workers, len(params_list)) or 1
        with ThreadPoolExecutor(max_workers=workers) as exe:
            return list(exe.map(self._run_single, params_list))
