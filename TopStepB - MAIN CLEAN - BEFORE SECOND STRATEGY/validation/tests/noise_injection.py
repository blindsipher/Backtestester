from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import extract_returns


def test_noise_injection(
    params: Dict[str, Any], *, simulations: int = 100, sigma: float = 0.01, threshold: float = 0.0, rng: np.random.Generator
) -> Tuple[float, bool]:
    data = extract_returns(params, "in_sample_returns")
    metrics = []
    for _ in range(simulations):
        noisy = data + rng.normal(0, sigma, size=data.size)
        metrics.append(float(np.mean(noisy)) if noisy.size else 0.0)
    metric = float(np.percentile(metrics, 5)) if metrics else 0.0
    return metric, metric >= threshold
