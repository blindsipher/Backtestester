from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import extract_returns


def test_monte_carlo(
    params: Dict[str, Any], *, simulations: int = 100, threshold: float = 0.0, rng: np.random.Generator
) -> Tuple[float, bool]:
    data = extract_returns(params, "in_sample_returns")
    metrics = []
    for _ in range(simulations):
        sample = (
            rng.choice(data, size=data.size, replace=True) if data.size else np.array([])
        )
        metrics.append(float(np.mean(sample)) if sample.size else 0.0)
    metric = float(np.percentile(metrics, 5)) if metrics else 0.0
    return metric, metric >= threshold
