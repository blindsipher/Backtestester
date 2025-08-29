from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def extract_returns(params: Dict[str, Any], key: str) -> np.ndarray:
    data = params.get(key)
    if data is None:
        data = [v for v in params.values() if isinstance(v, (int, float))]
    return np.asarray(list(data), dtype=float)


def permutation_metric(data: np.ndarray, permutations: int, rng: np.random.Generator) -> Tuple[float, float]:
    baseline = float(np.mean(data)) if data.size else 0.0
    count = 0
    for _ in range(permutations):
        if np.mean(rng.permutation(data)) >= baseline:
            count += 1
    p_value = (count + 1) / (permutations + 1)
    metric = 1.0 - p_value
    return metric, p_value
