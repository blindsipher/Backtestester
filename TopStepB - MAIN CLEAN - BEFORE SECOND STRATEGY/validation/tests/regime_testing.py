from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import extract_returns


def test_regime_testing(
    params: Dict[str, Any], *, threshold: float = 0.0, rng: np.random.Generator
) -> Tuple[float, bool]:
    data = extract_returns(params, "in_sample_returns")
    if data.size < 2:
        metric = float(np.mean(data)) if data.size else 0.0
        return metric, metric >= threshold
    mid = data.size // 2
    regime1 = float(np.mean(data[:mid])) if mid else 0.0
    regime2 = float(np.mean(data[mid:])) if data.size - mid else 0.0
    metric = min(regime1, regime2)
    return metric, metric >= threshold
