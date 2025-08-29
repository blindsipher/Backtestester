from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import extract_returns


def test_in_sample(
    params: Dict[str, Any], *, threshold: float = 0.0, rng: np.random.Generator
) -> Tuple[float, bool]:
    returns = extract_returns(params, "in_sample_returns")
    metric = float(np.mean(returns)) if returns.size else 0.0
    return metric, metric >= threshold
