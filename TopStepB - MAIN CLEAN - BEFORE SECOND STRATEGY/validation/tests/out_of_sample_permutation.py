from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .utils import extract_returns, permutation_metric


def test_out_of_sample_permutation(
    params: Dict[str, Any], *, permutations: int = 1000, threshold: float = 0.05, rng: np.random.Generator
) -> Tuple[float, bool]:
    data = extract_returns(params, "out_of_sample_returns")
    metric, p_value = permutation_metric(data, permutations, rng)
    return metric, p_value <= threshold
