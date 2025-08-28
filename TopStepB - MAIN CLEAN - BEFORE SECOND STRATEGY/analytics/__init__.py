from typing import Any, Dict, List


class AnalyticsEngine:
    """Cache validation results and return the best performers.

    Only parameter sets that pass their validation tests are cached.  This
    mirrors production behaviour where failed strategies are discarded rather
    than forwarded to downstream analytics or packaging stages.
    """

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._cache: List[Dict[str, Any]] = []

    def ingest(self, validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store results and keep only the highest scoring entries.

        Only results with ``passed=True`` are considered for caching.
        """

        passed = [r for r in validation_results if r.get("passed")]
        self._cache.extend(passed)
        self._cache.sort(key=lambda r: r.get("score", 0), reverse=True)
        self._cache = self._cache[: self.max_size]
        return self._cache

    def get_winners(self, top_n: int | None = None) -> List[Dict[str, Any]]:
        """Return cached winners, optionally limited to *top_n* entries."""

        if top_n is None or top_n >= len(self._cache):
            return list(self._cache)
        return self._cache[:top_n]
