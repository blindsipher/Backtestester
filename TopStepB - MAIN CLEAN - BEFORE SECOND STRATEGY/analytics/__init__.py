from typing import Any, Dict, List


class AnalyticsEngine:
    """Cache validation results and return the best performers."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._cache: List[Dict[str, Any]] = []

    def ingest(self, validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store results and keep only the highest scoring entries."""

        self._cache.extend(validation_results)
        self._cache.sort(key=lambda r: r.get("score", 0), reverse=True)
        self._cache = self._cache[: self.max_size]
        return self._cache

    def get_winners(self, top_n: int | None = None) -> List[Dict[str, Any]]:
        """Return cached winners, optionally limited to *top_n* entries."""

        if top_n is None or top_n >= len(self._cache):
            return list(self._cache)
        return self._cache[:top_n]
