from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from quantstats import reports as qsr
from quantstats import stats as qs


class AnalyticsEngine:
    """Augment validation results with quantstats tear sheets and select winners."""

    def __init__(self, max_size: int = 10, output_dir: str | os.PathLike[str] = "tear_sheets") -> None:
        self.max_size = max_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: List[Dict[str, Any]] = []

    def _generate_tear_sheet(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create quantstats tear sheet and compute key metrics."""

        params = result.get("params", {})
        returns = (
            params.get("in_sample_returns", [])
            + params.get("out_of_sample_returns", [])
        )

        if not returns:
            return {}

        index = pd.date_range("2000-01-01", periods=len(returns), freq="D")
        series = pd.Series(returns, index=index)

        def _safe(metric_fn: callable) -> float:
            try:
                value = metric_fn(series)
                return float(value) if value is not None else float("nan")
            except Exception:
                return float("nan")

        metrics = {
            "sharpe": _safe(qs.sharpe),
            "sortino": _safe(qs.sortino),
            "calmar": _safe(qs.calmar),
            "max_drawdown": _safe(qs.max_drawdown),
        }

        file_path = self.output_dir / f"tear_sheet_{uuid.uuid4().hex}.html"
        try:
            qsr.html(series, output=str(file_path), title="Strategy Tear Sheet")
        except Exception:
            file_path = Path()

        return {"metrics": metrics, "path": str(file_path)}

    def ingest(self, validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store passed results, attach tear sheets, and keep top performers."""

        for res in validation_results:
            if not res.get("passed"):
                continue
            res["tear_sheet"] = self._generate_tear_sheet(res)
            self._cache.append(res)

        self._cache.sort(key=lambda r: r.get("score", 0), reverse=True)
        self._cache = self._cache[: self.max_size]
        return list(self._cache)

    def get_winners(self, top_n: int | None = None) -> List[Dict[str, Any]]:
        """Return cached winners, optionally limited to *top_n* entries."""

        if top_n is None or top_n >= len(self._cache):
            return list(self._cache)
        return self._cache[:top_n]

