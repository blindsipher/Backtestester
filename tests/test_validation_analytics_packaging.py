from validation import ValidationEngine
from analytics import AnalyticsEngine
from packager import PackagingEngine


def test_validation_analytics_packaging_flow():
    params = [{"a": 1, "b": 2}, {"a": 0, "b": 0}]

    val_engine = ValidationEngine()
    val_results = val_engine.run(params)
    assert val_results[0]["score"] == 3
    assert "in_sample" in val_results[0]["tests"]

    analytics = AnalyticsEngine(max_size=1)
    analytics.ingest(val_results)
    winners = analytics.get_winners()
    assert winners[0]["params"] == {"a": 1, "b": 2}

    pkg_engine = PackagingEngine()
    pkg_result = pkg_engine.package("strat", winners)
    assert pkg_result["success"]
    assert pkg_result["packages"][0]["package_name"] == "strat_1.zip"
