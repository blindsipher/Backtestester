from validation import ValidationEngine
from analytics import AnalyticsEngine
from packager import PackagingEngine


def test_validation_analytics_packaging_flow():
    params = [
        {
            "in_sample_returns": [0.6, 0.7, 0.8],
            "out_of_sample_returns": [0.3, 0.2, 0.4],
        },
        {
            "in_sample_returns": [0.0, 0.1, -0.1],
            "out_of_sample_returns": [0.0, -0.2, 0.1],
        },
    ]

    val_engine = ValidationEngine()
    val_results = val_engine.run(params)
    assert "in_sample" in val_results[0]["tests"]
    assert val_results[0]["score"] > val_results[1]["score"]

    analytics = AnalyticsEngine(max_size=1)
    analytics.ingest(val_results)
    winners = analytics.get_winners()
    assert winners[0]["params"] == params[0]

    pkg_engine = PackagingEngine()
    pkg_result = pkg_engine.package("strat", winners)
    assert pkg_result["success"]
    assert pkg_result["packages"][0]["package_name"] == "strat_1.zip"
