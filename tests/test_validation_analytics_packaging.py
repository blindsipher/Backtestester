from validation import ValidationEngine
from analytics import AnalyticsEngine
from packager import PackagingEngine
import pandas as pd
import pytest
from data.data_structures import DataSplit


def _split(optimize, validate, test):
    return DataSplit(
        train=pd.DataFrame({"returns": optimize}),
        validation=pd.DataFrame({"returns": validate}),
        test=pd.DataFrame({"returns": test}),
        metadata={},
    )


def test_validation_analytics_packaging_flow():
    params = [
        {"data_split": _split([0.6, 0.7, 0.8], [0.1, 0.2, 0.3], [0.3, 0.2, 0.4])},
        {"data_split": _split([0.0, 0.1, -0.1], [0.0, 0.1, -0.1], [0.0, -0.2, 0.1])},
    ]

    val_engine = ValidationEngine()
    val_results = val_engine.run(params)
    assert "in_sample" in val_results[0]["tests"]
    assert val_results[0]["results"]["out_of_sample"]["metric"] == pytest.approx(0.3)
    assert val_results[0]["score"] > val_results[1]["score"]

    analytics = AnalyticsEngine(max_size=1)
    analytics.ingest(val_results)
    winners = analytics.get_winners()
    assert winners[0]["params"]["data_split"] == params[0]["data_split"]

    pkg_engine = PackagingEngine()
    pkg_result = pkg_engine.package("strat", winners)
    assert pkg_result["success"]
    assert pkg_result["packages"][0]["package_name"] == "strat_1.zip"
