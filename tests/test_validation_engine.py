import pytest

from validation import ValidationEngine, ValidationConfig, ValidationTestConfig


def test_validation_engine_runs_all_tests():
    config = ValidationConfig(
        in_sample=ValidationTestConfig(params={"threshold": 0.5}),
        out_of_sample=ValidationTestConfig(params={"threshold": 0.2}),
        in_sample_permutation=ValidationTestConfig(
            enabled=True, params={"permutations": 10, "threshold": 1.0}
        ),
        out_of_sample_permutation=ValidationTestConfig(
            enabled=True, params={"permutations": 10, "threshold": 1.0}
        ),
        noise_injection=ValidationTestConfig(
            enabled=True, params={"simulations": 10, "sigma": 0.01}
        ),
        monte_carlo=ValidationTestConfig(
            enabled=True, params={"simulations": 10}
        ),
        regime_testing=ValidationTestConfig(enabled=True),
    )

    engine = ValidationEngine(config=config, max_workers=1)
    params = [
        {
            "in_sample_returns": [0.6, 0.7, 0.8],
            "out_of_sample_returns": [0.3, 0.4, 0.5],
        }
    ]

    results = engine.run(params)
    assert len(results) == 1
    res = results[0]
    expected_tests = {
        "in_sample",
        "out_of_sample",
        "in_sample_permutation",
        "out_of_sample_permutation",
        "noise_injection",
        "monte_carlo",
        "regime_testing",
    }
    assert set(res["tests"]) == expected_tests
    assert set(res["results"].keys()) == expected_tests
    assert res["passed"] is True
    # Score is sum of individual metrics
    metric_sum = sum(v["metric"] for v in res["results"].values())
    assert res["score"] == pytest.approx(metric_sum)

