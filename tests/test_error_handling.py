import pytest
from utils.error_handling import (
    ErrorResultFactory,
    log_and_return_error,
    safe_execute,
    validate_required_keys,
    ensure_error_result_format,
)


def test_create_error_result_with_context_and_data():
    result = ErrorResultFactory.create_error_result(
        "failure", error_context="phase", additional_data={"extra": 1}
    )
    assert result["success"] is False
    assert result["error"] == "failure"
    assert result["error_context"] == "phase"
    assert result["extra"] == 1
    assert "timestamp" in result


def test_log_and_return_error():
    result = log_and_return_error("problem", context="ctx")
    assert not result["success"]
    assert result["error"] == "problem"
    assert result["error_context"] == "ctx"


def test_safe_execute_success_and_failure():
    def good(x):
        return x * 2

    def bad():
        raise ValueError("boom")

    res_good = safe_execute(good, 3)
    assert res_good["success"] and res_good["result"] == 6

    res_bad = safe_execute(bad)
    assert not res_bad["success"]
    assert "error" in res_bad


def test_validate_required_keys():
    data = {"a": 1, "b": 2}
    assert validate_required_keys(data, ["a", "b"], "ctx") is None
    assert "missing" in validate_required_keys(data, ["a", "c"], "ctx")


def test_ensure_error_result_format_adds_defaults():
    result = ensure_error_result_format({"error": "oops"})
    assert not result["success"]
    assert "best_parameters" in result
    assert "optimization_metadata" in result
