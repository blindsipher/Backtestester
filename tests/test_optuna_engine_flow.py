"""Tests for OptunaEngine flow."""

# ruff: noqa: E402

import sys
import types
import pandas as pd

# Provide minimal torch and psutil stubs before importing modules that require them
torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, set_device=lambda x: None))
psutil_stub = types.SimpleNamespace(
    virtual_memory=lambda: types.SimpleNamespace(total=0),
    Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0)),
)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("psutil", psutil_stub)

from optimization.engine import OptunaEngine
from app.core.state import PipelineState


class DummyStrategy:
    strategy_name = "dummy"

    def get_parameter_ranges(self):
        return {"x": (0, 1)}

    def validate_parameters(self, params):
        return True


class DummyTradingConfig:
    account_config = None


def build_state(**kwargs):
    defaults = dict(
        strategy_name="s",
        symbol="SYM",
        timeframe="1m",
        account_type="demo",
        slippage_ticks=0,
        commission_per_trade=0.0,
        contracts_per_trade=1,
        split_type="chronological",
    )
    defaults.update(kwargs)
    return PipelineState(**defaults)


def test_validate_pipeline_state_missing_strategy():
    engine = OptunaEngine()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=None,
    )
    result = engine._validate_pipeline_state(state)
    assert not result["valid"]
    assert "No strategy instance" in result["error"]


def test_run_returns_error_for_invalid_state():
    engine = OptunaEngine()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=None,
    )
    output = engine.run(state)
    assert not output["success"]
    assert "Pipeline validation failed" in output["error"]


def test_run_requires_secure_orchestrator():
    engine = OptunaEngine()
    strategy = DummyStrategy()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=strategy,
    )
    result = engine.run(state)
    assert not result["success"]
    assert "No secure orchestrator" in result["error"]


def test_run_success_flow(monkeypatch):
    engine = OptunaEngine()
    strategy = DummyStrategy()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=strategy,
        secure_orchestrator=True,
    )

    monkeypatch.setattr(engine, "_validate_pipeline_state", lambda s: {"valid": True})
    monkeypatch.setattr(engine, "_prepare_optimization_data", lambda s: {"success": True, "authorized_accesses": []})
    dummy_study = type("Study", (), {"trials": []})()
    monkeypatch.setattr(engine, "_create_optuna_study", lambda s: {"success": True, "study": dummy_study, "paths": {}})
    monkeypatch.setattr(engine.objective_factory, "create_objective", lambda *a, **k: lambda trial: 0)
    monkeypatch.setattr(engine, "_run_optimization", lambda obj, st: {"success": True})
    expected = {"success": True, "data": 1}
    monkeypatch.setattr(engine, "_process_optimization_results", lambda s, p: expected)

    output = engine.run(state)
    assert output == expected
