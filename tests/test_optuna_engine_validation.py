"""Additional tests for OptunaEngine validation and data preparation."""

import sys
import types
import time

import pandas as pd
import optuna

# Provide minimal torch and psutil stubs before importing modules that require them
torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, set_device=lambda x: None))
psutil_stub = types.SimpleNamespace(
    virtual_memory=lambda: types.SimpleNamespace(total=0),
    Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0)),
)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("psutil", psutil_stub)

from optimization.engine import OptunaEngine
from optimization.config.optuna_config import OptimizationConfig
from app.core.state import PipelineState


class DummyStrategy:
    strategy_name = "dummy"

    def get_parameter_ranges(self):
        return {"x": (0, 1)}

    def validate_parameters(self, params):
        return True


class DummyTradingConfig:
    account_config = None


class DummyAccess:
    def __init__(self, data):
        self.train_data = data
        self.validation_data = data
        self.test_data = None


class DummyOrchestrator:
    def __init__(self, data):
        self._access = DummyAccess(data)

    def get_authorized_data(self, phase, purpose):
        return self._access


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


def test_validate_pipeline_state_missing_trading_config():
    engine = OptunaEngine()
    state = build_state(
        trading_config=None,
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=DummyStrategy(),
        secure_orchestrator=True,
    )
    result = engine._validate_pipeline_state(state)
    assert not result["valid"]
    assert "No trading configuration" in result["error"]


def test_validate_pipeline_state_missing_data():
    engine = OptunaEngine()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=None,
        strategy_instance=DummyStrategy(),
        secure_orchestrator=True,
    )
    result = engine._validate_pipeline_state(state)
    assert not result["valid"]
    assert "No optimization data" in result["error"]


def test_prepare_optimization_data_requires_orchestrator():
    engine = OptunaEngine()
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=DummyStrategy(),
        secure_orchestrator=None,
    )
    result = engine._prepare_optimization_data(state)
    assert not result["success"]
    assert "No secure orchestrator" in result["error"]


def test_prepare_optimization_data_success():
    engine = OptunaEngine()
    data = pd.DataFrame({"a": [1, 2, 3]})
    orchestrator = DummyOrchestrator(data)
    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=data,
        strategy_instance=DummyStrategy(),
        secure_orchestrator=orchestrator,
    )
    result = engine._prepare_optimization_data(state)
    assert result["success"]
    assert result["authorized_accesses"][0].train_data.equals(data)


def test_run_optimization_completes_quickly():
    config = OptimizationConfig()
    config.limits.max_trials = 5
    config.limits.max_workers = 1
    engine = OptunaEngine(config)
    engine.study = optuna.create_study(direction="maximize")

    state = build_state(
        trading_config=DummyTradingConfig(),
        full_data=pd.DataFrame({"a": [1]}),
        strategy_instance=DummyStrategy(),
        secure_orchestrator=True,
    )

    start = time.time()
    result = engine._run_optimization(lambda trial: 1.0, state)
    duration = time.time() - start
    assert result["success"]
    assert duration < 5
