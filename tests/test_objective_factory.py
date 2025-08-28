import sys
import types
import pandas as pd
import pytest

# Stub torch and psutil before importing optimization modules
torch_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, set_device=lambda x: None))
psutil_stub = types.SimpleNamespace(
    virtual_memory=lambda: types.SimpleNamespace(total=0),
    Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0)),
)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("psutil", psutil_stub)

from optimization.objective import ObjectiveFactory
from optimization.config.optuna_config import OptimizationConfig
from app.core.pipeline_orchestrator import AuthorizedDataAccess


class DummyStrategy:
    strategy_name = "dummy"

    def get_parameter_ranges(self):
        return {"x": (0, 1)}

    def validate_parameters(self, params):
        return True


class EmptyRangeStrategy(DummyStrategy):
    def get_parameter_ranges(self):
        return {}


class DummyTradingConfig:
    account_config = None


def make_access(train=None, validation=None):
    df = pd.DataFrame({"x": [1]})
    return AuthorizedDataAccess(train_data=train or df, validation_data=validation or df)


def test_create_objective_requires_authorized_accesses():
    strategy = DummyStrategy()
    factory = ObjectiveFactory(OptimizationConfig())
    with pytest.raises(ValueError) as exc:
        factory.create_objective(strategy, [], DummyTradingConfig(), {})
    assert "No authorized data accesses" in str(exc.value)


def test_create_objective_requires_parameter_ranges():
    strategy = EmptyRangeStrategy()
    factory = ObjectiveFactory(OptimizationConfig())
    access = make_access()
    with pytest.raises(ValueError) as exc:
        factory.create_objective(strategy, [access], DummyTradingConfig(), {})
    assert "Strategy provided empty parameter ranges" in str(exc.value)


def test_stateful_objective_validates_access_data():
    strategy = DummyStrategy()
    factory = ObjectiveFactory(OptimizationConfig())
    empty_df = pd.DataFrame()
    access = AuthorizedDataAccess(train_data=pd.DataFrame({"x": [1]}), validation_data=empty_df)
    with pytest.raises(ValueError) as exc:
        factory.create_objective(strategy, [access], DummyTradingConfig(), {})
    assert "missing train or validation data" in str(exc.value)
