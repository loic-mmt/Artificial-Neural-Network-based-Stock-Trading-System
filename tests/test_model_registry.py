import sys

import numpy as np
import pytest

from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.factory import ModelRegistry, create_default_model_registry
from trading_system.models.manual_ann.sequence_adapter import ManualANNSequenceAdapter
from trading_system.models.specs import ModelBuildContext, ModelSelection


class DummySequenceClassifier:
    model_name = "dummy"
    classes_ = np.arange(3)

    def fit(self, X_train, y_train, *, X_val=None, y_val=None):
        return FitResult(1, "done", TrainingHistory())

    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3)

    def state_dict(self):
        return {}


def test_model_build_context_and_selection_validation():
    assert ModelBuildContext(4, 20, seed=0, device="mps").input_size == 4
    with pytest.raises(ValueError, match="positive"):
        ModelBuildContext(0, 20)
    with pytest.raises(TypeError, match="integer"):
        ModelBuildContext(True, 20)
    with pytest.raises(ValueError, match="device"):
        ModelBuildContext(4, 20, device="tpu")

    source = {"depth": 2}
    selection = ModelSelection("  Manual-ANN ", source)
    source["depth"] = 9
    assert selection == ModelSelection("manual_ann", {"depth": 2})
    with pytest.raises(ValueError, match="Model name"):
        ModelSelection("bad name")
    with pytest.raises(TypeError, match="JSON metadata"):
        ModelSelection("model", {"weights": np.arange(2)})


def test_registry_is_empty_deterministic_and_defensive():
    registry = ModelRegistry()
    assert registry.names() == ()
    received = {}

    def factory(context, parameters):
        received.update(parameters)
        parameters["changed"] = True
        return DummySequenceClassifier()

    registry.register("Z-model", factory)
    registry.register("a_model", factory)
    assert registry.names() == ("a_model", "z_model")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("z_model", factory)
    parameters = {"depth": 2}
    assert isinstance(
        registry.build("Z_MODEL", ModelBuildContext(4, 5), parameters),
        DummySequenceClassifier,
    )
    assert received == {"depth": 2}
    assert parameters == {"depth": 2}
    with pytest.raises(KeyError, match="available: a_model, z_model"):
        registry.build("missing", ModelBuildContext(4, 5))


def test_default_registry_is_lazy_and_builds_manual_ann():
    torch_before = sys.modules.get("torch")
    registry = create_default_model_registry()
    assert registry.names() == ("gru", "lstm", "manual_ann", "rnn", "transformer")
    assert sys.modules.get("torch") is torch_before
    model = registry.build(
        "manual_ann",
        ModelBuildContext(input_size=4, context_len=5, seed=7),
        {"hidden_size": 6, "epochs": 1},
    )
    assert isinstance(model, ManualANNSequenceAdapter)
    assert model.config.seed == 7
    with pytest.raises(ValueError, match="controlled"):
        registry.build(
            "manual_ann", ModelBuildContext(4, 5, seed=7), {"seed": 8}
        )
