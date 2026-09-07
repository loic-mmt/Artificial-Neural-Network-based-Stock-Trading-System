import numpy as np
import pandas as pd

from trading_system.experiments.comparison import (
    ComparisonRun,
    add_stability_flags,
    run_model_comparison,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import run_validation_experiment
from trading_system.experiments.walkforward import walk_forward_classifier
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.manual_ann.manual_nn import ManualANNClassifier, ManualANNConfig
from trading_system.models.specs import ModelSelection
from trading_system.models.factory import ModelRegistry
from trading_system.pipelines.compare_models import build_parser
from trading_system.pipelines.overfitting_arguments import overfitting_config_from_args
from trading_system.training.overfitting import (
    OverfittingControlConfig,
    TrainOnlyFeatureSelector,
    apply_overfitting_profile,
    generalization_diagnostics,
)


def test_compact_profile_preserves_explicit_parameters():
    control = OverfittingControlConfig()
    resolved = apply_overfitting_profile(
        ModelSelection("transformer", {"d_model": 48, "epochs": 7}), control
    )
    assert resolved.parameters["d_model"] == 48
    assert resolved.parameters["epochs"] == 7
    assert resolved.parameters["dim_feedforward"] == 64
    assert resolved.parameters["dropout"] == 0.3
    assert resolved.parameters["weight_decay"] == 1e-4
    assert apply_overfitting_profile(resolved, None) is resolved


def test_train_only_selector_filters_constant_correlated_and_low_rank_features():
    x = np.linspace(-1, 1, 60)
    train = pd.DataFrame(
        {
            "signal": x,
            "duplicate": x * 2,
            "noise": np.sin(np.arange(60) * 2.3),
            "constant": 1.0,
            "Label_id": np.where(x > 0.2, 2, np.where(x < -0.2, 0, 1)),
        }
    )
    selector = TrainOnlyFeatureSelector(
        OverfittingControlConfig(max_features=2, max_feature_correlation=0.95)
    ).fit(train, ("signal", "duplicate", "noise", "constant"))
    assert len(selector.columns) == 2
    assert "constant" not in selector.columns
    assert not {"signal", "duplicate"}.issubset(selector.columns)
    assert selector.state_dict()["fit_scope"] == "train_only"


def test_manual_ann_weight_decay_is_decoupled_and_biases_are_not_decayed():
    X = np.arange(24, dtype=np.float32).reshape(6, 4) / 10
    y = np.array([0, 1, 2, 0, 1, 2])
    common = dict(hidden_size=5, epochs=1, batch_size=6, learning_rate=0.01, seed=8)
    plain = ManualANNClassifier(ManualANNConfig(**common)).fit(X, y)
    regularized_model = ManualANNClassifier(
        ManualANNConfig(**common, weight_decay=0.2)
    )
    regularized_model.fit(X, y)
    plain_state = plain  # fit result retained only to ensure both fits complete
    assert plain_state.best_epoch == 1
    baseline_model = ManualANNClassifier(ManualANNConfig(**common))
    baseline_model.fit(X, y)
    decay = 1.0 - 0.01 * 0.2
    np.testing.assert_allclose(regularized_model.W0_, baseline_model.W0_ * decay, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(regularized_model.W1_, baseline_model.W1_ * decay, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(regularized_model.b0_, baseline_model.b0_)
    np.testing.assert_allclose(regularized_model.b1_, baseline_model.b1_)


def test_generalization_diagnostics_use_best_checkpoint_epoch():
    fit = FitResult(2, "early_stopping", TrainingHistory([0.9, 0.5, 0.4], [1.0, 0.7, 0.8]))
    diagnostics = generalization_diagnostics(fit)
    assert diagnostics["train_loss_at_best"] == 0.5
    assert diagnostics["val_loss_at_best"] == 0.7
    assert np.isclose(diagnostics["loss_gap"], 0.2)
    assert diagnostics["epochs_ran"] == 3


def test_stability_gate_uses_validation_dispersion_gap_and_seed_coverage():
    summary = pd.DataFrame(
        {
            "model_name": ["stable", "variable", "overfit", "incomplete"],
            "success_count": [5, 5, 5, 4],
            "val_macro_f1_std": [0.02, 0.08, 0.02, 0.02],
            "train_macro_f1_mean": [0.55, 0.55, 0.80, 0.55],
            "val_macro_f1_mean": [0.50, 0.50, 0.50, 0.50],
        }
    )
    flagged = add_stability_flags(
        summary,
        expected_runs=5,
        max_validation_metric_std=0.05,
        max_train_validation_gap=0.15,
        require_all_seeds=True,
    )
    assert flagged.set_index("model_name")["stable"].to_dict() == {
        "stable": True,
        "variable": False,
        "overfit": False,
        "incomplete": False,
    }
    empty = add_stability_flags(
        pd.DataFrame(columns=["model_name"]), expected_runs=5,
        max_validation_metric_std=0.05, max_train_validation_gap=0.15,
        require_all_seeds=True,
    )
    assert empty.empty and "stable" in empty


def test_overfitting_cli_is_opt_in_and_configurable():
    parser = build_parser()
    assert overfitting_config_from_args(parser.parse_args([])) is None
    args = parser.parse_args(
        ["--overfitting-control", "--overfitting-max-features", "12"]
    )
    assert overfitting_config_from_args(args).max_features == 12


def test_static_validation_applies_profile_and_frozen_train_only_selection():
    rows = 150
    x = np.arange(rows, dtype=float)
    close = 100 + 0.04 * x + 2 * np.sin(x / 6)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1e6 + 1000 * np.cos(x / 5),
        }
    )
    result = run_validation_experiment(
        frame,
        ExperimentConfig(
            label_mode="forward_return",
            context_len=3,
            decision_mode="argmax",
            model=ModelSelection("manual_ann", {"epochs": 2, "batch_size": 32}),
            overfitting_control=OverfittingControlConfig(max_features=5),
        ),
    )
    assert len(result.bundle.feature_columns) == 5
    assert result.bundle.overfitting_selector.state_dict()["fit_scope"] == "train_only"
    assert result.bundle.model_selection.parameters["weight_decay"] == 1e-4
    assert result.bundle.model_selection.parameters["dropout_probability"] == 0.3


def test_multi_seed_stability_gate_keeps_test_sealed_for_unstable_candidate():
    class ConstantModel:
        classes_ = np.arange(3)

        def __init__(self, name, label):
            self.model_name, self.label = name, label

        def fit(self, X_train, y_train, *, X_val=None, y_val=None):
            return FitResult(1, "constant", TrainingHistory([0.5], [0.5]))

        def predict_proba(self, X):
            probabilities = np.full((len(X), 3), 0.05, dtype=np.float32)
            probabilities[:, self.label] = 0.9
            return probabilities

        def state_dict(self):
            return {"label": np.array([self.label])}

    rows = 150
    x = np.arange(rows, dtype=float)
    close = 100 + 0.04 * x + 2 * np.sin(x / 6)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows),
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close,
            "volume": 1e6 + 1000 * np.cos(x / 5),
        }
    )
    registry = ModelRegistry()
    registry.register("stable_model", lambda context, parameters: ConstantModel("stable_model", 1))
    registry.register("variable_model", lambda context, parameters: ConstantModel("variable_model", context.seed % 3))
    runs = [
        ComparisonRun(model, seed=seed)
        for model in ("stable_model", "variable_model")
        for seed in (1, 2)
    ]
    result = run_model_comparison(
        frame,
        ExperimentConfig(
            label_mode="forward_return", context_len=3, decision_mode="argmax",
            overfitting_control=OverfittingControlConfig(
                max_features=5, max_validation_metric_std=0.001,
            ),
        ),
        runs,
        registry,
    )
    by_model = result.runs.groupby("model_name")
    assert by_model.final_test_evaluated.all()["stable_model"]
    assert not by_model.final_test_evaluated.any()["variable_model"]
    assert by_model.test_macro_f1.count()["variable_model"] == 0


def test_walkforward_refits_train_only_selector_per_history_chunk():
    class HoldModel:
        classes_ = np.arange(3)
        model_name = "hold_model"

        def fit(self, X_train, y_train, *, X_val=None, y_val=None):
            return FitResult(1, "constant", TrainingHistory([0.5], [0.5]))

        def predict_proba(self, X):
            probabilities = np.full((len(X), 3), 0.05, dtype=np.float32)
            probabilities[:, 1] = 0.9
            return probabilities

        def state_dict(self):
            return {"label": np.array([1])}

    rows = 140
    x = np.arange(rows, dtype=float)
    close = 100 + 0.03 * x + np.sin(x / 5)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows),
            "adj_close": close,
            "signal": np.sin(x / 4),
            "duplicate": np.sin(x / 4) * 2,
        }
    )
    registry = ModelRegistry()
    registry.register("hold_model", lambda context, parameters: HoldModel())
    result = walk_forward_classifier(
        frame,
        ("signal", "duplicate"),
        train_ratio=0.7,
        val_ratio=0.15,
        walkforward_step=20,
        context_len=3,
        label_mode="forward_return",
        model_selection=ModelSelection("hold_model"),
        registry=registry,
        evaluation_split="validation",
        overfitting_control=OverfittingControlConfig(
            max_features=1, max_feature_correlation=0.9
        ),
    )
    assert result["retrain_logs"]
    assert all(
        log["overfitting_feature_selection"]["fit_scope"] == "train_only"
        and len(log["overfitting_feature_selection"]["selected_columns"]) == 1
        for log in result["retrain_logs"]
    )
