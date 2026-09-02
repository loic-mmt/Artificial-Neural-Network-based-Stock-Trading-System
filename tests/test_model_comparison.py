import numpy as np
import pandas as pd
import pytest
import json

from trading_system.experiments.comparison import (
    ComparisonRun,
    build_comparison_runs,
    run_model_comparison,
    save_comparison_result,
    summarize_comparison_runs,
    validate_fair_comparison,
)
from trading_system.experiments.config import ExperimentConfig
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.factory import ModelRegistry


class ConstantSequenceModel:
    classes_ = np.arange(3)

    def __init__(self, name, label):
        self.model_name = name
        self.label = label

    def fit(self, X_train, y_train, *, X_val=None, y_val=None):
        assert X_train.ndim == X_val.ndim == 3
        return FitResult(1, "constant", TrainingHistory([1.0], [1.0]))

    def predict_proba(self, X):
        values = np.full((len(X), 3), 0.05, dtype=np.float32)
        values[:, self.label] = 0.9
        return values

    def state_dict(self):
        return {"label": np.asarray([self.label])}


def market_frame(rows=160):
    x = np.arange(rows, dtype=float)
    close = 100 + 0.03 * x + 2 * np.sin(x / 5)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=rows),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1e6 + 1000 * np.cos(x / 7),
        }
    )


def test_comparison_matrix_requires_equal_budgets_and_stable_order():
    runs = build_comparison_runs(
        {"z-model": [{"label": 1}], "a_model": [{"label": 2}]}, [3, 1]
    )
    assert [(run.model_name, run.seed) for run in runs] == [
        ("a_model", 3),
        ("a_model", 1),
        ("z_model", 3),
        ("z_model", 1),
    ]
    with pytest.raises(ValueError, match="same parameter-set budget"):
        build_comparison_runs(
            {"a": [{}], "b": [{}, {"depth": 2}]}, [1]
        )
    with pytest.raises(ValueError, match="unique"):
        build_comparison_runs({"a": [{}]}, [1, 1])


def test_fairness_rejects_one_model_seed_drift_and_oracle():
    config = ExperimentConfig(label_mode="forward_return")
    with pytest.raises(ValueError, match="at least two"):
        validate_fair_comparison(config, [ComparisonRun("a")])
    with pytest.raises(ValueError, match="identical seed"):
        validate_fair_comparison(
            config,
            [ComparisonRun("a", seed=1), ComparisonRun("b", seed=2)],
        )
    with pytest.raises(ValueError, match="Oracle"):
        validate_fair_comparison(
            ExperimentConfig(label_mode="oracle_all"),
            [ComparisonRun("a"), ComparisonRun("b")],
        )


def test_model_comparison_runs_registry_models_summarizes_and_saves_artifacts(tmp_path):
    registry = ModelRegistry()
    registry.register(
        "hold_model",
        lambda context, parameters: ConstantSequenceModel("hold_model", parameters["label"]),
    )
    registry.register(
        "buy_model",
        lambda context, parameters: ConstantSequenceModel("buy_model", parameters["label"]),
    )
    runs = build_comparison_runs(
        {"hold_model": [{"label": 1}], "buy_model": [{"label": 2}]}, [4]
    )
    result = run_model_comparison(
        market_frame(),
        ExperimentConfig(
            label_mode="forward_return",
            context_len=3,
            decision_mode="argmax",
        ),
        runs,
        registry,
        artifact_directory=tmp_path / "artifacts",
    )
    assert result.failures.empty
    assert result.runs.status.tolist() == ["ok", "ok"]
    assert set(result.summary.model_name) == {"buy_model", "hold_model"}
    assert "test_macro_f1_mean" in result.summary
    assert "test_macro_f1_median" in result.summary
    assert result.runs.config_hash.str.len().eq(64).all()
    assert result.runs.artifact_path.notna().all()
    assert len(list((tmp_path / "artifacts").iterdir())) == 2


def test_summary_excludes_failures_and_validates_schema():
    frame = pd.DataFrame(
        [
            {"model_name": "a", "status": "ok", "metric": 1.0},
            {"model_name": "a", "status": "error", "metric": 100.0},
        ]
    )
    summary = summarize_comparison_runs(frame)
    assert summary.metric_mean.tolist() == [1.0]
    assert summary.metric_median.tolist() == [1.0]
    assert summary.failure_count.tolist() == [1]
    with pytest.raises(ValueError, match="Missing comparison"):
        summarize_comparison_runs(pd.DataFrame({"status": ["ok"]}))


def test_comparison_result_persists_csv_and_strict_json(tmp_path):
    from trading_system.experiments.comparison import ComparisonResult

    result = ComparisonResult(
        runs=pd.DataFrame([{"model_name": "rnn", "status": "ok", "score": 1.0}]),
        summary=pd.DataFrame([{"model_name": "rnn", "score_std": np.nan}]),
        failures=pd.DataFrame(columns=["model_name", "error"]),
    )
    destination = save_comparison_result(
        result, tmp_path / "comparison", metadata={"seeds": [1, 7]}
    )
    assert {path.name for path in destination.iterdir()} == {
        "runs.csv",
        "summary.csv",
        "failures.csv",
        "report.json",
    }
    report = json.loads((destination / "report.json").read_text())
    assert report["metadata"]["seeds"] == [1, 7]
    assert report["summary"][0]["score_std"] is None
    with pytest.raises(FileExistsError):
        save_comparison_result(result, destination)


def test_comparison_cli_parses_frozen_models_and_seed_list(tmp_path):
    from trading_system.pipelines.compare_models import build_parser, load_ticker_selection

    parameter_file = tmp_path / "models.json"
    parameter_file.write_text('{"manual_ann": [{"epochs": 2}]}')
    ticker_file = tmp_path / "tickers.json"
    ticker_file.write_text(
        '{"tickers": [{"ticker": "BNP.PA"}, {"ticker": "SAN.PA"}]}'
    )
    args = build_parser().parse_args(
        [
            "--models",
            "manual_ann",
            "--seeds",
            "1,7,19,42,1337",
            "--model-parameter-sets",
            str(parameter_file),
            "--ticker-selection",
            str(ticker_file),
        ]
    )
    assert args.models == ["manual_ann"]
    assert args.seeds == [1, 7, 19, 42, 1337]
    assert args.model_parameter_sets["manual_ann"][0]["epochs"] == 2
    assert args.ticker_selection == ticker_file
    assert load_ticker_selection(ticker_file) == ["BNP.PA", "SAN.PA"]


def test_ticker_selection_rejects_duplicates_and_invalid_shape(tmp_path):
    from trading_system.pipelines.compare_models import load_ticker_selection

    duplicate_file = tmp_path / "duplicates.json"
    duplicate_file.write_text('{"tickers": ["BNP.PA", "BNP.PA"]}')
    with pytest.raises(ValueError, match="unique"):
        load_ticker_selection(duplicate_file)

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{"symbols": ["BNP.PA"]}')
    with pytest.raises(ValueError, match="tickers"):
        load_ticker_selection(invalid_file)
