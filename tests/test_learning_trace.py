"""Synthetic numerical parity: diagnostic observations never change training."""

from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.training.learning_trace import LearningTrace, gradient_norm_l2, position_summary


def test_position_summary_and_trace_are_explicit_json_safe_observations():
    positions = np.array([-1., 0., .5, .5])
    summary = position_summary(positions)
    assert summary["count"] == 4
    assert summary["mean_position"] == 0
    assert summary["mean_abs_position"] == .5
    assert summary["long_fraction"] == .5
    assert summary["short_fraction"] == .25
    assert summary["flat_fraction"] == .25
    trace = LearningTrace()
    phase = trace.phase(np.float64(-.3), positions)
    trace.record(1, train=phase, validation=phase, gradient_norm_pre_clip=2.,
                 improved=True, stale=0, best_epoch=1)
    phase["positions"]["mean_position"] = .9
    output = trace.finish(stop_reason="max_epochs", best_epoch=1)
    assert output["epochs"][0]["train"]["positions"]["mean_position"] == 0
    assert "pre_optimizer_update_train_mode" in output["phase_semantics"]["train"]
    assert "post_optimizer_update_eval_mode" in output["phase_semantics"]["validation"]
    assert json.loads(json.dumps(output, allow_nan=False)) == output


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [True], [".5"], [1j]])
def test_position_summary_rejects_invalid_inputs(values):
    with pytest.raises(ValueError, match="positions"):
        position_summary(values)


def test_trace_rejects_invalid_epoch_checkpoint_phase_and_stop_reason():
    trace = LearningTrace()
    with pytest.raises(ValueError, match="at least one"):
        trace.finish(stop_reason="max_epochs", best_epoch=1)
    phase = trace.phase(0., [0.])
    kwargs = dict(train=phase, validation=phase, gradient_norm_pre_clip=0.,
                  improved=True, stale=0, best_epoch=1)
    for epoch in (True, 0, 2):
        with pytest.raises(ValueError):
            trace.record(epoch, **kwargs)
    with pytest.raises(ValueError, match="staleness"):
        trace.record(1, **{**kwargs, "stale": 1})
    with pytest.raises(ValueError, match="phase"):
        trace.record(1, **{**kwargs, "train": {}})
    with pytest.raises(ValueError, match="finite"):
        trace.phase(float("nan"), [0.])
    trace.record(1, **kwargs)
    with pytest.raises(ValueError, match="consecutive"):
        trace.record(1, **kwargs)
    with pytest.raises(ValueError, match="stop reason"):
        trace.finish(stop_reason="fabricated", best_epoch=1)
    with pytest.raises(ValueError, match="best_epoch"):
        trace.finish(stop_reason="max_epochs", best_epoch=2)


def test_gradient_norm_is_pre_clip_detached_and_read_only():
    torch = pytest.importorskip("torch")
    first = torch.nn.Parameter(torch.tensor([0., 0.]))
    second = torch.nn.Parameter(torch.tensor([0.]))
    unused = torch.nn.Parameter(torch.tensor([0.]))
    first.grad = torch.tensor([3., 4.])
    second.grad = torch.tensor([12.])
    before = [p.grad.clone() for p in (first, second)]
    rng = torch.get_rng_state().clone()
    assert gradient_norm_l2((first, second, unused)) == 13
    assert gradient_norm_l2((unused,)) == 0
    for parameter, original in zip((first, second), before):
        assert torch.equal(parameter.grad, original)
        assert not parameter.grad.requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    torch.nn.utils.clip_grad_norm_((first, second), 1.)
    assert gradient_norm_l2((first, second)) == pytest.approx(1., abs=1e-6)
    first.grad[0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        gradient_norm_l2((first,))


class _TinyDataset:
    tickers = ("A", "B")

    def __init__(self, torch, *, masked=False):
        self.features = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) / 15 - .5
        self.available = torch.ones((6, 2), dtype=torch.bool)
        if masked:
            self.available[::2, 1] = False

    def __len__(self):
        return 6

    def iter_batches(self, batch_dates):
        for start in range(0, len(self), batch_dates):
            end = min(start + batch_dates, len(self))
            yield SimpleNamespace(features=self.features[start:end],
                                  available=self.available[start:end],
                                  row_positions=np.arange(start * 2, end * 2).reshape(-1, 2))


class _ObservedPanel:
    def __init__(self, panel):
        self.panel = panel
        self.observations = []

    def loss_and_gradient(self, positions, loss):
        value, gradient = self.panel.loss_and_gradient(positions, loss)
        self.observations.append((value, positions.copy()))
        return value, gradient


@pytest.mark.parametrize("kind", ["graph", "news"])
@pytest.mark.parametrize("clip", [None, 1e-6])
@pytest.mark.parametrize("stop_reason", ["max_epochs", "early_stopping"])
def test_opt_in_does_not_change_weights_predictions_rng_or_checkpoint(kind, clip, stop_reason):
    torch = pytest.importorskip("torch")
    from trading_system.experiments.graph_ablation import GraphAblationConfig, _fit, _positions
    from trading_system.experiments.news_sentiment_ablation import NewsSentimentAblationConfig, _fit_sentiment, _masked_positions
    from trading_system.models.neural.config import GRUConfig
    from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(.35)
            self.linear = torch.nn.Linear(2, 3)
            self.forwards = 0

        def forward(self, batch):
            self.forwards += 1
            return SimpleNamespace(logits=self.linear(self.dropout(batch.features)), availability=batch.available)

    dataset = _TinyDataset(torch, masked=kind == "news")
    frame = pd.DataFrame({"date": np.repeat(pd.date_range("2020-01-01", periods=6, tz="UTC"), 2),
                          "ticker": ["A", "B"] * 6,
                          "adj_close": [100., 100., 101., 99., 102., 100., 101., 102., 104., 103., 103., 105.]})
    panel = ReturnPanel(frame, group_col="ticker")
    loss = FinancialLossConfig("combined", cost_bps=5.)
    training = GRUConfig(epochs=4, seed=7, learning_rate=.01, gradient_clip_norm=clip,
                         early_stopping_min_delta=100. if stop_reason == "early_stopping" else 0.,
                         early_stopping_patience=2 if stop_reason == "early_stopping" else 10)
    config = SimpleNamespace(resolved_backtest_position_mode=lambda: "long_short")
    ablation_cls = GraphAblationConfig if kind == "graph" else NewsSentimentAblationConfig
    ablation = ablation_cls(date_batch_size=3)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        outcomes = []
        for enabled in (False, True):
            torch.manual_seed(19)
            model = TinyModel()
            train_panel, val_panel = _ObservedPanel(panel), _ObservedPanel(panel)
            run_ablation = replace(ablation, learning_diagnostics=enabled)
            fit = (_fit if kind == "graph" else _fit_sentiment)(
                model, dataset, dataset, train_panel, val_panel, loss, config, training, run_ablation, torch)
            state = {key: value.clone() for key, value in model.state_dict().items()}
            forwards = model.forwards
            rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
            with torch.no_grad():
                predicted = (_positions(model, dataset, mode="tiny", config=config, batch_dates=3, torch=torch)
                             if kind == "graph" else _masked_positions(model, dataset, config, run_ablation, torch))
            outcomes.append((fit, state, forwards, rng, predicted, train_panel, val_panel))
        off, on = outcomes
        assert "learning_trace" not in off[0]
        assert {key: value for key, value in off[0].items() if key != "seconds"} == {
            key: value for key, value in on[0].items() if key not in {"seconds", "learning_trace"}}
        assert off[2] == on[2] == on[0]["epochs_run"] * 6  # two TRAIN passes, one validation pass
        for key in off[1]:
            assert torch.equal(off[1][key], on[1][key])
        assert torch.equal(off[3][0], on[3][0])
        assert off[3][1][0] == on[3][1][0]
        np.testing.assert_array_equal(off[3][1][1], on[3][1][1])
        assert off[3][1][2:] == on[3][1][2:]
        assert off[3][2] == on[3][2]
        np.testing.assert_array_equal(off[4], on[4])
        trace = on[0]["learning_trace"]
        assert trace["stop_reason"] == stop_reason
        assert trace["best_epoch"] == on[0]["best_epoch"]
        assert len(trace["epochs"]) == on[0]["epochs_run"]
        for index, epoch in enumerate(trace["epochs"]):
            for phase, panel_index in (("train", 5), ("validation", 6)):
                observed_loss, observed_positions = on[panel_index].observations[index]
                assert epoch[phase]["loss"] == observed_loss
                assert epoch[phase]["positions"] == position_summary(observed_positions)
            assert epoch["gradient_norm_pre_clip"] > 0
            if clip is not None:
                assert epoch["gradient_norm_pre_clip"] > clip
        assert len(on[5].observations) == len(on[6].observations) == len(trace["epochs"])
        if kind == "news":
            assert all(row["train"]["positions"]["flat_fraction"] >= .25 for row in trace["epochs"])
        if stop_reason == "early_stopping":
            assert [row["improved"] for row in trace["epochs"]] == [True, False, False]
            assert [row["stale"] for row in trace["epochs"]] == [0, 1, 2]
            assert trace["best_epoch"] == 1
        json.dumps(trace, allow_nan=False)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("kind", ["graph", "news"])
def test_config_and_cli_diagnostics_are_opt_in(kind):
    from trading_system.experiments.graph_ablation import GraphAblationConfig
    from trading_system.experiments.news_sentiment_ablation import NewsSentimentAblationConfig
    from trading_system.pipelines.compare_gnn_graphs import build_parser as graph_parser
    from trading_system.pipelines.compare_news_sentiment import build_parser as news_parser
    config = GraphAblationConfig if kind == "graph" else NewsSentimentAblationConfig
    parser = graph_parser() if kind == "graph" else news_parser()
    assert config().learning_diagnostics is False
    for value in (0, 1, None, "yes"):
        with pytest.raises(ValueError, match="learning_diagnostics"):
            config(learning_diagnostics=value)
    arguments = ["--data", "synthetic.parquet"]
    if kind == "news":
        arguments += ["--news-sentiment-export", "synthetic-news.parquet"]
    assert parser.parse_args(arguments).learning_diagnostics is False
    assert parser.parse_args(arguments + ["--learning-diagnostics"]).learning_diagnostics is True


def test_graph_diagnostic_request_is_signed_but_opt_out_keeps_legacy_spec_shape():
    pytest.importorskip("torch")
    from trading_system.experiments.graph_ablation import GraphAblationConfig, plan_run_graph_ablation
    from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
    from trading_system.training.financial_loss import FinancialLossConfig

    days = pd.date_range("2020-01-01", periods=160, freq="B", tz="UTC")
    rows = []
    for asset, ticker in enumerate(("A", "B")):
        for index, day in enumerate(days):
            price = 80 + asset * 20 + index * .1 + 2 * np.sin(index / (5 + asset))
            rows.append({"date": day, "ticker": ticker, "sector": "Finance", "open": price * .999,
                         "high": price * 1.01, "low": price * .99, "close": price,
                         "adj_close": price, "volume": 1000 + index})
    config = replace(DEFAULT_CONFIG, context_len=4, device="cpu", label_mode="forward_return", forward_horizon=3)
    ablation = GraphAblationConfig(graph_lookback=5, candidates=("gru",))
    args = (pd.DataFrame(rows), config, FinancialLossConfig("sharpe"), {"hidden_size": 4, "epochs": 1}, [1])
    plans = [plan_run_graph_ablation(*args, ablation=replace(ablation, learning_diagnostics=enabled),
                                    n_splits=2, gap_bars=2) for enabled in (False, True)]
    assert "learning_diagnostics" not in plans[0]["metadata"]["ablation"]
    assert plans[1]["metadata"]["ablation"]["learning_diagnostics"] is True
    for off, on in zip(plans[0]["task_specs"], plans[1]["task_specs"]):
        assert "learning_diagnostics" not in off["spec"]
        assert on["spec"]["learning_diagnostics"] is True
        assert off["signature"] != on["signature"]
        assert off["spec"] == {key: value for key, value in on["spec"].items() if key != "learning_diagnostics"}
