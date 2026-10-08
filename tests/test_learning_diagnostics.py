"""Offline diagnostics on signed synthetic exports; never fit or load a model."""

from copy import deepcopy
from dataclasses import dataclass
import builtins
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.analysis import learning_diagnostics as diagnostics
from trading_system.artifacts.multimodal_study import (
    atomic_write_json, completion_manifest, file_record, sha256_file,
    stable_digest, training_signature,
)
from trading_system.pipelines.diagnose_learning import main as diagnostic_main
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel
from trading_system.training.learning_trace import LearningTrace


@dataclass
class SignedRun:
    root: Path
    metadata: dict
    rows: list[dict]

    def predictions(self, candidate="gru", partition="inner"):
        row = next(row for row in self.rows if row["candidate"] == candidate)
        return self.root / row["prediction_artifacts"][partition]


def _fingerprints(root):
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in root.rglob("*") if path.is_file()}


def _daily(frame, metadata, candidate, partition):
    loss = FinancialLossConfig(**metadata["loss_config"])
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=1)
    net, executed, _, turnover, costs = panel.path(frame.position.to_numpy(), loss)
    buy_hold, *_ = panel.path(np.ones(len(frame)), loss)
    return pd.DataFrame({
        "variant_id": candidate, "fold": 0, "seed": 7, "partition": partition,
        "date": pd.to_datetime(panel.dates, utc=True),
        "net_return": net, "gross_return": (executed * panel.returns).mean(axis=0),
        "cost": costs.mean(axis=0), "turnover": turnover.mean(axis=0),
        "gross_exposure": np.abs(executed).mean(axis=0),
        "net_exposure": executed.mean(axis=0), "buy_hold_return": buy_hold,
        "buy_hold_gross_return": panel.returns.mean(axis=0),
    }), panel.metrics(frame.position.to_numpy(), loss, metadata["config"]["initial_capital"])


def _predictions(dates, candidate, partition, *, news=False, flip=False):
    frame = pd.DataFrame([(day, ticker) for day in dates for ticker in ("A", "B")],
                         columns=["date", "ticker"])
    probabilities = np.tile(np.array([
        [.8, .1, .1], [.1, .1, .8], [.05, .9, .05], [.01, .02, .97],
    ], dtype=np.float32), (len(frame) // 4 + 1, 1))[:len(frame)]
    labels = probabilities.argmax(axis=1)
    if flip:
        probabilities = probabilities[:, ::-1].copy()
    frame["candidate"] = frame["variant_id"] = candidate
    frame["fold"], frame["seed"], frame["partition"] = 0, 7, partition
    frame["backtest_key"] = frame.date.map(lambda day: day.isoformat()) + "|" + frame.ticker
    frame["row_position"] = np.arange(len(frame))
    frame["available"], frame["label_known"] = True, True
    frame["label"] = labels
    frame.loc[1, ["label_known", "label"]] = [False, -1]
    day_number = np.repeat(np.arange(len(dates)), 2)
    frame["adj_close"] = np.where(frame.ticker.eq("A"),
                                  100 + day_number * 1.7 + np.sin(day_number),
                                  80 - day_number * .5 + np.cos(day_number))
    logits = np.log(probabilities)
    if news:
        frame["news_protocol"] = "fnspid-exploratory"
        frame["news_available"] = True
        frame["coverage_status"] = "unknown"
        frame["observation_status"] = "observed"
        frame["availability_kind"] = "publication_plus_delay_assumption"
        frame["export_row_present"] = True
        frame["news_count"] = 1.
        frame["available_at"] = frame.date - pd.Timedelta(hours=1)
        frame["probability_kind"] = "softmax"
        # Missing standalone news is a signed policy fallback, not an actual
        # model prediction. It stays labelled and must not count as Hold.
        frame.loc[0, ["available", "news_available"]] = [False, False]
        frame.loc[0, "observation_status"] = "unobserved"
        frame.loc[0, "news_count"] = 0.
        frame.loc[0, "probability_kind"] = "flat_fallback"
        probabilities[0] = [0., 1., 0.]
        logits[0] = np.nan
    for index, name in enumerate(("sell", "hold", "buy")):
        frame[f"p_{name}"] = probabilities[:, index]
        frame[f"logit_{name}"] = logits[:, index]
    frame["position"] = probabilities[:, 2] - probabilities[:, 0]
    if news:
        frame["prediction"] = probabilities.argmax(axis=1)
    return frame


def _save_rows(run):
    atomic_write_json(run.root / "folds.json", run.rows)


def _resign(run, candidate="gru"):
    """Update fixture records so semantic-invalid bytes reach validation."""
    row = next(row for row in run.rows if row["candidate"] == candidate)
    paths = [record["path"] for record in row["completion_manifest"]["files"]]
    row["completion_manifest"] = completion_manifest(
        row["task_signature"], [file_record(run.root / path, run.root) for path in paths])
    _save_rows(run)


def _rewrite_result(run, update, candidate="gru"):
    row = next(row for row in run.rows if row["candidate"] == candidate)
    path = run.root / row["result_artifact"]
    result = json.loads(path.read_text())
    update(result)
    atomic_write_json(path, result)
    _resign(run, candidate)


def _recorded_trace():
    trace = LearningTrace()
    positions = np.array([-.5, 0., .75])
    for epoch, train_loss, validation_loss, improved, stale, best in (
        (1, -.2, -.1, True, 0, 1), (2, -.4, -.3, True, 0, 2),
        (3, -.5, -.25, False, 1, 2),
    ):
        trace.record(epoch, train=trace.phase(train_loss, positions),
                     validation=trace.phase(validation_loss, positions),
                     gradient_norm_pre_clip=epoch * .5,
                     improved=improved, stale=stale, best_epoch=best)
    return trace.finish(stop_reason="early_stopping", best_epoch=2)


def _signed_run(root, *, news=False, candidates=None):
    root.mkdir(parents=True)
    candidates = candidates or (("sentiment",) if news else ("gru",))
    inner = pd.date_range("2020-01-27", periods=8, freq="B", tz="UTC")
    outer = pd.date_range("2020-02-17", periods=9, freq="B", tz="UTC")
    metadata = {
        "schema_version": 2, "protocol": ("matched_purged_news_sentiment_ablation"
                                              if news else "matched_purged_graph_ablation"),
        "class_names": ["Sell", "Hold", "Buy"], "final_holdout_opened": False,
        "final_split": {"test_start": "2020-03-02T00:00:00+00:00"},
        "cv_folds": [{"fold": 0, "end": outer[-1].isoformat(),
                      "split": {"validation_start": inner[0].isoformat(),
                                "test_start": outer[0].isoformat()}}],
        "n_splits": 1, "seeds": [7], "ablation": {"candidates": list(candidates)},
        "calendar": {"tickers": ["A", "B"],
                     "sessions": list(pd.date_range("2020-01-01", "2020-02-28", freq="B", tz="UTC").map(lambda d: d.isoformat()))},
        "config": {"position_mode": "long_short", "execution_delay": 1, "initial_capital": 1000.},
        "loss_config": {"objective": "combined", "cost_bps": 5., "annualization": 252},
        "export_contract": {"schema_version": 1, "unknown_label": -1,
                            "signal_availability_mask": "available", "label_availability_mask": "label_known"},
    }
    if news:
        metadata.update(news_protocol="fnspid-exploratory", point_in_time=False,
                        historical_availability_verified=False,
                        availability_kind="publication_plus_delay_assumption")
    atomic_write_json(root / "metadata.json", metadata)
    rows = []
    for candidate in candidates:
        stem = f"fold-0-{candidate}-seed-7"
        signature = training_signature({"candidate": candidate, "fold": 0, "seed": 7,
                                        "config": metadata["config"], "features": ["f1", "f2"]})
        checkpoint = root / f"{stem}.pt"
        checkpoint.write_bytes(b"synthetic checkpoint bytes: MUST NOT be deserialized")
        row = {
            "candidate": candidate, "fold": 0, "seed": 7, "status": "ok",
            "task_signature": signature, "preprocessing_signature": stable_digest(["f1", "f2"]),
            "fit": {"best_epoch": 1, "epochs_run": 3, "parameter_count": 12, "seconds": .01},
            "feature_columns": ["f1", "f2"], "model_artifact": checkpoint.name,
            "result_artifact": f"{stem}-result.json", "prediction_artifacts": {},
            "daily_path_artifacts": {}, "graph_artifact": None, "outer_graph_artifact": None,
            "eligible_sessions": {"train": ["2020-01-01T00:00:00+00:00", "2020-01-10T00:00:00+00:00"]},
        }
        for partition, dates in (("inner", inner), ("outer", outer)):
            frame = _predictions(dates, candidate, partition, news=news,
                                 flip=candidate == "identity")
            prediction_path = root / f"{stem}-{partition}-predictions.parquet"
            frame.to_parquet(prediction_path, index=False)
            daily, metrics = _daily(frame, metadata, candidate, partition)
            daily_path = root / f"{stem}-{partition}-daily.parquet"
            daily.to_parquet(daily_path, index=False)
            row["prediction_artifacts"][partition] = prediction_path.name
            row["daily_path_artifacts"][partition] = daily_path.name
            row[f"{partition}_metrics"] = metrics
            row["eligible_sessions"][partition] = list(dates.map(lambda d: d.isoformat()))
        row["score"] = row["outer_metrics"]["regularized_sharpe"]
        if news:
            row.update(effective_temporal_columns=["f1", "f2"], news_control={"mode": "original"})
        atomic_write_json(root / row["result_artifact"], row)
        paths = [checkpoint, root / row["result_artifact"],
                 *(root / value for value in row["prediction_artifacts"].values()),
                 *(root / value for value in row["daily_path_artifacts"].values())]
        row["completion_manifest"] = completion_manifest(signature, [file_record(path, root) for path in paths])
        rows.append(row)
    run = SignedRun(root, metadata, rows)
    _save_rows(run)
    return run


@pytest.fixture
def graph_run(tmp_path):
    return _signed_run(tmp_path / "graph")


@pytest.fixture
def news_run(tmp_path):
    return _signed_run(tmp_path / "news", news=True)


def test_graph_report_default_inner_exact_statistics_and_pnl(graph_run, tmp_path):
    before = _fingerprints(graph_run.root)
    target = tmp_path / "diagnostic"
    report = diagnostics.diagnose_learning(graph_run.root, target)
    assert report["complete"] and report["task_partitions"] == 1
    assert report["partitions"] == ["inner"]
    assert not report["training_performed"] and not report["checkpoint_loaded"]
    assert not report["final_holdout_opened"] and report["selected"] is None
    task = report["tasks"][0]
    signals = task["signals"]
    assert (signals["rows"], signals["available_rows"], signals["known_labels"], signals["unknown_labels"]) == (16, 16, 15, 1)
    assert signals["classification"]["acc"] == 1.
    assert signals["long_fraction"] == .5
    assert signals["short_fraction"] == .25
    assert signals["flat_fraction"] == .25
    assert signals["saturated_fraction"] == .25
    assert signals["mean_position"] == pytest.approx(.24, abs=1e-6)
    assert signals["mean_abs_position"] == pytest.approx(.59, abs=1e-6)
    assert task["best_epoch_is_first"] and task["epochs_after_best"] == 2
    assert not task["trace"]["trace_available"] and task["trace"]["stop_reason"] is None
    assert "cannot reconstruct" in task["trace"]["trace_limitation"]
    assert report["trace_tasks"] == 0
    assert (target / "epochs.csv").read_text() == "no_records\n"
    assert "Missing epoch traces are unavailable, never fabricated" in (target / "report.md").read_text()
    assert json.loads((target / "report.json").read_text()) == report
    assert _fingerprints(graph_run.root) == before

    tickers = pd.read_csv(target / "tickers.csv")
    assert set(tickers.ticker) == {"A", "B"}
    assert tickers.pnl_contribution.sum() == pytest.approx(task["metrics"]["net_pnl"], abs=1e-9)
    frame = pd.read_parquet(graph_run.predictions())
    loss = FinancialLossConfig(**graph_run.metadata["loss_config"])
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=1)
    net, executed, _, _, costs = panel.path(frame.position.to_numpy(), loss)
    capital_before = 1000. * np.r_[1., np.cumprod(1 + net)[:-1]]
    expected = ((executed * panel.returns - costs) * capital_before / 2).sum(axis=1)
    assert tickers.set_index("ticker").loc[["A", "B"], "pnl_contribution"].to_numpy() == pytest.approx(expected)
    assert task["metrics"]["net_pnl"] == pytest.approx(graph_run.rows[0]["inner_metrics"]["net_pnl"])
    monthly = pd.read_csv(target / "monthly.csv")
    assert len(monthly) == 2
    assert np.prod(1 + monthly.net_return) - 1 == pytest.approx(task["metrics"]["net_return"])
    assert monthly.cost_return_sum.sum() == pytest.approx(task["metrics"]["cost_return_sum"])


def test_news_fallback_and_unknown_labels_are_not_scored_as_hold(news_run, tmp_path):
    report = diagnostics.diagnose_learning(news_run.root, tmp_path / "diagnostic")
    signals = report["tasks"][0]["signals"]
    assert (signals["available_rows"], signals["unavailable_rows"], signals["known_labels"], signals["unknown_labels"]) == (15, 1, 15, 1)
    assert signals["coverage"] == 15 / 16
    assert signals["classification"]["acc"] == 1.
    assert signals["flat_fraction"] == pytest.approx(4 / 15)
    assert signals["long_fraction"] == pytest.approx(8 / 15)
    frame = pd.read_parquet(news_run.predictions("sentiment"))
    usable = frame.available & frame.label_known
    probabilities = frame.loc[usable, ["p_sell", "p_hold", "p_buy"]].to_numpy()
    labels = frame.loc[usable, "label"].to_numpy()
    assert signals["classification"]["nll"] == pytest.approx(-np.log(probabilities[np.arange(14), labels]).mean())
    assert report["tasks"][0]["metrics"]["net_pnl"] == pytest.approx(news_run.rows[0]["inner_metrics"]["net_pnl"])


def test_exploratory_news_context_and_warnings_are_preserved(news_run, tmp_path):
    news_run.metadata["warnings"] = ["Publication-delay availability is assumed, not verified PIT."]
    news_run.metadata["survivor_bias_warning"] = "Current-universe survivorship remains."
    atomic_write_json(news_run.root / "metadata.json", news_run.metadata)
    report = diagnostics.diagnose_learning(news_run.root, tmp_path / "diagnostic")
    context = report["sources"][0]["source_context"]
    assert context["news_protocol"] == "fnspid-exploratory"
    assert context["point_in_time"] is False
    assert context["historical_availability_verified"] is False
    assert context["availability_kind"] == "publication_plus_delay_assumption"
    assert context["warnings"] == news_run.metadata["warnings"]
    assert context["survivor_bias_warning"] == news_run.metadata["survivor_bias_warning"]


def test_recorded_trace_exports_observed_losses_and_gradients(graph_run, tmp_path):
    trace = _recorded_trace()
    _rewrite_result(graph_run, lambda row: row["fit"].update(best_epoch=2, learning_trace=trace))
    target = tmp_path / "diagnostic"
    report = diagnostics.diagnose_learning(graph_run.root, target)
    assert report["trace_tasks"] == 1
    recorded = report["tasks"][0]["trace"]
    assert recorded["trace_available"] is True
    assert recorded["stop_reason"] == "early_stopping"
    assert recorded["train_loss_at_best"] == -.4
    assert recorded["validation_loss_at_best"] == -.3
    assert recorded["phase_semantics"] == trace["phase_semantics"]
    epochs = pd.read_csv(target / "epochs.csv")
    assert epochs.epoch.tolist() == [1, 2, 3]
    assert epochs.train_loss.tolist() == [-.2, -.4, -.5]
    assert epochs.validation_loss.tolist() == [-.1, -.3, -.25]
    assert epochs.gradient_norm_pre_clip.tolist() == [.5, 1., 1.5]
    assert epochs.best_epoch.tolist() == [1, 2, 2]
    assert len(report["summary"]) == 1
    assert report["summary"][0]["trace_available_fraction"] == 1.


@pytest.mark.parametrize("damage", ["schema", "nonconsecutive", "epoch_count", "best_epoch", "negative_gradient", "nonfinite_loss", "bad_positions", "bad_stop_reason"])
def test_malformed_signed_learning_trace_rejected(graph_run, tmp_path, damage):
    trace = _recorded_trace()
    if damage == "schema":
        trace["schema_version"] = 2
    elif damage == "nonconsecutive":
        trace["epochs"][1]["epoch"] = 9
    elif damage == "epoch_count":
        trace["epochs"].pop()
    elif damage == "best_epoch":
        trace["best_epoch"] = 1
    elif damage == "negative_gradient":
        trace["epochs"][0]["gradient_norm_pre_clip"] = -1.
    elif damage == "nonfinite_loss":
        # Null also tests the strict signed JSON representation: no NaN bytes
        # are generated in an otherwise valid completion manifest.
        trace["epochs"][0]["train"]["loss"] = None
    elif damage == "bad_positions":
        trace["epochs"][0]["train"]["positions"]["long_fraction"] = 2.
    elif damage == "bad_stop_reason":
        trace["stop_reason"] = "selected_on_outer"
    _rewrite_result(graph_run, lambda row: row["fit"].update(best_epoch=2, learning_trace=trace))
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists()


def test_all_unavailable_has_no_invented_signal_or_classifier(news_run):
    frame = pd.read_parquet(news_run.predictions("sentiment"))
    frame["available"] = False
    frame["position"] = 0.
    frame[["p_sell", "p_hold", "p_buy"]] = [0., 1., 0.]
    stats = diagnostics.prediction_statistics(frame)
    assert stats["classification"] is None
    assert stats["mean_position"] is None
    assert stats["flat_fraction"] is None
    assert stats["available_rows"] == 0 and stats["unavailable_rows"] == len(frame)


def test_inner_and_outer_remain_distinct(graph_run, tmp_path):
    report = diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic", partitions=("inner", "outer"))
    assert report["task_partitions"] == 2
    tasks = {task["partition"]: task for task in report["tasks"]}
    assert tasks["inner"]["signals"]["rows"] == 16
    assert tasks["outer"]["signals"]["rows"] == 18
    assert tasks["inner"]["metrics"]["net_pnl"] == pytest.approx(graph_run.rows[0]["inner_metrics"]["net_pnl"])
    assert tasks["outer"]["metrics"]["net_pnl"] == pytest.approx(graph_run.rows[0]["outer_metrics"]["net_pnl"])


def test_paired_candidates_and_filtering_are_exact(tmp_path):
    run = _signed_run(tmp_path / "graph", candidates=("gru", "identity"))
    report = diagnostics.diagnose_learning(run.root, tmp_path / "both")
    pair = report["paired_signals"][0]
    assert pair["candidate"] == "identity" and pair["reference"] == "gru"
    assert pair["paired_available_rows"] == 16
    assert pair["mean_abs_position_delta"] == pytest.approx(0.)
    assert pair["mean_position_delta"] == pytest.approx(-.48, abs=1e-6)
    assert pair["sign_disagreement_fraction"] == .75
    assert pair["position_correlation"] == pytest.approx(-1.)
    metrics = {task["candidate"]: task["metrics"] for task in report["tasks"]}
    assert set(pair["metric_deltas"]) == {"net_return", "net_pnl", "net_sharpe", "regularized_sharpe",
                                          "max_drawdown", "mean_abs_position", "mean_position", "turnover", "cost_return_sum"}
    for name, delta in pair["metric_deltas"].items():
        assert delta == pytest.approx(metrics["identity"][name] - metrics["gru"][name])
    filtered = diagnostics.diagnose_learning(run.root, tmp_path / "identity", candidates=("identity",), folds=(0,), seeds=(7,))
    assert [task["candidate"] for task in filtered["tasks"]] == ["identity"]
    assert filtered["paired_signals"] == []


def test_offline_path_never_imports_torch_loads_pickle_or_computes_gradient(graph_run, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline diagnostics tried checkpoint loading or training")

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(ReturnPanel, "loss_and_gradient", forbidden)
    report = diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic")
    assert report["checkpoint_loaded"] is False and report["training_performed"] is False


@pytest.mark.parametrize("partition", [("test",), ("final",), ("inner", "inner"), ()])
def test_final_test_and_bad_partition_requests_fail_without_output(graph_run, tmp_path, partition):
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="partitions|final test|unique partitions"):
        diagnostics.diagnose_learning(graph_run.root, target, partitions=partition)
    assert not target.exists()


def test_opened_final_holdout_rejected(graph_run, tmp_path):
    graph_run.metadata["final_holdout_opened"] = True
    atomic_write_json(graph_run.root / "metadata.json", graph_run.metadata)
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="sealed holdout"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists()


def test_holdout_dates_rejected_even_with_complete_signed_predictions(graph_run, tmp_path):
    graph_run.metadata["final_split"]["test_start"] = "2020-02-03T00:00:00+00:00"
    atomic_write_json(graph_run.root / "metadata.json", graph_run.metadata)
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists()


@pytest.mark.parametrize("mutation", ["unknown_as_hold", "known_as_unknown", "key", "duplicate", "missing_cell", "missing_mask", "bad_probability", "bad_position", "wrong_partition", "variant_id", "duplicate_row_position", "noninteger_row_position", "bad_logits", "missing_logits"])
def test_prediction_contract_rejects_semantic_corruption(graph_run, mutation):
    frame = pd.read_parquet(graph_run.predictions())
    if mutation == "unknown_as_hold":
        frame.loc[1, "label"] = 1
    elif mutation == "known_as_unknown":
        frame.loc[0, "label"] = -1
    elif mutation == "key":
        frame.loc[0, "backtest_key"] = "different"
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_cell":
        frame = frame.drop(index=0)
    elif mutation == "missing_mask":
        frame["available"] = frame.available.astype(object)
        frame.loc[0, "available"] = None
    elif mutation == "bad_probability":
        frame.loc[0, "p_buy"] = np.nan
    elif mutation == "bad_position":
        frame.loc[0, "position"] = .2
    elif mutation == "wrong_partition":
        frame.loc[0, "partition"] = "outer"
    elif mutation == "variant_id":
        frame.loc[0, "variant_id"] = "another_candidate"
    elif mutation == "duplicate_row_position":
        frame.loc[1, "row_position"] = 0
    elif mutation == "noninteger_row_position":
        frame["row_position"] = frame.row_position.astype(float)
    elif mutation == "bad_logits":
        frame.loc[0, "logit_buy"] += 1.
    elif mutation == "missing_logits":
        frame = frame.drop(columns="logit_buy")
    with pytest.raises(ValueError):
        diagnostics.validate_predictions(frame, graph_run.rows[0], graph_run.metadata, "inner")


def test_signed_calendar_mismatch_rejected(graph_run):
    frame = pd.read_parquet(graph_run.predictions())
    row = deepcopy(graph_run.rows[0])
    row["eligible_sessions"]["inner"] = row["eligible_sessions"]["inner"][:-1]
    with pytest.raises(ValueError, match="calendar"):
        diagnostics.validate_predictions(frame, row, graph_run.metadata, "inner")


@pytest.mark.parametrize("mutation", ["inner_lower", "inner_upper", "outer_lower", "outer_upper", "overlap", "duplicate_fold", "no_fold"])
def test_cv_boundaries_and_partition_separation_are_enforced(graph_run, mutation):
    partition = "outer" if mutation.startswith("outer") else "inner"
    frame = pd.read_parquet(graph_run.predictions(partition=partition))
    metadata, row = deepcopy(graph_run.metadata), deepcopy(graph_run.rows[0])
    fold = metadata["cv_folds"][0]
    if mutation == "inner_lower":
        fold["split"]["validation_start"] = "2020-01-28T00:00:00+00:00"
    elif mutation == "inner_upper":
        fold["split"]["test_start"] = "2020-02-03T00:00:00+00:00"
    elif mutation == "outer_lower":
        fold["split"]["test_start"] = "2020-02-18T00:00:00+00:00"
    elif mutation == "outer_upper":
        fold["end"] = "2020-02-25T00:00:00+00:00"
    elif mutation == "overlap":
        row["eligible_sessions"]["outer"][0] = row["eligible_sessions"]["inner"][-1]
    elif mutation == "duplicate_fold":
        metadata["cv_folds"].append(deepcopy(fold))
    elif mutation == "no_fold":
        metadata["cv_folds"] = []
    with pytest.raises(ValueError, match="CV partition|calendars overlap|fold boundaries"):
        diagnostics.validate_predictions(frame, row, metadata, partition)


def test_unavailable_nonflat_cannot_be_invented(news_run):
    frame = pd.read_parquet(news_run.predictions("sentiment"))
    frame.loc[0, "position"] = .2
    with pytest.raises(ValueError, match="FLAT"):
        diagnostics.validate_predictions(frame, news_run.rows[0], news_run.metadata, "inner")


@pytest.mark.parametrize("damage", ["changed_bytes", "missing", "signature"])
def test_integrity_corruption_prevents_output(graph_run, tmp_path, damage):
    path = graph_run.predictions()
    if damage == "changed_bytes":
        contents = path.read_bytes()
        path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    elif damage == "missing":
        path.unlink()
    else:
        graph_run.rows[0]["completion_manifest"]["training_signature"] = "f" * 64
        _save_rows(graph_run)
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists()


def test_editable_fold_metrics_are_not_authoritative(graph_run, tmp_path):
    graph_run.rows[0]["score"] = 999.
    graph_run.rows[0]["inner_metrics"]["net_pnl"] = 999.
    _save_rows(graph_run)
    report = diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic")
    assert report["tasks"][0]["metrics"]["net_pnl"] != 999.


def test_daily_corruption_detected_after_valid_hashes(graph_run, tmp_path):
    path = graph_run.root / graph_run.rows[0]["daily_path_artifacts"]["inner"]
    daily = pd.read_parquet(path)
    daily.loc[2, "cost"] += .01
    daily.to_parquet(path, index=False)
    _resign(graph_run)
    target = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists()


def test_daily_candidate_identity_is_not_merely_a_cosmetic_column(graph_run, tmp_path):
    path = graph_run.root / graph_run.rows[0]["daily_path_artifacts"]["inner"]
    daily = pd.read_parquet(path)
    daily["variant_id"] = "another_candidate"
    daily.to_parquet(path, index=False)
    _resign(graph_run)
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic")


def test_recomputed_metrics_reject_signed_mismatch(graph_run, tmp_path):
    _rewrite_result(graph_run, lambda row: row["inner_metrics"].update(net_pnl=999.))
    with pytest.raises(ValueError, match="No valid selected task"):
        diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic")


def test_partial_rejection_is_reported_without_zero_filled_task(tmp_path):
    run = _signed_run(tmp_path / "graph", candidates=("gru", "identity"))
    run.predictions("identity").unlink()
    report = diagnostics.diagnose_learning(run.root, tmp_path / "diagnostic")
    assert report["complete"] is False
    assert [task["candidate"] for task in report["tasks"]] == ["gru"]
    assert report["paired_signals"] == []
    assert any(issue.get("candidate") == "identity" for issue in report["issues"])


def test_existing_destination_preserved(graph_run, tmp_path):
    target = tmp_path / "diagnostic"
    target.mkdir()
    (target / "keep.txt").write_text("user data")
    before = _fingerprints(target)
    with pytest.raises(FileExistsError, match="new directory"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert _fingerprints(target) == before


def test_source_descendant_destination_rejected_without_mutation(graph_run):
    before = _fingerprints(graph_run.root)
    target = graph_run.root / "diagnostic"
    with pytest.raises(ValueError, match="outside"):
        diagnostics.diagnose_learning(graph_run.root, target)
    assert not target.exists() and _fingerprints(graph_run.root) == before


def test_study_source_descendant_destination_also_rejected(tmp_path):
    study = tmp_path / "study"
    _signed_run(study / "reference")
    before = _fingerprints(study)
    target = study / "diagnostic"
    with pytest.raises(ValueError, match="outside"):
        diagnostics.diagnose_learning(study, target)
    assert not target.exists() and _fingerprints(study) == before


def test_relative_artifact_root_dot_resolves_against_run(graph_run, tmp_path):
    graph_run.rows[0]["artifact_root"] = "."
    _save_rows(graph_run)
    report = diagnostics.diagnose_learning(graph_run.root, tmp_path / "diagnostic")
    assert report["complete"] and report["task_partitions"] == 1


@pytest.mark.parametrize("reference", ["", "/tmp/external", "C:/external", "..\\external", "../external"])
def test_external_artifact_roots_need_portability_and_verified_provenance(graph_run, reference):
    row = {**graph_run.rows[0], "artifact_root": reference}
    with pytest.raises(ValueError, match="portable|verified reuse"):
        diagnostics._immutable(row, graph_run.root)


def test_verified_reuse_resolves_original_signed_task(graph_run, tmp_path):
    target = tmp_path / "reused"
    target.mkdir()
    row = deepcopy(graph_run.rows[0])
    row.update(artifact_root="../graph", reuse={"verified": True, "source_run": "../graph"})
    atomic_write_json(target / "metadata.json", graph_run.metadata)
    atomic_write_json(target / "folds.json", [row])
    before = _fingerprints(graph_run.root)
    report = diagnostics.diagnose_learning(target, tmp_path / "diagnostic")
    assert report["complete"] and report["task_partitions"] == 1
    assert _fingerprints(graph_run.root) == before


def test_cyclic_reuse_provenance_is_refused(graph_run):
    graph_run.rows[0].update(artifact_root="../graph", reuse={"verified": True, "source_run": "."})
    _save_rows(graph_run)
    with pytest.raises(ValueError, match="Cyclic"):
        diagnostics._immutable(graph_run.rows[0], graph_run.root)


def test_missing_training_signature_cannot_skip_identity_validation(graph_run):
    with pytest.raises(ValueError, match="declare its training signature"):
        diagnostics._immutable({**graph_run.rows[0], "task_signature": None}, graph_run.root)


def test_cli_default_is_inner_and_reports_success(graph_run, tmp_path, capsys):
    target = tmp_path / "diagnostic"
    assert diagnostic_main(["--run-dir", str(graph_run.root), "--output-dir", str(target)]) == 0
    captured = capsys.readouterr().out
    assert "tasks=1" in captured and "traces=0" in captured and "final_holdout_opened=False" in captured
    assert json.loads((target / "report.json").read_text())["partitions"] == ["inner"]


def test_cli_partial_failure_returns_one(tmp_path, capsys):
    run = _signed_run(tmp_path / "graph", candidates=("gru", "identity"))
    run.predictions("identity").unlink()
    assert diagnostic_main(["--run-dir", str(run.root), "--output-dir", str(tmp_path / "diagnostic")]) == 1
    assert "complete=False" in capsys.readouterr().out


@pytest.mark.parametrize("requested,already_configured", [(False, False), (True, False), (True, True)])
def test_us_study_cli_propagates_learning_flag_once_without_training(tmp_path, monkeypatch, capsys, requested, already_configured):
    from trading_system.pipelines import us_multimodal_study as cli

    config = {"common_arguments": ["--overfitting-control"], "feature_caps": [32, 64]}
    if already_configured:
        config["common_arguments"].append("--learning-diagnostics")
    captured = {}

    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run CLI must not train or execute a study")

    def plan(configured, stage, output_dir, **options):
        captured["common_arguments"] = list(configured["common_arguments"])
        captured["stage"] = stage
        return {"stage": stage, "graph_choice": options["graph_choice"], "feature_choice": None,
                "counts": {}, "blocked": [], "incompatibilities": [], "runs": [], "feature_audit": {}}

    monkeypatch.setattr(cli, "load_study_config", lambda path: deepcopy(config))
    monkeypatch.setattr(cli, "plan_study", plan)
    monkeypatch.setattr(cli, "execute_study", forbidden)
    monkeypatch.setitem(sys.modules, "trading_system.pipelines.compare_gnn_graphs",
                        SimpleNamespace(prepare_graph_run=forbidden))
    monkeypatch.setitem(sys.modules, "trading_system.experiments.graph_ablation",
                        SimpleNamespace(plan_run_graph_ablation=forbidden, run_graph_ablation=forbidden))
    args = ["--stage", "features", "--output-dir", str(tmp_path / "study"),
            "--graph-choice", "rolling_residual_topk", "--dry-run"]
    if requested:
        args.append("--learning-diagnostics")
    result = cli.main(args)
    assert result["stage"] == captured["stage"] == "features"
    assert captured["common_arguments"].count("--learning-diagnostics") == int(requested or already_configured)
    assert not (tmp_path / "study").exists()
    assert '"stage": "features"' in capsys.readouterr().out
