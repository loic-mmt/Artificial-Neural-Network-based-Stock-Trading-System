"""No-training fixtures with real signed checkpoints and prediction Parquet."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from trading_system.artifacts.multimodal_study import (
    atomic_write_json, completion_manifest, file_record, stable_digest, training_signature,
)
from trading_system.experiments import feature_gate_interaction as interaction
from trading_system.experiments.graph_ablation import GraphAblationConfig
from trading_system.experiments.exposure_comparison import _metrics, normalize_positions
from trading_system.experiments.multimodal_study import _planned_feature_preprocessing
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


CANDIDATES = ("gru", "gru_market", "rolling_topk", "rolling_topk_market", "identity")


def _write(path, value):
    atomic_write_json(path, value)


def _read(path):
    return json.loads(path.read_text())


def _seal(root, row):
    """Re-sign changed fixture bytes; tests then exercise semantic guards."""
    saved = {key: value for key, value in row.items() if key != "completion_manifest"}
    _write(root / row["result_artifact"], saved)
    refs = [row["model_artifact"], row["result_artifact"],
            *row["prediction_artifacts"].values(), *row["daily_path_artifacts"].values()]
    row["completion_manifest"] = completion_manifest(row["task_signature"],
                                                     [file_record(ref, root) for ref in refs])


def _fixture(tmp_path, *, high_columns=("f1", "f2", "f3"), high_scaler_delta=0., seeds=(1,),
             score_override=None, missing=None):
    study = tmp_path / "study"
    dates = pd.date_range("2020-01-01", periods=18, freq="B", tz="UTC")
    sessions = {"train": [day.isoformat() for day in dates[:5]],
                "inner": [day.isoformat() for day in dates[5:8]],
                "outer": [day.isoformat() for day in dates[8:16]]}
    split = {"validation_start": dates[5].isoformat(), "test_start": dates[8].isoformat(),
             "gap_bars": 0, "embargo_bars": 0}
    loss = FinancialLossConfig("sharpe", cost_bps=17)
    price_rows = [{"date": day, "ticker": ticker,
                   "adj_close": 80 + 20 * number + index * .27 + np.sin(index * .9 + number)}
                  for index, day in enumerate(dates[8:16]) for number, ticker in enumerate(("A", "B"))]
    prices = pd.DataFrame(price_rows)
    panel = ReturnPanel(prices, group_col="ticker", execution_delay=1)
    runs = []
    for cap in (32, 64):
        root = study / "02-features" / f"features-{cap}"
        root.mkdir(parents=True)
        columns = ["f1", "f2"] if cap == 32 else list(high_columns)
        scaler_mean = np.zeros((1, len(columns)))
        if cap == 64:
            scaler_mean[0, 0] = high_scaler_delta
        stock = {"schema_version": 1, "feature_columns": columns, "tickers": ["A", "B"],
                 "fill_values": {name: 0. for name in ("f1", "f2", "f3")},
                 "scaler": {"mean": scaler_mean.tolist(), "scale": np.ones((1, len(columns))).tolist()},
                 "feature_selector": {"input_columns": ["f1", "f2", "f3"]},
                 "overfitting_selector": {
                     "fit_scope": "train_only", "supervised": False, "input_columns": ["f1", "f2", "f3"],
                     "selected_columns": columns, "dropped_columns": [name for name in ("f1", "f2", "f3") if name not in columns],
                     "scores": {"f1": 3., "f2": 2., "f3": 1.}, "variances": {"f1": 1., "f2": 1., "f3": 1.},
                     "config": {"max_features": cap, "min_feature_variance": 1e-10, "max_feature_correlation": .99}},
                 "fracdiff": None, "purging": {"train": 0, "inner": 0}}
        metadata = {"schema_version": 2, "final_holdout_opened": False, "dataset_sha256": "a" * 64,
                    "graph_context_sha256": "b" * 64, "market_context_sha256": "c" * 64,
                    "market_columns": ["m1"], "calendar": {"tickers": ["A", "B"],
                        "sessions": [day.isoformat() for day in dates], "sha256": stable_digest(dates.tolist())},
                    "config": {"overfitting_control": {"max_features": cap}, "initial_capital": 10000.,
                               "execution_delay": 1, "price_col": "adj_close", "date_col": "date", "group_col": "ticker"},
                    "loss_config": asdict(loss), "gru_parameters": {"hidden_size": 4},
                    "cv_spec": {"n_splits": 1, "initial_train_fraction": .5, "inner_val_fraction": .2,
                                "final_test_fraction": .1, "gap_bars": 0, "embargo_bars": 0},
                    "cv_folds": [{"fold": 0, "split": split, "end": dates[15].isoformat()}],
                    "final_split": {"validation_start": dates[8].isoformat(), "test_start": dates[16].isoformat()},
                    "provenance": {"source_sha256": "d" * 64, "runtime": {"packages": {"numpy": "test"}}},
                    "ablation": {**asdict(GraphAblationConfig(candidates=CANDIDATES)), "candidates": list(CANDIDATES)},
                    "n_splits": 1, "seeds": list(seeds)}
        _write(root / "metadata.json", metadata)
        tasks, rows = [], []
        for candidate in CANDIDATES:
            for seed in seeds:
                state = {**deepcopy(stock), "market_columns": ["m1"] if candidate.endswith("_market") else [],
                         "market_scaler": {"mean": [[0.]], "scale": [[1.]]} if candidate.endswith("_market") else None}
                spec = {"schema_version": 2, "candidate": candidate, "fold": 0, "seed": seed,
                        **{key: deepcopy(metadata[key]) for key in ("config", "loss_config", "gru_parameters",
                                                                   "dataset_sha256", "calendar", "cv_spec", "provenance")},
                        "model": {"candidate": candidate, "width": len(columns)},
                        "fold_boundaries": {"split": split, "end": dates[15].isoformat()},
                        "eligible_sessions": {name: sessions[name] for name in ("train", "inner")},
                        "preprocessing": state}
                mode, gated = candidate.removesuffix("_market"), candidate.endswith("_market")
                ablation = metadata["ablation"]
                if mode != "gru":
                    spec["model"].update({key: ablation[key] for key in ("gnn_hidden_size", "gnn_layers", "gnn_dropout")})
                if gated:
                    spec["model"].update({key: ablation[key] for key in (
                        "market_transformer_width", "market_transformer_heads", "market_transformer_layers")})
                    spec["model"]["market_gate_temperature"] = 1.0 if mode == "gru" else ablation["market_gate_temperature"]
                spec["graph"] = None if mode in ("gru", "identity") else {
                    "mode": mode, "lookback": ablation["graph_lookback"], "threshold": ablation["graph_threshold"],
                    "weight_mode": ablation["graph_weight_mode"], "neighbors": ablation["graph_neighbors"],
                    "rebalance_bars": ablation["graph_rebalance_bars"], "context_sha256": None, "sector_context_columns": None}
                spec.update(shared_graph_warmup=ablation["graph_lookback"], date_batch_size=ablation["date_batch_size"],
                            market_context_sha256=metadata["market_context_sha256"] if gated else None)
                signature = training_signature(spec)
                tasks.append({"candidate": candidate, "fold": 0, "seed": seed, "signature": signature, "spec": spec})
                if missing == (cap, candidate, seed):
                    continue
                stem = f"fold0-{candidate}-seed{seed}"
                checkpoint = {"schema_version": 2, "task_signature": signature, "training_spec": spec,
                              "preprocessing": state, "feature_columns": columns, "model_spec": spec["model"],
                              "scaler_mean": scaler_mean, "scaler_scale": np.ones((1, len(columns))),
                              "market_columns": ["m1"], "market_scaler_mean": np.array([[0.]]),
                              "market_scaler_scale": np.array([[1.]]),
                              "mode": candidate, "fold": 0, "seed": seed, "model_state": {}}
                torch.save(checkpoint, root / f"{stem}.pt")
                number = CANDIDATES.index(candidate)
                amplitude = .10 + number * .065 + (cap == 64) * .025
                wave = np.array([1., .7, -.5, 1., .3, -.8, .5, .9])
                position = np.repeat(amplitude * wave, 2) * np.tile([1., -.7], 8)
                financial = panel.metrics(position, loss, 10000.)
                if score_override is not None:
                    score = score_override.get((cap, candidate), 0.) + (seed - 1) * .1
                    financial.update({key: score for key in interaction.METRICS})
                frame = prices.copy()
                frame["candidate"] = frame["variant_id"] = candidate
                frame["fold"], frame["seed"], frame["partition"] = 0, seed, "outer"
                frame["position"], frame["available"] = position, True
                frame["backtest_key"] = frame.date.map(lambda value: value.isoformat()) + "|" + frame.ticker
                frame["label"], frame["label_known"] = 1, True
                frame.to_parquet(root / f"{stem}-outer.parquet", index=False)
                frame.assign(partition="inner").to_parquet(root / f"{stem}-inner.parquet", index=False)
                for partition in ("inner", "outer"):
                    pd.DataFrame({"date": dates[8:16], "net_return": 0.}).to_parquet(root / f"{stem}-{partition}-daily.parquet", index=False)
                row = {"candidate": candidate, "fold": 0, "seed": seed, "status": "ok", "task_signature": signature,
                       "score": financial["regularized_sharpe"], "outer_metrics": financial, "feature_columns": columns,
                       "preprocessing_signature": stable_digest(state), "eligible_sessions": sessions,
                       "model_artifact": f"{stem}.pt", "result_artifact": f"{stem}-result.json",
                       "prediction_artifacts": {part: f"{stem}-{part}.parquet" for part in ("inner", "outer")},
                       "daily_path_artifacts": {part: f"{stem}-{part}-daily.parquet" for part in ("inner", "outer")}}
                _seal(root, row)
                rows.append(row)
        _write(root / "folds.json", rows)
        runs.append({"variant_id": f"features-{cap}", "feature_cap": cap, "metadata": metadata,
                     "tasks": [{key: value for key, value in task.items() if key != "spec"} for task in tasks],
                     "feature_preprocessing": _planned_feature_preprocessing(tasks)})
    _write(study / "study.json", {"schema_version": 1, "final_holdout_opened": False,
                                  "stages": {"features": {"plan": {"stage": "features", "graph_choice": "rolling_topk", "runs": runs}}}})
    return study, prices, loss


def _mutate_prediction(study, transform, *, cap=64, candidate="gru_market"):
    root = study / "02-features" / f"features-{cap}"
    rows = _read(root / "folds.json")
    row = next(row for row in rows if row["candidate"] == candidate)
    path = root / row["prediction_artifacts"]["outer"]
    transform(pd.read_parquet(path)).to_parquet(path, index=False)
    _seal(root, row)
    _write(root / "folds.json", rows)


def _mutate_checkpoint(study, transform, *, candidate="gru_market", resign=False):
    root = study / "02-features" / "features-64"
    rows = _read(root / "folds.json")
    config = _read(study / "study.json")
    run = config["stages"]["features"]["plan"]["runs"][1]
    for row in rows:
        if row["candidate"] != candidate:
            continue
        checkpoint = interaction._load_checkpoint(root / row["model_artifact"])
        transform(checkpoint)
        if resign:
            signature = training_signature(checkpoint["training_spec"])
            checkpoint["task_signature"] = row["task_signature"] = signature
            row["preprocessing_signature"] = stable_digest(checkpoint["preprocessing"])
            task = next(task for task in run["tasks"] if task["candidate"] == candidate and task["seed"] == row["seed"])
            task["signature"] = signature
        torch.save(checkpoint, root / row["model_artifact"])
        _seal(root, row)
    _write(root / "folds.json", rows)
    _write(study / "study.json", config)


def test_signed_score_formula_simple_effects_and_aggregates(tmp_path):
    scores = {(32, "gru"): 1., (64, "gru"): 3., (32, "gru_market"): 0., (64, "gru_market"): 2.5}
    study, _, _ = _fixture(tmp_path, seeds=(1, 2), score_override=scores)
    result = interaction.build_feature_interaction_report(study)
    assert result["complete"], result["unavailable"]
    assert len(result["rows"]) == 4
    gru = next(row for row in result["rows"] if row["branch"] == "gru")
    assert gru["score_interaction"] == .5
    assert gru["gain_high_plain"]["score"] == 2
    assert gru["gate_gain_high"]["score"] == -.5
    assert gru["gain_high_gate"]["score"] == 2.5
    aggregate = next(row for row in result["aggregates"] if row["branch"] == "gru")
    assert aggregate["mean_interaction"]["regularized_sharpe"] == .5
    assert aggregate["mean_gate_gain_high"]["score"] == -.5
    assert aggregate["paired_count"] == 2 and len(aggregate["by_seed"]) == 2
    assert result["selected"] is None and result["final_holdout_opened"] is False
    assert result["feature_audit"][0]["actual_high_count"] == 3  # requested cap need not be reached


def test_exposure_all_ten_plus_one_controls_recompute_costs(tmp_path):
    study, prices, loss = _fixture(tmp_path)
    result = interaction.build_feature_interaction_report(study, exposure_comparison=True)
    assert result["complete"], result["unavailable"]
    assert len(result["exposure"]["rows"]) == 33
    assert len(result["exposure"]["interaction_rows"]) == len(result["rows"]) == 6
    assert len([row for row in result["exposure"]["rows"] if row["candidate"] == "buy_hold"]) == 3
    positions = {}
    for cap in (32, 64):
        root = study / "02-features" / f"features-{cap}"
        for row in _read(root / "folds.json"):
            positions[f"features-{cap}/{row['candidate']}"] = pd.read_parquet(root / row["prediction_artifacts"]["outer"]).position.to_numpy()
    positions["buy_hold"] = np.ones(len(prices))
    panel = ReturnPanel(prices, group_col="ticker", execution_delay=1)
    adjusted, info = normalize_positions(panel, positions, "daily_min")
    for row in result["exposure"]["rows"]:
        if row["method"] == "daily_min":
            expected, _ = _metrics(panel, adjusted[row["variant_id"]], loss, 10000.)
            assert row["metrics"]["cost_return_sum"] == pytest.approx(expected["cost_return_sum"])
            assert row["metrics"]["mean_abs_position"] == pytest.approx(info["target_mean_exposure"])
    assert any("ex post" in note for note in result["notes"])


def test_missing_identity_keeps_raw_pairs_but_blocks_exposure_subset(tmp_path):
    study, _, _ = _fixture(tmp_path, missing=(64, "identity", 1))
    result = interaction.build_feature_interaction_report(study, exposure_comparison=True)
    assert not result["complete"] and len(result["rows"]) == 2
    assert result["exposure"]["rows"] == []
    assert any("all ten" in item["reason"] for item in result["unavailable"])


@pytest.mark.parametrize("change,reason", [
    (lambda frame: frame.assign(adj_close=frame.adj_close + 1), "prices"),
    (lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True), "Duplicate"),
    (lambda frame: frame.assign(backtest_key="bad"), "keys"),
    (lambda frame: frame.drop(columns="partition"), "incomplete"),
    (lambda frame: frame.assign(available=False), "Unavailable"),
    (lambda frame: frame.assign(label=-1), "labels"),
    (lambda frame: frame.assign(ticker=frame.ticker.map({"A": "C", "B": "D"})), "roster"),
    (lambda frame: frame.assign(fold=1), "identity"),
    (lambda frame: frame.assign(position=frame.position * .1), "metric"),
])
def test_semantic_prediction_corruption_never_claims_interaction(tmp_path, change, reason):
    study, _, _ = _fixture(tmp_path)
    _mutate_prediction(study, change)
    result = interaction.build_feature_interaction_report(study, exposure_comparison=True)
    assert not result["complete"] and not result["rows"]
    assert reason.lower() in result["unavailable"][-1]["reason"].lower()


@pytest.mark.parametrize("kwargs", [{"high_columns": ("f1", "f3", "f2")}, {"high_scaler_delta": .02}])
def test_not_prefix_nested_or_common_scaler_drift_blocks_claim(tmp_path, kwargs):
    study, _, _ = _fixture(tmp_path, **kwargs)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "intervention" in result["unavailable"][-1]["reason"]


def test_no_actual_feature_growth_is_explicit_and_not_a_promotion(tmp_path):
    study, _, _ = _fixture(tmp_path, high_columns=("f1", "f2"))
    result = interaction.build_feature_interaction_report(study)
    assert result["complete"], result["unavailable"]
    assert result["feature_audit"][0]["actual_growth"] is False
    assert result["selected"] is None


@pytest.mark.parametrize("field", ["loss_config", "cv_spec", "dataset_sha256"])
def test_metadata_drift_is_strict(tmp_path, field):
    study, _, _ = _fixture(tmp_path)
    root = study / "02-features" / "features-64"
    meta = _read(root / "metadata.json")
    meta[field] = "changed"
    _write(root / "metadata.json", meta)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]


def test_folds_mutable_metric_tampering_is_ignored(tmp_path):
    study, _, _ = _fixture(tmp_path)
    before = interaction.build_feature_interaction_report(study)
    path = study / "02-features" / "features-64" / "folds.json"
    rows = _read(path)
    rows[0]["score"] = rows[0]["outer_metrics"]["regularized_sharpe"] = 999
    _write(path, rows)
    after = interaction.build_feature_interaction_report(study)
    assert after["complete"] and after["rows"] == before["rows"]


def test_unbound_checkpoint_training_spec_is_refused(tmp_path):
    study, _, _ = _fixture(tmp_path)
    root = study / "02-features" / "features-64"
    rows = _read(root / "folds.json")
    row = rows[0]
    checkpoint = interaction._load_checkpoint(root / row["model_artifact"])
    checkpoint["training_spec"]["seed"] = 99
    torch.save(checkpoint, root / row["model_artifact"])
    _seal(root, row)
    _write(root / "folds.json", rows)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "signature" in result["unavailable"][-1]["reason"]


def test_future_outer_sessions_cannot_open_holdout(tmp_path):
    study, _, _ = _fixture(tmp_path)
    root = study / "02-features" / "features-64"
    rows = _read(root / "folds.json")
    row = rows[0]
    row["eligible_sessions"]["outer"][-1] = "2020-02-01T00:00:00+00:00"
    _seal(root, row)
    _write(root / "folds.json", rows)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "calendar" in result["unavailable"][-1]["reason"] or "holdout" in result["unavailable"][-1]["reason"]


def test_missing_study_is_unavailable_not_exception(tmp_path):
    result = interaction.build_feature_interaction_report(tmp_path)
    assert not result["complete"] and result["unavailable"] and not result["rows"]


def test_undefined_ordinary_sharpe_remains_nullable(tmp_path):
    study, _, _ = _fixture(tmp_path)
    for cap in (32, 64):
        root = study / "02-features" / f"features-{cap}"
        rows = _read(root / "folds.json")
        for row in rows:
            row["outer_metrics"]["net_sharpe"] = None
            _seal(root, row)
        _write(root / "folds.json", rows)
    result = interaction.build_feature_interaction_report(study)
    assert result["complete"], result["unavailable"]
    assert all(row["interaction"]["net_sharpe"] is None for row in result["rows"])
    assert all(row["mean_interaction"]["net_sharpe"] is None and row["metric_counts"]["net_sharpe"] == 0
               for row in result["aggregates"])


def test_market_physical_scaler_is_bound_to_signed_preprocessing(tmp_path):
    study, _, _ = _fixture(tmp_path)
    _mutate_checkpoint(study, lambda checkpoint: checkpoint.update(market_scaler_mean=np.array([[9.]])))
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "market scaler" in result["unavailable"][-1]["reason"]


def test_individually_signed_market_drift_across_caps_is_not_an_interaction(tmp_path):
    study, _, _ = _fixture(tmp_path)
    def transform(checkpoint):
        checkpoint["preprocessing"]["market_scaler"]["mean"] = [[.2]]
        checkpoint["training_spec"]["preprocessing"] = checkpoint["preprocessing"]
        checkpoint["market_scaler_mean"] = np.array([[.2]])
    for candidate in ("gru_market", "rolling_topk_market"):
        _mutate_checkpoint(study, transform, candidate=candidate, resign=True)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "frozen market preprocessing" in result["unavailable"][-1]["reason"]


def test_signed_model_width_is_checked_against_actual_columns(tmp_path):
    study, _, _ = _fixture(tmp_path)
    def transform(checkpoint):
        checkpoint["training_spec"]["model"]["width"] = 99
        checkpoint["model_spec"] = checkpoint["training_spec"]["model"]
    _mutate_checkpoint(study, transform, resign=True)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "effective model" in result["unavailable"][-1]["reason"]


def test_missing_finite_return_cannot_be_complete(tmp_path):
    study, _, _ = _fixture(tmp_path)
    root = study / "02-features" / "features-64"
    rows = _read(root / "folds.json")
    rows[0]["outer_metrics"]["net_return"] = None
    _seal(root, rows[0])
    _write(root / "folds.json", rows)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "financial metrics" in result["unavailable"][-1]["reason"]


def test_omitted_declared_seed_cannot_claim_complete(tmp_path):
    study, _, _ = _fixture(tmp_path, seeds=(1, 2))
    config = _read(study / "study.json")
    for run in config["stages"]["features"]["plan"]["runs"]:
        run["tasks"] = [task for task in run["tasks"] if task["seed"] == 1]
    _write(study / "study.json", config)
    result = interaction.build_feature_interaction_report(study)
    assert not result["complete"] and not result["rows"]
    assert "declared CV folds, seeds" in result["unavailable"][-1]["reason"]


def test_prediction_label_availability_must_match_between_caps(tmp_path):
    study, _, _ = _fixture(tmp_path)
    _mutate_prediction(study, lambda frame: frame.assign(label_known=False, label=-1))
    result = interaction.build_feature_interaction_report(study, exposure_comparison=True)
    assert not result["complete"] and not result["rows"]
    assert "aligned label" in result["unavailable"][-1]["reason"]


@pytest.mark.parametrize("payload", [[], None, {"final_holdout_opened": False, "stages": []}])
def test_malformed_studies_are_structured_unavailable(tmp_path, payload):
    _write(tmp_path / "study.json", payload)
    result = interaction.build_feature_interaction_report(tmp_path)
    assert not result["complete"] and result["unavailable"][-1]["kind"] == "incompatible"


def test_prediction_byte_corruption_is_detected_before_inference(tmp_path):
    study, _, _ = _fixture(tmp_path)
    root = study / "02-features" / "features-64"
    row = _read(root / "folds.json")[0]
    (root / row["prediction_artifacts"]["outer"]).write_bytes(b"corrupted")
    result = interaction.build_feature_interaction_report(study, exposure_comparison=True)
    assert not result["complete"] and not result["rows"]
    assert "integrity mismatch" in result["unavailable"][-1]["reason"]
