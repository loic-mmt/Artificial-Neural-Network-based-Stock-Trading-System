"""Tiny CPU replays certify complete current inputs but never legacy defaults."""

from dataclasses import replace
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from test_graph_artifacts_p0 import inputs
from trading_system.artifacts.multimodal_study import sha256_file, validate_files
from trading_system.data.scaling import Standardizer
from trading_system.experiments import graph_ablation as graph
from trading_system.experiments import graph_replay as replay_module
from trading_system.experiments.graph_replay import replay_graph_ablation
from trading_system.experiments.market_gru_ablation import build_close_market_frame
from trading_system.pipelines.diagnose_gnn_information import run_information_tests


def feature_order_inputs():
    """Named frames deliberately have a different physical column order."""
    columns = ("beta", "alpha", "gamma")
    frames = {
        name: pd.DataFrame({"gamma": [3., 6.], "alpha": [1., 2.], "beta": [2., 4.],
                            "ticker": ["A", "B"]})
        for name in ("train", "validation", "history_train", "history_validation")
    }
    prepared = SimpleNamespace(
        columns=columns,
        scaler=Standardizer(mean_=np.array([[20., 10., 30.]]),
                            scale_=np.array([[2., 1., 3.]])),
        **frames,
    )
    checkpoint = {"feature_columns": ("gamma", "beta", "alpha"),
                  "scaler_mean": np.array([[30., 20., 10.]]),
                  "scaler_scale": np.array([[3., 2., 1.]])}
    return prepared, checkpoint


def assert_named_frames_unchanged(prepared, frames, snapshots):
    for name, frame in frames.items():
        assert getattr(prepared, name) is frame
        pd.testing.assert_frame_equal(frame, snapshots[name])


@pytest.mark.parametrize("already_ordered", [False, True])
def test_restore_feature_order_only_permutes_tensor_order_and_scaler(already_ordered):
    prepared, checkpoint = feature_order_inputs()
    if already_ordered:
        checkpoint = {"feature_columns": prepared.columns,
                      "scaler_mean": prepared.scaler.mean_.copy(),
                      "scaler_scale": prepared.scaler.scale_.copy()}
    frames = {name: getattr(prepared, name) for name in
              ("train", "validation", "history_train", "history_validation")}
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    old_scaler = prepared.scaler
    original_columns = prepared.columns

    audit = replay_module._restore_feature_order(prepared, checkpoint)

    assert prepared.columns == checkpoint["feature_columns"]
    np.testing.assert_array_equal(prepared.scaler.mean_, checkpoint["scaler_mean"])
    np.testing.assert_array_equal(prepared.scaler.scale_, checkpoint["scaler_scale"])
    assert audit == {"restored": not already_ordered,
                     "reconstructed_columns": list(original_columns),
                     "checkpoint_columns": list(checkpoint["feature_columns"]),
                     "permutation": [0, 1, 2] if already_ordered else [2, 0, 1]}
    assert not np.shares_memory(prepared.scaler.mean_, old_scaler.mean_)
    assert not np.shares_memory(prepared.scaler.scale_, old_scaler.scale_)
    assert not np.shares_memory(prepared.scaler.mean_, checkpoint["scaler_mean"])
    assert not np.shares_memory(prepared.scaler.scale_, checkpoint["scaler_scale"])
    assert_named_frames_unchanged(prepared, frames, snapshots)
    np.testing.assert_array_equal(old_scaler.mean_, [[20., 10., 30.]])
    np.testing.assert_array_equal(old_scaler.scale_, [[2., 1., 3.]])


@pytest.mark.parametrize("original,frozen", [
    (("beta", "alpha", "gamma"), ("gamma", "beta", "changed")),
    (("beta", "alpha", "gamma"), ("beta", "alpha")),
    (("beta", "alpha", "gamma"), ("gamma", "beta", "alpha", "extra")),
    (("beta", "beta", "gamma"), ("gamma", "beta", "alpha")),
    (("beta", "alpha", "gamma"), ("gamma", "beta", "beta")),
    (("beta", "alpha", "gamma"), ()),
    ((), ()),
])
def test_restore_feature_order_refuses_feature_selection_changes_or_duplicates(original, frozen):
    prepared, checkpoint = feature_order_inputs()
    prepared.columns = original
    checkpoint["feature_columns"] = frozen
    scaler = prepared.scaler
    with pytest.raises(ValueError, match="identical named feature set"):
        replay_module._restore_feature_order(prepared, checkpoint)
    assert prepared.columns == original
    assert prepared.scaler is scaler


@pytest.mark.parametrize("owner,suffix,value", [
    ("prepared", "mean", np.array([20., 10., 30.])),
    ("prepared", "scale", np.array([[2., 1.]])),
    ("checkpoint", "mean", np.array([30., 20., 10.])),
    ("checkpoint", "scale", np.array([[3., 2.]])),
    ("prepared", "mean", np.array([[20., np.nan, 30.]])),
    ("checkpoint", "mean", np.array([[30., np.inf, 10.]])),
    ("prepared", "scale", np.array([[2., np.nan, 3.]])),
    ("checkpoint", "scale", np.array([[3., np.inf, 1.]])),
    ("prepared", "scale", np.array([[2., 0., 3.]])),
    ("prepared", "scale", np.array([[2., -1., 3.]])),
    ("checkpoint", "scale", np.array([[3., 0., 1.]])),
    ("checkpoint", "scale", np.array([[3., -2., 1.]])),
])
def test_restore_feature_order_refuses_invalid_scalers_without_partial_mutation(owner, suffix, value):
    prepared, checkpoint = feature_order_inputs()
    if owner == "prepared":
        setattr(prepared.scaler, f"{suffix}_", value)
    else:
        checkpoint[f"scaler_{suffix}"] = value
    original_columns, scaler = prepared.columns, prepared.scaler
    frames = {name: getattr(prepared, name) for name in
              ("train", "validation", "history_train", "history_validation")}
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with pytest.raises(ValueError, match="Invalid scaler state"):
        replay_module._restore_feature_order(prepared, checkpoint)
    assert prepared.columns == original_columns
    assert prepared.scaler is scaler
    assert_named_frames_unchanged(prepared, frames, snapshots)


@pytest.mark.parametrize("suffix", ["mean", "scale"])
def test_restore_feature_order_refuses_changed_named_statistics(suffix):
    prepared, checkpoint = feature_order_inputs()
    # A permutation alone is safe; importing different training statistics is not.
    checkpoint[f"scaler_{suffix}"][0, 1] += .1
    original_columns, scaler = prepared.columns, prepared.scaler
    with pytest.raises(ValueError, match="Named feature scalers differ"):
        replay_module._restore_feature_order(prepared, checkpoint)
    assert prepared.columns == original_columns
    assert prepared.scaler is scaler


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("graph-replay")
    candidates = ("gru", "identity", "rolling_topk", "rolling_residual_topk",
                  "gru_market", "identity_market", "rolling_topk_market", "rolling_residual_topk_market")
    args, options = inputs(root / "source", candidates)
    dates = pd.DatetimeIndex(args[0].date.unique()).sort_values()
    t = np.arange(len(dates))
    context = pd.DataFrame({"date": dates, "spy_close": 300 + .2 * t + np.sin(t / 9),
                            "xlf_close": 100 + .1 * t + np.sin(t / 7)})
    context["source_end"] = dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    market, audit = build_close_market_frame(args[0], "date", ("spy_close",), context_frame=context)
    options = {**options,
               "ablation": replace(options["ablation"], graph_weight_mode="absolute", graph_neighbors=1,
                                    market_transformer_width=8, market_transformer_heads=2),
               "graph_context": context, "sector_context_columns": {"Finance": "xlf_close"},
               "market_frame": market, "market_columns": tuple(audit["features"]), "market_audit": audit}
    report = graph.run_graph_ablation(*args, **options)
    return root, args, options, report, context, market


def test_replay_all_current_controls_inner_outer_and_source_read_only(benchmark):
    root, args, _, report, context, market = benchmark
    before = {path.name: sha256_file(path) for path in args[-1].iterdir() if path.is_file()}
    replay = replay_graph_ablation(args[-1], root / "replay", args[0],
                                   graph_context=context, market_frame=market, device="cpu")
    assert replay["reusable"] is True
    assert replay["final_holdout_opened"] is False
    assert len(replay["tasks"]) == len(report["folds"])
    validate_files(replay["files"], root / "replay")
    predictions = pd.read_parquet(root / "replay" / "predictions.parquet")
    assert set(predictions.candidate) == set(report["metadata"]["ablation"]["candidates"])
    assert set(predictions.partition) == {"inner", "outer"}
    assert not predictions.duplicated(["candidate", "fold", "seed", "partition", "date", "ticker"]).any()
    assert np.allclose(predictions.position, predictions.p_buy - predictions.p_sell)
    logits = predictions[["logit_sell", "logit_hold", "logit_buy"]].to_numpy()
    exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert np.allclose(exponents / exponents.sum(axis=1, keepdims=True),
                       predictions[["p_sell", "p_hold", "p_buy"]].to_numpy(), atol=1e-7)
    daily = pd.read_parquet(root / "replay" / "daily_paths.parquet")
    assert np.allclose(daily.net_return, daily.gross_return - daily.cost)
    assert (daily.date.max() < pd.Timestamp(report["metadata"]["final_split"]["test_start"]))
    assert before == {path.name: sha256_file(path) for path in args[-1].iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        replay_graph_ablation(args[-1], root / "replay", args[0])
    with pytest.raises(ValueError, match="outside"):
        replay_graph_ablation(args[-1], args[-1] / "replay", args[0])


def test_replay_imported_checkpoint_from_verified_source(benchmark, monkeypatch):
    root, args, options, original, context, market = benchmark
    monkeypatch.setattr(graph, "_fit", lambda *a, **k: pytest.fail("Imported checkpoint retrained"))
    imported_path = root / "imported"
    imported = graph.run_graph_ablation(*(*args[:-1], imported_path), **options, reuse_from=[args[-1]])
    assert all(row.get("reuse", {}).get("verified") for row in imported["folds"])
    assert not list(imported_path.glob("*.pt"))
    replay = replay_graph_ablation(imported_path, root / "replay-imported", args[0],
                                   graph_context=context, market_frame=market, device="cpu")
    assert replay["reusable"] is True
    assert {task["task_signature"] for task in replay["tasks"]} == {
        row["task_signature"] for row in original["folds"]}


def test_replay_refuses_changed_data_context_folds_and_source(benchmark):
    root, args, _, _, context, market = benchmark
    changed = args[0].copy()
    changed.loc[0, "adj_close"] += .01
    with pytest.raises(ValueError, match="differ from benchmark"):
        replay_graph_ablation(args[-1], root / "bad-data", changed, graph_context=context, market_frame=market)
    changed_context = context.copy()
    changed_context.loc[0, "spy_close"] += .01
    with pytest.raises(ValueError, match="graph_context"):
        replay_graph_ablation(args[-1], root / "bad-context", args[0],
                              graph_context=changed_context, market_frame=market)
    with pytest.raises(ValueError, match="CV setting differs"):
        replay_graph_ablation(args[-1], root / "bad-cv", args[0], graph_context=context,
                              market_frame=market, inner_val_fraction=.25)
    copy = root / "changed-source"
    shutil.copytree(args[-1], copy)
    report_path = copy / "report.json"
    report = json.loads(report_path.read_text())
    report["metadata"]["provenance"]["source_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source/runtime fingerprint"):
        replay_graph_ablation(copy, root / "bad-source", args[0], graph_context=context, market_frame=market)
    assert not any((root / name).exists() for name in ("bad-data", "bad-context", "bad-cv", "bad-source"))


def test_legacy_requires_explicit_cv_and_stays_non_reusable(benchmark):
    root, args, _, _, context, market = benchmark
    legacy = root / "legacy"
    shutil.copytree(args[-1], legacy)
    report_path = legacy / "report.json"
    report = json.loads(report_path.read_text())
    for key in ("schema_version", "cv_spec", "cv_folds", "calendar", "provenance"):
        report["metadata"].pop(key, None)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Legacy replay requires explicit"):
        replay_graph_ablation(legacy, root / "legacy-missing-cv", args[0],
                              graph_context=context, market_frame=market, candidates=("gru",))
    replay = replay_graph_ablation(legacy, root / "legacy-replay", args[0],
                                   graph_context=context, market_frame=market, candidates=("gru",),
                                   initial_train_fraction=.5, inner_val_fraction=.2, device="cpu")
    assert replay["reusable"] is False
    assert replay["source_provenance"]["verified"] is False
    assert replay["tasks"][0]["metrics"]["inner"] == report["folds"][0]["inner_metrics"]


def test_legacy_order_restoration_is_opt_in_reproduces_metrics_and_stays_non_reusable(
        benchmark, tmp_path, monkeypatch):
    _, args, _, report, context, market = benchmark
    legacy = tmp_path / "legacy-permuted"
    shutil.copytree(args[-1], legacy)
    report_path = legacy / "report.json"
    legacy_report = json.loads(report_path.read_text())
    for key in ("schema_version", "cv_spec", "cv_folds", "calendar", "provenance"):
        legacy_report["metadata"].pop(key, None)
    report_path.write_text(json.dumps(legacy_report))
    before = {path.relative_to(legacy): sha256_file(path)
              for path in legacy.rglob("*") if path.is_file()}
    original_outer_fold = replay_module._outer_fold
    permutations = {}

    def swapped_order(*values):
        prepared, outer, history = original_outer_fold(*values)
        fold_id = values[-1]["fold"]
        if fold_id == 1:
            assert len(prepared.columns) >= 2
            permutation = [1, 0, *range(2, len(prepared.columns))]
            prepared.columns = tuple(prepared.columns[index] for index in permutation)
            prepared.scaler = Standardizer(
                mean_=prepared.scaler.mean_[:, permutation].copy(),
                scale_=prepared.scaler.scale_[:, permutation].copy(),
            )
            permutations[fold_id] = permutation
        return prepared, outer, history

    monkeypatch.setattr(replay_module, "_outer_fold", swapped_order)
    replay_options = dict(graph_context=context, market_frame=market, device="cpu",
                          candidates=("gru", "identity", "gru_market"),
                          initial_train_fraction=.5, inner_val_fraction=.2)
    rejected = tmp_path / "order-not-authorized"
    with pytest.raises(ValueError, match="Checkpoint/preprocessing mismatch.*fold=1"):
        replay_graph_ablation(legacy, rejected, args[0], **replay_options)
    assert not rejected.exists()
    assert not list(tmp_path.glob(f".{rejected.name}-*"))

    output = tmp_path / "order-restored"
    result = replay_graph_ablation(
        legacy, output, args[0], **replay_options, restore_checkpoint_feature_order=True,
    )

    assert result["reusable"] is False
    assert result["source_provenance"]["verified"] is False
    assert result["checkpoint_feature_order_requested"] is True
    assert len(result["checkpoint_feature_order_audits"]) == 2
    audits = {audit["fold"]: audit for audit in result["checkpoint_feature_order_audits"]}
    assert audits[0]["restored"] is False
    assert audits[1]["restored"] is True
    assert audits[1]["permutation"] == permutations[1]
    assert audits[1]["reconstructed_columns"] != audits[1]["checkpoint_columns"]
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in report["folds"]}
    assert len(result["tasks"]) == 6
    for task in result["tasks"]:
        row = saved[task["candidate"], task["fold"], task["seed"]]
        assert task["metrics"]["inner"] == row["inner_metrics"]
        assert task["metrics"]["outer"] == row["outer_metrics"]
    validate_files(result["files"], output)
    assert before == {path.relative_to(legacy): sha256_file(path)
                      for path in legacy.rglob("*") if path.is_file()}


def test_current_schema_refuses_legacy_feature_order_restoration(benchmark, tmp_path):
    _, args, _, _, context, market = benchmark
    output = tmp_path / "current-order-override"
    with pytest.raises(ValueError, match="explicit legacy-only diagnostic"):
        replay_graph_ablation(args[-1], output, args[0], graph_context=context,
                              market_frame=market, device="cpu",
                              restore_checkpoint_feature_order=True)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}-*"))


def test_information_pipeline_accepts_market_topk_and_keeps_inner_logits(benchmark):
    root, args, _, _, context, _ = benchmark
    prices = root / "prices.parquet"
    context_path = root / "context.parquet"
    selection = root / "tickers.json"
    args[0].to_parquet(prices, index=False)
    context.to_parquet(context_path, index=False)
    selection.write_text(json.dumps({"tickers": ["A", "B"]}))
    result = run_information_tests(
        args[-1], root / "information", prices, selection,
        market_context_data=context_path, modes=("rolling_topk_market", "rolling_residual_topk_market"),
        folds=[0], device="cpu", block_length=5, bootstrap_samples=100,
    )
    assert result["reusable"] is True
    assert len(result["statistics"]) == 2
    assert set(pd.read_parquet(root / "information" / "predictions.parquet").partition) == {"inner", "outer"}


def test_replay_checks_immutable_metrics_and_refuses_corrupt_checkpoint_exports(benchmark):
    root, args, _, _, context, market = benchmark
    copied = root / "editable-report"
    shutil.copytree(args[-1], copied)
    path = copied / "report.json"
    report = json.loads(path.read_text())
    report["folds"][0]["inner_metrics"]["net_return"] = 12345
    report["folds"][0]["outer_metrics"]["net_return"] = 12345
    path.write_text(json.dumps(report))
    replay = replay_graph_ablation(copied, root / "immutable-replay", args[0],
                                   graph_context=context, market_frame=market, candidates=("gru",), device="cpu")
    assert replay["tasks"][0]["metrics"]["outer"]["net_return"] != 12345
    broken = root / "corrupt-export"
    shutil.copytree(args[-1], broken)
    row = report["folds"][0]
    (broken / row["prediction_artifacts"]["inner"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity mismatch"):
        replay_graph_ablation(broken, root / "corrupt-replay", args[0], graph_context=context,
                              market_frame=market, candidates=("gru",), device="cpu")
    assert not (root / "corrupt-replay").exists()


def test_explicit_runtime_override_is_exploratory_never_reusable(benchmark, monkeypatch):
    from trading_system.artifacts import multimodal_study

    root, args, _, report, context, market = benchmark
    changed = json.loads(json.dumps(report["metadata"]["provenance"]))
    changed["runtime"]["packages"]["numpy"] = "different-runtime"
    monkeypatch.setattr(multimodal_study, "runtime_provenance", lambda: changed)
    with pytest.raises(ValueError, match="source/runtime fingerprint"):
        replay_graph_ablation(args[-1], root / "unverified-runtime", args[0], graph_context=context,
                              market_frame=market, candidates=("gru",), device="cpu")
    result = replay_graph_ablation(args[-1], root / "exploratory-runtime", args[0], graph_context=context,
                                    market_frame=market, candidates=("gru",), device="cpu",
                                    allow_provenance_mismatch=True)
    assert result["reusable"] is False
    assert result["source_provenance"]["verified"] is False


def test_replay_only_pipeline_accepts_standalone_candidate_without_gru(tmp_path):
    args, options = inputs(tmp_path / "standalone", candidates=("identity",))
    graph.run_graph_ablation(*args, **options)
    data = tmp_path / "prices.parquet"
    selected = tmp_path / "tickers.json"
    args[0].to_parquet(data, index=False)
    selected.write_text(json.dumps({"tickers": ["A", "B"]}))
    result = run_information_tests(args[-1], tmp_path / "standalone-replay", data, selected,
                                   replay_only=True, device="cpu")
    assert result["reusable"] is True
    assert {task["candidate"] for task in result["tasks"]} == {"identity"}
    with pytest.raises(ValueError, match="Information tests require GRU"):
        run_information_tests(args[-1], tmp_path / "invalid-information", data, selected, device="cpu")
