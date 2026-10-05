"""P0 portable grid planning, immutable reuse, resume and paired reporting."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.artifacts.multimodal_study import (
    atomic_write_json, completion_manifest, file_record, stable_digest,
)
from trading_system.experiments.multimodal_study import (
    compare_study, execute_study, expand_stage, load_study_config, plan_study,
    verified_task_index,
)


def _config():
    return {"schema_version": 1, "common_arguments": ["--overfitting-control"],
            "feature_caps": [32, 64], "gnn_depths": [1, 2], "market_depths": [1, 2]}


def _prepare(args):
    def value(flag):
        return args[args.index(flag) + 1]
    return {"destination": Path(value("--output-dir")),
            "cap": int(value("--overfitting-max-features")),
            "gnn": int(value("--gnn-layers")),
            "market": int(value("--market-transformer-layers")),
            "candidates": value("--graph-candidates").split(",")}


def _planner(*, cap, gnn, market, candidates):
    tasks = []
    for candidate in candidates:
        for fold in range(3):
            for seed in (1, 7, 19):
                effective = {"candidate": candidate, "cap": cap, "fold": fold, "seed": seed,
                             "gnn": gnn if not candidate.startswith("gru") else None,
                             "market": market if candidate.endswith("_market") else None}
                tasks.append({"candidate": candidate, "fold": fold, "seed": seed,
                              "signature": stable_digest(effective), "spec": effective})
    return {"metadata": {"schema_version": 2}, "task_specs": tasks}


def _plan(tmp_path, stage="features", **kwargs):
    return plan_study(_config(), stage, tmp_path / "study", graph_choice="rolling_topk",
                      prepare=_prepare, planner=_planner, **kwargs)


def _source(path, tasks, *, cap=32, score=1.0):
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "metadata.json", {"schema_version": 2})
    rows = []
    for index, task in enumerate(tasks):
        checkpoint = f"checkpoint-{index}.pt"
        predictions = {partition: f"prediction-{index}-{partition}.parquet" for partition in ("inner", "outer")}
        paths = {partition: f"path-{index}-{partition}.parquet" for partition in ("inner", "outer")}
        for name in (checkpoint, *predictions.values(), *paths.values()):
            (path / name).write_bytes(b"synthetic artifact for manifest validation")
        row = {"candidate": task["candidate"], "fold": task["fold"], "seed": task["seed"],
               "score": score, "outer_metrics": {"net_return": .1, "max_drawdown": -.2,
                                                    "mean_abs_position": .2, "turnover": 3.0,
                                                    "cost_return_sum": .002},
               "fit": {"seconds": .01, "parameter_count": 100},
               "status": "ok", "task_signature": task["signature"],
               "feature_columns": [f"feature-{item}" for item in range(cap)],
               "model_artifact": checkpoint, "prediction_artifacts": predictions,
               "daily_path_artifacts": paths, "result_artifact": f"result-{index}.json"}
        atomic_write_json(path / row["result_artifact"], row)
        row["completion_manifest"] = completion_manifest(task["signature"], [
            file_record(name, path) for name in (checkpoint, *predictions.values(),
                                                *paths.values(), row["result_artifact"])])
        rows.append(row)
    atomic_write_json(path / "folds.json", rows)
    return rows


def test_expansion_matches_predeclared_small_grids():
    for stage, caps, depths, candidates in (
        ("features", [32, 64], [1, 1], 6),
        ("gnn-depth", [64, 64], [1, 2], 4),
        ("market-depth", [64, 64], [1, 1], 3),
    ):
        runs, blocked = expand_stage(_config(), stage, graph_choice="sector", feature_choice=64)
        assert not blocked
        assert [run.feature_cap for run in runs] == caps
        assert [run.gnn_layers for run in runs] == depths
        assert all(len(run.candidates) == candidates for run in runs)
        assert all("sector_market" in run.candidates for run in runs)
        assert all(len(set(run.candidates)) == candidates for run in runs)
        assert all(("identity_market" in run.candidates) == (stage == "features") for run in runs)
    runs, _ = expand_stage(_config(), "market-depth", graph_choice="sector", feature_choice=64)
    assert [run.market_layers for run in runs] == [1, 2]


@pytest.mark.parametrize("stage,choice,feature,message", [
    ("features", None, None, "graph-choice"),
    ("gnn-depth", "rolling_topk", None, "feature-choice"),
    ("fusion", "rolling_topk", 64, "P3"),
])
def test_missing_dependencies_block_without_loading_or_training(tmp_path, stage, choice, feature, message):
    def forbidden(*args, **kwargs):
        pytest.fail("Blocked stage must not prepare data or train.")
    plan = plan_study(_config(), stage, tmp_path / "output", graph_choice=choice,
                      feature_choice=feature, prepare=forbidden, planner=forbidden)
    assert message in plan["blocked"][0]
    assert plan["counts"]["new_trainings"] == 0
    with pytest.raises(ValueError, match=message):
        execute_study(plan, prepare=forbidden, runner=forbidden)
    assert not (tmp_path / "output").exists()


def test_dry_plan_counts_signatures_without_storing_big_specs_or_writing(tmp_path):
    plan = _plan(tmp_path)
    assert plan["counts"]["new_trainings"] == 108
    assert plan["counts"]["calibrations"] == 0
    assert all(run["command"][0] == sys.executable for run in plan["runs"])
    assert all("spec" not in task for run in plan["runs"] for task in run["tasks"])
    assert not (tmp_path / "study").exists()
    depth = _plan(tmp_path, "gnn-depth", feature_choice=64)
    assert depth["counts"]["requested_tasks"] == 72
    assert depth["counts"]["new_trainings"] == 63
    assert depth["counts"]["shared_trainings"] == 9
    market = _plan(tmp_path, "market-depth", feature_choice=64)
    assert market["counts"]["new_trainings"] == 45


def test_verified_reference_imports_matching_cap_and_corrupt_row_does_not_hide_valid_rows(tmp_path):
    plan = _plan(tmp_path)
    source = tmp_path / "reference"
    _source(source, plan["runs"][0]["tasks"])
    reused = _plan(tmp_path, reference_runs=[source])
    assert reused["counts"]["reused_trainings"] == 54
    assert reused["counts"]["new_trainings"] == 54
    (source / "checkpoint-0.pt").write_bytes(b"corrupt")
    index, incompatible = verified_task_index([source])
    assert len(index) == 53
    assert plan["runs"][0]["tasks"][0]["signature"] not in index
    assert any("integrity" in item["reason"] for item in incompatible)


def test_reference_without_identity_market_reuses_only_verified_existing_tasks(tmp_path):
    plan = _plan(tmp_path)
    source = tmp_path / "previous-grid"
    tasks = [task for task in plan["runs"][0]["tasks"] if task["candidate"] != "identity_market"]
    _source(source, tasks)
    updated = _plan(tmp_path, reference_runs=[source])
    assert updated["counts"]["requested_tasks"] == 108
    assert updated["counts"]["reused_trainings"] == 45
    assert updated["counts"]["new_trainings"] == 63
    controls = [task for run in updated["runs"] for task in run["tasks"]
                if task["candidate"] == "identity_market"]
    assert len(controls) == 18
    assert all(task["state"] == "new" for task in controls)


def test_legacy_status_alone_is_never_reused(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    atomic_write_json(source / "metadata.json", {})
    atomic_write_json(source / "folds.json", [{"status": "ok", "candidate": "gru"}])
    index, incompatible = verified_task_index([source])
    assert not index
    assert "replay audit" in incompatible[0]["reason"]


def test_interruption_leaves_atomic_resumable_study_and_refuses_protocol_or_signature_change(tmp_path):
    plan = _plan(tmp_path)
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("intentional interruption")
    with pytest.raises(RuntimeError, match="intentional"):
        execute_study(plan, prepare=_prepare, runner=runner)
    state = json.loads((tmp_path / "study" / "study.json").read_text())
    assert state["stages"]["features"]["status"] == "interrupted"
    assert not calls[0]["resume"] and callable(calls[0]["progress_callback"])
    changed = _plan(tmp_path)
    changed["graph_choice"] = "sector"
    with pytest.raises(ValueError, match="choices do not match"):
        execute_study(changed, prepare=_prepare, runner=runner, resume=True)
    changed = _plan(tmp_path)
    changed["runs"][0]["tasks"][0]["signature"] = "0" * 64
    with pytest.raises(ValueError, match="effective data"):
        execute_study(changed, prepare=_prepare, runner=runner, resume=True)


def test_resume_refuses_silent_expansion_of_previous_five_candidate_grid(tmp_path):
    plan = _plan(tmp_path)
    target = tmp_path / "study"
    previous = json.loads(json.dumps(plan))
    for run in previous["runs"]:
        run["tasks"] = [task for task in run["tasks"] if task["candidate"] != "identity_market"]
    # Create a real interrupted state so its identity follows the execution contract.
    def interrupt(**kwargs):
        raise RuntimeError("intentional interruption")
    with pytest.raises(RuntimeError, match="intentional"):
        execute_study(plan, prepare=_prepare, runner=interrupt)
    state = json.loads((target / "study.json").read_text())
    state["stages"]["features"]["plan"] = previous
    atomic_write_json(target / "study.json", state)
    def forbidden(**kwargs):
        pytest.fail("A different grid must not resume training.")
    with pytest.raises(ValueError, match="effective data"):
        execute_study(plan, prepare=_prepare, runner=forbidden, resume=True)


def test_reports_pending_and_empty_without_winning_claims(tmp_path):
    plan = _plan(tmp_path)
    target = tmp_path / "study"
    target.mkdir()
    atomic_write_json(target / "study.json", {"stages": {"features": {"plan": plan}}})
    report = compare_study(target)
    assert len(report["variants"]) == 12
    assert not report["complete"]
    assert all(row["completed"] == 0 and row["mean_score"] is None for row in report["variants"])
    assert not report["paired_deltas"] and report["selected"] is None
    atomic_write_json(target / "study.json", {"stages": {}})
    empty = compare_study(target)
    assert not empty["complete"] and empty["selected"] is None


def test_ctrl_c_records_interruption_and_keeps_resume_protocol(tmp_path):
    plan = _plan(tmp_path)
    def interrupt(**kwargs):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute_study(plan, prepare=_prepare, runner=interrupt)
    state = json.loads((tmp_path / "study" / "study.json").read_text())
    assert state["stages"]["features"]["status"] == "interrupted"
    assert state["stages"]["features"]["error"].startswith("KeyboardInterrupt")
    with pytest.raises(KeyboardInterrupt):
        execute_study(plan, prepare=_prepare, runner=interrupt, resume=True)


def test_paired_report_uses_immutable_metrics_retains_losses_and_marks_missing_pairs(tmp_path):
    plan = _plan(tmp_path)
    target = tmp_path / "study"
    target.mkdir()
    atomic_write_json(target / "study.json", {"stages": {"features": {"plan": plan}}})
    for index, run in enumerate(plan["runs"]):
        rows = _source(Path(run["output_dir"]), run["tasks"], cap=run["feature_cap"], score=1.0 - index)
        rows[0]["score"] = 12345  # Editable folds.json cannot inject winning scores.
        if index == 1:
            rows.pop()
        atomic_write_json(Path(run["output_dir"]) / "folds.json", rows)
    report = compare_study(target)
    feature_pairs = [row for row in report["paired_deltas"] if row["dimension"] == "feature_cap"]
    assert not report["complete"] and report["selected"] is None
    assert len(feature_pairs) == 53
    assert all(row["score_delta"] == -1 for row in feature_pairs)
    assert sum(not row["complete_pair"] for row in feature_pairs) == 8
    assert {row["fold"] for row in feature_pairs} == {0, 1, 2}
    assert {row["seed"] for row in feature_pairs} == {1, 7, 19}
    assert {row["dimension"] for row in report["paired_deltas"]} == {
        "feature_cap", "market_gate", "graph", "graph_market", "branch"}
    gate_controls = [row for row in report["paired_deltas"]
                     if row["dimension"] == "market_gate" and row["baseline"].endswith("/identity")]
    assert len(gate_controls) == 17
    assert all(row["variant"].endswith("/identity_market") for row in gate_controls)
    relational_controls = [row for row in report["paired_deltas"] if row["dimension"] == "graph_market"]
    assert len(relational_controls) == 17
    assert all(row["baseline"].endswith("/identity_market")
               and row["variant"].endswith("/rolling_topk_market") for row in relational_controls)
    assert report["variants"][0]["mean_abs_position"] == .2
    assert report["variants"][0]["mean_parameter_count"] == 100
    assert len(report["paired_aggregates"][0]["by_fold"]) == 3
    assert len(report["paired_aggregates"][0]["by_seed"]) == 3


def test_report_follows_study_local_paths_when_directory_moves(tmp_path):
    plan = _plan(tmp_path)
    target = tmp_path / "study"
    target.mkdir()
    atomic_write_json(target / "study.json", {"stages": {"features": {"plan": plan}}})
    for run in plan["runs"]:
        _source(Path(run["output_dir"]), run["tasks"], cap=run["feature_cap"])
    moved = tmp_path / "moved-study"
    target.rename(moved)
    report = compare_study(moved)
    assert report["complete"] and not report["unavailable"]


def test_resume_progress_counts_local_completion_even_when_reference_is_preferred(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    reference = tmp_path / "reference"
    _source(reference, [task for run in plan["runs"] for task in run["tasks"]])
    initial_values = []
    class Bar:
        def __init__(self, **kwargs):
            initial_values.append(kwargs["initial"])
        def update(self, count):
            pass
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr("tqdm.auto.tqdm", Bar)
    def runner(destination, **kwargs):
        selected = next(run["tasks"] for run in plan["runs"] if Path(run["output_dir"]) == destination)
        return {"folds": _source(destination, selected)}
    execute_study(plan, prepare=_prepare, runner=runner)
    resumed = _plan(tmp_path, reference_runs=[reference])
    assert all(task["reuse_source"] == str(reference) for run in resumed["runs"] for task in run["tasks"])
    execute_study(resumed, prepare=_prepare, runner=runner, resume=True)
    assert initial_values == [0, 108]


def test_report_refuses_signed_tasks_from_different_effective_protocol(tmp_path):
    plan = _plan(tmp_path)
    target = tmp_path / "study"
    target.mkdir()
    atomic_write_json(target / "study.json", {"stages": {"features": {"plan": plan}}})
    for run in plan["runs"]:
        changed = [dict(task, signature="0" * 64) for task in run["tasks"]]
        _source(Path(run["output_dir"]), changed)
    report = compare_study(target)
    assert not report["complete"]
    assert all(row["completed"] == 0 for row in report["variants"])
    assert any("signature" in row["reason"] for row in report["unavailable"])


def test_default_config_frozen_options_and_portable_cli():
    root = Path(__file__).resolve().parents[1]
    config = load_study_config(root / "configs/benchmark/us_multimodal_optimization.json")
    assert config["feature_caps"] == [32, 64]
    assert config["gnn_depths"] == config["market_depths"] == [1, 2]
    assert not any("MT5/" in arg or ".venv/bin" in arg for arg in config["common_arguments"])
    from trading_system.pipelines.us_multimodal_study import build_parser
    args = build_parser().parse_args(["--stage", "features", "--output-dir", "C:/study",
                                      "--data", "C:/prices.parquet", "--dry-run"])
    assert args.dry_run and args.graph_choice is None


def test_invalid_predeclared_grid_rejected(tmp_path):
    config = _config()
    config["gnn_depths"] = [1, 1]
    path = tmp_path / "config.json"
    atomic_write_json(path, config)
    with pytest.raises(ValueError, match="unique"):
        load_study_config(path)


@pytest.mark.parametrize("platform,peak", [("darwin", 128 * 1024 * 1024), ("linux", 128 * 1024)])
def test_memory_helper_normalizes_unix_units_without_initializing_cuda(monkeypatch, platform, peak):
    from trading_system.experiments import multimodal_study as study
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=peak)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: False,
        max_memory_allocated=lambda: pytest.fail("Uninitialized CUDA queried"))))
    assert study._peak_memory_usage() == {"peak_ram_mb": 128, "peak_vram_mb": None}


def test_memory_helper_simulated_windows_structure_and_initialized_cuda(monkeypatch):
    import ctypes
    from trading_system.experiments import multimodal_study as study
    calls = []
    class Function:
        def __init__(self, callback):
            self.callback = callback
        def __call__(self, *args):
            return self.callback(*args)
    def read_info(handle, pointer, size):
        counters = pointer._obj
        assert handle == 123
        assert counters.cb == size == ctypes.sizeof(counters)
        assert [name for name, _ in counters._fields_][:3] == ["cb", "PageFaultCount", "PeakWorkingSetSize"]
        counters.PeakWorkingSetSize = 256 * 1024 * 1024
        calls.append(size)
        return 1
    libraries = {"kernel32": SimpleNamespace(GetCurrentProcess=Function(lambda: 123)),
                 "psapi": SimpleNamespace(GetProcessMemoryInfo=Function(read_info))}
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: libraries[name], raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, max_memory_allocated=lambda: 512 * 1024 * 1024)))
    assert study._peak_memory_usage() == {"peak_ram_mb": 256, "peak_vram_mb": 512}
    assert len(calls) == 1


def test_memory_diagnostics_unavailable_never_fail_training(monkeypatch):
    import ctypes
    from trading_system.experiments import multimodal_study as study
    def unavailable(*args, **kwargs):
        raise OSError("Diagnostic API unavailable")
    monkeypatch.setattr(ctypes, "WinDLL", unavailable, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=unavailable)))
    assert study._peak_memory_usage() == {"peak_ram_mb": None, "peak_vram_mb": None}


@pytest.mark.parametrize("feature_caps,compare_exposure", [([1, 2], False), ([32, 64], True)])
def test_tiny_real_cli_dryrun_execute_resume_and_paired_exports(tmp_path, monkeypatch, capsys, feature_caps, compare_exposure):
    pytest.importorskip("torch")
    from trading_system.data.us_market_context import US_MARKET_CONTEXT_TICKERS
    from trading_system.experiments import graph_ablation as graph
    from trading_system.pipelines.us_multimodal_study import compare_main, main
    dates = pd.date_range("2020-01-01", periods=240, freq="B", tz="UTC")
    records = []
    for number, ticker in enumerate(("A", "B")):
        for index, day in enumerate(dates):
            close = 80 + number * 20 + index * .12 + np.sin(index / (5 + number))
            records.append(dict(date=day, ticker=ticker, sector="Financial", open=close,
                                high=close * 1.01, low=close * .99, close=close,
                                adj_close=close, volume=1000 + index * 10,
                                vix_close=20 + np.sin(index / 7)))
    data_path = tmp_path / "prices.parquet"
    pd.DataFrame(records).to_parquet(data_path, index=False)
    context = pd.DataFrame({"date": dates, **{
        column: 100 + number * 5 + np.arange(len(dates)) * .1 + np.sin(np.arange(len(dates)) / (7 + number))
        for number, column in enumerate(US_MARKET_CONTEXT_TICKERS.values())}})
    context["source_end"] = dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    context_path = tmp_path / "market.parquet"
    context.to_parquet(context_path, index=False)
    selection_path = tmp_path / "tickers.json"
    atomic_write_json(selection_path, {"tickers": ["A", "B"]})
    parameters = tmp_path / "parameters.json"
    atomic_write_json(parameters, {"gru": [{"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1}]})
    config = _config()
    config["feature_caps"] = feature_caps
    config["common_arguments"] = [
        "--data", str(data_path), "--ticker-selection", str(selection_path),
        "--models", "gru", "--model-parameter-sets", str(parameters),
        "--losses", "sharpe", "--context-len", "5", "--feature-set", "expanded",
        "--feature-groups", "technical", "--no-external-features", "--overfitting-control",
        "--market-context-data", str(context_path), "--market-close-columns", "vix_close,spy_close",
        "--graph-lookback", "10", "--gnn-hidden-size", "4", "--market-transformer-width", "4",
        "--market-transformer-heads", "1", "--cv-folds", "2", "--cv-gap-bars", "2",
        "--seeds", "1", "--device", "cpu"]
    config_path = tmp_path / "config.json"
    atomic_write_json(config_path, config)
    output = tmp_path / "study"
    argv = ["--study-config", str(config_path), "--stage", "features", "--graph-choice", "sector",
            "--output-dir", str(output)]
    if compare_exposure:
        argv.append("--compare-exposure")
    original_fit = graph._fit
    monkeypatch.setattr(graph, "_fit", lambda *a, **k: pytest.fail("Dry-run trained a model"))
    dry = main([*argv, "--dry-run"])
    printed = capsys.readouterr().out
    assert dry["counts"]["new_trainings"] == 24
    assert '"task_specs"' not in printed and '"provenance"' not in printed
    assert not output.exists()
    monkeypatch.setattr(graph, "_fit", original_fit)
    result = main(argv)
    assert len(result["reports"]) == 2
    assert all(len(report["folds"]) == 12 for report in result["reports"])
    assert all("market_context__vix_level" in report["metadata"]["market_columns"]
               for report in result["reports"])
    if compare_exposure:
        assert result["comparison"]["complete"]
        assert result["comparison"]["feature_interaction"]["exposure"]["complete"]
        assert len(result["comparison"]["feature_interaction"]["rows"]) == 18
        assert len(result["comparison"]["feature_interaction"]["exposure"]["interaction_rows"]) == 18
        assert {row["branch"] for row in result["comparison"]["feature_interaction"]["rows"]} == {
            "gru", "sector", "identity"}
    monkeypatch.setattr(graph, "_fit", lambda *a, **k: pytest.fail("Resume retrained completed tasks"))
    main([*argv, "--resume"])
    report_dir = output / "reports" / "reanalysis" if compare_exposure else output / "reports"
    report = compare_main(["--study-dir", str(output), "--output-dir", str(report_dir),
                           *(["--exposure-comparison"] if compare_exposure else [])])
    assert report["complete"] and report["selected"] is None
    assert all(row["mean_ece_10"] is not None for row in report["variants"])
    assert all(row["mean_net_pnl"] is not None for row in report["variants"])
    assert (report_dir / "summary.csv").is_file()
    assert (report_dir / "paired_deltas.csv").is_file()
    assert (report_dir / "report.md").is_file()
    if compare_exposure:
        assert (report_dir / "feature_interactions.csv").is_file()
        assert (report_dir / "exposure_summary.csv").is_file()
