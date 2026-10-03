"""Small, dependency-aware grids for the sealed US multimodal study.

This module orchestrates existing graph controls.  It does not implement the
later frozen-branch fusion experiment and never selects a graph automatically.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import statistics
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

from trading_system.artifacts.multimodal_study import atomic_write_json, stable_digest


STAGES = ("features", "gnn-depth", "market-depth", "fusion")
GRAPH_CHOICES = ("sector", "rolling_topk", "rolling_residual_topk")
STAGE_DIRECTORIES = {
    "features": "02-features", "gnn-depth": "03-gnn-depth",
    "market-depth": "04-market-depth", "fusion": "05-fusion",
}


def _windows_peak_memory_mb() -> float | None:
    """Read peak working set through the Windows API, without dependencies."""
    try:
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(ProcessMemoryCounters), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int32
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        # Optional diagnostics must never interrupt an otherwise valid fit.
        return None


def _peak_memory_usage() -> dict[str, float | None]:
    result = {"peak_ram_mb": None, "peak_vram_mb": None}
    try:
        if sys.platform == "win32":
            result["peak_ram_mb"] = _windows_peak_memory_mb()
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_ram_mb"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except Exception:
        pass
    # Do not import torch, initialize CUDA or reset its allocator for display.
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return result


@dataclass(frozen=True)
class StudyRun:
    variant_id: str
    stage: str
    feature_cap: int
    gnn_layers: int
    market_layers: int
    candidates: tuple[str, ...]

    def arguments(self, common: Iterable[str], destination: Path) -> list[str]:
        return [*common, "--overfitting-max-features", str(self.feature_cap),
                "--gnn-layers", str(self.gnn_layers),
                "--market-transformer-layers", str(self.market_layers),
                "--graph-candidates", ",".join(self.candidates),
                "--output-dir", str(destination)]


def load_study_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Study configuration must have schema_version=1.")
    arguments = payload.get("common_arguments")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("common_arguments must be a list of CLI strings.")
    forbidden = {"--output-dir", "--resume", "--graph-candidates",
                 "--overfitting-max-features", "--gnn-layers",
                 "--market-transformer-layers", "--final-test", "--cv-final-test"}
    if any(item.split("=", 1)[0] in forbidden for item in arguments):
        raise ValueError("Stage settings, output, resume and final-test flags do not belong in common_arguments.")
    if "--overfitting-control" not in arguments:
        raise ValueError("The feature-cap study requires --overfitting-control.")
    for name, default in (("feature_caps", [32, 64]), ("gnn_depths", [1, 2]),
                          ("market_depths", [1, 2])):
        values = payload.get(name, default)
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)):
            raise ValueError(f"{name} must contain unique positive integers.")
        payload[name] = values
    return payload


def expand_stage(config: dict[str, Any], stage: str, *, graph_choice: str | None,
                 feature_choice: int | None = None) -> tuple[list[StudyRun], list[str]]:
    if stage not in STAGES:
        raise ValueError(f"Unknown study stage: {stage}.")
    if stage == "fusion":
        return [], ["Fusion requires P3 frozen-branch adapters/calibration and is not implemented by P0."]
    if graph_choice is None:
        return [], ["An explicit --graph-choice is required; the unfinished reference run is never auto-selected."]
    if graph_choice not in GRAPH_CHOICES:
        raise ValueError(f"Graph choice must be one of {GRAPH_CHOICES}.")
    graph = graph_choice
    if stage == "features":
        runs = [StudyRun(f"features-{cap}", stage, cap, 1, 1,
                         ("gru", "gru_market", graph, graph + "_market", "identity"))
                for cap in config.get("feature_caps", [32, 64])]
    else:
        if feature_choice is None:
            return [], ["Depth stages require an explicit --feature-choice from the completed feature comparison."]
        if isinstance(feature_choice, bool) or feature_choice not in config.get("feature_caps", [32, 64]):
            raise ValueError("feature_choice must be one of the predeclared feature caps.")
        if stage == "gnn-depth":
            runs = [StudyRun(f"gnn-{depth}-features-{feature_choice}", stage,
                             feature_choice, depth, 1,
                             ("gru", graph, graph + "_market", "identity"))
                    for depth in config.get("gnn_depths", [1, 2])]
        else:
            runs = [StudyRun(f"market-{depth}-features-{feature_choice}", stage,
                             feature_choice, 1, depth,
                             ("gru", "gru_market", graph + "_market"))
                    for depth in config.get("market_depths", [1, 2])]
    return runs, []


def _read_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    path = root / "folds.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if not isinstance(metadata, dict) or not isinstance(rows, list):
        raise ValueError("Run metadata/folds have an invalid schema.")
    return metadata, rows


def verified_task_index(roots: Iterable[str | Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Only signed completed rows with verified files are reusable.

    Legacy checkpoints are diagnostic-only when provenance is missing. A replay
    audit does not invent unknown training code, runtime or split settings.
    """
    index: dict[str, dict[str, Any]] = {}
    incompatibilities = []
    from trading_system.experiments.graph_ablation import _validated_row
    for location in dict.fromkeys(str(Path(item).expanduser().resolve()) for item in roots):
        root = Path(location)
        try:
            metadata, rows = _read_rows(root)
            if metadata.get("schema_version") != 2:
                raise ValueError("Legacy run lacks verified provenance; explicit replay audit required.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            incompatibilities.append({"source": location, "reason": str(error)})
            continue
        for row in rows:
            if row.get("status") != "ok":
                continue
            try:
                signature = row.get("task_signature")
                if not signature:
                    raise ValueError("Completed row lacks a verified signature; replay audit required.")
                immutable = _validated_row(row, root, signature)
                index.setdefault(signature, {"source": location, "row": immutable})
            except (OSError, ValueError, TypeError, KeyError) as error:
                incompatibilities.append({"source": location, "reason": str(error)})
    unique = {(item["source"], item["reason"]): item for item in incompatibilities}
    return index, list(unique.values())


def _planned_feature_preprocessing(tasks):
    """Keep a compact, train-only audit, not all copies of large task specs."""
    records = {}
    for task in tasks:
        spec = task.get("spec", {})
        state = spec.get("preprocessing")
        if state is None:
            continue
        selector = state.get("overfitting_selector") or {}
        sessions = spec.get("eligible_sessions", {})
        record = {
            "fold": task["fold"], "feature_columns": state["feature_columns"],
            "input_columns": selector.get("input_columns"),
            "fill_values": state["fill_values"], "scaler": state["scaler"],
            "selector_config": selector.get("config"),
            "train_only": selector.get("fit_scope") == "train_only" and selector.get("supervised") is False,
            "train_calendar_sha256": stable_digest(sessions.get("train")),
            "inner_calendar_sha256": stable_digest(sessions.get("inner")),
            "tickers": state["tickers"], "fracdiff": state.get("fracdiff"),
            "purging": state.get("purging"),
        }
        old = records.setdefault(task["fold"], record)
        if old != record:
            raise ValueError("Plain and gated tasks must use identical stock preprocessing and calendars.")
    return [records[fold] for fold in sorted(records)]


def audit_feature_grid(runs):
    """Verify an isolated feature-cap intervention before any model is fitted."""
    import numpy as np
    from trading_system.experiments.graph_ablation import _resume_metadata

    result = {"available": False, "complete": False, "folds": [], "warnings": [], "errors": []}
    if not runs or not all(run.get("feature_preprocessing") for run in runs):
        result["warnings"].append("Feature preprocessing audit unavailable from the supplied planner.")
        return result
    result["available"] = True
    ordered = sorted(runs, key=lambda run: run["feature_cap"])
    baseline = ordered[0]

    def protocol(run):
        metadata = deepcopy(_resume_metadata(run["metadata"]))
        metadata["config"]["overfitting_control"].pop("max_features", None)
        return metadata

    base_protocol = protocol(baseline)
    if base_protocol.get("final_holdout_opened") is not False:
        result["errors"].append("Feature study must keep the final holdout sealed.")
    if any(protocol(run) != base_protocol for run in ordered[1:]):
        result["errors"].append("Feature caps must share the exact data, CV calendars, loss, groups and architecture.")
    base_folds = {item["fold"]: item for item in baseline["feature_preprocessing"]}
    for run in ordered[1:]:
        high_folds = {item["fold"]: item for item in run["feature_preprocessing"]}
        if set(base_folds) != set(high_folds):
            result["errors"].append("Feature caps have different fold sets.")
            continue
        for fold, low in sorted(base_folds.items()):
            high = high_folds[fold]
            low_columns, high_columns = low["feature_columns"], high["feature_columns"]
            nested = (bool(low_columns) and len(set(low_columns)) == len(low_columns)
                      and len(set(high_columns)) == len(high_columns)
                      and high_columns[:len(low_columns)] == low_columns)
            shared = all(low[key] == high[key] for key in (
                "input_columns", "train_calendar_sha256", "inner_calendar_sha256", "tickers", "fracdiff", "purging"))
            low_selector, high_selector = deepcopy(low["selector_config"]), deepcopy(high["selector_config"])
            if isinstance(low_selector, dict) and isinstance(high_selector, dict):
                low_selector.pop("max_features", None)
                high_selector.pop("max_features", None)
            shared &= low_selector == high_selector
            train_only = low["train_only"] and high["train_only"]
            scaler_match, fills_match = True, True
            low_stats = {field: np.asarray(low["scaler"][field], dtype=np.float64).reshape(-1)
                         for field in ("mean", "scale")}
            high_stats = {field: np.asarray(high["scaler"][field], dtype=np.float64).reshape(-1)
                          for field in ("mean", "scale")}
            valid_stats = all(len(values) == len(columns) and np.isfinite(values).all()
                              and (field != "scale" or (values > 0).all())
                              for stats, columns in ((low_stats, low_columns), (high_stats, high_columns))
                              for field, values in stats.items())
            scaler_match &= valid_stats
            for name in low_columns:
                if name not in high_columns:
                    scaler_match = fills_match = False
                    break
                left, right = low_columns.index(name), high_columns.index(name)
                if valid_stats:
                    for field in ("mean", "scale"):
                        scaler_match &= bool(np.allclose(low_stats[field][left], high_stats[field][right],
                                                         rtol=1e-6, atol=1e-6, equal_nan=False))
                fills_match &= low["fill_values"].get(name) == high["fill_values"].get(name)
            counts_valid = len(low_columns) <= baseline["feature_cap"] and len(high_columns) <= run["feature_cap"]
            comparable = bool(nested and shared and train_only and scaler_match and fills_match and counts_valid)
            item = {"fold": fold, "low_cap": baseline["feature_cap"], "high_cap": run["feature_cap"],
                    "low_columns": low_columns, "high_columns": high_columns,
                    "actual_low_count": len(low_columns), "actual_high_count": len(high_columns),
                    "prefix_nested": nested, "train_only": bool(train_only),
                    "common_scaler_match": bool(scaler_match), "common_fills_match": bool(fills_match),
                    "same_calendar_and_pool": bool(shared), "comparable": comparable,
                    "train_calendar_sha256": low["train_calendar_sha256"],
                    "inner_calendar_sha256": low["inner_calendar_sha256"],
                    "additional_columns": [name for name in high_columns if name not in low_columns],
                    "actual_growth": len(high_columns) > len(low_columns)}
            result["folds"].append(item)
            if not comparable:
                result["errors"].append(f"Feature cap comparison is not isolated in fold {fold} ({item['low_cap']}/{item['high_cap']}).")
            if len(high_columns) < run["feature_cap"]:
                result["warnings"].append(f"Fold {fold}: cap {run['feature_cap']} retains {len(high_columns)} actual features.")
            if not item["actual_growth"]:
                result["warnings"].append(f"Fold {fold}: no additional features survive; not an effective capacity intervention.")
    result["complete"] = bool(result["folds"]) and not result["errors"]
    return result


def plan_study(config: dict[str, Any], stage: str, destination: str | Path, *,
               graph_choice: str | None = None, feature_choice: int | None = None,
               reference_runs: Iterable[str | Path] = (),
               prepare: Callable, planner: Callable) -> dict[str, Any]:
    target = Path(destination).expanduser().resolve()
    runs, blocked = expand_stage(config, stage, graph_choice=graph_choice,
                                 feature_choice=feature_choice)
    roots = list(reference_runs)
    # Earlier immutable stage runs are safe reuse candidates after file checks.
    if target.is_dir():
        roots.extend(path.parent for path in sorted(target.glob("0*-*/*/metadata.json")))
    available, incompatible = verified_task_index(roots)
    planned, seen = [], set()
    for run in runs:
        output = target / STAGE_DIRECTORIES[stage] / run.variant_id
        args = run.arguments(config["common_arguments"], output)
        inputs = prepare(args)
        plan = planner(**{key: value for key, value in inputs.items()
                          if key not in {"destination", "resume", "reuse_from", "task_filter"}})
        tasks = []
        for item in plan["task_specs"]:
            signature = item["signature"]
            if signature in available:
                state, source = "reused", available[signature]["source"]
            elif signature in seen:
                state, source = "shared", None
            else:
                state, source = "new", None
            seen.add(signature)
            tasks.append({**{key: item[key] for key in ("candidate", "fold", "seed", "signature")},
                          "state": state, "reuse_source": source,
                          "variant_id": f"{run.variant_id}/{item['candidate']}"})
        planned.append({"variant_id": run.variant_id, "output_dir": str(output),
                        "arguments": args, "command": [sys.executable,
                            "scripts/run_gnn_graph_comparison.py", *args],
                        "metadata": plan["metadata"], "tasks": tasks,
                        "feature_preprocessing": _planned_feature_preprocessing(plan["task_specs"]),
                        "feature_cap": run.feature_cap, "gnn_layers": run.gnn_layers,
                        "market_layers": run.market_layers})
    feature_audit = audit_feature_grid(planned) if stage == "features" else None
    if feature_audit:
        blocked.extend(feature_audit["errors"])
    tasks = [task for run in planned for task in run["tasks"]]
    return {"schema_version": 1, "stage": stage, "graph_choice": graph_choice,
            "feature_choice": feature_choice, "study_config": deepcopy(config),
            "output_dir": str(target), "reference_runs": list(dict.fromkeys(
                str(Path(item).expanduser().resolve()) for item in roots)),
            "blocked": blocked, "incompatibilities": incompatible, "runs": planned,
            "feature_audit": feature_audit,
            "counts": {"requested_tasks": len(tasks),
                       "new_trainings": sum(task["state"] == "new" for task in tasks),
                       "reused_trainings": sum(task["state"] == "reused" for task in tasks),
                       "shared_trainings": sum(task["state"] == "shared" for task in tasks),
                       "calibrations": 0, "evaluations": sum(task["state"] == "new" for task in tasks)}}


def execute_study(plan: dict[str, Any], *, prepare: Callable, runner: Callable,
                  resume: bool = False) -> dict[str, Any]:
    if plan["blocked"]:
        raise ValueError(" ".join(plan["blocked"]))
    target = Path(plan["output_dir"])
    study_path = target / "study.json"
    identity = deepcopy({key: plan[key] for key in ("schema_version", "stage", "graph_choice", "feature_choice", "study_config")})
    # Locations are informational. The complete task signature below still
    # binds actual dataset bytes, parameters, calendar, code and runtime.
    location_flags = {"--data", "--market-context-data", "--ticker-selection", "--model-parameter-sets"}
    arguments = identity["study_config"]["common_arguments"]
    for index, argument in enumerate(arguments[:-1]):
        if argument in location_flags:
            arguments[index + 1] = "<content-bound-by-task-signature>"
    if study_path.exists():
        saved = json.loads(study_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"Study exists; use --resume: {target}")
        stages = saved.get("stages", {})
        previous = stages.get(plan["stage"])
        if previous is not None and previous["identity"] != identity:
            raise ValueError("Resume study protocol or explicit choices do not match.")
        if previous is not None:
            old = {run["variant_id"]: [task["signature"] for task in run["tasks"]]
                   for run in previous["plan"]["runs"]}
            current = {run["variant_id"]: [task["signature"] for task in run["tasks"]]
                       for run in plan["runs"]}
            if old != current:
                raise ValueError("Resume study effective data, dates, preprocessing or runtime changed.")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume missing study: {target}")
        stages = {}
    target.mkdir(parents=True, exist_ok=True)
    stages[plan["stage"]] = {"identity": identity, "plan": plan, "status": "pending"}
    atomic_write_json(study_path, {"schema_version": 1, "stages": stages, "final_holdout_opened": False})
    sources = list(plan["reference_runs"])
    results = []
    registry_path = target / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    from tqdm.auto import tqdm
    initial = 0
    for run in plan["runs"]:
        output = Path(run["output_dir"])
        if output.is_dir():
            local, _ = verified_task_index([output])
            initial += sum(task["signature"] in local for task in run["tasks"])
    progress = tqdm(total=plan["counts"]["requested_tasks"], initial=initial,
                    desc=f"US {plan['stage']}", unit="task", disable=not sys.stderr.isatty())

    def completed(row, total):
        progress.update(1)
        progress.set_postfix({key: f"{value:.0f}" if value is not None else "unavailable"
                              for key, value in _peak_memory_usage().items()}, refresh=False)

    try:
        for run in plan["runs"]:
            output = Path(run["output_dir"])
            inputs = prepare(run["arguments"])
            inputs.pop("resume", None)
            report = runner(**inputs, resume=output.is_dir(), reuse_from=sources,
                            progress_callback=completed)
            expected = {(task["candidate"], task["fold"], task["seed"]): task["signature"]
                        for task in run["tasks"]}
            actual = {(row["candidate"], row["fold"], row["seed"]): row.get("task_signature")
                      for row in report["folds"] if row.get("status") == "ok"}
            if actual != expected or len(actual) != len(report["folds"]):
                raise ValueError("Runner returned incomplete or incompatible study tasks.")
            results.append(report)
            sources.append(str(output))
            for row in report["folds"]:
                signature = row.get("task_signature") or row.get("signature")
                if signature:
                    registry[signature] = {"run": str(output), "candidate": row["candidate"],
                                           "fold": row["fold"], "seed": row["seed"],
                                           "reuse": row.get("reuse"),
                                           "completion_manifest": row.get("completion_manifest"),
                                           "artifact_root": row.get("artifact_root", ".")}
            atomic_write_json(registry_path, registry)
    except (Exception, KeyboardInterrupt) as error:
        stages[plan["stage"]].update(status="interrupted", error=f"{type(error).__name__}: {error}")
        atomic_write_json(study_path, {"schema_version": 1, "stages": stages, "final_holdout_opened": False})
        raise
    finally:
        progress.close()
    stages[plan["stage"]]["status"] = "complete"
    atomic_write_json(study_path, {"schema_version": 1, "stages": stages, "final_holdout_opened": False})
    return {"plan": plan, "reports": results}


def compare_study(destination: str | Path, *, exposure_comparison: bool = False) -> dict[str, Any]:
    """Build paired rows without promoting incomplete or empty experiments."""
    target = Path(destination).expanduser().resolve()
    study = json.loads((target / "study.json").read_text(encoding="utf-8"))
    variants = []
    observed: dict[str, dict[tuple[int, int], dict]] = {}
    paired, unavailable = [], []
    from trading_system.experiments.graph_ablation import _validated_row

    def mean(values):
        valid = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
        return statistics.fmean(valid) if valid else None

    for stage, state in study["stages"].items():
        for run in state["plan"]["runs"]:
            # Run layout is study-local; saved absolute locations do not bind
            # reporting to the machine on which the study was first launched.
            output = target / STAGE_DIRECTORIES[stage] / run["variant_id"]
            expected: dict[str, dict[tuple[int, int], str]] = {}
            for task in run["tasks"]:
                expected.setdefault(task["candidate"], {})[task["fold"], task["seed"]] = task["signature"]
            try:
                _, rows = _read_rows(output)
            except (OSError, ValueError) as error:
                rows = []
                unavailable.append({"run": str(output), "reason": str(error)})
            for candidate, keys in expected.items():
                variant = f"{run['variant_id']}/{candidate}"
                values = {}
                for row in rows:
                    if row.get("candidate") != candidate or row.get("status") != "ok":
                        continue
                    key = (row["fold"], row["seed"])
                    if key in values:
                        raise ValueError(f"Duplicate fold/seed in {variant}: {key}.")
                    try:
                        if key not in keys:
                            raise ValueError("Unexpected fold/seed in study output.")
                        row = _validated_row(row, output, keys[key])
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        unavailable.append({"run": str(output), "reason": str(error)})
                        continue
                    if key not in keys or not isinstance(row.get("score"), (int, float)) or not math.isfinite(row["score"]):
                        continue
                    values[key] = row
                scores = [row["score"] for row in values.values()]
                variants.append({"stage": stage, "variant_id": variant, "candidate": candidate,
                                 "feature_cap": run["feature_cap"], "gnn_layers": run["gnn_layers"],
                                 "market_layers": run["market_layers"], "completed": len(values),
                                 "expected": len(keys), "complete": bool(keys) and set(values) == set(keys),
                                 "mean_score": sum(scores) / len(scores) if scores else None,
                                 "std_score": statistics.pstdev(scores) if scores else None,
                                 "min_score": min(scores) if scores else None,
                                 **{name if name.startswith("mean_") else f"mean_{name}": mean(row.get("outer_metrics", {}).get(name)
                                                        for row in values.values())
                                    for name in ("net_pnl", "net_return", "net_sharpe", "max_drawdown", "mean_abs_position", "turnover", "cost_return_sum")},
                                 **{f"mean_{name}": mean(row.get("fit", {}).get(name)
                                                        for row in values.values())
                                    for name in ("seconds", "parameter_count")},
                                 **{f"mean_{name}": mean((row.get("classification") or {}).get(name)
                                                        for row in values.values())
                                    for name in ("macro_f1", "nll", "ece_10")},
                                 "actual_feature_counts": sorted({len(row.get("feature_columns", [])) for row in values.values()})})
                observed[variant] = values
    # Same branch, same fold/seed, one grid dimension at a time.
    for index, left in enumerate(variants):
        for right in variants[index + 1:]:
            dimensions = ("feature_cap", "gnn_layers", "market_layers")
            changed = [name for name in dimensions if left[name] != right[name]]
            if left["stage"] != right["stage"]:
                continue
            if left["candidate"] == right["candidate"] and len(changed) == 1:
                dimension = changed[0]
                baseline, variant = (left, right) if left[dimension] < right[dimension] else (right, left)
            elif not changed and left["variant_id"].split("/")[0] == right["variant_id"].split("/")[0]:
                candidates = {left["candidate"], right["candidate"]}
                plain = next((name for name in candidates if name + "_market" in candidates), None)
                if plain:
                    dimension, base_candidate = "market_gate", plain
                elif "identity" in candidates and not any(name.startswith("gru") or name.endswith("_market") for name in candidates):
                    dimension, base_candidate = "graph", "identity"
                elif "gru" in candidates and not any(name.endswith("_market") for name in candidates):
                    dimension, base_candidate = "branch", "gru"
                elif "gru_market" in candidates and all(name.endswith("_market") for name in candidates):
                    dimension, base_candidate = "branch", "gru_market"
                else:
                    continue
                baseline, variant = (left, right) if left["candidate"] == base_candidate else (right, left)
            else:
                continue
            base_rows, rows = observed[baseline["variant_id"]], observed[variant["variant_id"]]
            for fold, seed in sorted(set(base_rows) & set(rows)):
                base, row = base_rows[fold, seed], rows[fold, seed]
                paired.append({"stage": left["stage"], "dimension": dimension,
                               "baseline": baseline["variant_id"], "variant": variant["variant_id"],
                               "fold": fold, "seed": seed,
                               "score_delta": row["score"] - base["score"],
                               **{f"{name}_delta": row.get("outer_metrics", {}).get(name) - base.get("outer_metrics", {}).get(name)
                                  if isinstance(row.get("outer_metrics", {}).get(name), (int, float))
                                  and isinstance(base.get("outer_metrics", {}).get(name), (int, float)) else None
                                  for name in ("net_return", "max_drawdown", "mean_abs_position", "turnover", "cost_return_sum")},
                               "complete_pair": baseline["complete"] and variant["complete"]})
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in paired:
        groups.setdefault((row["dimension"], row["baseline"], row["variant"]), []).append(row)
    aggregates = []
    for (dimension, baseline, variant), rows in groups.items():
        deltas = [row["score_delta"] for row in rows]
        aggregates.append({"dimension": dimension, "baseline": baseline, "variant": variant,
                           "paired_count": len(rows), "complete": all(row["complete_pair"] for row in rows),
                           "mean_score_delta": statistics.fmean(deltas),
                           "std_score_delta": statistics.pstdev(deltas), "min_score_delta": min(deltas),
                           "by_fold": [{"fold": fold, "mean_score_delta": statistics.fmean(
                               row["score_delta"] for row in rows if row["fold"] == fold)}
                               for fold in sorted({row["fold"] for row in rows})],
                           "by_seed": [{"seed": seed, "mean_score_delta": statistics.fmean(
                               row["score_delta"] for row in rows if row["seed"] == seed)}
                               for seed in sorted({row["seed"] for row in rows})]})
    feature_interaction = None
    if "features" in study["stages"]:
        from trading_system.experiments.feature_gate_interaction import build_feature_interaction_report
        feature_interaction = build_feature_interaction_report(target, exposure_comparison=exposure_comparison)
    elif exposure_comparison:
        raise ValueError("Exposure interaction reporting requires a features stage.")
    return {"schema_version": 1, "variants": variants, "paired_deltas": paired,
            "paired_aggregates": aggregates,
            "feature_interaction": feature_interaction,
            "complete": bool(variants) and all(row["complete"] for row in variants),
            "selected": None, "promotion": "No automatic winner: explicit exploratory decision required.",
            "unavailable": unavailable, "final_holdout_opened": False}


__all__ = ["StudyRun", "STAGES", "GRAPH_CHOICES", "load_study_config", "expand_stage",
           "verified_task_index", "audit_feature_grid", "plan_study", "execute_study", "compare_study"]
