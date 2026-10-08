"""Read-only learning diagnostics from signed multimodal CV exports.

No checkpoint deserialization, inference, training, threshold search or final
test access. Process one task/partition at a time, preserving missing signals.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path, PureWindowsPath
from typing import Sequence

import numpy as np
import pandas as pd

from trading_system.artifacts.multimodal_study import (
    artifact_path, atomic_write_json, atomic_write_text, file_record,
    stable_digest, validate_completion,
)
from trading_system.evaluation.classification import evaluate_predictions
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel
from trading_system.training.learning_trace import LearningTrace


PARTITIONS = ("inner", "outer")
_PROBABILITIES = ("p_sell", "p_hold", "p_buy")
_LOGITS = ("logit_sell", "logit_hold", "logit_buy")
_DAILY = ("net_return", "gross_return", "cost", "turnover", "gross_exposure",
          "net_exposure", "buy_hold_return", "buy_hold_gross_return")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_runs(source: str | Path) -> tuple[Path, ...]:
    """Accept one graph/news run or a parent study; do not scan checkpoints."""
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")
    paths = [root / "folds.json"] if (root / "folds.json").is_file() else sorted(root.rglob("folds.json"))
    runs = tuple(path.parent for path in paths if (path.parent / "metadata.json").is_file())
    if not runs:
        raise ValueError("No metadata.json/folds.json run found; copy the completed artifacts first.")
    return runs


def _identity(row):
    return row["candidate"], int(row["fold"]), int(row["seed"])


def _artifact_root(root, row, seen=()):
    """Resolve portable reuse provenance without importing experiment runners."""
    reference = row.get("artifact_root", ".")
    if (not isinstance(reference, str) or not reference or "\\" in reference
            or Path(reference).is_absolute() or PureWindowsPath(reference).drive):
        raise ValueError("Artifact roots must be portable relative paths.")
    current = Path(root).resolve()
    if current in seen:
        raise ValueError("Cyclic reuse provenance.")
    physical = (current / reference).resolve()
    if reference == ".":
        return physical
    reuse = row.get("reuse", {})
    if not isinstance(reuse, dict) or reuse.get("verified") is not True:
        raise ValueError("External artifact roots require verified reuse provenance.")
    source_reference = reuse.get("source_run")
    if (not isinstance(source_reference, str) or not source_reference or "\\" in source_reference
            or Path(source_reference).is_absolute() or PureWindowsPath(source_reference).drive):
        raise ValueError("Reuse source must be a portable relative run reference.")
    source = (current / source_reference).resolve()
    metadata = _json(source / "metadata.json")
    if metadata.get("schema_version") != 2 or metadata.get("final_holdout_opened") is not False:
        raise ValueError("Reused source must have a sealed schema2 holdout.")
    rows = _json(source / "folds.json")
    matching = [item for item in rows if all(item.get(key) == row.get(key)
                    for key in ("candidate", "fold", "seed", "task_signature"))]
    if len(matching) != 1 or _artifact_root(source, matching[0], (*seen, current)) != physical:
        raise ValueError("Artifact root does not match its declared source task.")
    if matching[0].get("completion_manifest") != row.get("completion_manifest"):
        raise ValueError("Reused completion manifest differs from its source task.")
    return physical


def _immutable(row, root):
    """Same immutable-result contract as training resume, without importing torch."""
    if not isinstance(row.get("task_signature"), str):
        raise ValueError("Task must declare its training signature.")
    physical = _artifact_root(root, row)
    files = validate_completion(row.get("completion_manifest"), physical, row.get("task_signature"))
    result_path = artifact_path(row.get("result_artifact"), physical)
    if result_path not in files:
        raise ValueError("Immutable result is not covered by the completion manifest.")
    result = _json(result_path)
    for key in ("candidate", "fold", "seed", "task_signature"):
        if result.get(key) != row.get(key):
            raise ValueError("Immutable task identity differs from folds.json.")
    if result.get("status") != "ok":
        raise ValueError("Immutable task is not completed successfully.")
    required = [result.get("model_artifact"), result.get("result_artifact")]
    for name in ("prediction_artifacts", "daily_path_artifacts"):
        mapping = result.get(name, {})
        if set(mapping) != set(PARTITIONS):
            raise ValueError(f"Missing inner/outer {name}; legacy exports are diagnostic-incomplete.")
        required.extend(mapping.values())
    if any(artifact_path(name, physical) not in files for name in required):
        raise ValueError("Required artifact is not covered by the completion manifest.")
    return result, physical


def _dates(values):
    result = pd.to_datetime(values, utc=True, errors="raise")
    if pd.isna(result).any():
        raise ValueError("Missing diagnostic date.")
    return result


def validate_predictions(frame, row, metadata, partition):
    required = {"date", "ticker", "backtest_key", "available", "label_known", "label",
                "position", "adj_close", "candidate", "fold", "seed", "partition",
                "row_position", "variant_id", *_PROBABILITIES, *_LOGITS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Prediction export is empty or missing required columns.")
    frame = frame.copy()
    frame["date"] = _dates(frame.date)
    if frame.ticker.isna().any() or not frame.ticker.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Ticker keys must be nonempty strings.")
    if frame.duplicated(["date", "ticker"]).any() or frame.backtest_key.duplicated().any():
        raise ValueError("Duplicate prediction/backtest keys.")
    if (not pd.api.types.is_integer_dtype(frame.row_position) or pd.api.types.is_bool_dtype(frame.row_position)
            or not np.array_equal(np.sort(frame.row_position.to_numpy()), np.arange(len(frame)))):
        raise ValueError("row_position must be a complete permutation of exported rows.")
    expected_keys = frame.date.map(lambda day: day.isoformat()) + "|" + frame.ticker
    if not frame.backtest_key.eq(expected_keys).all():
        raise ValueError("Prediction does not match its date/ticker backtest key.")
    for name, value in (("candidate", row["candidate"]), ("variant_id", row["candidate"]), ("fold", row["fold"]),
                        ("seed", row["seed"]), ("partition", partition)):
        if not frame[name].eq(value).all():
            raise ValueError(f"Prediction {name} differs from task/partition identity.")
    for name in ("available", "label_known"):
        if not pd.api.types.is_bool_dtype(frame[name]) or frame[name].isna().any():
            raise ValueError(f"{name} must be a nonmissing boolean mask.")
    known, available = frame.label_known.to_numpy(), frame.available.to_numpy()
    labels = frame.label.to_numpy(dtype=float)
    if not np.isin(labels[known], (0, 1, 2)).all() or not np.all(labels[~known] == -1):
        raise ValueError("Unknown labels must remain -1, distinct from Hold.")
    probabilities = frame.loc[:, list(_PROBABILITIES)].to_numpy(dtype=float)
    valid = probabilities[available]
    if (not np.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any()
            or not np.allclose(valid.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError("Available probabilities must be finite and normalized.")
    logits = frame.loc[available, list(_LOGITS)].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError("Available logits must be finite.")
    exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
    if not np.allclose(valid, exponents / exponents.sum(axis=1, keepdims=True), atol=1e-6, rtol=0):
        raise ValueError("Probabilities disagree with exported logits.")
    position = frame.position.to_numpy(dtype=float)
    if not np.isfinite(position).all() or (np.abs(position) > 1 + 1e-6).any():
        raise ValueError("Positions must be finite and bounded.")
    mode = metadata.get("config", {}).get("position_mode", "long_short")
    if mode not in ("long_only", "long_short"):
        raise ValueError("Unsupported position decoder.")
    decoded = valid[:, 2] - valid[:, 0] if mode == "long_short" else valid[:, 2]
    if not np.allclose(position[available], decoded, atol=1e-6, rtol=0):
        raise ValueError("Positions differ from the recorded probability decoder.")
    if np.any(position[~available] != 0):
        raise ValueError("Unavailable signals require the runner's explicit FLAT fallback.")
    boundary = metadata.get("final_split", {}).get("test_start")
    if boundary is None or (frame.date >= pd.to_datetime(boundary, utc=True)).any():
        raise ValueError("Missing holdout boundary or predictions entering the final holdout.")
    sessions = pd.DatetimeIndex(frame.date.unique()).sort_values()
    expected = row.get("eligible_sessions", {}).get(partition)
    if expected is None or not sessions.equals(pd.DatetimeIndex(_dates(expected))):
        raise ValueError("Prediction calendar differs from the immutable eligible sessions.")
    folds = [item for item in metadata.get("cv_folds", []) if item.get("fold") == row["fold"]]
    if len(folds) != 1:
        raise ValueError("Missing or ambiguous fold boundaries in metadata.")
    split = folds[0]["split"]
    lower = pd.to_datetime(split["validation_start" if partition == "inner" else "test_start"], utc=True)
    upper = pd.to_datetime(split["test_start"] if partition == "inner" else folds[0]["end"], utc=True)
    if (frame.date < lower).any() or (frame.date >= upper if partition == "inner" else frame.date > upper).any():
        raise ValueError("Prediction dates are outside their declared CV partition.")
    calendars = row["eligible_sessions"]
    if (pd.DatetimeIndex(_dates(calendars["inner"])).max()
            >= pd.DatetimeIndex(_dates(calendars["outer"])).min()):
        raise ValueError("INNER and OUTER calendars overlap or are not chronological.")
    tickers = metadata.get("calendar", {}).get("tickers")
    if (not tickers or len(tickers) != len(set(tickers)) or set(frame.ticker) != set(tickers)
            or len(frame) != len(sessions) * len(tickers)):
        raise ValueError("Incomplete ticker/date panel; missing rows cannot become FLAT.")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def prediction_statistics(frame, *, flat_tolerance=1e-6, saturation_threshold=.95):
    """Signal statistics, not executed exposure or calibrated confidence."""
    available = frame.available.to_numpy(dtype=bool)
    known = frame.label_known.to_numpy(dtype=bool)
    positions = frame.loc[available, "position"].to_numpy(dtype=float)
    probabilities = frame.loc[available, list(_PROBABILITIES)].to_numpy(dtype=float)
    result = {"rows": len(frame), "available_rows": int(available.sum()),
              "unavailable_rows": int((~available).sum()), "known_labels": int(known.sum()),
              "unknown_labels": int((~known).sum()), "coverage": float(available.mean())}
    for name, values in (
        ("long_fraction", positions > flat_tolerance), ("short_fraction", positions < -flat_tolerance),
        ("flat_fraction", np.abs(positions) <= flat_tolerance),
        ("saturated_fraction", np.abs(positions) >= saturation_threshold),
    ):
        result[name] = float(values.mean()) if len(values) else None
    result["mean_position"] = float(positions.mean()) if len(positions) else None
    result["mean_abs_position"] = float(np.abs(positions).mean()) if len(positions) else None
    result["std_position"] = float(positions.std()) if len(positions) else None
    result["max_probability_mean"] = float(probabilities.max(axis=1).mean()) if len(probabilities) else None
    result["probability_margin_mean"] = (float(np.diff(np.sort(probabilities, axis=1)[:, -2:], axis=1).mean())
                                          if len(probabilities) else None)
    for index, name in enumerate(("sell", "hold", "buy")):
        result[f"mean_p_{name}"] = float(probabilities[:, index].mean()) if len(probabilities) else None
        result[f"argmax_{name}_fraction"] = float((probabilities.argmax(axis=1) == index).mean()) if len(probabilities) else None
    usable = available & known
    result["classification"] = None
    if usable.any():
        labels = frame.loc[usable, "label"].to_numpy(dtype=int)
        probs = frame.loc[usable, list(_PROBABILITIES)].to_numpy(dtype=float)
        result["classification"] = evaluate_predictions(labels, probs.argmax(axis=1))
        result["classification"]["nll"] = float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean())
    return result


def _paths(frame, daily, metadata, row, partition):
    """Verify delayed fills/costs and compute additive portfolio PnL attribution."""
    if not {"date", "variant_id", "fold", "seed", "partition", *_DAILY}.issubset(daily) or daily.empty:
        raise ValueError("Daily export is missing required columns.")
    daily = daily.copy()
    daily["date"] = _dates(daily.date)
    daily = daily.sort_values("date").reset_index(drop=True)
    if daily.date.duplicated().any():
        raise ValueError("Duplicate daily dates.")
    for name, value in (("variant_id", row["candidate"]), ("fold", row["fold"]),
                        ("seed", row["seed"]), ("partition", partition)):
        if not daily[name].eq(value).all():
            raise ValueError("Daily task/partition identity mismatch.")
    config = metadata["config"]
    loss = FinancialLossConfig(**metadata["loss_config"])
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=config["execution_delay"])
    net, executed, _, turnover, costs = panel.path(frame.position.to_numpy(), loss)
    buy_hold, *_ = panel.path(np.ones(len(frame)), loss)
    expected = np.column_stack((net, (executed * panel.returns).mean(axis=0), costs.mean(axis=0),
                                turnover.mean(axis=0), np.abs(executed).mean(axis=0),
                                executed.mean(axis=0), buy_hold, panel.returns.mean(axis=0)))
    observed = daily.loc[:, list(_DAILY)].to_numpy(dtype=float)
    if (not np.isfinite(observed).all()
            or not pd.DatetimeIndex(daily.date).equals(pd.DatetimeIndex(_dates(panel.dates)))
            or not np.allclose(observed, expected, rtol=5e-5, atol=1e-8)):
        raise ValueError("Daily returns/costs/exposure disagree with delayed prediction execution.")
    capital = float(config.get("initial_capital", 10000.))
    metrics = panel.metrics(frame.position.to_numpy(), loss, initial_capital=capital)
    metrics["mean_position"] = float(executed.mean())
    metrics["buy_hold"] = panel.metrics(np.ones(len(frame)), loss, initial_capital=capital)
    for name, value in row.get(f"{partition}_metrics", {}).items():
        if name in metrics and isinstance(value, (int, float)):
            if not np.isclose(metrics[name], value, rtol=5e-4, atol=5e-4):
                raise ValueError(f"Recomputed metric mismatch: {name}.")
    # Portfolio capital before each return, not an independently compounded
    # single-ticker backtest. These contributions sum exactly to portfolio PnL.
    wealth_before = capital * np.r_[1., np.cumprod(1 + net)[:-1]]
    asset_net = executed * panel.returns - costs
    contribution = asset_net * wealth_before[None, :] / len(panel.indices)
    pnl = contribution.sum(axis=1)
    if not np.isclose(pnl.sum(), metrics["net_pnl"], atol=1e-6, rtol=1e-8):
        raise ValueError("Ticker contributions do not reconcile with portfolio PnL.")
    tickers = frame.iloc[panel.indices[:, 0]].ticker.to_numpy()
    attribution = {ticker: {"pnl_contribution": float(pnl[index]),
                           "executed_mean_abs_position": float(np.abs(executed[index]).mean()),
                           "executed_mean_position": float(executed[index].mean()),
                           "turnover_sum": float(turnover[index].sum()),
                           "cost_return_sum": float(costs[index].sum())}
                   for index, ticker in enumerate(tickers)}
    monthly = []
    for month, part in daily.groupby(daily.date.dt.strftime("%Y-%m"), sort=True):
        monthly.append({"month": month, "sessions": len(part),
                        "net_return": float(np.prod(1 + part.net_return) - 1),
                        "buy_hold_return": float(np.prod(1 + part.buy_hold_return) - 1),
                        "mean_gross_exposure": float(part.gross_exposure.mean()),
                        "mean_net_exposure": float(part.net_exposure.mean()),
                        "turnover_sum": float(part.turnover.sum()), "cost_return_sum": float(part.cost.sum())})
    return metrics, attribution, monthly


def _trace_summary(fit):
    trace = fit.get("learning_trace")
    if trace is None:
        return {"trace_available": False, "stop_reason": None,
                "trace_limitation": "Epoch curves and gradients were not recorded; cannot reconstruct them."}, []
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        raise ValueError("Unsupported learning trace schema.")
    epochs = trace.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("Recorded learning trace has no epochs.")
    if [item.get("epoch") for item in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError("Learning trace epochs must be consecutive from one.")
    if len(epochs) != fit.get("epochs_run") or trace.get("best_epoch") != fit.get("best_epoch"):
        raise ValueError("Learning trace disagrees with fit summary.")
    validated = LearningTrace()
    for item in epochs:
        validated.record(**item)
    validated.finish(stop_reason=trace.get("stop_reason"), best_epoch=fit["best_epoch"])
    best = epochs[fit["best_epoch"] - 1]
    return {"trace_available": True, "stop_reason": trace.get("stop_reason"),
            "train_loss_at_best": best["train"]["loss"],
            "validation_loss_at_best": best["validation"]["loss"],
            "phase_semantics": trace.get("phase_semantics"),
            "trace_limitation": "TRAIN pre-update with dropout; validation post-update in eval mode; not identical conditions."}, epochs


def _csv(path, records):
    if not records:
        atomic_write_text(path, "no_records\n")
        return
    flattened = pd.json_normalize(records, sep="_")
    for name in flattened:
        flattened[name] = flattened[name].map(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)
    atomic_write_text(path, flattened.to_csv(index=False))


def diagnose_learning(source: str | Path, destination: str | Path, *,
                      partitions: Sequence[str] = ("inner",), candidates: Sequence[str] | None = None,
                      folds: Sequence[int] | None = None, seeds: Sequence[int] | None = None,
                      flat_tolerance: float = 1e-6, saturation_threshold: float = .95):
    """Write a new diagnostic report; source artifacts are strictly read-only."""
    if not partitions or len(set(partitions)) != len(partitions) or not set(partitions) <= set(PARTITIONS):
        raise ValueError("Choose unique partitions from inner,outer; final test is forbidden.")
    if not np.isfinite(flat_tolerance) or not 0 <= flat_tolerance < 1:
        raise ValueError("flat_tolerance must be finite and in [0,1).")
    if not np.isfinite(saturation_threshold) or not flat_tolerance < saturation_threshold <= 1:
        raise ValueError("saturation_threshold must be above flat_tolerance and at most one.")
    source, target = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    runs = discover_runs(source)
    if target.exists():
        raise FileExistsError("Diagnostic output must be a new directory; no overwrite.")
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Diagnostic output must be outside the source run directories.")
    report = {"schema_version": 1, "source": str(source), "partitions": list(partitions),
              "training_performed": False, "checkpoint_loaded": False, "final_holdout_opened": False,
              "selected": None, "complete": True, "issues": [], "sources": [], "source_containers": [], "tasks": [],
              "flat_tolerance": flat_tolerance, "saturation_threshold": saturation_threshold,
              "limitations": ["Diagnostic only; no feature, threshold or checkpoint selection on OUTER.",
                              "Probability maxima are not validated or calibrated confidence.",
                              "Seeds share market dates; fold/seed observations are not independent.",
                              "Gross exposure is not net exposure, beta or equal risk.",
                              "Ticker PnL contributions describe this portfolio, not standalone ticker backtests.",
                              "Checks cover file integrity and recorded task identity, not re-derivation of the original training spec."]}
    # Container paths can be old Windows paths. Only inspect local manifests;
    # never execute their plans or resolve remote output_dir/registry references.
    for name in ("study.json", "study.manifest.json"):
        path = source / name
        if path.is_file():
            container = _json(path)
            if container.get("final_holdout_opened") is not False:
                raise ValueError("Study manifest must explicitly keep the final holdout sealed.")
            report["source_containers"].append(file_record(path, source))
    ticker_rows, monthly_rows, epoch_rows, paired, accepted = [], [], [], [], []
    for run in runs:
        meta, rows = _json(run / "metadata.json"), _json(run / "folds.json")
        if (meta.get("schema_version") != 2 or not isinstance(rows, list)
                or meta.get("class_names") != ["Sell", "Hold", "Buy"]
                or meta.get("final_holdout_opened") is not False):
            raise ValueError("Only schema2 signed runs with explicit class order and sealed holdout are supported.")
        selected = lambda item: ((candidates is None or item["candidate"] in candidates)
                                 and (folds is None or item["fold"] in folds)
                                 and (seeds is None or item["seed"] in seeds))
        configured = meta.get("ablation", {}).get("candidates", [])
        expected = {(candidate, fold, seed) for candidate, fold, seed in product(
            configured, range(meta["n_splits"]), meta["seeds"])
            if selected({"candidate": candidate, "fold": fold, "seed": seed})}
        actual = [_identity(row) for row in rows if selected(row)]
        if len(actual) != len(set(actual)):
            raise ValueError("Duplicate task identities in a run.")
        for identity in sorted(expected - set(actual)):
            report["issues"].append({"run": str(run), "task": list(identity), "reason": "Expected task missing."})
        if set(actual) - expected:
            raise ValueError("Task identity not declared by run metadata.")
        report["sources"].append({"run": str(run), "metadata": file_record(run / "metadata.json", run),
                                   "folds": file_record(run / "folds.json", run), "expected_tasks": len(expected),
                                   "source_context": {key: meta.get(key) for key in (
                                       "protocol", "news_protocol", "point_in_time", "historical_availability_verified",
                                       "availability_kind", "warnings", "survivor_bias_warning")}})
        for editable in rows:
            if not selected(editable):
                continue
            identity = {"run": str(run), "candidate": editable["candidate"],
                        "fold": editable["fold"], "seed": editable["seed"]}
            try:
                if editable.get("status") != "ok":
                    raise ValueError("Task is failed or incomplete.")
                row, physical = _immutable(editable, run)
                if target.is_relative_to(physical) or physical.is_relative_to(target):
                    raise ValueError("Output overlaps a reused source artifact directory.")
                fit = row["fit"]
                if (type(fit.get("best_epoch")) is not int or type(fit.get("epochs_run")) is not int
                        or not 1 <= fit["best_epoch"] <= fit["epochs_run"]):
                    raise ValueError("Invalid best_epoch/epochs_run.")
                trace, epochs = _trace_summary(fit)
                task_partitions = []
                for partition in partitions:
                    frame = validate_predictions(pd.read_parquet(artifact_path(row["prediction_artifacts"][partition], physical)),
                                                 row, meta, partition)
                    daily = pd.read_parquet(artifact_path(row["daily_path_artifacts"][partition], physical))
                    metrics, attribution, monthly = _paths(frame, daily, meta, row, partition)
                    stats = prediction_statistics(frame, flat_tolerance=flat_tolerance, saturation_threshold=saturation_threshold)
                    task = identity | {"partition": partition, "task_signature": row["task_signature"],
                                       "objective": meta["loss_config"]["objective"],
                                       "feature_count": len(row.get("effective_temporal_columns", row.get("feature_columns", []))),
                                       "base_feature_count": len(row.get("feature_columns", [])),
                                       "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
                                       "best_epoch_is_first": fit["best_epoch"] == 1,
                                       "epochs_after_best": fit["epochs_run"] - fit["best_epoch"],
                                       "trace": trace, "signals": stats, "metrics": metrics}
                    task_partitions.append((task, frame, attribution, monthly))
                # Do not label half-validated partitions as a successful task.
                for task, frame, attribution, monthly in task_partitions:
                    report["tasks"].append(task)
                    base = identity | {"partition": task["partition"]}
                    for ticker, part in frame.groupby("ticker", sort=True):
                        ticker_rows.append(base | {"ticker": ticker} | prediction_statistics(
                            part, flat_tolerance=flat_tolerance, saturation_threshold=saturation_threshold) | attribution[ticker])
                    monthly_rows.extend(base | item for item in monthly)
                    # Retain only paths/scalars, not every portfolio's predictions.
                    paired.append((base, artifact_path(row["prediction_artifacts"][task["partition"]], physical), metrics))
                epoch_rows.extend(identity | item for item in epochs)
                accepted.append((editable, run))
            except (OSError, ValueError, TypeError, KeyError) as error:
                report["issues"].append(identity | {"reason": str(error)})
    if not report["tasks"]:
        raise ValueError("No valid selected task; no diagnostic output written.")
    report["complete"] = not report["issues"]
    report["trace_tasks"] = sum(task["trace"]["trace_available"] for task in report["tasks"])
    report["task_partitions"] = len(report["tasks"])
    report["manifest_sha256"] = stable_digest(report["sources"])
    # Only compare branches from one run/fold/seed on exactly the same keys.
    differences = []
    baselines = {(item["run"], item["fold"], item["seed"], item["partition"]): (path, metrics)
                 for item, path, metrics in paired if item["candidate"] == "gru"}
    columns = ["date", "ticker", "available", "position", "adj_close", "label", "label_known"]
    for item, path, metrics in paired:
        reference = baselines.get((item["run"], item["fold"], item["seed"], item["partition"]))
        if reference is None or item["candidate"] == "gru":
            continue
        left_frame = pd.read_parquet(reference[0], columns=columns)
        right_frame = pd.read_parquet(path, columns=columns)
        merged = left_frame.merge(right_frame, on=["date", "ticker"], suffixes=("_gru", "_candidate"),
                                  how="outer", indicator=True, validate="one_to_one")
        comparable = (merged._merge.eq("both").all()
                      and merged.label_gru.eq(merged.label_candidate).all()
                      and merged.label_known_gru.eq(merged.label_known_candidate).all()
                      and np.array_equal(merged.adj_close_gru.to_numpy(), merged.adj_close_candidate.to_numpy()))
        if not comparable:
            report["issues"].append(item | {"reason": "Unmatched keys, labels or prices for paired comparison."})
            report["complete"] = False
            continue
        valid = merged.available_gru & merged.available_candidate
        left, right = merged.loc[valid, "position_gru"].to_numpy(), merged.loc[valid, "position_candidate"].to_numpy()
        differences.append(item | {"reference": "gru", "paired_available_rows": len(left),
                                    "mean_abs_position_delta": float(np.mean(np.abs(right) - np.abs(left))) if len(left) else None,
                                    "mean_position_delta": float(np.mean(right - left)) if len(left) else None,
                                    "sign_disagreement_fraction": float(np.mean(
                                        np.where(np.abs(right) <= flat_tolerance, 0, np.sign(right))
                                        != np.where(np.abs(left) <= flat_tolerance, 0, np.sign(left)))) if len(left) else None,
                                    "position_correlation": float(np.corrcoef(left, right)[0, 1])
                                    if len(left) > 1 and left.std() > 0 and right.std() > 0 else None,
                                    "metric_deltas": {key: (metrics[key] - reference[1][key]
                                                           if metrics[key] is not None and reference[1][key] is not None else None)
                                        for key in ("net_return", "net_pnl", "net_sharpe", "regularized_sharpe",
                                                    "max_drawdown", "mean_abs_position", "mean_position", "turnover", "cost_return_sum")}})
        del left_frame, right_frame, merged
    # A run being modified during diagnosis must not receive a complete report.
    for record in report["source_containers"]:
        if file_record(artifact_path(record["path"], source), source) != record:
            report["issues"].append({"run": str(source), "reason": "Study manifest changed during diagnosis."})
    for entry in report["sources"]:
        run = Path(entry["run"])
        if any(file_record(run / f"{name}.json", run) != entry[name] for name in ("metadata", "folds")):
            report["issues"].append({"run": str(run), "reason": "Source metadata/folds changed during diagnosis."})
    for editable, run in accepted:
        try:
            _immutable(editable, run)
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["issues"].append({"run": str(run), "task": list(_identity(editable)),
                                     "reason": f"Source changed during diagnosis: {error}"})
    report["complete"] = not report["issues"]
    summary = []
    for (run, candidate, partition), part in pd.json_normalize(report["tasks"]).groupby(
            ["run", "candidate", "partition"], sort=True):
        description = {"run": run, "candidate": candidate, "partition": partition, "task_partitions": len(part),
                       "best_epoch_is_first_fraction": float(part.best_epoch_is_first.mean()),
                       "trace_available_fraction": float(part["trace.trace_available"].mean())}
        for column in ("best_epoch", "epochs_run", "metrics.net_return", "metrics.net_sharpe", "metrics.regularized_sharpe",
                       "metrics.mean_abs_position", "metrics.mean_position", "metrics.turnover", "metrics.cost_return_sum"):
            values = part[column].dropna().to_numpy(dtype=float)
            description[column.replace(".", "_")] = {
                "count": len(values), "mean": float(values.mean()) if len(values) else None,
                "std_descriptive": float(values.std()) if len(values) else None,
                "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None}
        summary.append(description)
    target.mkdir(parents=True, exist_ok=False)
    _csv(target / "tasks.csv", report["tasks"])
    _csv(target / "tickers.csv", ticker_rows)
    _csv(target / "monthly.csv", monthly_rows)
    _csv(target / "epochs.csv", epoch_rows)
    _csv(target / "paired_signals.csv", differences)
    _csv(target / "summary.csv", summary)
    report["paired_signals"] = differences
    report["summary"] = summary
    atomic_write_json(target / "report.json", report)
    lines = ["# Learning diagnostics", "", f"Complete: {report['complete']}. Task/partitions: {len(report['tasks'])}.",
             f"Epoch traces available: {report['trace_tasks']}/{len(report['tasks'])}.", "",
             "No training, checkpoint loading, automatic selection or final holdout access.",
             "INNER is the default; OUTER, when explicitly requested, is descriptive only.", "",
             "| Run | Candidate | Fold | Seed | Partition | Best epoch / epochs run | Trace | Net Sharpe | Executed gross | Signal mean |",
             "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |"]
    for task in report["tasks"]:
        sharpe = task["metrics"]["net_sharpe"]
        sharpe_text = "unavailable" if sharpe is None else f"{sharpe:.4f}"
        lines.append(f"| {Path(task['run']).name} | {task['candidate']} | {task['fold']} | {task['seed']} | {task['partition']} | "
                     f"{task['best_epoch']}/{task['epochs_run']} | {task['trace']['trace_available']} | "
                     f"{sharpe_text} | {task['metrics']['mean_abs_position']:.2%} | {task['signals']['mean_position']} |")
    lines += ["", "Missing epoch traces are unavailable, never fabricated. Best epoch 1 does not mean no training.",
              "TRAIN loss (pre-update, train mode) and validation loss (post-update, eval mode) have different semantics.",
              "Signal mean position is not delayed executed net exposure. See monthly.csv for executed gross/net.", "",
              "Files: tasks.csv, summary.csv, tickers.csv, monthly.csv, epochs.csv, paired_signals.csv, report.json.", "", "## Limits", ""]
    lines.extend("- " + item for item in report["limitations"])
    if report["issues"]:
        lines += ["", "## Incomplete or rejected tasks", ""]
        lines.extend("- " + json.dumps(item, ensure_ascii=False) for item in report["issues"])
    atomic_write_text(target / "report.md", "\n".join(lines) + "\n")
    return report
