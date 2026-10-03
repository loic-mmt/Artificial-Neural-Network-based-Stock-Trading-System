"""Descriptive equal-gross-exposure controls for frozen benchmark predictions.

No model is trained and no return is used to choose normalization factors.
Whole-fold mean matching is explicitly ex post. Daily matching instead changes
the common exposure path and therefore requires fresh turnover/cost accounting.
Neither procedure equalizes net exposure, market beta or volatility.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata
from trading_system.artifacts.multimodal_study import (
    atomic_write_json, atomic_write_text, file_record, validate_files,
)
from trading_system.experiments.graph_ablation import _daily_paths
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


METHODS = ("raw", "mean_min", "daily_min")


def aligned_prediction_group(predictions, candidates):
    """Return identical complete date/ticker panels, never imputing predictions."""
    required = {"candidate", "date", "ticker", "adj_close", "position", "available"}
    if required - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions))}")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidates must be nonempty and unique.")
    if set(predictions.candidate) != set(candidates):
        raise ValueError("Prediction candidates do not match the frozen candidate set.")
    work = predictions.copy()
    work["date"] = pd.to_datetime(work.date, utc=True, errors="raise")
    if work[["date", "ticker", "available"]].isna().any().any():
        raise ValueError("Missing dates, tickers or availability.")
    if not work.available.eq(True).all():
        raise ValueError("Unavailable predictions cannot be treated as flat positions.")
    if work.duplicated(["candidate", "date", "ticker"]).any():
        raise ValueError("Duplicate prediction date/ticker keys.")
    work["ticker"] = work.ticker.astype(str)
    values = work[["adj_close", "position"]].to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values[:, 0] <= 0).any()
            or (np.abs(values[:, 1]) > 1 + 1e-6).any()):
        raise ValueError("Predictions/prices must be finite, positive and positions bounded.")
    positions, frame = {}, None
    for candidate in candidates:
        group = work.loc[work.candidate.eq(candidate)].sort_values(["date", "ticker"]).reset_index(drop=True)
        current = group[["date", "ticker", "adj_close"]]
        if frame is None:
            frame = current.copy()
            if not (frame.groupby("date").ticker.nunique() == frame.ticker.nunique()).all():
                raise ValueError("Every session must contain every ticker.")
        elif not current.equals(frame):
            raise ValueError("Candidates have different date/ticker calendars or prices.")
        positions[candidate] = group.position.to_numpy(dtype=np.float64)
    return frame, positions


def normalize_positions(panel, positions, method="mean_min"):
    """Match mean or daily executed gross exposure using downscaling only.

    The last delay+1 signal dates never earn a return in ReturnPanel. They do
    not determine a target; daily normalization sets these unused targets flat.
    """
    if method not in ("mean_min", "daily_min") or not positions:
        raise ValueError("Expected positions and method mean_min or daily_min.")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in positions.items()}
    for value in arrays.values():
        if (value.shape != (panel.rows,) or not np.isfinite(value).all()
                or (np.abs(value) > 1 + 1e-6).any()):
            raise ValueError("Positions must be finite, aligned and bounded.")
    eligible = panel.returns.shape[1] - panel.delay
    gross = {name: np.abs(value[panel.indices][:, :eligible]).mean(axis=0)
             for name, value in arrays.items()}
    adjusted, scales = {}, {}
    if method == "mean_min":
        means = {name: value.sum() / len(panel.dates) for name, value in gross.items()}
        target = min(means.values())
        for name, value in arrays.items():
            scales[name] = target / means[name] if means[name] > 0 else 0.0
            adjusted[name] = value * scales[name]
        info = {"target_mean_exposure": float(target), "ex_post": True}
    else:
        target = np.minimum.reduce(tuple(gross.values()))
        for name, value in arrays.items():
            factors = np.divide(target, gross[name], out=np.zeros_like(target), where=gross[name] > 0)
            matrix = np.zeros(panel.indices.shape, dtype=np.float64)
            matrix[:, :eligible] = factors
            scale = np.zeros(panel.rows, dtype=np.float64)
            scale[panel.indices] = matrix
            scales[name], adjusted[name] = scale, value * scale
        info = {"target_mean_exposure": float(target.sum() / len(panel.dates)), "ex_post": False}
    return adjusted, {**info, "method": method, "scales": scales,
                      "eligible_decision_dates": int(eligible),
                      "downscaling_only": True}


def _metrics(panel, positions, loss, capital):
    metrics = panel.metrics(positions, loss, capital)
    daily = _daily_paths(panel, positions, loss)
    metrics.update(mean_net_exposure=float(daily.net_exposure.mean()),
                   daily_net_volatility=float(daily.net_return.std(ddof=0)),
                   annualized_net_volatility=float(daily.net_return.std(ddof=0) * np.sqrt(loss.annualization)))
    return metrics, daily


def _summary(rows):
    def mean_available(numbers):
        available = [value for value in numbers if value is not None]
        return statistics.fmean(available) if available else None

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], row["candidate"]].append(row)
    result = []
    for (method, candidate), values in grouped.items():
        fields = {}
        for key in values[0]["metrics"]:
            numbers = [row["metrics"][key] for row in values if row["metrics"][key] is not None]
            fields[key] = statistics.fmean(numbers) if numbers else None
        result.append({"method": method, "candidate": candidate, "tasks": len(values),
                       "mean_metrics": fields,
                       "worst_drawdown": min(row["metrics"]["max_drawdown"] for row in values),
                       "by_fold": [{"fold": fold,
                                    "net_return": statistics.fmean(row["metrics"]["net_return"]
                                        for row in values if row["fold"] == fold),
                                    "net_sharpe": mean_available(row["metrics"]["net_sharpe"]
                                        for row in values if row["fold"] == fold)}
                                   for fold in sorted({row["fold"] for row in values})]})
    return result


def _pairs(rows):
    index = {(row["method"], row["candidate"], row["fold"], row["seed"]): row for row in rows}
    candidates = sorted({row["candidate"] for row in rows})
    comparisons = [(candidate, "gru") for candidate in candidates if candidate != "gru"]
    comparisons += [(name, "identity") for name in candidates
                    if name not in {"gru", "gru_market", "identity", "buy_hold"}
                    and not name.endswith("_market")]
    comparisons += [(name, name.removesuffix("_market")) for name in candidates
                    if name.endswith("_market") and name.removesuffix("_market") in candidates]
    comparisons = list(dict.fromkeys(comparisons))
    result = []
    for method in METHODS:
        for candidate, baseline in comparisons:
            pairs = [(row, index[method, baseline, row["fold"], row["seed"]])
                     for row in rows if row["method"] == method and row["candidate"] == candidate]
            deltas = [{"fold": row["fold"], "seed": row["seed"],
                       **{name: row["metrics"][name] - ref["metrics"][name]
                          if row["metrics"][name] is not None and ref["metrics"][name] is not None else None
                          for name in ("net_return", "net_sharpe", "regularized_sharpe", "max_drawdown",
                                       "cost_return_sum", "annualized_net_volatility")}}
                      for row, ref in pairs]
            result.append({"method": method, "candidate": candidate, "baseline": baseline,
                           "pairs": deltas,
                           "mean_return_delta": statistics.fmean(row["net_return"] for row in deltas),
                           "mean_sharpe_delta": statistics.fmean(row["net_sharpe"] for row in deltas
                               if row["net_sharpe"] is not None)
                               if any(row["net_sharpe"] is not None for row in deltas) else None,
                           "return_wins": sum(row["net_return"] > 0 for row in deltas)})
    return result


def _reconstruction_audit(path, replay, metadata, run_dir):
    """Admit only an explicit, audited derived-market reconstruction mismatch.

    Identical raw inputs and fold preprocessing are necessary, not sufficient:
    all frozen financial metrics are independently checked below as well. An
    audit never certifies a legacy checkpoint or turns it into a reusable run.
    """
    if path is None:
        if replay.get("mismatches"):
            raise ValueError("Comparison requires matching replay inputs or an explicit market reconstruction audit.")
        return None
    audit = json.loads(Path(path).read_text())
    if (audit.get("schema_version") != 1 or audit.get("source_run") != str(run_dir)
            or replay.get("mismatches") != ["market_context"] or replay.get("reusable") is not False):
        raise ValueError("Audit can admit only a non-reusable derived-market mismatch.")
    for name, field in (("dataset", "dataset_sha256"), ("graph_context", "graph_context_sha256"),
                        ("market_context", "market_context_sha256")):
        record = audit.get("hashes", {}).get(name, {})
        actual, expected = record.get("actual"), record.get("expected")
        if (expected != metadata.get(field)
                or any(not isinstance(value, str) or len(value) != 64
                       or any(character not in "0123456789abcdef" for character in value)
                       for value in (actual, expected))):
            raise ValueError(f"Invalid audited {name} input hash.")
        if name != "market_context" and actual != metadata.get(field):
            raise ValueError("Raw price/context inputs must match exactly, even with an audit.")
        if name == "market_context" and actual == metadata.get(field):
            raise ValueError("Derived-market audit must describe the reported hash mismatch.")
    if (audit["hashes"]["dataset"]["actual"] != replay.get("replay_dataset_sha256")
            or replay.get("source_dataset_sha256") != metadata["dataset_sha256"]):
        raise ValueError("Audit/replay price hashes disagree.")
    folds = audit.get("folds", [])
    if ({fold.get("fold") for fold in folds} != set(range(metadata["n_splits"]))
            or len(folds) != metadata["n_splits"]
            or any(fold.get(flag) is not True for fold in folds for flag in (
                "feature_columns_match", "stock_scaler_allclose", "market_scaler_exact",
                "market_columns_match", "train_calendar_match", "inner_calendar_match",
                "outer_calendar_match"))):
        raise ValueError("Audit must verify preprocessing and calendars for every fold.")
    return audit


def compare_replay_exposure(replay_dir, run_dir, output_dir, *, market_reconstruction_audit=None):
    """Write a fresh diagnostic report from fully checked replay outputs."""
    replay_dir, run_dir, target = map(lambda value: Path(value).resolve(), (replay_dir, run_dir, output_dir))
    if target.exists():
        raise FileExistsError(f"Exposure output already exists: {target}")
    if target == run_dir or run_dir in target.parents or target == replay_dir or replay_dir in target.parents:
        raise ValueError("Exposure output must be outside source benchmark/replay directories.")
    replay = json.loads((replay_dir / "replay.json").read_text())
    report = json.loads((run_dir / "report.json").read_text())
    metadata = report["metadata"]
    if (replay.get("final_holdout_opened") is not False
            or replay.get("source_run") != str(run_dir)
            or report.get("final_test") != [] or metadata.get("final_holdout_opened") is not False):
        raise ValueError("Comparison requires matching replay inputs and a sealed final holdout.")
    audit = _reconstruction_audit(market_reconstruction_audit, replay, metadata, run_dir)
    files = validate_files(replay["files"], replay_dir)
    path = replay_dir / "predictions.parquet"
    if path not in files:
        raise ValueError("Verified replay predictions are missing.")
    candidates = tuple(metadata["ablation"]["candidates"])
    expected = {(candidate, fold, seed) for candidate in candidates
                for fold in range(metadata["n_splits"]) for seed in metadata["seeds"]}
    actual = {(task["candidate"], task["fold"], task["seed"]) for task in replay["tasks"]}
    if actual != expected or len(actual) != len(replay["tasks"]):
        raise ValueError("Replay does not contain every expected benchmark task.")
    loss = FinancialLossConfig(**metadata["loss_config"])
    capital, delay = metadata["config"]["initial_capital"], metadata["config"]["execution_delay"]
    original = {(row["candidate"], row["fold"], row["seed"]): row for row in report["folds"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows, paired_sensitivity, writer = [], [], None
    raw_metric_errors = defaultdict(float)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        try:
            for fold in range(metadata["n_splits"]):
                for seed in metadata["seeds"]:
                    predictions = pd.read_parquet(path, filters=[("partition", "=", "outer"),
                        ("fold", "=", fold), ("seed", "=", seed)],
                        columns=["candidate", "date", "ticker", "adj_close", "position", "available"])
                    frame, positions = aligned_prediction_group(predictions, candidates)
                    panel = ReturnPanel(frame, group_col="ticker", execution_delay=delay)
                    # Existing ReturnPanel reference: equal-weight always-long
                    # portfolio, identical delayed execution and linear costs.
                    positions["buy_hold"] = np.ones(panel.rows, dtype=np.float64)
                    for method in METHODS:
                        if method == "raw":
                            adjusted, info = positions, {"scales": {name: 1.0 for name in positions}}
                        else:
                            adjusted, info = normalize_positions(panel, positions, method)
                        exposures = []
                        for name, value in adjusted.items():
                            metrics, daily = _metrics(panel, value, loss, capital)
                            exposures.append(daily.gross_exposure.to_numpy())
                            if method == "raw" and name != "buy_hold":
                                from trading_system.experiments.graph_replay import _metric_match
                                _metric_match({key: metrics[key] for key in original[name, fold, seed]["outer_metrics"]},
                                              original[name, fold, seed]["outer_metrics"], candidate=name,
                                              fold=fold, seed=seed, partition="exposure_raw")
                                for key, expected_value in original[name, fold, seed]["outer_metrics"].items():
                                    if expected_value is not None and metrics[key] is not None:
                                        raw_metric_errors[key] = max(raw_metric_errors[key],
                                            abs(float(metrics[key]) - float(expected_value)))
                            factor = np.asarray(info["scales"][name])
                            if factor.ndim:
                                factor = factor[panel.indices[:, :panel.returns.shape[1] - panel.delay]].ravel()
                            rows.append({"method": method, "candidate": name, "fold": fold, "seed": seed,
                                         "factor_min": float(factor.min()), "factor_max": float(factor.max()),
                                         "factor_mean": float(factor.mean()), "metrics": metrics})
                            daily.insert(0, "seed", seed)
                            daily.insert(0, "fold", fold)
                            daily.insert(0, "candidate", name)
                            daily.insert(0, "method", method)
                            table = pa.Table.from_pandas(daily, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(stage / "daily_paths.parquet", table.schema)
                            writer.write_table(table)
                        if method != "raw":
                            check = np.stack(exposures)
                            axis = check.mean(axis=1) if method == "mean_min" else check
                            if not np.allclose(axis, axis[0], atol=1e-12, rtol=1e-10):
                                raise ValueError("Executed gross exposures failed to match.")
                    # Sensitivity to the all-candidate minimum: match each pair
                    # against GRU, but never rank returns across these targets.
                    for name in positions:
                        if name == "gru":
                            continue
                        pair, info = normalize_positions(panel, {"gru": positions["gru"], name: positions[name]}, "mean_min")
                        a, _ = _metrics(panel, pair[name], loss, capital)
                        b, _ = _metrics(panel, pair["gru"], loss, capital)
                        paired_sensitivity.append({"candidate": name, "fold": fold, "seed": seed,
                            "target_mean_exposure": info["target_mean_exposure"], "candidate_metrics": a,
                            "gru_metrics": b, "net_return_delta": a["net_return"] - b["net_return"]})
                    print(f"exposure compared fold={fold} seed={seed}", flush=True)
        finally:
            if writer is not None:
                writer.close()
        result = _nullable_metadata({"schema_version": 1, "source_run": str(run_dir),
            "source_replay": str(replay_dir), "final_holdout_opened": False,
            "descriptive_only": True, "training_performed": False,
            "methods": {"raw": "unchanged positions", "mean_min": "ex-post whole-fold common mean gross; constant downscaling",
                        "daily_min": "decision-time common daily gross; dynamic downscaling with fresh costs"},
            "candidates": [*candidates, "buy_hold"], "loss_config": metadata["loss_config"],
            "initial_capital": capital, "execution_delay": delay,
            "source_reusable": replay.get("reusable", False),
            "replay_mismatches": replay.get("mismatches", []),
            "market_reconstruction_audited": audit is not None,
            "checkpoint_feature_order_audits": replay.get("checkpoint_feature_order_audits", []),
            "raw_metric_verification": {"tasks": len(expected), "atol": 5e-4, "rtol": 5e-4,
                                        "max_absolute_error": dict(raw_metric_errors)},
            "notes": ["Gross exposure is not net exposure, beta or equal volatility.",
                      "Constant scaling preserves ordinary net Sharpe; fixed-epsilon regularized Sharpe is not invariant.",
                      "Targets use only executable predictions; no leverage or position clipping.",
                      "Whole-fold scales are ex post, not a deployable strategy or model-selection validation.",
                      "Daily matching is a different shared sizing overlay; it may raise turnover.",
                      "Buy-and-hold is the repository ReturnPanel equal-weight always-long reference.",
                      "Mean fold returns are not a concatenated backtest; seeds share market dates."],
            "tasks": rows, "summary": _summary(rows), "paired": _pairs(rows),
            "pairwise_gru_mean_sensitivity": paired_sensitivity})
        result["files"] = [file_record(stage / "daily_paths.parquet", stage)]
        if audit is not None:
            atomic_write_json(stage / "input-audit.json", audit)
            result["files"].append(file_record(stage / "input-audit.json", stage))
            result["notes"].append("Raw PC prices/context hashes and fold preprocessing match; the derived market hash differs. "
                                   "Every raw financial metric is rechecked, but legacy results remain exploratory and non-reusable.")
        if any(row.get("restored") for row in result["checkpoint_feature_order_audits"]):
            result["notes"].append("Legacy tied feature rankings were permuted to the recorded checkpoint order; "
                                   "the feature sets and named scalers were still checked, with no model retraining.")
        atomic_write_json(stage / "report.json", result)
        lines = ["# Equal-gross-exposure diagnostic", "", "No training. Final holdout remains sealed.", "",
                 "Mean matching is ex post. Daily matching changes sizing and recomputes costs.", "",
                 "| Method | Candidate | Mean exposure | Net return | Net Sharpe | Regularized Sharpe | Mean drawdown |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in result["summary"]:
            metric = row["mean_metrics"]
            sharpe = "NA" if metric["net_sharpe"] is None else f"{metric['net_sharpe']:.4f}"
            lines.append(f"| {row['method']} | {row['candidate']} | {metric['mean_abs_position']:.2%} | "
                         f"{metric['net_return']:+.2%} | {sharpe} | {metric['regularized_sharpe']:.4f} | {metric['max_drawdown']:.2%} |")
        lines += ["", *[f"- {note}" for note in result["notes"]], ""]
        atomic_write_text(stage / "report.md", "\n".join(lines))
        if target.exists():
            raise FileExistsError(f"Exposure output already exists: {target}")
        stage.rename(target)
    return result


__all__ = ["aligned_prediction_group", "normalize_positions", "compare_replay_exposure"]
