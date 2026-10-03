"""Signed, descriptive feature-cap × market-gate comparisons.

No training, promotion or final-holdout evaluation occurs here. Missing cells
remain missing; invalid provenance prevents an interaction claim.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import pickle
import statistics

import numpy as np
import pandas as pd

from trading_system.artifacts.multimodal_study import artifact_path, stable_digest, training_signature
from trading_system.experiments.exposure_comparison import (
    METHODS, _metrics, aligned_prediction_group, normalize_positions,
)
from trading_system.experiments.graph_ablation import _resume_metadata, _row_root, _validated_row
from trading_system.experiments.graph_replay import _metric_match
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


METRICS = ("regularized_sharpe", "net_sharpe", "net_return", "net_pnl", "max_drawdown",
           "mean_abs_position", "turnover", "cost_return_sum")
SIMPLE_EFFECTS = ("gain_high_plain", "gain_high_gate", "gate_gain_low", "gate_gain_high")
RELATION_EFFECTS = tuple(f"relation_{effect}_{cap}" for effect in (
    "gain_plain", "gain_gate", "gate_interaction") for cap in ("low", "high"))
FEATURE_DIRECTORY = "02-features"
NOTES = [
    "Primary result is the raw signed outer score; no automatic feature or gate promotion.",
    "Interaction = (high gate - low gate) - (high plain - low plain). A positive interaction does not imply that the gate wins or that the higher cap is better.",
    "Actual retained feature counts, not requested caps, define the effective capacity intervention.",
    "mean_min is descriptive ex post whole-outer-fold mean matching; daily_min changes the common sizing path. Neither is a deployable sizing rule.",
    "All twelve variants and one buy-and-hold share each exposure target. Gross matching does not equalize net exposure, beta or volatility.",
    "Identity cap × gate interactions control for the gated GNN architecture without cross-stock edges. Matched relation gains compare graph versus identity at the same cap and gate state; relation gate interaction subtracts the identity gate gain from the graph gate gain.",
    "Turnover and costs are recomputed from normalized positions. Returns are fractions; PnL is in initial-capital units per fold/seed, averaged rather than multiplied or compounded across overlapping runs.",
]


def _initial(requested):
    return {"schema_version": 1, "complete": False, "rows": [], "aggregates": [],
            "feature_audit": [], "unavailable": [], "notes": list(NOTES),
            "primary_method": "raw", "selected": None, "final_holdout_opened": False,
            "exposure": {"requested": requested, "complete": False, "rows": [], "aggregates": [],
                         "interaction_rows": []}}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(fold, seed):
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (fold, seed)) or fold < 0:
        raise ValueError("Fold and seed identities must be integers, not booleans.")
    return fold, seed


def _protocol(metadata, cap):
    if not isinstance(metadata, dict):
        raise ValueError("Run metadata must be a JSON object.")
    if metadata.get("schema_version") != 2 or metadata.get("final_holdout_opened") is not False:
        raise ValueError("Feature interactions require signed schema-2 runs and a sealed holdout.")
    required = ("dataset_sha256", "calendar", "cv_spec", "cv_folds", "final_split", "loss_config",
                "gru_parameters", "provenance", "market_columns", "market_context_sha256")
    if any(key not in metadata for key in required):
        raise ValueError("Run lacks complete data, calendar, CV, loss or runtime provenance.")
    result = deepcopy(_resume_metadata(metadata))
    control = result["config"]["overfitting_control"]
    if not isinstance(control, dict):
        raise ValueError("Run must declare its feature selection configuration.")
    if control.get("max_features") != cap:
        raise ValueError("Requested feature cap disagrees with the signed run configuration.")
    control.pop("max_features")
    return result


def _load_checkpoint(path):
    """Read signature/preprocessing only; mapped weights are never evaluated."""
    import torch
    core = getattr(np, "_core", None) or np.core
    allowed = [core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype("float32")), type(np.dtype("float64"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _sessions(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Signed eligible-session calendars must be nonempty lists.")
    dates = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Eligible-session calendars must be unique and sorted.")
    return [day.isoformat() for day in dates]


def _stock_state(state):
    return {key: value for key, value in state.items() if key not in {"market_columns", "market_scaler"}}


def _signed_cell(row, root, task, metadata, cap, planned_features):
    row = _validated_row(row, root, task["signature"])
    identity = (task["candidate"], task["fold"], task["seed"])
    if (row.get("candidate"), row.get("fold"), row.get("seed")) != identity or row.get("status") != "ok":
        raise ValueError("Immutable result differs from its planned task identity/status.")
    physical = _row_root(root, row)
    checkpoint = _load_checkpoint(artifact_path(row["model_artifact"], physical))
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a signed state mapping.")
    spec, state = checkpoint.get("training_spec", {}), checkpoint.get("preprocessing", {})
    if not isinstance(spec, dict) or not isinstance(state, dict):
        raise ValueError("Checkpoint training/preprocessing state is unavailable.")
    signature = task["signature"]
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("task_signature") != signature
            or training_signature(spec) != signature or spec.get("preprocessing") != state
            or (spec.get("candidate"), spec.get("fold"), spec.get("seed")) != identity
            or checkpoint.get("fold") != task["fold"] or checkpoint.get("seed") != task["seed"]
            or checkpoint.get("mode") != task["candidate"]):
        raise ValueError("Checkpoint training signature or identity does not match the planned task.")
    columns = list(row.get("feature_columns", []))
    if (not columns or len(set(columns)) != len(columns) or len(columns) > cap
            or columns != state.get("feature_columns") or columns != list(checkpoint.get("feature_columns", []))
            or row.get("preprocessing_signature") != stable_digest(state)
            or checkpoint.get("model_spec") != spec.get("model")):
        raise ValueError("Immutable result and checkpoint feature/preprocessing contracts disagree.")
    for key in ("config", "loss_config", "gru_parameters", "dataset_sha256", "calendar", "cv_spec"):
        if spec.get(key) != metadata[key]:
            raise ValueError(f"Checkpoint differs from signed run metadata: {key}.")
    if training_signature(spec.get("provenance")) != training_signature(metadata["provenance"]):
        raise ValueError("Checkpoint source/runtime provenance differs from the run.")
    mode, gated = task["candidate"].removesuffix("_market"), task["candidate"].endswith("_market")
    ablation = metadata["ablation"]
    expected_model = {"candidate": task["candidate"], "width": len(columns)}
    if mode != "gru":
        expected_model.update({key: ablation[key] for key in ("gnn_hidden_size", "gnn_layers", "gnn_dropout")})
    if gated:
        expected_model.update({key: ablation[key] for key in (
            "market_transformer_width", "market_transformer_heads", "market_transformer_layers")})
        expected_model["market_gate_temperature"] = 1.0 if mode == "gru" else ablation["market_gate_temperature"]
    expected_graph = None if mode in {"gru", "identity"} else {
        "mode": mode, "lookback": ablation["graph_lookback"], "threshold": ablation["graph_threshold"],
        "weight_mode": ablation["graph_weight_mode"], "neighbors": ablation["graph_neighbors"],
        "rebalance_bars": ablation["graph_rebalance_bars"],
        "context_sha256": metadata["graph_context_sha256"] if mode == "rolling_residual_topk" else None,
        "sector_context_columns": metadata["sector_context_columns"] if mode == "rolling_residual_topk" else None}
    if (spec.get("model") != expected_model or spec.get("graph") != expected_graph
            or spec.get("shared_graph_warmup") != ablation["graph_lookback"]
            or spec.get("date_batch_size") != ablation["date_batch_size"]
            or spec.get("market_context_sha256") != (metadata["market_context_sha256"] if gated else None)):
        raise ValueError("Checkpoint effective model/graph/market protocol differs from its signed run.")
    if state.get("tickers") != metadata["calendar"]["tickers"]:
        raise ValueError("Checkpoint ticker universe differs from signed run calendar.")
    if gated:
        market = state.get("market_scaler") or {}
        if (state.get("market_columns") != metadata["market_columns"]
                or list(checkpoint.get("market_columns", [])) != metadata["market_columns"]):
            raise ValueError("Checkpoint gated market columns differ from the signed inputs.")
        for suffix in ("mean", "scale"):
            stored = np.asarray(checkpoint.get(f"market_scaler_{suffix}"), dtype=np.float64)
            frozen = np.asarray(market.get(suffix), dtype=np.float64)
            if (stored.shape != (1, len(metadata["market_columns"])) or stored.shape != frozen.shape
                    or not np.isfinite(stored).all() or not np.array_equal(stored, frozen)
                    or suffix == "scale" and (stored <= 0).any()):
                raise ValueError("Checkpoint gated market scaler differs from its signed preprocessing.")
    elif state.get("market_columns") != [] or state.get("market_scaler") is not None:
        raise ValueError("Ungated signed preprocessing unexpectedly contains a market branch.")
    boundary = [item for item in metadata["cv_folds"] if item["fold"] == task["fold"]]
    if len(boundary) != 1 or spec.get("fold_boundaries") != {key: boundary[0][key] for key in ("split", "end")}:
        raise ValueError("Checkpoint CV boundary differs from the declared fold.")
    calendars = {name: _sessions(row.get("eligible_sessions", {}).get(name)) for name in ("train", "inner", "outer")}
    for name in ("train", "inner"):
        if calendars[name] != _sessions(spec.get("eligible_sessions", {}).get(name)):
            raise ValueError("Immutable result/calendar differs from its training signature.")
    if not (calendars["train"][-1] < calendars["inner"][0]
            and calendars["inner"][-1] < calendars["outer"][0]):
        raise ValueError("Eligible train/inner/outer calendars overlap or are not chronological.")
    source_sessions = set(_sessions(metadata["calendar"]["sessions"]))
    if any(set(values) - source_sessions for values in calendars.values()):
        raise ValueError("Eligible calendar contains sessions absent from the signed dataset/holdout calendar.")
    holdout = pd.to_datetime(metadata["final_split"]["test_start"], utc=True)
    outer_dates = pd.to_datetime(calendars["outer"], utc=True)
    split = boundary[0]["split"]
    if (outer_dates.max() >= holdout or outer_dates.max() > pd.to_datetime(boundary[0]["end"], utc=True)
            or outer_dates.min() < pd.to_datetime(split["test_start"], utc=True)
            or pd.to_datetime(calendars["inner"], utc=True).max() >= pd.to_datetime(split["test_start"], utc=True)
            or pd.to_datetime(calendars["inner"], utc=True).min() < pd.to_datetime(split["validation_start"], utc=True)
            or pd.to_datetime(calendars["train"], utc=True).max() >= pd.to_datetime(split["validation_start"], utc=True)):
        raise ValueError("Eligible sessions cross their signed CV/final-holdout boundaries.")
    selector = state.get("overfitting_selector") or {}
    if (selector.get("fit_scope") != "train_only" or selector.get("supervised") is not False
            or selector.get("selected_columns") != columns
            or selector.get("config", {}).get("max_features") != cap):
        raise ValueError("Feature selection must be the frozen unsupervised train-only cap intervention.")
    for suffix in ("mean", "scale"):
        stored = np.asarray(checkpoint.get(f"scaler_{suffix}"), dtype=np.float64)
        frozen = np.asarray(state.get("scaler", {}).get(suffix), dtype=np.float64)
        if (stored.shape != (1, len(columns)) or frozen.shape != stored.shape
                or not np.isfinite(stored).all() or not np.array_equal(stored, frozen)
                or suffix == "scale" and (stored <= 0).any()):
            raise ValueError("Checkpoint stock scaler differs from its signed preprocessing.")
    if planned_features is not None:
        expected = {"feature_columns": columns, "fill_values": state["fill_values"], "scaler": state["scaler"],
                    "input_columns": selector.get("input_columns"), "selector_config": selector.get("config"),
                    "train_only": True, "tickers": state["tickers"], "fracdiff": state.get("fracdiff"),
                    "purging": state.get("purging"),
                    "train_calendar_sha256": stable_digest(calendars["train"]),
                    "inner_calendar_sha256": stable_digest(calendars["inner"])}
        if any(planned_features.get(key) != value for key, value in expected.items()):
            raise ValueError("Actual checkpoint preprocessing differs from the predeclared feature audit.")
    metrics = row.get("outer_metrics", {})
    if any(name not in metrics or metrics[name] is None and name != "net_sharpe" or metrics[name] is not None and
           (isinstance(metrics[name], bool) or not isinstance(metrics[name], (int, float))
            or not math.isfinite(metrics[name])) for name in METRICS):
        raise ValueError("Signed outer financial metrics are missing or nonfinite.")
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("Signed raw selection score must be finite.")
    _metric_match({"score": score}, {"score": metrics["regularized_sharpe"]}, candidate=identity[0],
                  fold=identity[1], seed=identity[2], partition="feature_raw_score")
    return {"row": row, "root": physical, "state": state, "calendars": calendars, "cap": cap}


def _feature_audits(cells, caps, keys):
    audits = []
    for fold in sorted({fold for fold, _ in keys}):
        canonical = {}
        gated = [cell for (width, name, current, _), cell in cells.items()
                 if current == fold and name.endswith("_market")]
        if gated and any(any(cell["state"].get(key) != gated[0]["state"].get(key)
                             for key in ("market_columns", "market_scaler")) for cell in gated[1:]):
            raise ValueError("Gated candidates/caps/seeds have different frozen market preprocessing.")
        for cap in caps:
            values = [cell for (width, _, current, _), cell in cells.items() if width == cap and current == fold]
            if not values:
                continue
            first = values[0]
            if any(_stock_state(value["state"]) != _stock_state(first["state"])
                   or value["calendars"] != first["calendars"] for value in values[1:]):
                raise ValueError("Candidates/seeds within a feature cap have different stock preprocessing or calendars.")
            canonical[cap] = first
        if len(canonical) != 2:
            audits.append({"fold": fold, "comparable": False, "reason": "Both caps need a verified checkpoint."})
            continue
        low, high = (canonical[cap] for cap in caps)
        a, b = low["state"], high["state"]
        lc, hc = a["feature_columns"], b["feature_columns"]
        prefix = hc[:len(lc)] == lc
        selector_a, selector_b = deepcopy(a["overfitting_selector"]), deepcopy(b["overfitting_selector"])
        for selector in (selector_a, selector_b):
            selector.pop("selected_columns", None)
            selector.pop("dropped_columns", None)
            selector["config"].pop("max_features", None)
        shared = (low["calendars"] == high["calendars"] and selector_a == selector_b
                  and all(a.get(key) == b.get(key) for key in ("feature_selector", "tickers", "fracdiff", "purging")))
        fills, scalers = True, True
        for name in lc:
            if name not in hc:
                fills = scalers = False
                break
            fills &= a["fill_values"].get(name) == b["fill_values"].get(name)
            for suffix in ("mean", "scale"):
                left, right = (np.asarray(state["scaler"][suffix], dtype=np.float64).reshape(-1)
                               for state in (a, b))
                scalers &= bool(np.isclose(left[lc.index(name)], right[hc.index(name)], rtol=1e-6, atol=1e-6))
        item = {"fold": fold, "low_cap": caps[0], "high_cap": caps[1],
                "low_columns": lc, "high_columns": hc, "actual_low_count": len(lc),
                "actual_high_count": len(hc), "prefix_nested": prefix, "ordered_nested": prefix,
                "actual_growth": len(hc) > len(lc), "additional_columns": hc[len(lc):] if prefix else [],
                "common_fills_match": bool(fills), "common_scaler_match": bool(scalers),
                "same_calendar_and_pool": shared, "train_only": True,
                "train_calendar_sha256": stable_digest(low["calendars"]["train"]),
                "inner_calendar_sha256": stable_digest(low["calendars"]["inner"]),
                "comparable": bool(prefix and shared and fills and scalers)}
        audits.append(item)
        if not item["comparable"]:
            raise ValueError(f"Feature-cap intervention is not isolated/prefix-nested in fold {fold}.")
    return audits


def _interaction(values, branch, caps, fold, seed, method):
    lp, lg, hp, hg = (values[cap, candidate] for cap, candidate in (
        (caps[0], branch), (caps[0], branch + "_market"), (caps[1], branch), (caps[1], branch + "_market")))
    def subtract(left, right):
        return {name: left[name] - right[name] if left[name] is not None and right[name] is not None else None
                for name in (*METRICS, "score")}
    plain, gate = subtract(hp, lp), subtract(hg, lg)
    result = {"method": method, "branch": branch, "fold": fold, "seed": seed,
              "low_cap": caps[0], "high_cap": caps[1], "gain_high_plain": plain,
              "gain_high_gate": gate, "gate_gain_low": subtract(lg, lp), "gate_gain_high": subtract(hg, hp),
              "interaction": subtract(gate, plain), "score_interaction": subtract(gate, plain)["score"]}
    if branch not in {"gru", "identity"} and all(
            (cap, candidate) in values for cap in caps for candidate in ("identity", "identity_market")):
        for label, cap, graph_plain, graph_gate in (("low", caps[0], lp, lg), ("high", caps[1], hp, hg)):
            relation_plain = subtract(graph_plain, values[cap, "identity"])
            relation_gate = subtract(graph_gate, values[cap, "identity_market"])
            result[f"relation_gain_plain_{label}"] = relation_plain
            result[f"relation_gain_gate_{label}"] = relation_gate
            result[f"relation_gate_interaction_{label}"] = subtract(relation_gate, relation_plain)
    return result


def _aggregates(rows, expected):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["branch"]].append(row)
    result = []
    for (method, branch), values in sorted(groups.items()):
        def summary(group):
            means, stds, counts = {}, {}, {}
            for name in (*METRICS, "score"):
                numbers = [item["interaction"][name] for item in group if item["interaction"][name] is not None]
                means[name] = statistics.fmean(numbers) if numbers else None
                stds[name] = statistics.pstdev(numbers) if numbers else None
                counts[name] = len(numbers)
            effects = {}
            for effect in (*SIMPLE_EFFECTS, *RELATION_EFFECTS):
                # An absent identity gate must not invent an edge-attribution
                # control or silently aggregate a different subset of pairs.
                if not all(effect in item for item in group):
                    continue
                effects[f"mean_{effect}"] = {
                    name: statistics.fmean([item[effect][name] for item in group if item[effect][name] is not None])
                    if any(item[effect][name] is not None for item in group) else None for name in (*METRICS, "score")}
            return {"mean_interaction": means, "std_interaction": stds, "metric_counts": counts,
                    "mean_score_interaction": means["score"], **effects}
        result.append({"method": method, "branch": branch, "paired_count": len(values),
                       "expected_count": len(expected), "complete": {(row["fold"], row["seed"]) for row in values} == expected,
                       **summary(values),
                       **{f"by_{field}": [{field: value, **summary([row for row in values if row[field] == value])}
                                           for value in sorted({row[field] for row in values})] for field in ("fold", "seed")}})
    return result


def _exposure(cells, caps, candidates, keys, metadata, unavailable):
    rows, comparisons = [], []
    loss = FinancialLossConfig(**metadata["loss_config"])
    capital, delay = metadata["config"]["initial_capital"], metadata["config"]["execution_delay"]
    for fold, seed in sorted(keys):
        if any((cap, candidate, fold, seed) not in cells for cap in caps for candidate in candidates):
            unavailable.append({"kind": "missing", "fold": fold, "seed": seed,
                                "reason": "Exposure normalization needs all twelve verified variants, including identity_market at both caps, never a subset."})
            continue
        frames, names, aliases, labels = [], [], {}, None
        for cap in caps:
            for candidate in candidates:
                cell = cells[cap, candidate, fold, seed]
                path = artifact_path(cell["row"]["prediction_artifacts"]["outer"], cell["root"])
                frame = pd.read_parquet(path)
                identity_columns = {"variant_id", "candidate", "fold", "seed", "partition", "backtest_key",
                                    "date", "ticker", "adj_close", "position", "available", "label", "label_known"}
                if identity_columns - set(frame):
                    raise ValueError(f"Prediction identity/label contract is incomplete: {sorted(identity_columns - set(frame))}.")
                for key, expected in (("variant_id", candidate), ("candidate", candidate),
                                      ("fold", fold), ("seed", seed), ("partition", "outer")):
                    if not frame[key].eq(expected).all():
                        raise ValueError("Verified prediction artifact task/partition identity differs.")
                dates = pd.to_datetime(frame.date, utc=True, errors="raise")
                if frame.duplicated(["date", "ticker"]).any():
                    raise ValueError("Duplicate prediction date/ticker keys.")
                backtest_keys = dates.map(lambda value: value.isoformat()) + "|" + frame.ticker.astype(str)
                if (not frame.backtest_key.equals(backtest_keys.rename("backtest_key"))
                        or sorted(frame.ticker.astype(str).unique()) != sorted(cell["state"]["tickers"])):
                    raise ValueError("Prediction backtest keys/ticker roster differ from the checkpoint.")
                if (not pd.api.types.is_bool_dtype(frame.label_known) or not pd.api.types.is_bool_dtype(frame.available)
                        or frame.label_known.isna().any()
                        or not frame.loc[frame.label_known, "label"].isin([0, 1, 2]).all()
                        or not frame.loc[~frame.label_known, "label"].eq(-1).all()):
                    raise ValueError("Prediction known/unknown labels are inconsistent.")
                current_labels = frame[["date", "ticker", "label", "label_known"]].sort_values(["date", "ticker"]).reset_index(drop=True)
                if labels is not None and not current_labels.equals(labels):
                    raise ValueError("Candidates/caps have different aligned label or label-availability states.")
                labels = current_labels
                if _sessions(sorted(pd.to_datetime(frame.date, utc=True).unique().tolist())) != cell["calendars"]["outer"]:
                    raise ValueError("Outer predictions differ from the immutable eligible-session calendar.")
                name = f"features-{cap}/{candidate}"
                frame["candidate"] = name
                frames.append(frame[["candidate", "date", "ticker", "adj_close", "position", "available"]])
                names.append(name)
                aliases[name] = cap, candidate
        frame, positions = aligned_prediction_group(pd.concat(frames, ignore_index=True), names)
        panel = ReturnPanel(frame, group_col="ticker", execution_delay=delay)
        positions["buy_hold"] = np.ones(panel.rows, dtype=np.float64)
        for method in METHODS:
            adjusted, info = (positions, {}) if method == "raw" else normalize_positions(panel, positions, method)
            values, exposures = {}, []
            for name, value in adjusted.items():
                metrics, daily = _metrics(panel, value, loss, capital)
                exposures.append(daily.gross_exposure.to_numpy())
                if name != "buy_hold":
                    cap, candidate = aliases[name]
                    if method == "raw":
                        expected = cells[cap, candidate, fold, seed]["row"]["outer_metrics"]
                        _metric_match({key: metrics[key] for key in expected}, expected, candidate=candidate,
                                      fold=fold, seed=seed, partition="feature_exposure_raw")
                    values[cap, candidate] = {**{key: metrics[key] for key in METRICS}, "score": metrics["regularized_sharpe"]}
                rows.append({"method": method, "variant_id": name, "fold": fold, "seed": seed,
                             "feature_cap": aliases[name][0] if name != "buy_hold" else None,
                             "candidate": aliases[name][1] if name != "buy_hold" else "buy_hold",
                             "metrics": metrics, "target_mean_exposure": info.get("target_mean_exposure")})
            if method != "raw":
                common = np.stack(exposures)
                check = common.mean(axis=1) if method == "mean_min" else common
                if not np.allclose(check, check[0], atol=1e-12, rtol=1e-10):
                    raise ValueError("Normalized executed gross exposures do not match.")
            for branch in ("gru", candidates[2], "identity"):
                comparisons.append(_interaction(values, branch, caps, fold, seed, method))
    aggregates = []
    for method in METHODS:
        for name in sorted({row["variant_id"] for row in rows}):
            group = [row for row in rows if row["method"] == method and row["variant_id"] == name]
            if group:
                aggregates.append({"method": method, "variant_id": name, "tasks": len(group),
                    "mean_metrics": {key: statistics.fmean([row["metrics"][key] for row in group if row["metrics"][key] is not None])
                                     if any(row["metrics"][key] is not None for row in group) else None for key in group[0]["metrics"]}})
    return rows, comparisons, aggregates


def build_feature_interaction_report(study_dir, *, exposure_comparison=False):
    """Report verified feature32/64 × gate differences; partial studies stay partial."""
    result = _initial(exposure_comparison)
    target = Path(study_dir).expanduser().resolve()
    cells, metadata_by_cap = {}, {}
    try:
        study = _read(target / "study.json")
        if not isinstance(study, dict):
            raise ValueError("Study must be a JSON object.")
        if study.get("final_holdout_opened") is not False:
            raise ValueError("Study must retain a sealed final holdout.")
        plan = study.get("stages", {}).get("features", {}).get("plan")
        if not plan or plan.get("stage") != "features":
            raise ValueError("Feature-stage plan is unavailable.")
        runs = sorted(plan["runs"], key=lambda run: run["feature_cap"])
        caps = tuple(run["feature_cap"] for run in runs)
        if caps != (32, 64) or any(isinstance(cap, bool) or not isinstance(cap, int) for cap in caps):
            raise ValueError("Interaction requires exactly the predeclared feature32/64 grid.")
        graph = plan["graph_choice"]
        if graph not in {"sector", "rolling_topk", "rolling_residual_topk"}:
            raise ValueError("Graph choice must be explicit and supported.")
        candidates = ("gru", "gru_market", graph, graph + "_market", "identity", "identity_market")
        keys, protocol = None, None
        for run in runs:
            cap = run["feature_cap"]
            if run["variant_id"] != f"features-{cap}":
                raise ValueError("Feature run layout differs from its predeclared cap identity.")
            planned_metadata = run["metadata"]
            current = _protocol(planned_metadata, cap)
            if protocol is not None and protocol != current:
                raise ValueError("Feature caps differ beyond max_features (data/CV/loss/architecture/runtime drift).")
            protocol = current
            declared_candidates = tuple(planned_metadata["ablation"]["candidates"])
            if declared_candidates not in (candidates, candidates[:-1]):
                raise ValueError("Feature-stage plan requires the six-candidate grid or its legacy five-candidate grid.")
            expected = {}
            for task in run["tasks"]:
                key = task["candidate"], *_ids(task["fold"], task["seed"])
                if key in expected or key[0] not in declared_candidates:
                    raise ValueError("Duplicate or unexpected task in feature-stage plan.")
                expected[key] = task
            grid = {(fold, seed) for _, fold, seed in expected}
            if not grid or set(expected) != {(candidate, fold, seed) for candidate in declared_candidates for fold, seed in grid}:
                raise ValueError("Each cap requires its complete declared candidate fold/seed grid.")
            declared_seeds = planned_metadata["seeds"]
            declared_folds = [item["fold"] for item in planned_metadata["cv_folds"]]
            if (len(set(declared_seeds)) != len(declared_seeds)
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in declared_seeds)
                    or len(declared_folds) != planned_metadata["n_splits"]
                    or set(declared_folds) != set(range(planned_metadata["n_splits"]))
                    or grid != {(fold, seed) for fold in declared_folds for seed in declared_seeds}):
                raise ValueError("Planned task grid differs from all declared CV folds, seeds or candidates.")
            if keys is not None and keys != grid:
                raise ValueError("Feature caps declare different fold/seed grids.")
            keys = grid
            root = target / FEATURE_DIRECTORY / run["variant_id"]
            metadata = _read(root / "metadata.json") if (root / "metadata.json").is_file() else planned_metadata
            if _protocol(metadata, cap) != current:
                raise ValueError("Actual run protocol differs from its feature-stage plan.")
            metadata_by_cap[cap] = metadata
            if (root / "report.json").is_file() and _read(root / "report.json").get("final_test") != []:
                raise ValueError("Run final holdout has been opened.")
            rows = _read(root / "folds.json") if (root / "folds.json").is_file() else []
            observed = {}
            audits = {item["fold"]: item for item in run.get("feature_preprocessing", [])}
            for row in rows:
                key = row["candidate"], *_ids(row["fold"], row["seed"])
                if key in observed or key not in expected:
                    raise ValueError("Duplicate or unplanned task in feature run output.")
                observed[key] = row
                if row.get("status") != "ok":
                    continue
                cells[cap, *key] = _signed_cell(row, root, expected[key], metadata, cap, audits.get(key[1]))
            for key in expected:
                if (cap, *key) not in cells:
                    result["unavailable"].append({"kind": "missing", "feature_cap": cap,
                        "candidate": key[0], "fold": key[1], "seed": key[2], "reason": "Signed completed task is unavailable."})
            if declared_candidates != candidates:
                for fold, seed in sorted(grid):
                    result["unavailable"].append({"kind": "missing", "feature_cap": cap,
                        "candidate": "identity_market", "fold": fold, "seed": seed,
                        "reason": "Legacy five-candidate feature plan has no identity_market task; the expanded control grid is incomplete."})
        result["feature_audit"] = _feature_audits(cells, caps, keys)
        if any(row.get("comparable") and not row["actual_growth"] for row in result["feature_audit"]):
            result["notes"].append("At least one fold retained identical feature counts: the requested caps do not produce an effective capacity increase there.")
        comparable = {row["fold"] for row in result["feature_audit"] if row["comparable"]}
        for fold, seed in sorted(keys):
            values = {(cap, candidate): {**{name: cell["row"]["outer_metrics"][name] for name in METRICS},
                                         "score": cell["row"]["score"]}
                      for (cap, candidate, f, s), cell in cells.items() if (f, s) == (fold, seed)}
            for branch in ("gru", graph, "identity"):
                if fold in comparable and all((cap, candidate) in values for cap in caps for candidate in (branch, branch + "_market")):
                    result["rows"].append(_interaction(values, branch, caps, fold, seed, "raw"))
        raw_complete = len(cells) == 2 * len(candidates) * len(keys) and len(comparable) == len({f for f, _ in keys})
        if exposure_comparison:
            exposure_rows, comparisons, summary = _exposure(cells, caps, candidates, keys, metadata_by_cap[caps[0]], result["unavailable"])
            result["exposure"].update(rows=exposure_rows, aggregates=summary, interaction_rows=comparisons,
                complete=len(exposure_rows) == len(METHODS) * (len(caps) * len(candidates) + 1) * len(keys))
            # Raw remains the immutable signed score, not a numerically close
            # reconstructed score. Reconstructed raw is retained in exposure.
            result["rows"].extend(row for row in comparisons if row["method"] != "raw")
        result["aggregates"] = _aggregates(result["rows"], keys)
        result["complete"] = raw_complete and (not exposure_comparison or result["exposure"]["complete"])
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, pickle.UnpicklingError) as error:
        # Reporting an interrupted or old study must not invent valid cells or
        # crash its ordinary comparison. Incompatible provenance is explicit
        # and suppresses every interaction claim, not just its failing cell.
        result.update(complete=False, rows=[], aggregates=[])
        result["exposure"].update(complete=False, rows=[], aggregates=[], interaction_rows=[])
        result["unavailable"].append({"kind": "missing" if isinstance(error, FileNotFoundError) else "incompatible",
                                      "reason": str(error)})
    return result


__all__ = ["build_feature_interaction_report", "METRICS"]
