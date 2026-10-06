"""Discover supported result families without entering checkpoint trees."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .adapters import IDENTITY_COLUMNS, parquet_columns, parquet_identities
from .schemas import Catalog, RunRecord


SKIP_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", "folds", "final",
    "checkpoints", "models", "model", "data", "strategy", "future", "viz",
    "backtest", "metrics", "graphs", "graph-cache", "graphs-cache",
}
MARKERS = {"report.json", "selection.json", "folds.json", "results.json", "replay.json", "manifest.json", "metadata.json", "config.json", "metrics.json", "portfolio_metrics.json", "runs.csv", "summary.csv", "equity-curves.parquet", "equity_curve.parquet", "equity_curve_sign.parquet", "positions_sign.parquet", "sign_portfolio_metrics.json", "daily_paths.parquet", "equity.parquet", "positions.parquet", "orders.parquet", "trades.parquet"}


def _gridsearch_files(names: set[str]) -> set[str]:
    return {name for name in names if name.startswith("gridsearch_walkforward_") and Path(name).suffix in (".csv", ".json")}


def _directories(root: Path):
    if not root.is_dir():
        return
    for directory, children, files in os.walk(root):
        children[:] = sorted(name for name in children if name not in SKIP_DIRS and not name.startswith("."))
        path = Path(directory)
        names = set(files)
        if names & MARKERS or _gridsearch_files(names):
            yield path, names
        if "manifest.json" in names and (path / "backtest").is_dir():
            children[:] = []


def catalog_signature(root) -> tuple:
    """Bound cache invalidation to result directories, excluding checkpoints."""
    root = Path(root).expanduser().resolve()
    entries = []
    for directory, names in _directories(root):
        watched = set(names & MARKERS) | _gridsearch_files(names)
        if (directory / "backtest").is_dir():
            watched.update(relative for relative in ("backtest/equity_curve.parquet", "backtest/positions.parquet", "backtest/trades.parquet", "data/raw_market.parquet", "metrics/core_metrics.json", "config.json") if (directory / relative).is_file())
        watched.update(name for name in names if name.startswith("positions-") and name.endswith(".parquet"))
        watched.update(name for name in names if name.startswith("fold-") and name.endswith("-positions.parquet"))
        paths = {directory / name for name in watched}
        paths.update(_referenced_files(directory, names))
        for path in sorted(paths):
            try:
                stat = path.stat()
                entries.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                entries.append((str(path), None, None))
    return str(root), tuple(entries)


def _reference(directory: Path, value) -> Path:
    path = Path(value)
    if not path.is_absolute():
        if (directory / path).is_file() or (directory / path).is_dir():
            return directory / path
        if path.is_file():
            return path.resolve()
        path = directory / path
    return path


def _referenced_files(directory: Path, names: set[str]) -> set[Path]:
    """Follow declared outputs only, never walk model/checkpoint directories."""
    references = set()
    for name in names & {"report.json", "folds.json", "results.json", "replay.json", "metadata.json", "manifest.json"}:
        try:
            payload = json.loads((directory / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = payload if isinstance(payload, list) else []
        if isinstance(payload, dict):
            for key in ("prices_path", "signals_path", "dataset_path", "data_path"):
                if isinstance(payload.get(key), str):
                    references.add(_source_reference(directory, payload[key]))
                metadata = payload.get("metadata", {})
                if isinstance(metadata, dict) and isinstance(metadata.get(key), str):
                    references.add(_source_reference(directory, metadata[key]))
            rows = []
            for key in ("runs", "folds", "rows", "tasks"):
                if isinstance(payload.get(key), list):
                    rows.extend(payload[key])
        for row in rows:
            if not isinstance(row, dict):
                continue
            daily = row.get("daily_path_artifacts", {})
            if isinstance(daily, dict):
                references.update(_reference(directory, value) for value in daily.values() if isinstance(value, str))
            if isinstance(row.get("artifact_path"), str):
                references.add(_reference(directory, row["artifact_path"]) / "training_history.json")
    return references


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _source_reference(directory: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (directory, *directory.parents):
        candidate = base / path
        if candidate.is_file() or candidate.is_dir():
            return candidate
    return directory / path


def _attach_market(record: RunRecord) -> None:
    if "market" in record.tables:
        return
    meta = record.metadata
    candidates = (
        ("prices_path", meta.get("inputs_sha256", {}).get("data") if isinstance(meta.get("inputs_sha256"), dict) else None),
        ("signals_path", meta.get("inputs_sha256", {}).get("signals") if isinstance(meta.get("inputs_sha256"), dict) else None),
        ("dataset_path", meta.get("dataset_sha256")),
        ("data_path", meta.get("dataset_sha256", meta.get("data_sha256"))),
    )
    for key, expected in candidates:
        if not isinstance(meta.get(key), str):
            continue
        path = _source_reference(record.path, meta[key])
        # Trading quotes must be tied to a hash supplied by their replay.
        if record.family == "trading" and not isinstance(expected, str):
            continue
        if path.is_file() or path.is_dir():
            record.tables["market"] = path
            record.metadata.update(market_source=key, market_requires_hash=isinstance(expected, str), market_sha256=expected)
            return


def _metadata(payload: dict, row: dict | None = None) -> dict:
    row = row or {}
    meta = dict(payload.get("metadata", {}))
    # Report-level protocol and provenance are needed for formats with no
    # metadata wrapper; exclude embedded result arrays.
    for key, value in payload.items():
        if key not in ("metadata", "folds", "runs", "rows", "tasks", "summary", "comparisons", "failures", "final_test"):
            meta.setdefault(key, value)
    config = meta.get("config", payload.get("config", {}))
    if isinstance(config, dict):
        for key, value in config.items():
            meta.setdefault(key, value)
    loss = meta.get("loss_config", meta.get("loss", meta.get("financial_loss", {})))
    if isinstance(loss, str):
        try:
            loss = json.loads(loss)
        except ValueError:
            loss = {}
    if isinstance(loss, dict):
        for key in ("cost_bps", "fees_bps", "slippage_bps", "execution_delay", "annualization", "initial_capital"):
            if key in loss:
                meta.setdefault(key, loss[key])
    for key in ("candidate", "model", "model_name", "seed", "fold", "partition", "method", "decoder", "retrain_every_sessions", "split", "outer_end", "run_dir", "trial_id", "selected"):
        if key in row:
            meta[key] = row[key]
    meta.setdefault("model", meta.get("model_name", meta.get("candidate")))
    if isinstance(meta.get("model"), dict):
        meta["model_config"] = meta["model"]
        meta["model"] = meta["model"].get("name", meta["model"].get("model_name"))
    meta.setdefault("tickers", meta.get("selected_tickers", meta.get("ticker_selection", meta.get("symbol"))))
    return meta


def _numeric(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in IDENTITY_COLUMNS and key != "run_id":
            result[key] = value
    for key in ("metrics", "outer_metrics", "classification"):
        if isinstance(row.get(key), dict):
            for metric, value in row[key].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[("classification_" if key == "classification" else "") + metric] = value
    return result


def _record(path: Path, family: str, row: dict, metadata: dict, *, selectors=None, tables=None, suffix="", warnings=()) -> RunRecord:
    selectors = dict(selectors or {})
    parts = [str(metadata.get("source_file", path.name))]
    identity = selectors or {key: row[key] for key in ("run_id", "trial_id", "candidate", "model_name", "fold", "seed", "partition", "method", "decoder", "retrain_every_sessions") if key in row}
    parts.extend(f"{key}={value}" for key, value in identity.items())
    label = " · ".join(parts)
    digest = hashlib.sha256(json.dumps([str(path), family, identity, suffix], sort_keys=True, default=str).encode()).hexdigest()[:20]
    status = str(row.get("status", "complete"))
    if status == "ok":
        status = "complete"
    tables = {key: Path(value) for key, value in (tables or {}).items() if Path(value).is_file()}
    if not tables.get("equity"):
        warnings = (*warnings, "Time-series export unavailable; metrics only.")
    return RunRecord(digest, label, family, path, _numeric(row), metadata, tables, selectors, status, tuple(warnings))


def _existing(path: Path, entries: dict[str, str]) -> dict[str, Path]:
    return {key: path / relative for key, relative in entries.items() if (path / relative).is_file()}


def _advanced(path: Path) -> list[RunRecord]:
    manifest = _json(path / "manifest.json")
    config = _json(path / "config.json") if (path / "config.json").is_file() else {}
    metrics_path = path / "metrics/core_metrics.json"
    metrics = _json(metrics_path) if metrics_path.is_file() else {}
    meta = _metadata({"metadata": {**manifest, **config}})
    meta["partition"] = "train" if config.get("run_train_only", True) else "unknown"
    meta.update(benchmark_convention="legacy_close_buy_hold_no_fees", benchmark_price_column="close", benchmark_provenance="Persisted raw_market, dates aligned to equity, without costs", price_basis="raw")
    tables = _existing(path, {"equity": "backtest/equity_curve.parquet", "positions": "backtest/positions.parquet", "trades": "backtest/trades.parquet", "market": "data/raw_market.parquet"})
    missing = [key for key in ("equity", "positions", "trades") if key not in tables]
    warnings = (f"Incomplete backtest: missing {', '.join(missing)}.",) if missing else ()
    return [_record(path, "backtest", {**metrics, "status": "partial" if missing else "complete"}, meta, tables=tables, warnings=warnings)]


def _trading(path: Path, payload: dict) -> list[RunRecord]:
    records = []
    for comparison in payload.get("comparisons", []):
        if "variant" in comparison:
            variant = str(comparison["variant"])
            folder = path / variant
            metadata = _metadata(payload, {"fold": payload.get("fold"), "seed": payload.get("seed")})
            metadata.update(arm=variant, config=payload.get("variants", {}).get(variant, {}))
            metadata = _metadata({"metadata": metadata})
            tables = _existing(folder, {"equity": "equity.parquet", "positions": "positions.parquet", "trades": "trades.parquet", "orders": "orders.parquet"})
            row = dict(comparison.get("metrics", {}))
            if "equity" not in tables:
                row["status"] = "partial"
            records.append(_record(folder, "trading", row, metadata, tables=tables, suffix=variant))
            continue
        directory = path / comparison.get("directory", "")
        for arm in ("without_rules", "with_rules"):
            folder = directory / arm
            if not folder.is_dir():
                metadata = _metadata(payload, comparison.get("group", {}))
                metadata["arm"] = arm
                records.append(_record(folder, "trading", {**comparison.get(arm, {}), "status": "partial"}, metadata, suffix=arm, warnings=("Trading result directory unavailable.",)))
                continue
            metadata = _metadata(payload, comparison.get("group", {}))
            if (folder / "metadata.json").is_file():
                metadata.update(_metadata({"metadata": _json(folder / "metadata.json")}))
            metadata["arm"] = arm
            row = dict(comparison.get(arm, {}))
            if not row and (folder / "metrics.json").is_file():
                row = _json(folder / "metrics.json")
            tables = _existing(folder, {"equity": "equity.parquet", "positions": "positions.parquet", "trades": "trades.parquet", "orders": "orders.parquet"})
            if "equity" not in tables:
                row["status"] = "partial"
            records.append(_record(folder, "trading", row, metadata, tables=tables, suffix=arm))
    return records


def _direct_trading(path: Path) -> list[RunRecord]:
    row = _json(path / "metrics.json") if (path / "metrics.json").is_file() else {}
    metadata = _metadata({"metadata": _json(path / "metadata.json")}) if (path / "metadata.json").is_file() else {}
    for parent in list(path.parents)[:2]:
        for name in ("report.json", "manifest.json"):
            source = parent / name
            if not source.is_file():
                continue
            payload = _json(source)
            for key in ("prices_path", "signals_path", "inputs_sha256", "fold", "seed", "final_holdout_opened"):
                if key in payload:
                    metadata.setdefault(key, payload[key])
            variants = payload.get("variants", {})
            config = variants.get(path.name, {}) if isinstance(variants, dict) else {}
            if isinstance(config, dict):
                for key, value in config.items():
                    metadata.setdefault(key, value)
    tables = _existing(path, {"equity": "equity.parquet", "positions": "positions.parquet", "trades": "trades.parquet", "orders": "orders.parquet"})
    return [_record(path, "trading", row, metadata, tables=tables)]


def _table_records(path: Path, payload: dict, table: Path, family: str) -> list[RunRecord]:
    identities = parquet_identities(table)
    rows = payload.get("tasks", payload.get("runs", payload.get("rows", payload.get("folds", []))))
    if not isinstance(rows, list):
        rows = []
    records = []
    for identity in identities:
        matches = [row for row in rows if isinstance(row, dict) and all(row.get(key) == value for key, value in identity.items() if key != "partition" or "partition" in row)]
        row = matches[0] if len(matches) == 1 else dict(identity)
        partition = identity.get("partition")
        if partition and isinstance(row.get("metrics"), dict) and isinstance(row["metrics"].get(partition), dict):
            row = {**row, "metrics": row["metrics"][partition]}
        if identity.get("partition") == "inner" and isinstance(row.get("inner_metrics"), dict):
            row = {**row, "metrics": row["inner_metrics"]}
            row.pop("outer_metrics", None)
            row.pop("classification", None)
        metadata = _metadata(payload, {**row, **identity})
        tables = {"equity": table}
        if family == "mt5":
            decoder = identity.get("decoder", metadata.get("decoder"))
            positions = path / f"positions-{decoder}.parquet" if decoder else path / "positions.parquet"
            if decoder == "sign" and not positions.is_file() and (path / "positions_sign.parquet").is_file():
                positions = path / "positions_sign.parquet"
            if not positions.is_file() and row.get("run_dir"):
                folder = Path(row["run_dir"])
                positions = folder / f"positions-{decoder}.parquet" if decoder else folder / "positions.parquet"
            if positions.is_file():
                tables["positions"] = positions
        elif (path / "predictions.parquet").is_file():
            tables["positions"] = path / "predictions.parquet"
        records.append(_record(path, family, row, metadata, selectors=identity, tables=tables))
    return records


def _direct_mt5(path: Path) -> list[RunRecord]:
    metadata = _json(path / "metadata.json") if (path / "metadata.json").is_file() else {}
    metrics = _json(path / "portfolio_metrics.json") if (path / "portfolio_metrics.json").is_file() else {}
    if isinstance(metrics.get("model"), dict):
        metrics = {**metrics["model"], **{"benchmark_" + key: value for key, value in metrics.get("buy_hold", {}).items()}, **{key: value for key, value in metrics.items() if not isinstance(value, dict)}}
    tables = _existing(path, {"equity": "equity_curve.parquet", "positions": "positions.parquet"})
    warnings = ()
    if "equity" not in tables:
        metrics["status"] = "partial"
        warnings = ("MT5 equity export unavailable.",)
    records = [_record(path, "mt5", metrics, _metadata({"metadata": {**metadata, "decoder": metadata.get("decoder", "raw")}}), tables=tables, warnings=warnings)]
    if (path / "equity_curve_sign.parquet").is_file():
        sign_metrics = _json(path / "sign_portfolio_metrics.json") if (path / "sign_portfolio_metrics.json").is_file() else {}
        if isinstance(sign_metrics.get("model"), dict):
            sign_metrics = {**sign_metrics["model"], **{"benchmark_" + key: value for key, value in sign_metrics.get("buy_hold", {}).items()}}
        sign_tables = _existing(path, {"equity": "equity_curve_sign.parquet", "positions": "positions_sign.parquet"})
        records.append(_record(path, "mt5", sign_metrics, _metadata({"metadata": {**metadata, "decoder": "sign"}}), selectors={"decoder": "sign"}, tables=sign_tables))
    return records


def _benchmarks(path: Path, payload: dict) -> list[RunRecord]:
    family = "cv" if "folds" in payload else "benchmark"
    rows = payload.get("folds", payload.get("runs", payload.get("rows", [])))
    if not isinstance(rows, list):
        return []
    records = []
    complete = (path / "report.json").is_file()
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row = dict(row)
        metadata = _metadata(payload, row)
        daily = row.get("daily_path_artifacts")
        if isinstance(daily, dict):
            for partition, relative in daily.items():
                table = _reference(path, relative)
                partition_row = {**row, "metrics": row.get(f"{partition}_metrics", {})}
                partition_row.pop("outer_metrics", None)
                partition_row.pop("classification", None)
                selectors = {key: row[key] for key in ("candidate", "seed", "fold") if key in row}
                selectors["partition"] = partition
                partition_metadata = {**metadata, "partition": partition}
                if not table.is_file():
                    partition_row["status"] = "partial"
                records.append(_record(path, "graph", partition_row, partition_metadata, selectors=selectors, tables={"equity": table}, suffix=str(number)))
            continue
        metadata.setdefault("partition", "outer" if family == "cv" else "test")
        if family == "benchmark" and row.get("final_test_evaluated") is False:
            metadata["partition"] = "validation"
        tables = {}
        artifact = row.get("artifact_path")
        if artifact:
            artifact = _reference(path, artifact)
            if (artifact / "training_history.json").is_file():
                tables["history"] = artifact / "training_history.json"
        warnings = []
        if not complete:
            warnings.append("Incremental results; final report unavailable.")
            if row.get("status", "ok") == "ok":
                row["status"] = "partial"
        if row.get("status") == "error":
            warnings.append(str(row.get("error", "Run failed")))
        history = {}
        for source, target in (("train_loss", "train_loss"), ("inner_loss", "val_loss"), ("validation_loss", "val_loss")):
            if isinstance(row.get(source), list):
                history[target] = row[source]
        fit = row.get("fit", {})
        if isinstance(fit, dict):
            for key in ("train_loss", "val_loss"):
                if isinstance(fit.get(key), list):
                    history[key] = fit[key]
        if history:
            metadata["history"] = history
        # These are signal exports, not synthetic executed trades or equity.
        if "fold" in row and "seed" in row and "candidate" in row:
            positions = path / f"fold-{row['fold']}-seed-{row['seed']}-{row['candidate']}-positions.parquet"
            if positions.is_file():
                tables["positions"] = positions
        records.append(_record(path, family, row, metadata, tables=tables, suffix=str(number), warnings=tuple(warnings)))
    return records


def _gridsearch(path: Path, stem: str) -> list[RunRecord]:
    source = path / f"{stem}.json"
    csv = path / f"{stem}.csv"
    payload = _json(source) if source.is_file() else {}
    meta = dict(payload.get("meta", {}))
    meta.update({key: value for key, value in payload.items() if key not in ("meta", "top", "failures", "final_test", "validation_retrain_logs")})
    if "ticker" in meta:
        meta["tickers"] = [meta["ticker"]]
    rows = pd.read_csv(csv).where(lambda frame: frame.notna(), None).to_dict("records") if csv.is_file() else payload.get("top", [])
    records = []
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row = dict(row)
        partition = row.get("selection_split", payload.get("selection_split"))
        warnings = []
        if partition not in ("validation", "test", "final_test"):
            partition = "unknown"
            warnings.append("Legacy grid search does not declare its evaluation partition.")
        metadata = _metadata({"metadata": meta}, row)
        metadata.update(partition=partition, source_file=stem, kind="gridsearch")
        if not source.is_file() or not csv.is_file():
            if row.get("status", "ok") == "ok":
                row["status"] = "partial"
            warnings.append("Grid-search CSV/JSON pair incomplete; ranking may be partial.")
        if row.get("status") == "error":
            warnings.append(str(row.get("error", "Trial failed")))
        records.append(_record(path, "benchmark", row, metadata, suffix=f"{stem}/{number}", warnings=tuple(warnings)))
    final = payload.get("final_test")
    if isinstance(final, dict) and final:
        best = payload.get("best_parameters", {})
        best = best if isinstance(best, dict) else {}
        row = {**best, **{key: value for key, value in final.items() if not isinstance(value, (dict, list))}, "metrics": final.get("test_metrics", {}), "partition": "final_test", "selected": True}
        if isinstance(final.get("benchmark_comparison"), dict):
            row.update(final["benchmark_comparison"])
        metadata = _metadata({"metadata": meta}, row)
        metadata.update(partition="final_test", source_file=stem, kind="gridsearch", selected=True)
        records.append(_record(path, "benchmark", row, metadata, suffix=f"{stem}/final_test"))
    return records


def discover_runs(root) -> Catalog:
    root = Path(root).expanduser().resolve()
    catalog = Catalog()
    if not root.is_dir():
        catalog.issues.append(f"Artifacts directory unavailable: {root}")
        return catalog
    consumed = set()
    for path, names in _directories(root):
        if path in consumed:
            continue
        try:
            records = []
            grid_records = []
            grid_stems = {Path(name).stem for name in _gridsearch_files(names)}
            for stem in sorted(grid_stems):
                try:
                    grid_records.extend(_gridsearch(path, stem))
                except (OSError, ValueError, TypeError, KeyError) as error:
                    catalog.issues.append(f"{stem}: {error}")
            if "manifest.json" in names and (path / "backtest").is_dir():
                records.extend(_advanced(path))
            else:
                payload = _json(path / "report.json") if "report.json" in names else {}
                if not payload and "replay.json" in names:
                    payload = _json(path / "replay.json")
                if not payload and "metadata.json" in names:
                    payload = {"metadata": _json(path / "metadata.json")}
                if "comparisons" in payload:
                    records = _trading(path, payload)
                    consumed.update(record.path for record in records)
                elif "equity-curves.parquet" in names:
                    records = _table_records(path, payload, path / "equity-curves.parquet", "mt5")
                elif "daily_paths.parquet" in names:
                    records = _table_records(path, payload, path / "daily_paths.parquet", "graph")
                elif "equity.parquet" in names:
                    records = _direct_trading(path)
                elif "equity_curve.parquet" in names or "portfolio_metrics.json" in names or ("positions.parquet" in names and "metadata.json" in names and payload.get("metadata", {}).get("protocol", "").startswith("expanding_walk_forward")):
                    records = _direct_mt5(path)
                elif any(key in payload for key in ("folds", "runs", "rows")):
                    records = _benchmarks(path, payload)
                elif "results.json" in names:
                    rows = json.loads((path / "results.json").read_text(encoding="utf-8"))
                    if isinstance(rows, list):
                        records = _benchmarks(path, {**payload, "folds": rows})
                elif "folds.json" in names:
                    rows = json.loads((path / "folds.json").read_text(encoding="utf-8"))
                    metadata = _json(path / "metadata.json") if "metadata.json" in names else {}
                    if isinstance(rows, list):
                        records = _benchmarks(path, {"metadata": metadata, "folds": rows})
                elif "runs.csv" in names:
                    records = _benchmarks(path, {"runs": pd.read_csv(path / "runs.csv").where(lambda frame: frame.notna(), None).to_dict("records")})
            records = [*grid_records, *records]
            for record in records:
                _attach_market(record)
            catalog.records.extend(records)
            for record in records:
                if record.status in ("partial", "error"):
                    catalog.issues.append(f"{record.label}: {record.status}")
        except (OSError, ValueError, TypeError, KeyError) as error:
            catalog.issues.append(f"{path.relative_to(root) or '.'}: {error}")
    return catalog


def catalog_frame(records: Iterable[RunRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {"run_id": record.run_id, "label": record.label, "family": record.family, "status": record.status}
        for key in ("model", "candidate", "seed", "fold", "partition", "method", "decoder", "retrain_every_sessions", "tickers", "selected"):
            row[key] = record.selectors.get(key, record.metadata.get(key))
            if isinstance(row[key], (list, tuple)):
                row[key] = ", ".join(str(value) for value in row[key])
            elif isinstance(row[key], dict):
                row[key] = json.dumps(row[key], sort_keys=True, default=str)
        row.update({key: value for key, value in record.metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in row})
        for capability in ("equity", "positions", "trades", "history", "market", "orders"):
            row[f"has_{capability}"] = capability in record.tables or (capability == "history" and bool(record.metadata.get("history")))
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_value(record: RunRecord, key: str):
    meta = record.metadata
    if key == "dataset":
        value = meta.get("dataset_sha256", meta.get("dataset_hash"))
        if value is None:
            hashes = meta.get("hashes", meta.get("inputs_sha256", {}))
            if isinstance(hashes, dict):
                value = hashes.get("raw_market_hash", hashes.get("data"))
        return value
    if key == "universe":
        value = meta.get("tickers", meta.get("universe", meta.get("symbol")))
        if isinstance(value, (list, tuple)):
            return tuple(sorted(str(item) for item in value))
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True, default=str)
        return value
    if key == "period":
        split = meta.get("split")
        if isinstance(split, dict):
            return json.dumps(split, sort_keys=True, default=str), str(meta.get("outer_end"))
        start = meta.get("evaluation_start", meta.get("start"))
        end = meta.get("evaluation_end", meta.get("end"))
        if start is not None and end is not None:
            return str(start), str(end), json.dumps(meta.get("split_ratios"), default=str)
        return None
    if key == "costs":
        return tuple((name, meta[name]) for name in ("fees_bps", "slippage_bps", "cost_bps") if name in meta) or None
    if key == "annualization":
        value = meta.get("annualization", meta.get("annualization_factor"))
        return float(value) if isinstance(value, (float, int)) else value
    if key == "capital_basis":
        return meta.get("capital_basis", {"backtest": "first_recorded_equity", "graph": "initial_capital", "mt5": "initial_capital", "trading": "initial_capital"}.get(record.family))
    if key == "initial_capital":
        return meta.get("initial_capital", record.metrics.get("initial_capital"))
    return record.selectors.get(key, meta.get(key))


def comparison_issues(records: Iterable[RunRecord]) -> list[str]:
    records = list(records)
    if len(records) < 2:
        return []
    issues = []
    if len({record.family for record in records}) > 1:
        issues.append("Different result families; accounting conventions may differ.")
    for key in ("dataset", "universe", "period", "partition", "price_basis", "execution", "signal_timing", "execution_delay", "costs", "annualization", "initial_capital", "capital_basis"):
        values = [_comparison_value(record, key) for record in records]
        if any(value is None for value in values):
            issues.append(f"Comparison provenance incomplete: {key}.")
        known = {json.dumps(value, sort_keys=True, default=str) for value in values if value is not None}
        if len(known) > 1:
            issues.append(f"Comparison mismatch: {key}.")
    if any(record.status != "complete" for record in records):
        issues.append("Comparison includes incomplete or failed runs.")
    return issues
